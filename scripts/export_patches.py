"""Export parcellation geometries as NBT-core ``.patches`` files.

Each configuration selects an atlas, an (optional) simplification algorithm and
level, and (optional) boundary-filter settings, builds the boundary geometry for
both hemispheres, and writes a single ``.patches`` file that NBT-core can load
via ``nbt.topography.eeg_topography._load_parcellation_polygons``.

A ``.patches`` file is ``gzip(msgpack)`` of a dict::

    { "<region>-<hemi>": {"type": "Polygon", "x": [...], "y": [...]} }
    { "<region>-<hemi>": {"type": "MultiPolygon",
                          "children_polygons": [{"type": "Polygon", ...}, ...]} }

This is the exact encoding produced by ``nbt.serialization.write_serialized``;
we write the bytes directly so the script does not depend on NBT-core's runtime
storage-backend configuration. Coordinates already match NBT's flatmap frame
(see ``load_freesurfer_data.load_hemisphere``), so no re-projection is needed.

NBT's loader keeps only exterior rings (holes are dropped), but we still emit the
exterior of every polygon faithfully.

Usage
-----
    uv run python scripts/export_patches.py

Edit the ``CONFIGS`` list below to choose what to export. Output is written to
``results/patches/``.
"""
from __future__ import annotations

import gzip
import sys
from dataclasses import dataclass
from pathlib import Path

import msgpack
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from parcellation_boundaries.boundary import build_mesh_cached
from parcellation_boundaries.load_freesurfer_data import Atlas
from parcellation_boundaries import simplify as S

# Algorithm name → simplify dispatch key (mirrors simplification_explorer.ipynb).
ALGORITHMS = {
    'Douglas-Peucker':    S.dp,
    'Visvalingam-Whyatt': S.vw,
    'Saalfeld':           'saalfeld',
    'TopoVW':             'topovw',
    'TopoVW (modified)':  'topovw_modified',
    'de Berg':            'deberg',
}

OUT_DIR = Path('results/patches')


@dataclass
class Config:
    """One ``.patches`` export.

    algorithm / epsilon = None  → no simplification (original geometry).
    min_area / min_points = None → defaults (0 and 1, i.e. no filtering).
    """
    atlas: Atlas
    algorithm: str | None = None
    # Distance threshold for DP / VW / Saalfeld / de Berg; for TopoVW it is the
    # fraction (0–1) of interior vertices to remove.
    epsilon: float | None = None
    min_area: float | None = None
    min_points: int | None = None
    hemis: tuple[str, ...] = ('lh', 'rh')
    out: str | None = None  # output filename; None → auto-generated

    def filename(self) -> str:
        if self.out:
            return self.out
        ma = 0 if self.min_area is None else self.min_area
        mp = 1 if self.min_points is None else self.min_points
        parts = [self.atlas.name]
        if self.algorithm and self.epsilon is not None:
            algo = self.algorithm.replace(' ', '').replace('(', '').replace(')', '')
            parts.append(f'{algo}_eps{self.epsilon:g}')
        else:
            parts.append('original')
        parts.append(f'ma{ma:g}_mp{mp}')
        return '_'.join(parts) + '.patches'


# ── Configurations to export ──────────────────────────────────────────────────
# Defaults: no simplification, min_area=0, min_points=1.
CONFIGS: list[Config] = [
    Config(Atlas.DK),
    Config(Atlas.DESTRIEUX),
    Config(Atlas.DK, algorithm='TopoVW', epsilon=0.5),
]


def _simplify_mesh(mesh, data, algorithm, epsilon):
    """Apply a simplification algorithm to a mesh (see the explorer notebook)."""
    if algorithm is None or epsilon is None:
        return mesh
    fn = ALGORITHMS[algorithm]
    if fn == 'saalfeld':
        return S.simplify_saalfeld(mesh, epsilon)
    if fn == 'topovw':
        # True TopoVW: epsilon is the fraction of interior vertices to remove.
        n_remove = max(0, int(round(epsilon * S.count_interior_points(mesh))))
        return S.simplify_topovw(mesh, n_remove)
    if fn == 'topovw_modified':
        face_seeds = {n: np.column_stack([r['x'], r['y']]) for n, r in data.items()}
        return S.simplify_topovw_modified(mesh, epsilon, face_seeds=face_seeds)
    if fn == 'deberg':
        return S.simplify_deberg(mesh, epsilon)
    return S.simplify(mesh, epsilon, fn)


def _polygon_entry(poly) -> dict:
    x, y = poly.exterior.coords.xy
    return {'type': 'Polygon', 'x': list(x), 'y': list(y)}


def _geometry_to_patch(geom) -> dict | None:
    """Convert a Shapely geometry to NBT's patch dict (exterior rings only)."""
    polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
    polys = [p for p in polys if p.geom_type == 'Polygon' and not p.is_empty]
    if not polys:
        return None
    if len(polys) == 1:
        return _polygon_entry(polys[0])
    return {'type': 'MultiPolygon',
            'children_polygons': [_polygon_entry(p) for p in polys]}


def build_patches(cfg: Config) -> dict:
    """Build the region → patch dict for one configuration, both hemispheres."""
    min_area = 0 if cfg.min_area is None else cfg.min_area
    min_points = 1 if cfg.min_points is None else cfg.min_points

    patches: dict = {}
    for hemi in cfg.hemis:
        mesh, data = build_mesh_cached(
            cfg.atlas, hemi, min_area=min_area, min_points=min_points)
        mesh = _simplify_mesh(mesh, data, cfg.algorithm, cfg.epsilon)
        for name, geom in mesh.to_shapes().items():
            entry = _geometry_to_patch(geom)
            if entry is not None:
                patches[name] = entry
    return patches


def write_patches(obj: dict, path: Path) -> None:
    """Write a ``.patches`` file: gzip(msgpack), matching write_serialized."""
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = msgpack.packb(obj, use_bin_type=True)
    path.write_bytes(gzip.compress(packed))


def main() -> None:
    for cfg in CONFIGS:
        patches = build_patches(cfg)
        out_path = OUT_DIR / cfg.filename()
        write_patches(patches, out_path)
        print(f'{out_path}  ({len(patches)} regions)')


if __name__ == '__main__':
    main()

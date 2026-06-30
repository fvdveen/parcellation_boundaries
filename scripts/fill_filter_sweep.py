"""
Build and cache baseline geometry for the filter-size sweep.

For each (atlas, hemi, min_area, min_points) combination, builds the mesh
and writes the original (unsimplified) shapes and mesh object to the standard
results/geometries/ and results/meshes/ pickle files under the ('original', None)
key — exactly as evaluate_all.py does, so all downstream scripts can reuse them.

Existing ('original', None) entries are skipped. All other keys in existing
pickles (e.g. simplified variants from evaluate_all.py) are preserved.

Usage
-----
    uv run python3 fill_filter_sweep.py
    uv run python3 fill_filter_sweep.py --workers 4
    uv run python3 fill_filter_sweep.py --atlas DK YEO7
"""
from __future__ import annotations

import argparse
import pickle
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from parcellation_boundaries.boundary import build_mesh_cached
from parcellation_boundaries.load_freesurfer_data import Atlas

GEO_DIR = Path('results/geometries')
MESH_DIR = Path('results/meshes')

MIN_AREAS = [0, 0.25, 0.5, 1, 2, 5, 10, 20, 50]
MIN_POINTS = [1, 2, 3, 5, 10]


def _fill_config(atlas_name: str, hemi: str,
                 min_area: float, min_points: int,
                 force: bool = False) -> str:
    atlas = Atlas[atlas_name]
    geo_key = f'{atlas_name}_{hemi}_ma{min_area:g}_mp{min_points}'

    geo_path = GEO_DIR / f'{geo_key}.pkl'
    mesh_path = MESH_DIR / f'{geo_key}.pkl'

    geo_store = {}
    mesh_store = {}
    if geo_path.exists():
        with open(geo_path, 'rb') as fh:
            geo_store = pickle.load(fh)
    if mesh_path.exists():
        with open(mesh_path, 'rb') as fh:
            mesh_store = pickle.load(fh)

    if not force and ('original', None) in geo_store and ('original', None) in mesh_store:
        return 'skipped'

    mesh, _ = build_mesh_cached(
        atlas, hemi, min_area=min_area, min_points=min_points)
    shapes = mesh.to_shapes()

    geo_store[('original', None)] = shapes
    mesh_store[('original', None)] = mesh

    GEO_DIR.mkdir(parents=True, exist_ok=True)
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    with open(geo_path, 'wb') as fh:
        pickle.dump(geo_store, fh)
    with open(mesh_path, 'wb') as fh:
        pickle.dump(mesh_store, fh)

    return 'done'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--atlas',   nargs='+', default=None)
    parser.add_argument('--workers', type=int,
                        default=max(os.cpu_count() - 2, 1))
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing ("original", None) entries')
    ns = parser.parse_args()

    atlas_names = [a.name for a in Atlas]
    if ns.atlas:
        unknown = set(ns.atlas) - set(atlas_names)
        if unknown:
            parser.error(f'Unknown atlases: {sorted(unknown)}')
        atlas_names = ns.atlas

    configs = [
        (atlas, hemi, ma, mp)
        for atlas in atlas_names
        for hemi in ('lh', 'rh')
        for ma in MIN_AREAS
        for mp in MIN_POINTS
        if not (ma == 0 and mp == 1)
    ]
    print(f'Configs: {len(configs)}  Workers: {ns.workers}', flush=True)

    t_start = time.perf_counter()
    n_done = n_skipped = 0

    if ns.workers == 1:
        for i, (atlas, hemi, ma, mp) in enumerate(configs, 1):
            status = _fill_config(atlas, hemi, ma, mp, force=ns.force)
            if status == 'done':
                n_done += 1
            else:
                n_skipped += 1
            print(f'  [{i}/{len(configs)}] {atlas} {hemi} ma={ma:g} mp={mp}  {status}',
                  flush=True)
    else:
        with ProcessPoolExecutor(max_workers=ns.workers) as pool:
            futures = {
                pool.submit(_fill_config, a, h, ma, mp, ns.force): (a, h, ma, mp)
                for a, h, ma, mp in configs
            }
            done = 0
            for fut in as_completed(futures):
                done += 1
                a, h, ma, mp = futures[fut]
                try:
                    status = fut.result()
                    if status == 'done':
                        n_done += 1
                    else:
                        n_skipped += 1
                    print(f'  [{done}/{len(configs)}] {a} {h} ma={ma:g} mp={mp}  {status}',
                          flush=True)
                except Exception as exc:
                    print(f'  [{done}/{len(configs)}] {a} {h} ma={ma:g} mp={mp}  ERROR: {exc}',
                          flush=True)

    elapsed = time.perf_counter() - t_start
    print(f'\nDone in {elapsed:.1f}s  |  {n_done} built, {n_skipped} skipped')


if __name__ == '__main__':
    main()

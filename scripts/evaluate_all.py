#!/usr/bin/env python3
"""Batch simplification quality evaluation across all atlases and filter settings.

Outputs
-------
results/data/{atlas}_{hemi}_ma{min_area}_mp{min_points}.json
    Full evaluation and validation results for every (algorithm, epsilon) pair.
    Contains complete per-region breakdowns, overlapping-pair lists, lost/gained
    edges, etc. — everything returned by evaluate.* and validate.check_*.

results/metrics.csv
    Scalar summary derived from the JSON files; convenient for plotting and
    quick pandas analysis.  Rebuilt from all JSON files at the end of each run.

results/geometries/{atlas}_{hemi}_ma{min_area}_mp{min_points}.pkl
    Dict mapping (algorithm, epsilon) → {region_name: shapely_geom}.
    Key ('original', None) holds the unsimplified shapes for that config.

Usage
-----
    python evaluate_all.py                  # use all CPU cores
    python evaluate_all.py --workers 4
    python evaluate_all.py --workers 1      # serial, verbose per-epsilon output

Re-running is incremental: already-computed (algorithm, epsilon) pairs are
skipped.  Add new epsilons to EPSILONS_BY_ALGO and re-run to extend results.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from parcellation_boundaries.boundary import build_mesh_cached
from parcellation_boundaries.load_freesurfer_data import Atlas
from parcellation_boundaries import evaluate
from parcellation_boundaries import simplify as S
from parcellation_boundaries import validate

# ── Configuration ──────────────────────────────────────────────────────────────

HEMIS = ['lh', 'rh']
MIN_AREAS = [0]
MIN_POINTS = [1]

EPSILONS_BY_ALGO: dict[str, list[float]] = {
    # 'Douglas-Peucker':    [0.05, 0.06, 0.07, 0.085, 0.1, 0.12, 0.15, 0.175, 0.2, 0.25, 0.35, 0.5, 1.0, 2.0, 16.0, 10000],
    # 'Visvalingam-Whyatt': [0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 5.0, 16.0, 10000],
    # 'Saalfeld':           [0.05, 0.06, 0.07, 0.085, 0.1, 0.12, 0.15, 0.175, 0.2, 0.25, 0.35, 0.5, 1.0, 2.0, 16.0, 18.0, 20.0, 22.0, 24.0, 10000],
    # # 'TopoVW (modified)':  [0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 16.0],
    # 'de Berg':            [0.05, 0.06, 0.07, 0.085, 0.1, 0.12, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.2, 1.3, 1.4, 1.5, 10000],
    # # Fractions of interior vertices to remove (0.5 = paper's 50%-removal experiment)
    # 'TopoVW':             [0.5, 0.6, 0.7, 0.8, 0.9, 0.9125, 0.925, 0.9375, 0.95, 0.96, 0.975, 0.9775, 0.98, 0.9875, 1],

    'Douglas-Peucker':    [0.05, 10000],
    'Visvalingam-Whyatt': [0.005, 10000],
    'Saalfeld':           [0.05, 10000],
    'TopoVW (modified)':  [0.005, 10000],
    'de Berg':            [0.05,  10000],
    # Fractions of interior vertices to remove (0.5 = paper's 50%-removal experiment)
    'TopoVW':             [0.5, 1]
}

OUT_DIR = Path('results')
GEO_DIR = OUT_DIR / 'geometries'
MESH_DIR = OUT_DIR / 'meshes'
DATA_DIR = OUT_DIR / 'data'


# ── JSON serialisation ─────────────────────────────────────────────────────────

class _Encoder(json.JSONEncoder):
    """Convert numpy scalars/arrays to plain Python types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dispatch(algo_name: str, mesh, epsilon: float,
              face_seeds: dict | None = None):
    """Call the right simplification function by name (lambdas aren't picklable)."""
    if algo_name == 'Douglas-Peucker':
        return S.simplify(mesh, epsilon, S.dp)
    if algo_name == 'Visvalingam-Whyatt':
        return S.simplify(mesh, epsilon, S.vw)
    if algo_name == 'Saalfeld':
        return S.simplify_saalfeld(mesh, epsilon)
    if algo_name == 'TopoVW (modified)':
        return S.simplify_topovw_modified(mesh, epsilon, face_seeds=face_seeds)
    if algo_name == 'de Berg':
        return S.simplify_deberg(mesh, epsilon)
    if algo_name == 'TopoVW':
        # epsilon is a removal fraction [0, 1]; convert to point count
        n_total = S.count_interior_points(mesh)
        n_remove = max(0, int(round(epsilon * n_total)))
        return S.simplify_topovw(mesh, n_remove)
    raise ValueError(f'Unknown algorithm: {algo_name}')


def _extract_arcs(mesh) -> tuple[list, list]:
    """Split mesh arcs into (inner_arcs, outer_arcs) as coordinate lists."""
    seen: set[int] = set()
    inner, outer = [], []
    for h in mesh.half_edges:
        if id(h.arc) in seen:
            continue
        seen.add(id(h.arc))
        h_fwd = h if h.forward else h.twin
        coords = h_fwd.arc.coords.tolist()
        if h_fwd.face == '__hull__' or h_fwd.twin.face == '__hull__':
            outer.append(coords)
        else:
            inner.append(coords)
    return inner, outer


def _check_detail(name: str, r: dict) -> str:
    if r['passed']:
        return ''
    overlapping = r.get('overlapping_pairs')
    lost = r.get('lost_edges')
    gained = r.get('gained_edges')
    changed = r.get('changed')
    invalid = r.get('invalid')
    missing = r.get('missing')
    extra = r.get('extra')

    if overlapping:
        detail = (f'{len(overlapping)} pairs: '
                  + ', '.join(f'{a}&{b}({area:.2g})' for a, b, area in overlapping[:3]))
    elif lost and gained:
        detail = f'lost {len(lost)}: {lost[:2]}  gained {len(gained)}: {gained[:2]}'
    elif lost:
        detail = f'{len(lost)} lost: {lost[:2]}'
    elif gained:
        detail = f'{len(gained)} gained: {gained[:2]}'
    elif changed:
        parts = []
        for n, v in list(changed.items())[:3]:
            part = f'{n}: {v["original"]}→{v["simplified"]}'
            removed = r.get('removed_components', {}).get(n, [])
            if removed:
                locs = ', '.join(
                    f'({cx:.1f},{cy:.1f}) a={a:.2g}' for (cx, cy), a in removed[:2]
                )
                part += f' [{locs}]'
            parts.append(part)
        detail = '; '.join(parts)
    elif invalid:
        detail = ', '.join(invalid[:3])
    elif missing:
        detail = 'missing: ' + ', '.join(missing[:3])
    elif extra:
        detail = 'extra: ' + ', '.join(extra[:3])
    else:
        detail = ''

    return f'{name}({detail})' if detail else name


def _csv_row(entry: dict) -> dict:
    """Flatten one full result entry to a scalar CSV row."""
    ev = entry.get('evaluate', {})
    return {
        'atlas':        entry['atlas'],
        'hemi':         entry['hemi'],
        'min_area':     entry['min_area'],
        'min_points':   entry['min_points'],
        'algorithm':    entry['algorithm'],
        'epsilon':      entry['epsilon'],
        'orig_vertices': ev.get('orig_vertex_count'),
        'vertices':     ev.get('vertex_count', {}).get('total'),
        'n_traces':     ev.get('trace_count'),
        'vert_retain':  ev.get('vertex_retention'),
        'iou_mean':     ev.get('iou', {}).get('mean'),
        'iou_min':      ev.get('iou', {}).get('min'),
        'hd_mean':      ev.get('hausdorff', {}).get('mean'),
        'hd_max':       ev.get('hausdorff', {}).get('max'),
        'hd1_mean':     ev.get('hausdorff_one_sided', {}).get('mean'),
        'hd1_max':      ev.get('hausdorff_one_sided', {}).get('max'),
        'hd_pc_mean':         ev.get('hausdorff_per_component', {}).get('mean'),
        'hd_pc_max':          ev.get('hausdorff_per_component', {}).get('max'),
        'collapsed_fids':     len(ev.get('hausdorff_per_component', {}).get('collapsed_fids', [])),
        'n_collapsed_faces':  entry.get('validate', {}).get('collapsed_face_components', {}).get('n_collapsed'),
        'collapsed_fraction': entry.get('validate', {}).get('collapsed_face_components', {}).get('collapsed_fraction'),
        'rae_mean':     ev.get('relative_area_error', {}).get('mean'),
        'rae_max':      ev.get('relative_area_error', {}).get('max'),
        'px_accuracy':  ev.get('pixel_comparison', {}).get('accuracy'),
        'la_accuracy':  ev.get('label_accuracy', {}).get('accuracy'),
        'has_error':    entry['has_error'],
        'errors':       entry['errors'],
    }


def _backfill_combined() -> int:
    """Compute all metrics on lh+rh merged shapes, writing hemi='both' JSON files.

    Loads existing per-hemisphere geo and mesh PKLs — no re-simplification needed.
    Skips (atlas, algo, eps) triples already present in the 'both' JSON.
    Returns the number of new entries written.
    """
    patched = 0

    for lh_json_path in sorted(DATA_DIR.glob('*_lh_ma*_mp*.json')):
        stem = lh_json_path.stem                        # e.g. DK_lh_ma0_mp1
        parts = stem.split('_lh_ma')
        if len(parts) != 2:
            continue
        atlas_name = parts[0]
        ma_mp = parts[1].split('_mp')
        if len(ma_mp) != 2:
            continue
        min_area, min_points = float(ma_mp[0]), int(ma_mp[1])

        rh_stem = f'{atlas_name}_rh_ma{min_area:g}_mp{min_points}'
        both_stem = f'{atlas_name}_both_ma{min_area:g}_mp{min_points}'
        rh_json_path = DATA_DIR / f'{rh_stem}.json'
        both_json_path = DATA_DIR / f'{both_stem}.json'

        if not rh_json_path.exists():
            continue

        lh_geo_path = GEO_DIR / f'{stem}.pkl'
        rh_geo_path = GEO_DIR / f'{rh_stem}.pkl'
        lh_mesh_path = MESH_DIR / f'{stem}.pkl'
        rh_mesh_path = MESH_DIR / f'{rh_stem}.pkl'

        if not all(p.exists() for p in
                   [lh_geo_path, rh_geo_path, lh_mesh_path, rh_mesh_path]):
            continue

        tag = f'{atlas_name} both ma={min_area:g} mp={min_points}'

        # Load existing 'both' results so we can skip done pairs
        existing_results: list[dict] = []
        done_pairs: set[tuple] = set()
        if both_json_path.exists():
            with open(both_json_path) as fh:
                both_payload = json.load(fh)
            existing_results = both_payload.get('results', [])
            done_pairs = {(r['algorithm'], r['epsilon'])
                          for r in existing_results if not r.get('has_error')}

        with open(lh_geo_path,  'rb') as fh:
            lh_geo = pickle.load(fh)
        with open(rh_geo_path,  'rb') as fh:
            rh_geo = pickle.load(fh)
        with open(lh_mesh_path, 'rb') as fh:
            lh_meshes = pickle.load(fh)
        with open(rh_mesh_path, 'rb') as fh:
            rh_meshes = pickle.load(fh)

        lh_orig_shapes = lh_geo.get(('original', None))
        rh_orig_shapes = rh_geo.get(('original', None))
        lh_orig_mesh = lh_meshes.get(('original', None))
        rh_orig_mesh = rh_meshes.get(('original', None))

        if any(x is None for x in
               [lh_orig_shapes, rh_orig_shapes, lh_orig_mesh, rh_orig_mesh]):
            continue

        combined_orig = {**lh_orig_shapes, **rh_orig_shapes}
        orig_vc = evaluate.vertex_count(combined_orig)['total']

        lh_orig_inner, lh_orig_outer = _extract_arcs(lh_orig_mesh)
        rh_orig_inner, rh_orig_outer = _extract_arcs(rh_orig_mesh)

        # Point cloud for label_accuracy
        try:
            atlas_obj = Atlas[atlas_name]
            _, lh_data = build_mesh_cached(atlas_obj, 'lh',
                                           min_area=min_area, min_points=min_points)
            _, rh_data = build_mesh_cached(atlas_obj, 'rh',
                                           min_area=min_area, min_points=min_points)
            combined_data = {**lh_data, **rh_data}
            pts = np.column_stack([
                np.concatenate([r['x'] for r in combined_data.values()]),
                np.concatenate([r['y'] for r in combined_data.values()]),
            ])
            labels = np.concatenate([
                np.full(len(r['x']), name, dtype=object)
                for name, r in combined_data.items()
            ])
        except Exception:
            pts = labels = None

        common_keys = (
            {k for k in lh_geo if k != ('original', None)} &
            {k for k in rh_geo if k != ('original', None)} &
            {k for k in lh_meshes if k != ('original', None)} &
            {k for k in rh_meshes if k != ('original', None)}
        )

        todo_keys = sorted(k for k in common_keys if k not in done_pairs)
        if not todo_keys:
            print(f'  → {tag}  (all {len(done_pairs)} pairs cached, skipping)',
                  flush=True)
            continue
        print(f'  → {tag}  ({len(done_pairs)} cached, {len(todo_keys)} to run)',
              flush=True)

        new_results: list[dict] = []
        for algo_name, eps in todo_keys:

            combined_simp = {**lh_geo[(algo_name, eps)],
                             **rh_geo[(algo_name, eps)]}
            lh_simp_mesh = lh_meshes[(algo_name, eps)]
            rh_simp_mesh = rh_meshes[(algo_name, eps)]
            lh_simp_inner, lh_simp_outer = _extract_arcs(lh_simp_mesh)
            rh_simp_inner, rh_simp_outer = _extract_arcs(rh_simp_mesh)

            entry: dict = {
                'atlas':      atlas_name,
                'hemi':       'both',
                'min_area':   min_area,
                'min_points': min_points,
                'algorithm':  algo_name,
                'epsilon':    eps,
            }
            try:
                ev: dict = {
                    'orig_vertex_count':      orig_vc,
                    'vertex_count':           evaluate.vertex_count(combined_simp),
                    'trace_count':            evaluate.trace_count(combined_simp),
                    'vertex_retention':       evaluate.vertex_retention(combined_orig, combined_simp),
                    'iou':                    evaluate.iou(combined_orig, combined_simp),
                    'hausdorff':              evaluate.hausdorff(combined_orig, combined_simp),
                    'hausdorff_one_sided':    evaluate.hausdorff_one_sided(combined_orig, combined_simp),
                    'hausdorff_per_component': evaluate.hausdorff_per_component_both(
                        lh_orig_mesh, lh_simp_mesh,
                        rh_orig_mesh, rh_simp_mesh),
                    'relative_area_error':    evaluate.relative_area_error(combined_orig, combined_simp),
                    'pixel_comparison':       evaluate.pixel_comparison(combined_orig, combined_simp),
                }
                if pts is not None:
                    ev['label_accuracy'] = evaluate.label_accuracy(
                        combined_simp, pts, labels)
                entry['evaluate'] = ev

                val: dict = {
                    'completeness':   validate.check_completeness(combined_orig, combined_simp),
                    'validity':       validate.check_validity(combined_simp),
                    'component_count': validate.check_component_count(combined_orig, combined_simp),
                    'no_overlap':     validate.check_no_overlap(combined_simp),
                    'adjacency_graph': validate.check_adjacency_graph(combined_orig, combined_simp),
                    'junction_count': validate.check_junction_count_both(
                        lh_orig_inner, lh_simp_inner,
                        rh_orig_inner, rh_simp_inner),
                    'no_gaps':        validate.check_no_gaps_both(
                        lh_geo[(algo_name, eps)], lh_simp_outer,
                        rh_geo[(algo_name, eps)], rh_simp_outer),
                    'no_crossings':   validate.check_no_crossings_both(
                        lh_simp_inner, lh_simp_outer,
                        rh_simp_inner, rh_simp_outer),
                    'collapsed_face_components': validate.check_collapsed_face_components_both(
                        lh_orig_mesh, lh_simp_mesh,
                        rh_orig_mesh, rh_simp_mesh),
                }
                entry['validate'] = val

                failed = [k for k, v in val.items()
                          if not v.get('passed', True) and k != 'component_count']
                entry['has_error'] = bool(failed)
                entry['errors'] = ' | '.join(
                    _check_detail(k, val[k]) for k in failed)

            except Exception as exc:
                entry['evaluate'] = None
                entry['validate'] = None
                entry['has_error'] = True
                entry['errors'] = f'exception: {exc}'

            status = 'ERR' if entry['has_error'] else 'ok '
            print(f'       ε={eps:<6g}  {algo_name}  [{status}]', flush=True)
            new_results.append(entry)
            patched += 1

        if new_results:
            all_results = existing_results + new_results
            payload = {
                'atlas':      atlas_name,
                'hemi':       'both',
                'min_area':   min_area,
                'min_points': min_points,
                'results':    all_results,
            }
            with open(both_json_path, 'w') as fh:
                json.dump(payload, fh, cls=_Encoder)
            print(f'  ✓ {tag}  (+{len(new_results)} new, {len(all_results)} total)'
                  f'  → {both_json_path}', flush=True)

    return patched


def _rebuild_csv(csv_path: Path) -> int:
    """Rebuild metrics.csv from all JSON files in DATA_DIR. Returns row count."""
    rows = []
    for json_path in sorted(DATA_DIR.glob('*.json')):
        with open(json_path) as fh:
            payload = json.load(fh)
        rows.extend(_csv_row(e) for e in payload.get('results', []))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return len(rows)


def _scrub_algos(algos: frozenset[str]) -> int:
    """Remove entries for the given algorithms from every JSON in DATA_DIR.

    Also removes the corresponding keys from geometry and mesh pickles so
    stale shapes don't linger.  Returns the total number of entries removed.
    """
    total_removed = 0

    for json_path in sorted(DATA_DIR.glob('*.json')):
        with open(json_path) as fh:
            payload = json.load(fh)
        before = payload.get('results', [])
        after = [r for r in before if r['algorithm'] not in algos]
        n_removed = len(before) - len(after)
        if n_removed:
            payload['results'] = after
            with open(json_path, 'w') as fh:
                json.dump(payload, fh, cls=_Encoder)
            total_removed += n_removed

    for pkl_dir in (GEO_DIR, MESH_DIR):
        for pkl_path in sorted(pkl_dir.glob('*.pkl')):
            with open(pkl_path, 'rb') as fh:
                store = pickle.load(fh)
            scrubbed = {k: v for k, v in store.items()
                        if not (isinstance(k, tuple) and k[0] in algos)}
            if len(scrubbed) != len(store):
                with open(pkl_path, 'wb') as fh:
                    pickle.dump(scrubbed, fh)

    return total_removed


# ── Per-config worker ──────────────────────────────────────────────────────────

def _run_config(atlas_name: str, hemi: str,
                min_area: float, min_points: int,
                verbose: bool = False,
                rerun_algos: frozenset[str] = frozenset()) -> tuple[list[dict], int]:
    """Evaluate all algorithms for one (atlas, hemi, min_area, min_points) config.

    Returns (all_results, n_new) where all_results includes both previously
    computed and newly computed entries, and n_new is the count of new ones.
    Skips (algorithm, epsilon) pairs already present in the JSON on disk.
    Saves/updates the geometry and mesh pickles as a side effect.
    """
    atlas = Atlas[atlas_name]
    geo_key = f'{atlas_name}_{hemi}_ma{min_area:g}_mp{min_points}'
    tag = f'{atlas_name} {hemi} ma={min_area} mp={min_points}'

    json_path = DATA_DIR / f'{geo_key}.json'
    geo_path = GEO_DIR / f'{geo_key}.pkl'
    mesh_path = MESH_DIR / f'{geo_key}.pkl'

    # ── Load existing results ─────────────────────────────────────────────────
    existing_results: list[dict] = []
    done_pairs: set[tuple] = set()
    if json_path.exists():
        with open(json_path) as fh:
            payload = json.load(fh)
        existing_results = payload.get('results', [])
        done_pairs = {(r['algorithm'], r['epsilon']) for r in existing_results
                      if r['algorithm'] not in rerun_algos}

    # ── Determine what still needs running ────────────────────────────────────
    todo = [
        (algo, eps)
        for algo, epsilons in EPSILONS_BY_ALGO.items()
        for eps in epsilons
        if (algo, eps) not in done_pairs
    ]

    if not todo:
        print(f'  → {tag}  (all {len(existing_results)} pairs cached, skipping)',
              flush=True)
        return existing_results, 0

    print(f'  → {tag}  ({len(done_pairs)} cached, {len(todo)} to run)', flush=True)

    # ── Build mesh ────────────────────────────────────────────────────────────
    mesh, data = build_mesh_cached(
        atlas, hemi, min_area=min_area, min_points=min_points)
    orig = mesh.to_shapes()
    orig_verts = evaluate.vertex_count(orig)['total']
    orig_inner, orig_outer = _extract_arcs(mesh)

    pts = np.column_stack([
        np.concatenate([r['x'] for r in data.values()]),
        np.concatenate([r['y'] for r in data.values()]),
    ])
    labels = np.concatenate([
        np.full(len(r['x']), name, dtype=object) for name, r in data.items()
    ])
    face_seeds = {name: np.column_stack(
        [d['x'], d['y']]) for name, d in data.items()}

    if verbose:
        print(
            f'     original: {orig_verts:,} vertices, {len(orig)} regions', flush=True)

    # ── Load existing pickles (or start fresh) ────────────────────────────────
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    if geo_path.exists():
        with open(geo_path, 'rb') as fh:
            geo_store = pickle.load(fh)
    else:
        geo_store = {('original', None): orig}

    if mesh_path.exists():
        with open(mesh_path, 'rb') as fh:
            mesh_store = pickle.load(fh)
    else:
        mesh_store = {('original', None): mesh}

    # ── Run missing pairs ─────────────────────────────────────────────────────
    new_results: list[dict] = []

    # Group by algo so verbose output is grouped, and we only print the header once
    todo_by_algo: dict[str, list[float]] = {}
    for algo, eps in todo:
        todo_by_algo.setdefault(algo, []).append(eps)

    for algo_name, epsilons in EPSILONS_BY_ALGO.items():
        if algo_name not in todo_by_algo:
            continue
        if verbose:
            print(f'\n     {algo_name}', flush=True)

        for eps in epsilons:
            if (algo_name, eps) not in {(a, e) for a, e in todo}:
                continue

            entry: dict = {
                'atlas':      atlas_name,
                'hemi':       hemi,
                'min_area':   min_area,
                'min_points': min_points,
                'algorithm':  algo_name,
                'epsilon':    eps,
            }

            try:
                # modified variant uses interior anchor seeds; true TopoVW uses dummy points
                seeds = face_seeds if algo_name == 'TopoVW (modified)' else None
                m = _dispatch(algo_name, mesh, eps, face_seeds=seeds)
                simp = m.to_shapes()
                simp_inner, simp_outer = _extract_arcs(m)

                entry['evaluate'] = {
                    'orig_vertex_count':      orig_verts,
                    'vertex_count':           evaluate.vertex_count(simp),
                    'trace_count':            evaluate.trace_count(simp),
                    'vertex_retention':       evaluate.vertex_retention(orig, simp),
                    'iou':                    evaluate.iou(orig, simp),
                    'hausdorff':              evaluate.hausdorff(orig, simp),
                    'hausdorff_one_sided':    evaluate.hausdorff_one_sided(orig, simp),
                    'hausdorff_per_component': evaluate.hausdorff_per_component(mesh, m),
                    'relative_area_error':    evaluate.relative_area_error(orig, simp),
                    'pixel_comparison':       evaluate.pixel_comparison(orig, simp),
                    'label_accuracy':         evaluate.label_accuracy(simp, pts, labels),
                }

                entry['validate'] = {
                    'completeness':              validate.check_completeness(orig, simp),
                    'validity':                  validate.check_validity(simp),
                    'component_count':           validate.check_component_count(orig, simp),
                    'collapsed_face_components': validate.check_collapsed_face_components(mesh, m),
                    'no_overlap':                validate.check_no_overlap(simp),
                    'adjacency_graph':           validate.check_adjacency_graph(orig, simp),
                    'junction_count':            validate.check_junction_count(orig_inner, simp_inner),
                    'no_gaps':                   validate.check_no_gaps(simp, simp_outer),
                    'no_crossings':              validate.check_no_crossings(simp_inner, simp_outer),
                }

                failed = [k for k, v in entry['validate'].items()
                          if not v['passed'] and k != 'component_count']
                has_error = bool(failed)
                errors = ' | '.join(_check_detail(
                    k, entry['validate'][k]) for k in failed)

                geo_store[(algo_name, eps)] = simp
                mesh_store[(algo_name, eps)] = m

                if verbose:
                    vc = entry['evaluate']['vertex_count']['total']
                    iou_m = entry['evaluate']['iou']['mean']
                    hd_m = entry['evaluate']['hausdorff']['mean']
                    hd_pc = entry['evaluate']['hausdorff_per_component']
                    n_col = len(hd_pc['collapsed_fids'])
                    px_a = entry['evaluate']['pixel_comparison']['accuracy']
                    la_a = entry['evaluate']['label_accuracy']['accuracy']
                    status = 'ERR' if has_error else 'ok '
                    print(
                        f'       ε={eps:<6g}  verts={vc:>6,}'
                        f'  IoU={iou_m:.4f}  HD={hd_m:.4f}'
                        f'  HD_pc={hd_pc["mean"]:.4f}(col={n_col})'
                        f'  px={px_a:.4f}  la={la_a:.4f}  [{status}]',
                        flush=True,
                    )

            except Exception as exc:
                print(
                    f'       ε={eps:<6g}  {algo_name}  ERROR: {exc}', flush=True)
                entry['evaluate'] = None
                entry['validate'] = None
                has_error = True
                errors = f'exception: {exc}'

            entry['has_error'] = has_error
            entry['errors'] = errors
            new_results.append(entry)

    # ── Merge, save JSON + pickles ────────────────────────────────────────────
    kept_existing = [
        r for r in existing_results if r['algorithm'] not in rerun_algos]
    all_results = kept_existing + new_results

    payload = {
        'atlas':      atlas_name,
        'hemi':       hemi,
        'min_area':   min_area,
        'min_points': min_points,
        'results':    all_results,
    }
    with open(json_path, 'w') as fh:
        json.dump(payload, fh, cls=_Encoder)

    with open(geo_path, 'wb') as fh:
        pickle.dump(geo_store, fh)
    with open(mesh_path, 'wb') as fh:
        pickle.dump(mesh_store, fh)

    print(f'  ✓ {tag}  (+{len(new_results)} new, {len(all_results)} total)  → {json_path}',
          flush=True)
    return all_results, len(new_results)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Batch simplification evaluation')
    parser.add_argument(
        '--workers', type=int, default=max(os.cpu_count() - 1, 1),
        help='Parallel worker processes (default: cpu_count-1; use 1 for verbose serial)',
    )
    parser.add_argument(
        '--rerun', metavar='ALGO', nargs='+',
        help='Force recompute for these algorithms, ignoring cached results '
             '(e.g. --rerun TopoVW "TopoVW (modified)")',
    )
    parser.add_argument(
        '--scrub', metavar='ALGO', nargs='+',
        help='Remove results for these algorithms from all JSONs, pickles, and '
             'metrics.csv, then exit without running any evaluations',
    )
    parser.add_argument(
        '--backfill-only', action='store_true',
        help='Patch missing metrics into cached JSON entries without rerunning '
             'any simplification, then rebuild metrics.csv and exit',
    )
    ns = parser.parse_args()
    rerun_algos = frozenset(ns.rerun or [])

    OUT_DIR.mkdir(exist_ok=True)
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        (atlas.name, hemi, ma, mp)
        for atlas in Atlas
        for hemi in HEMIS
        for ma in MIN_AREAS
        for mp in MIN_POINTS
    ]

    csv_path = OUT_DIR / 'metrics.csv'
    total_new = 0

    scrub_algos = frozenset(ns.scrub or [])
    if scrub_algos:
        unknown = scrub_algos - set(EPSILONS_BY_ALGO)
        if unknown:
            parser.error(f'Unknown algorithms for --scrub: {sorted(unknown)}')
        print(f'Scrubbing: {sorted(scrub_algos)}', flush=True)
        n_removed = _scrub_algos(scrub_algos)
        csv_path = OUT_DIR / 'metrics.csv'
        n_rows = _rebuild_csv(csv_path)
        print(
            f'Removed {n_removed} entries.  CSV rebuilt: {n_rows} rows → {csv_path}')
        return

    if ns.backfill_only:
        print('Backfilling missing metrics...', flush=True)
        n = _backfill_combined()
        print(
            f'  combined (both hemispheres): {n} entries written', flush=True)
        n_rows = _rebuild_csv(csv_path)
        print(f'  CSV rebuilt: {n_rows} rows → {csv_path}')
        return

    if rerun_algos:
        unknown = rerun_algos - set(EPSILONS_BY_ALGO)
        if unknown:
            parser.error(f'Unknown algorithms for --rerun: {sorted(unknown)}')
        print(f'Force-rerun: {sorted(rerun_algos)}', flush=True)

    print(f'Configs: {len(configs)}   Workers: {ns.workers}', flush=True)

    if ns.workers == 1:
        for cfg in configs:
            _, n_new = _run_config(*cfg, verbose=True, rerun_algos=rerun_algos)
            total_new += n_new
    else:
        with ProcessPoolExecutor(max_workers=ns.workers) as pool:
            futures = {pool.submit(_run_config, *cfg, rerun_algos=rerun_algos): cfg
                       for cfg in configs}
            done = 0
            for fut in as_completed(futures):
                done += 1
                cfg = futures[fut]
                tag = f'{cfg[0]} {cfg[1]} ma={cfg[2]} mp={cfg[3]}'
                try:
                    _, n_new = fut.result()
                    total_new += n_new
                    print(
                        f'  [{done}/{len(configs)}] {tag}  +{n_new} new', flush=True)
                except Exception as exc:
                    print(
                        f'  [{done}/{len(configs)}] {tag}  FAILED: {exc}', flush=True)

    n_patched = _backfill_combined()
    if n_patched:
        print(
            f'  Combined both hemispheres: {n_patched} new entries', flush=True)

    n_rows = _rebuild_csv(csv_path)
    print(f'\nDone.  {total_new} new entries computed')
    print(f'  JSON → {DATA_DIR}/')
    print(f'  CSV  → {csv_path}  ({n_rows} rows total)')


if __name__ == '__main__':
    main()

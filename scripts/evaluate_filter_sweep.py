"""Evaluate fidelity metrics for every filter config, comparing to the
unfiltered baseline (min_area=0, min_points=1).

lh and rh are combined into a single 'both' object before evaluation —
averaging per-hemi metrics is incorrect because vertex counts, areas, and
region counts differ between hemispheres.

Reads cached meshes from results/meshes/ — run fill_filter_sweep.py first.

Output: results/filter_metrics.csv  (incremental — existing rows are skipped;
        (atlas, min_area, min_points) is the unique key)

Usage
-----
    uv run python3 evaluate_filter_sweep.py
    uv run python3 evaluate_filter_sweep.py --atlas DK YEO7
    uv run python3 evaluate_filter_sweep.py --workers 4
"""
from __future__ import annotations

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from parcellation_boundaries.boundary import build_mesh_cached
from parcellation_boundaries.load_freesurfer_data import Atlas
from parcellation_boundaries import evaluate
from parcellation_boundaries import validate

MESH_DIR = Path('results/meshes')
OUT_CSV  = Path('results/filter_metrics.csv')


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mesh_key(atlas: str, hemi: str, min_area: float, min_points: int) -> str:
    return f'{atlas}_{hemi}_ma{min_area:g}_mp{min_points}'


def _discover_configs(atlas_filter: list[str] | None) -> list[tuple]:
    """Return (atlas, min_area, min_points) for every config with both lh+rh cached."""
    configs = []
    for pkl in sorted(MESH_DIR.glob('*_lh_ma*_mp*.pkl')):
        stem  = pkl.stem
        parts = stem.split('_lh_ma')
        atlas = parts[0]
        if atlas_filter and atlas not in atlas_filter:
            continue
        ma_mp = parts[1].split('_mp')
        ma, mp = float(ma_mp[0]), int(ma_mp[1])
        rh_pkl = MESH_DIR / pkl.name.replace('_lh_', '_rh_')
        if rh_pkl.exists():
            configs.append((atlas, ma, mp))
    return configs


# ── Per-config worker ──────────────────────────────────────────────────────────

def _run_config(atlas_name: str, min_area: float, min_points: int) -> dict | None:
    def _load(hemi: str, ma: float, mp: int) -> object | None:
        pkl = MESH_DIR / f'{_mesh_key(atlas_name, hemi, ma, mp)}.pkl'
        if not pkl.exists():
            return None
        with open(pkl, 'rb') as fh:
            store = pickle.load(fh)
        return store.get(('original', None))

    ref_lh  = _load('lh', 0, 1)
    ref_rh  = _load('rh', 0, 1)
    filt_lh = _load('lh', min_area, min_points)
    filt_rh = _load('rh', min_area, min_points)

    if any(m is None for m in [ref_lh, ref_rh, filt_lh, filt_rh]):
        missing = [n for n, m in zip(['ref_lh','ref_rh','filt_lh','filt_rh'],
                                      [ref_lh, ref_rh, filt_lh, filt_rh])
                   if m is None]
        print(f'  SKIP {atlas_name} ma={min_area:g} mp={min_points}: '
              f'missing {missing}', flush=True)
        return None

    orig = {**ref_lh.to_shapes(),  **ref_rh.to_shapes()}
    filt = {**filt_lh.to_shapes(), **filt_rh.to_shapes()}


    # Point cloud for label_accuracy — from the filtered atlas data
    _, lh_data = build_mesh_cached(Atlas[atlas_name], 'lh',
                                   min_area=min_area, min_points=min_points)
    _, rh_data = build_mesh_cached(Atlas[atlas_name], 'rh',
                                   min_area=min_area, min_points=min_points)
    combined_data = {**lh_data, **rh_data}
    pts = np.column_stack([
        np.concatenate([r['x'] for r in combined_data.values()]),
        np.concatenate([r['y'] for r in combined_data.values()]),
    ])
    labels = np.concatenate([
        np.full(len(r['x']), name, dtype=object) for name, r in combined_data.items()
    ])

    ev = {
        'vertex_count':            evaluate.vertex_count(filt),
        'trace_count':             evaluate.trace_count(filt),
        'vertex_retention':        evaluate.vertex_retention(orig, filt),
        'iou':                     evaluate.iou(orig, filt),
        'hausdorff':               evaluate.hausdorff(orig, filt),
        'hausdorff_one_sided':     evaluate.hausdorff_one_sided(orig, filt),
        'hausdorff_per_component': evaluate.hausdorff_per_component_both(
                                       ref_lh, filt_lh, ref_rh, filt_rh),
        'relative_area_error':     evaluate.relative_area_error(orig, filt),
        'pixel_comparison':        evaluate.pixel_comparison(orig, filt),
        'label_accuracy':          evaluate.label_accuracy(filt, pts, labels),
    }

    val = {
        'collapsed_face_components': validate.check_collapsed_face_components_both(
            ref_lh, filt_lh, ref_rh, filt_rh),
    }

    return {
        'atlas':              atlas_name,
        'min_area':           min_area,
        'min_points':         min_points,
        'vertices':           ev['vertex_count']['total'],
        'n_traces':           ev['trace_count'],
        'vert_retain':        ev['vertex_retention'],
        'iou_mean':           ev['iou']['mean'],
        'iou_min':            ev['iou']['min'],
        'hd_mean':            ev['hausdorff']['mean'],
        'hd_max':             ev['hausdorff']['max'],
        'hd1_mean':           ev['hausdorff_one_sided']['mean'],
        'hd1_max':            ev['hausdorff_one_sided']['max'],
        'hd_pc_mean':         ev['hausdorff_per_component']['mean'],
        'hd_pc_max':          ev['hausdorff_per_component']['max'],
        'collapsed_fids':     len(ev['hausdorff_per_component']['collapsed_fids']),
        'n_collapsed_faces':  val['collapsed_face_components']['n_collapsed'],
        'collapsed_fraction': val['collapsed_face_components']['collapsed_fraction'],
        'rae_mean':           ev['relative_area_error']['mean'],
        'rae_max':            ev['relative_area_error']['max'],
        'px_accuracy':        ev['pixel_comparison']['accuracy'],
        'la_accuracy':        ev['label_accuracy']['accuracy'],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--atlas',   nargs='+', default=None)
    parser.add_argument('--workers', type=int,
                        default=max(os.cpu_count() - 1, 1))
    ns = parser.parse_args()

    configs = _discover_configs(ns.atlas)

    done_keys: set[tuple] = set()
    if OUT_CSV.exists():
        df_ex = pd.read_csv(OUT_CSV)
        if 'hemi' in df_ex.columns:
            print('WARNING: existing CSV has per-hemi rows — it will be '
                  'overwritten with combined both-hemi results.')
            done_keys = set()
        else:
            for _, row in df_ex.iterrows():
                done_keys.add((row['atlas'], float(row['min_area']),
                               int(row['min_points'])))

    todo = [(a, ma, mp) for a, ma, mp in configs
            if (a, ma, mp) not in done_keys]

    print(f'Configs available : {len(configs)}')
    print(f'Already done      : {len(done_keys)}')
    print(f'To run            : {len(todo)}')
    if not todo:
        print('Nothing to do.')
        return

    new_rows: list[dict] = []

    if ns.workers == 1:
        for i, (atlas, ma, mp) in enumerate(todo, 1):
            print(f'  [{i}/{len(todo)}] {atlas} ma={ma:g} mp={mp}', flush=True)
            row = _run_config(atlas, ma, mp)
            if row is not None:
                new_rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=ns.workers) as pool:
            futures = {
                pool.submit(_run_config, a, ma, mp): (a, ma, mp)
                for a, ma, mp in todo
            }
            done = 0
            for fut in as_completed(futures):
                done += 1
                a, ma, mp = futures[fut]
                try:
                    row = fut.result()
                    if row is not None:
                        new_rows.append(row)
                    print(f'  [{done}/{len(todo)}] {a} ma={ma:g} mp={mp}  done',
                          flush=True)
                except Exception as exc:
                    print(f'  [{done}/{len(todo)}] {a} ma={ma:g} mp={mp}  '
                          f'ERROR: {exc}', flush=True)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if OUT_CSV.exists() and 'hemi' not in pd.read_csv(OUT_CSV).columns:
            df_new = pd.concat([pd.read_csv(OUT_CSV), df_new], ignore_index=True)
        OUT_CSV.parent.mkdir(exist_ok=True)
        df_new.to_csv(OUT_CSV, index=False)
        print(f'\n{len(new_rows)} new rows  →  {OUT_CSV}  ({len(df_new)} total)')
    else:
        print('\nNo new rows written.')


if __name__ == '__main__':
    main()

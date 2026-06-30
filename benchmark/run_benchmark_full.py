"""
Full rendering benchmark suite.

Runs three renderer benchmarks (nbt, fast, svg) over all (atlas, algo, eps)
configurations, plus a filter-sweep benchmark over all (atlas, min_area,
min_points) configurations with original (unsimplified) geometry only.

Normal benchmark outputs (one file per renderer):
    results/rendering_benchmark.csv          — nbt / plot_flatmap_html
    results/rendering_benchmark_fast.csv     — FlatmapRenderer (Python lists)
    results/rendering_benchmark_fast_v2.csv  — FlatmapRenderer v2 (numpy arrays)
    results/rendering_benchmark_svg.csv      — SVGRenderer

Filter-sweep outputs (one file per renderer, incremental):
    results/rendering_benchmark_filter_nbt.csv
    results/rendering_benchmark_filter_fast.csv
    results/rendering_benchmark_filter_fast_v2.csv
    results/rendering_benchmark_filter_svg.csv

Usage
-----
    uv run python run_benchmark_full.py
    uv run python run_benchmark_full.py --workers 4
    uv run python run_benchmark_full.py --reps 5
    uv run python run_benchmark_full.py --atlas DK YEO7
    uv run python run_benchmark_full.py --renderers fast fast_v2 svg
    uv run python run_benchmark_full.py --skip-normal
    uv run python run_benchmark_full.py --skip-filter
    uv run python run_benchmark_full.py --min-area 1 --min-points 2
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'NBT-core'))

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import traceback
import pickle
import argparse



GEO_DIR = Path('results/geometries')
OUT_DIR = Path('results')

RENDERER_CSVS = {
    'nbt':     OUT_DIR / 'rendering_benchmark.csv',
    'fast':    OUT_DIR / 'rendering_benchmark_fast.csv',
    'fast_v2': OUT_DIR / 'rendering_benchmark_fast_v2.csv',
    'svg':     OUT_DIR / 'rendering_benchmark_svg.csv',
}
FILTER_CSVS = {
    'nbt':     OUT_DIR / 'rendering_benchmark_filter_nbt.csv',
    'fast':    OUT_DIR / 'rendering_benchmark_filter_fast.csv',
    'fast_v2': OUT_DIR / 'rendering_benchmark_filter_fast_v2.csv',
    'svg':     OUT_DIR / 'rendering_benchmark_filter_svg.csv',
}

RNG = np.random.default_rng(0)
N_PLOTS = 5 * 5   # 5 biomarkers × 5 frequency bands


# ── Geometry helpers ────────────────────────────────────────────────────────────

def _vertex_count(shapes: dict) -> int:
    total = 0
    for geom in shapes.values():
        for p in (geom.geoms if hasattr(geom, 'geoms') else [geom]):
            total += len(p.exterior.coords)
            for ring in p.interiors:
                total += len(ring.coords)
    return total


def _trace_count(shapes: dict) -> int:
    return sum(len(g.geoms) if hasattr(g, 'geoms') else 1 for g in shapes.values())


def _edge_count(shapes: dict) -> int:
    total = 0
    for geom in shapes.values():
        for p in (geom.geoms if hasattr(geom, 'geoms') else [geom]):
            total += len(p.exterior.coords) - 1
    return total


# ── Normal benchmark — per-renderer benchmark functions ────────────────────────

def _benchmark_shapes_nbt(shapes: dict, reps: int) -> dict:
    from nbt.visualization.plots.topography import plot_flatmap_html
    region_labels = list(shapes.keys())
    grids = [RNG.random(len(region_labels)).astype(np.float32)
             for _ in range(N_PLOTS)]

    times_render: list[float] = []
    times_serialize: list[float] = []
    total_json_bytes = None

    for _ in range(reps):
        t0 = time.perf_counter()
        figs = [
            plot_flatmap_html(
                biomarker_data=grids[i],
                biomarker_title=f'biomarker_{i}',
                test_type='benchmark',
                region_labels=region_labels,
                region_to_polygon_map=shapes,
                cmap_name='RdBu_r',
                cmin_val=0.0,
                cmax_val=1.0,
            )
            for i in range(N_PLOTS)
        ]
        t1 = time.perf_counter()
        jsons = [f.to_json() for f in figs]
        t2 = time.perf_counter()
        times_render.append(t1 - t0)
        times_serialize.append(t2 - t1)
        if total_json_bytes is None:
            total_json_bytes = sum(len(j) for j in jsons)

    totals = [r + s for r, s in zip(times_render, times_serialize)]
    return {
        'init_time':            0.0,
        'cold_time':            float(totals[0]),
        'render_time_mean':     float(np.mean(times_render)),
        'render_time_min':      float(np.min(times_render)),
        'serialize_time_mean':  float(np.mean(times_serialize)),
        'serialize_time_min':   float(np.min(times_serialize)),
        'total_time_mean':      float(np.mean(totals)),
        'total_time_min':       float(np.min(totals)),
        'json_bytes':           total_json_bytes,
        'n_plots':              N_PLOTS,
        'n_vertices':           _vertex_count(shapes),
        'n_traces':             _trace_count(shapes),
    }


def _benchmark_shapes_fast(shapes: dict, reps: int) -> dict:
    from parcellation_boundaries.benchmark.fast_renderer import FlatmapRenderer
    region_labels = list(shapes.keys())
    grids = [RNG.random(len(region_labels)).astype(np.float32)
             for _ in range(N_PLOTS)]

    t0 = time.perf_counter()
    renderer = FlatmapRenderer(region_labels, shapes)
    init_time = time.perf_counter() - t0

    times_render: list[float] = []
    times_serialize: list[float] = []
    total_json_bytes = None
    output_n_traces = None

    for _ in range(reps):
        t0 = time.perf_counter()
        figs = [
            renderer.render(
                biomarker_data=grids[i],
                biomarker_title=f'biomarker_{i}',
                cmap_name='RdBu_r',
                cmin_val=0.0,
                cmax_val=1.0,
            )
            for i in range(N_PLOTS)
        ]
        t1 = time.perf_counter()
        jsons = [f.to_json() for f in figs]
        t2 = time.perf_counter()
        times_render.append(t1 - t0)
        times_serialize.append(t2 - t1)
        if total_json_bytes is None:
            total_json_bytes = sum(len(j) for j in jsons)
            output_n_traces = len(figs[0].data)

    totals = [r + s for r, s in zip(times_render, times_serialize)]
    return {
        'init_time':            float(init_time),
        'cold_time':            float(init_time + totals[0]),
        'render_time_mean':     float(np.mean(times_render)),
        'render_time_min':      float(np.min(times_render)),
        'serialize_time_mean':  float(np.mean(times_serialize)),
        'serialize_time_min':   float(np.min(times_serialize)),
        'total_time_mean':      float(np.mean(totals)),
        'total_time_min':       float(np.min(totals)),
        'json_bytes':           total_json_bytes,
        'n_plots':              N_PLOTS,
        'n_vertices':           _vertex_count(shapes),
        'n_traces':             _trace_count(shapes),
        'output_n_traces':      output_n_traces,
    }


def _benchmark_shapes_svg(shapes: dict, reps: int) -> dict:
    from parcellation_boundaries.benchmark.svg_renderer import SVGRenderer
    region_labels = list(shapes.keys())
    grids = [RNG.random(len(region_labels)).astype(np.float32)
             for _ in range(N_PLOTS)]

    t0 = time.perf_counter()
    renderer = SVGRenderer(region_labels, shapes)
    init_time = time.perf_counter() - t0

    times_render: list[float] = []
    total_bytes = None

    for _ in range(reps):
        t0 = time.perf_counter()
        svgs = [
            renderer.render_svg(
                biomarker_data=grids[i],
                biomarker_title=f'biomarker_{i}',
                cmap_name='RdBu_r',
                cmin_val=0.0,
                cmax_val=1.0,
            )
            for i in range(N_PLOTS)
        ]
        times_render.append(time.perf_counter() - t0)
        if total_bytes is None:
            total_bytes = sum(len(s.encode()) for s in svgs)

    return {
        'init_time':            float(init_time),
        'cold_time':            float(init_time + times_render[0]),
        'render_time_mean':     float(np.mean(times_render)),
        'render_time_min':      float(np.min(times_render)),
        'serialize_time_mean':  0.0,
        'serialize_time_min':   0.0,
        'total_time_mean':      float(np.mean(times_render)),
        'total_time_min':       float(np.min(times_render)),
        'svg_bytes':            total_bytes,
        'n_plots':              N_PLOTS,
        'n_vertices':           _vertex_count(shapes),
        'n_traces':             _trace_count(shapes),
    }


def _benchmark_shapes_fast_v2(shapes: dict, reps: int) -> dict:
    from parcellation_boundaries.benchmark.fast_renderer_v2 import FlatmapRenderer
    region_labels = list(shapes.keys())
    grids = [RNG.random(len(region_labels)).astype(np.float32)
             for _ in range(N_PLOTS)]

    t0 = time.perf_counter()
    renderer = FlatmapRenderer(region_labels, shapes)
    init_time = time.perf_counter() - t0

    times_render: list[float] = []
    times_serialize: list[float] = []
    total_json_bytes = None
    output_n_traces = None

    for _ in range(reps):
        t0 = time.perf_counter()
        figs = [
            renderer.render(
                biomarker_data=grids[i],
                biomarker_title=f'biomarker_{i}',
                cmap_name='RdBu_r',
                cmin_val=0.0,
                cmax_val=1.0,
            )
            for i in range(N_PLOTS)
        ]
        t1 = time.perf_counter()
        jsons = [f.to_json() for f in figs]
        t2 = time.perf_counter()
        times_render.append(t1 - t0)
        times_serialize.append(t2 - t1)
        if total_json_bytes is None:
            total_json_bytes = sum(len(j) for j in jsons)
            output_n_traces = len(figs[0].data)

    totals = [r + s for r, s in zip(times_render, times_serialize)]
    return {
        'init_time':            float(init_time),
        'cold_time':            float(init_time + totals[0]),
        'render_time_mean':     float(np.mean(times_render)),
        'render_time_min':      float(np.min(times_render)),
        'serialize_time_mean':  float(np.mean(times_serialize)),
        'serialize_time_min':   float(np.min(times_serialize)),
        'total_time_mean':      float(np.mean(totals)),
        'total_time_min':       float(np.min(totals)),
        'json_bytes':           total_json_bytes,
        'n_plots':              N_PLOTS,
        'n_vertices':           _vertex_count(shapes),
        'n_traces':             _trace_count(shapes),
        'output_n_traces':      output_n_traces,
    }


_NORMAL_BENCH = {
    'nbt':     _benchmark_shapes_nbt,
    'fast':    _benchmark_shapes_fast,
    'fast_v2': _benchmark_shapes_fast_v2,
    'svg':     _benchmark_shapes_svg,
}


# ── Normal benchmark — per-atlas worker ────────────────────────────────────────

def _process_atlas_normal(args: tuple) -> list[dict]:
    atlas_name, lh_pkl, rh_pkl, reps, atlas_idx, n_atlases, renderer = args

    with open(lh_pkl, 'rb') as fh:
        lh_store: dict = pickle.load(fh)
    with open(rh_pkl, 'rb') as fh:
        rh_store: dict = pickle.load(fh)

    keys = sorted(lh_store.keys() & rh_store.keys(),
                  key=lambda k: (k[0] or '', k[1] or 0))
    print(
        f'  [{atlas_idx}/{n_atlases}] {atlas_name}  ({len(keys)} configs) ...', flush=True)

    orig_n_verts = None
    for k in keys:
        if k[0] in (None, 'original'):
            orig_n_verts = _vertex_count(
                {**lh_store.get(k, {}), **rh_store.get(k, {})})
            break

    bench_fn = _NORMAL_BENCH[renderer]
    rows = []
    for key in keys:
        lh_shapes = lh_store.get(key, {})
        rh_shapes = rh_store.get(key, {})
        if not lh_shapes and not rh_shapes:
            continue

        combined = {**lh_shapes, **rh_shapes}
        n_verts_lh = _vertex_count(lh_shapes)
        n_verts_rh = _vertex_count(rh_shapes)
        n_edges_lh = _edge_count(lh_shapes)
        n_edges_rh = _edge_count(rh_shapes)

        try:
            metrics = bench_fn(combined, reps)
        except Exception as exc:
            print(f'    ERROR {key}: {exc}')
            continue

        rows.append({
            'atlas':         atlas_name,
            'algorithm':     key[0],
            'epsilon':       key[1],
            'n_vertices_lh': n_verts_lh,
            'n_vertices_rh': n_verts_rh,
            'n_edges_lh':    n_edges_lh,
            'n_edges_rh':    n_edges_rh,
            'vert_retain':   (metrics['n_vertices'] / orig_n_verts if orig_n_verts else None),
            **metrics,
        })

    return rows


def run_normal_benchmark(atlas_filter: list[str] | None, reps: int, workers: int,
                         min_area: float, min_points: int,
                         renderers: list[str]) -> None:
    ma_str = f'ma{min_area:g}'
    mp_str = f'mp{min_points}'

    pkl_files = sorted(GEO_DIR.glob(f'*_lh_{ma_str}_{mp_str}.pkl'))
    if not pkl_files:
        raise FileNotFoundError(
            f'No geometry files matching {GEO_DIR}/*_lh_{ma_str}_{mp_str}.pkl\n'
            'Run evaluate_all.py first.')

    atlas_names = [f.stem.replace(
        f'_lh_{ma_str}_{mp_str}', '') for f in pkl_files]
    if atlas_filter:
        atlas_names = [a for a in atlas_names if a in set(atlas_filter)]
        if not atlas_names:
            raise ValueError(f'No atlases found matching: {atlas_filter}')

    # Base args without index/total — those are added per-renderer after filtering
    atlas_bases = []
    for name in atlas_names:
        lh_pkl = GEO_DIR / f'{name}_lh_{ma_str}_{mp_str}.pkl'
        rh_pkl = GEO_DIR / f'{name}_rh_{ma_str}_{mp_str}.pkl'
        if not lh_pkl.exists() or not rh_pkl.exists():
            print(f'  {name}: missing pkl, skip')
            continue
        atlas_bases.append((name, lh_pkl, rh_pkl, reps))

    OUT_DIR.mkdir(exist_ok=True)

    for renderer in renderers:
        out_csv = RENDERER_CSVS[renderer]
        print(
            f'\n=== Normal benchmark: {renderer} (workers={workers}, reps={reps}) ===')

        done_atlases: set[str] = set()
        if out_csv.exists():
            df_existing = pd.read_csv(out_csv)
            mask = ((df_existing['min_area'] == min_area) &
                    (df_existing['min_points'] == min_points))
            done_atlases = set(df_existing.loc[mask, 'atlas'])

        todo = [(name, lh_pkl, rh_pkl, reps)
                for name, lh_pkl, rh_pkl, reps in atlas_bases
                if name not in done_atlases]

        print(
            f'  {len(atlas_bases)} atlases, {len(done_atlases)} already done, {len(todo)} to run')
        if not todo:
            print('  Nothing to do.')
            continue

        worker_args = [
            (name, lh_pkl, rh_pkl, reps, i, len(todo), renderer)
            for i, (name, lh_pkl, rh_pkl, reps) in enumerate(todo, 1)
        ]

        def _append_atlas(atlas_rows: list[dict]) -> None:
            for row in atlas_rows:
                row['min_area'] = min_area
                row['min_points'] = min_points
            df_new = pd.DataFrame(atlas_rows)
            if out_csv.exists():
                df_new = pd.concat(
                    [pd.read_csv(out_csv), df_new], ignore_index=True)
            df_new.to_csv(out_csv, index=False)

        if workers == 1:
            for a in worker_args:
                _append_atlas(_process_atlas_normal(a))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(
                    _process_atlas_normal, a): a[0] for a in worker_args}
                for fut in as_completed(futures):
                    atlas_name = futures[fut]
                    try:
                        _append_atlas(fut.result())
                    except Exception as exc:
                        print(f'  ERROR {atlas_name}: {exc}')

        df = pd.read_csv(out_csv)
        print(f'Saved {len(df)} rows → {out_csv}')
        orig = df[df['algorithm'] == 'original']
        simp = df[df['algorithm'] != 'original']
        if not orig.empty and not simp.empty:
            print(
                f'  Original:   {orig["render_time_mean"].mean()*1000:.1f} ms mean render')
            print(
                f'  Simplified: {simp["render_time_mean"].mean()*1000:.1f} ms mean render')


def _run_filter_config(pkl_lh: Path, pkl_rh: Path,
                       atlas_name: str, min_area: float, min_points: int,
                       reps: int, renderer: str) -> dict:
    with open(pkl_lh, 'rb') as fh:
        lh_store = pickle.load(fh)
    with open(pkl_rh, 'rb') as fh:
        rh_store = pickle.load(fh)

    lh_shapes = lh_store.get(('original', None), {})
    rh_shapes = rh_store.get(('original', None), {})
    combined = {**lh_shapes, **rh_shapes}

    metrics = _NORMAL_BENCH[renderer](combined, reps)
    return {
        'atlas':      atlas_name,
        'min_area':   min_area,
        'min_points': min_points,
        'n_regions':  len(combined),
        **metrics,
    }


def run_filter_benchmark(atlas_filter: list[str] | None, reps: int, workers: int,
                         renderers: list[str]) -> None:
    lh_pkls = sorted(GEO_DIR.glob('*_lh_ma*_mp*.pkl'))
    if not lh_pkls:
        raise SystemExit(
            f'No geometry files in {GEO_DIR}/ — run fill_filter_sweep.py first.')

    configs = []
    for lh_pkl in lh_pkls:
        stem = lh_pkl.stem
        parts = stem.split('_lh_ma')
        atlas = parts[0]
        ma_mp = parts[1].split('_mp')
        ma, mp = float(ma_mp[0]), int(ma_mp[1])
        rh_pkl = GEO_DIR / lh_pkl.name.replace('_lh_', '_rh_')
        if not rh_pkl.exists():
            continue
        configs.append((atlas, ma, mp, lh_pkl, rh_pkl))

    if atlas_filter:
        configs = [(a, ma, mp, lh, rh) for a, ma, mp, lh, rh in configs
                   if a in set(atlas_filter)]

    for renderer in renderers:
        out_csv = FILTER_CSVS[renderer]
        print(f'\n=== Filter sweep: {renderer} ===')

        done_keys: set[tuple] = set()
        if out_csv.exists():
            df_existing = pd.read_csv(out_csv)
            for _, row in df_existing.iterrows():
                done_keys.add((row['atlas'], float(
                    row['min_area']), int(row['min_points'])))

        todo = [(a, ma, mp, lh, rh) for a, ma, mp, lh, rh in configs
                if (a, ma, mp) not in done_keys]

        print(
            f'  {len(configs)} configs, {len(done_keys)} already done, {len(todo)} to run')
        if not todo:
            print('  Nothing to do.')
            continue

        new_rows: list[dict] = []

        if workers == 1:
            for i, (atlas, ma, mp, lh_pkl, rh_pkl) in enumerate(todo, 1):
                print(
                    f'  [{i}/{len(todo)}] {atlas} ma={ma:g} mp={mp}', flush=True)
                new_rows.append(_run_filter_config(lh_pkl, rh_pkl, atlas, ma, mp,
                                                   reps, renderer))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_run_filter_config, lh, rh, a, ma, mp, reps, renderer): (a, ma, mp)
                    for a, ma, mp, lh, rh in todo
                }
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    a, ma, mp = futures[fut]
                    try:
                        new_rows.append(fut.result())
                        print(
                            f'  [{done}/{len(todo)}] {a} ma={ma:g} mp={mp}  done', flush=True)
                    except Exception as exc:
                        print(f'  [{done}/{len(todo)}] {a} ma={ma:g} mp={mp}  ERROR: {exc}',
                              flush=True)
                        print(traceback.format_exception(exc))

        if new_rows:
            df_new = pd.DataFrame(new_rows)
            if out_csv.exists():
                df_new = pd.concat(
                    [pd.read_csv(out_csv), df_new], ignore_index=True)
            OUT_DIR.mkdir(exist_ok=True)
            df_new.to_csv(out_csv, index=False)
            print(f'  {len(new_rows)} new rows → {out_csv} ({len(df_new)} total)')
        else:
            print('  No new rows written.')


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--atlas',       nargs='+', default=None,
                        help='Restrict to these atlas names')
    parser.add_argument('--min-area',    type=float, default=0,
                        help='Min polygon area filter for normal benchmark pkl lookup')
    parser.add_argument('--min-points',  type=int,   default=1,
                        help='Min polygon points filter for normal benchmark pkl lookup')
    parser.add_argument('--reps',        type=int,   default=3,
                        help='Repetitions per cell')
    parser.add_argument('--workers',     type=int,   default=1,
                        help='Parallel atlases (each runs serially inside its process)')
    parser.add_argument('--renderers',   nargs='+',
                        choices=['nbt', 'fast', 'fast_v2', 'svg'],
                        default=['nbt', 'fast', 'fast_v2', 'svg'], metavar='RENDERER',
                        help='Renderers to benchmark (default: all four)')
    parser.add_argument('--skip-normal', action='store_true',
                        help='Skip the normal (all algo/eps) benchmark')
    parser.add_argument('--skip-filter', action='store_true',
                        help='Skip the filter-sweep benchmark')
    ns = parser.parse_args()

    if not ns.skip_normal:
        run_normal_benchmark(
            atlas_filter=ns.atlas,
            reps=ns.reps,
            workers=ns.workers,
            min_area=ns.min_area,
            min_points=ns.min_points,
            renderers=ns.renderers,
        )

    if not ns.skip_filter:
        run_filter_benchmark(
            atlas_filter=ns.atlas,
            reps=ns.reps,
            workers=ns.workers,
            renderers=ns.renderers,
        )


if __name__ == '__main__':
    main()

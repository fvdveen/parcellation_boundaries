#!/usr/bin/env python3
"""
Per-atlas summary table + headline numbers for the section 4.1 claims.

Structure (refactored so figures are produced in one centralised sweep):

  STAGE 1  ingest        - load metrics.csv + per-region Hausdorff JSON,
                           detect the topology-preserving algorithm, load
                           parcellation-area scales.
  STAGE 2  cache         - CurveStore caches every (atlas, algorithm, column)
                           retention curve once; every figure reads from it
                           instead of regrouping the dataframe.
  STAGE 3  compute       - one pass builds the per-atlas figure bundle:
                             * floors / reach ratio / TopoVW peel-off
                             * THE single IoU-spread series (see below)
                             * Hausdorff ranking aggregates
  STAGE 4  report        - the print_* functions only format the bundle.

Single IoU-spread definition (fixes the r* inconsistency)
---------------------------------------------------------
Spread is computed exactly once per atlas, over a common retention grid that
spans the UNION of the reference algorithms' ranges. At every grid point we
record both the spread (max IoU - min IoU over the algorithms PRESENT there)
and the count of algorithms that reached that point. Every spread-derived
figure - r*, the all-support max, the partial-support max - is read off this
one series with an explicit `support` filter, so they can no longer disagree
about which retentions were included. See iou_spread_series() and the helpers.

Outputs:
  1. Per-atlas table - floors per algorithm, reach ratio, IoU spread,
     r* (agreement threshold), TopoVW peel-off.
  2. IoU spread section - per atlas r* (full support), the all-support max and
     the partial-support max (each tagged with the algorithm count), plus the
     max r* across atlases and the "spread stays < 0.01 above 0.05" tally.
  3. Floor ordering - how many atlases satisfy the expected ordering
     (DP/VW < Saalfeld/deBerg < TopoVW).
  4. Headline numbers - cross-atlas ranges for each claim.
  5. Hausdorff ranking - ranked over shared support (like-for-like); plus a
     full-range check that reveals where de Berg is beaten near the floors,
     now reporting each overtake band's maximum depth as %_of_map.
  6. Area calibration - per-atlas and global total-parcellation-area scales,
     and mean HD expressed BOTH raw and divided by sqrt(total area).
  7. HD threshold crossing, map-scale r*, worst-region tail, four-reference
     magnitude, baseline-pair convergence.

Usage:
    uv run python3 summarize_claims.py --csv results/metrics.csv

TODO(area): parcel areas are loaded by load_total_areas() (primary; total
parcellation area per atlas) and the optional load_parcel_areas() (per-region,
secondary). Both currently return None, so the area-calibration and
threshold-crossing-vs-sqrt(area) blocks are skipped; everything else runs
unchanged. load_total_areas can be a simple lookup of the six measured totals.
See those functions for the exact interface.
"""
import argparse
import json
import pickle
from pathlib import Path

from shapely.ops import unary_union

import numpy as np
import pandas as pd

GRID_N = 500
RANK_N = 5000
MIN_AREA = 0
MIN_POINTS = 1
HD_REGION_COL = 'hd_pc_region_mean_max'
# 95th-pct over regions (tail check)
HD_REGION_P95_COL = 'hd_pc_region_p95_max'
HD_REGION_MAX_COL = 'hd_pc_region_max_max'   # worst single region (tail check)
DATA_DIR = Path('results/data')
MESH_DIR = Path('results/meshes')

# Baseline pairs: (baseline, its topology-aware descendant)
PAIRS = [('Douglas-Peucker', 'Saalfeld'),
         ('Visvalingam-Whyatt', 'TopoVW')]
# Within-pair convergence thresholds (fraction of the pair's operating-range
# median HD). Swept so the 4/6-vs-2/6 split can be shown robust to the choice.
CONV_KS = [0.01, 0.02, 0.03, 0.05]
# Operating-range window (excludes near-floor noise) for pair convergence.
OP_LO, OP_HI = 0.10, 0.40

# --- IoU spread (single definition) -----------------------------------------
# All spread-derived figures read off ONE per-atlas series. These constants are
# the only knobs; r* and the "settled" tally are explicit functions of them.
SPREAD_THRESHOLD = 0.01   # IoU agreement tolerance: r* is the highest retention
#                           at which spread still reaches this.
SPREAD_CUTOFF = 0.05      # retention above which the "spread stays < threshold"
#                           claim (the 5/6 claim) is evaluated.
SPREAD_SUPPORT = 'full'   # default support for the headline r*: 'full' = every
#                           reference present; 'any' = >=2 references present.

# --- Area calibration --------------------------------------------------------
# Hausdorff is a length; we express it on the scale of the rendered map. The
# parcellation is drawn as a WHOLE (regions are not viewed in isolation), so the
# perceptually relevant scale is the TOTAL parcellation extent, not an individual
# parcel. We report HD raw AND divided by sqrt(total parcellation area), the
# characteristic linear extent of the whole map; HD / sqrt(total area) is the
# boundary excursion as a fraction of map width.
#
# Total area is a single measured number per atlas (no per-region distribution,
# so no mean/median or small-denominator question), and a reporting lens only:
# it does NOT change the raw-HD analysis (rankings, crossover, within-pair).
#
# OPTIONAL secondary: a per-parcel scale (sqrt of mean/median parcel area),
# reported only if per-region areas are also supplied via load_parcel_areas().
# Per-parcel normalization is NOT applied per region before averaging - that
# reintroduces a small-denominator blow-up (tiny parcels dominate the mean), the
# same pathology that makes the VW/TopoVW %-vs-deBerg numbers unusable.
PRIMARY_AREA_DENOM = 'sqrt_total_area'
# Threshold-crossing diagnostic: retention below which HD still exceeds T.
RAW_HD_THRESHOLDS = [1.0]        # raw coordinate units
# as a fraction of sqrt(total area), if present
AREA_FRAC_THRESHOLDS = [0.01]
# Map-scale Hausdorff r*: fractions of sqrt(total area) to sweep. The 1% level
# gives the "average worst-case deviation < 1% of map width above r*" claim;
# the lower levels are the discriminating versions.
HD_RSTAR_FRACS = [0.01, 0.005, 0.0025]   # 1%, 0.5%, 0.25% of map width
# Operating retentions at which to report the worst single region (tail check).
WORST_REGION_RETENTIONS = [0.1, 0.2, 0.3]

SHORT = {
    'Douglas-Peucker':    'DP',
    'Visvalingam-Whyatt': 'VW',
    'Saalfeld':           'Saalfeld',
    'de Berg':            'deBerg',
    'TopoVW':             'TopoVW',
}


def _short(name):
    return SHORT.get(name, name)


# === STAGE 1: AREA INGESTION =================================================
def load_total_areas(atlases):
    """Return total parcellation area per atlas, or None if not yet available.

    PRIMARY area scale. The whole parcellation is rendered at once, so boundary
    excursions are judged against the size of the entire map, not a single parcel.

    Loads results/meshes/{atlas}_{hemi}_ma{MIN_AREA}_mp{MIN_POINTS}.pkl for
    both hemispheres, calls to_shapes() on the original mesh, and sums the
    Shapely polygon areas. Atlases with missing pickle files are skipped.
    """
    suffix = f'_ma{MIN_AREA:g}_mp{MIN_POINTS}'
    result = {}
    for atl in atlases:
        total = 0.0
        found = True
        for hemi in ('lh', 'rh'):
            p = MESH_DIR / f'{atl}_{hemi}{suffix}.pkl'
            if not p.exists():
                found = False
                break
            with open(p, 'rb') as f:
                obj = pickle.load(f)
            orig = obj[('original', None)]
            shapes = orig.to_shapes()
            polys = list(shapes.values())
            hemi_area = sum(s.area for s in polys)
            union_area = unary_union(polys).area
            overlap = hemi_area - union_area
            if overlap > 1e-6 * hemi_area:
                print(f'  WARNING: {atl} {hemi} regions overlap '
                      f'(overlap={overlap:.4f}, {overlap/hemi_area*100:.3f}%)')
            total += hemi_area
        if found:
            result[atl] = total
    return result or None


def load_parcel_areas(atlases):
    """OPTIONAL secondary: per-region parcel areas, or None.

    Only needed for the secondary per-parcel scale columns. The PRIMARY
    calibration uses total parcellation area (see load_total_areas). Returns
    {atlas: {region_id: area}} with region_id matching the keys in
    hpc['region_max']. Returning None skips only the per-parcel columns.
    """
    suffix = f'_ma{MIN_AREA:g}_mp{MIN_POINTS}'
    result = {}
    for atl in atlases:
        regions = {}
        found = True
        for hemi in ('lh', 'rh'):
            p = MESH_DIR / f'{atl}_{hemi}{suffix}.pkl'
            if not p.exists():
                found = False
                break
            with open(p, 'rb') as f:
                obj = pickle.load(f)
            orig = obj[('original', None)]
            for region_id, shape in orig.to_shapes().items():
                regions[region_id] = shape.area
        if found:
            result[atl] = regions
    return result or None


def compute_area_scales(total_areas, parcel_areas=None):
    """Per-atlas and global area scales from total parcellation area (primary)
    and, optionally, per-region areas (secondary per-parcel scale).

    Global = mean total area across atlases (the '~11000 on average' figure);
    per-atlas totals are expected to be similar, so global is a cross-check and
    the natural denominator for the pooled cross-atlas HD table.
    """
    per_atlas = {}
    for atl, A in total_areas.items():
        if not (np.isfinite(A) and A > 0):
            continue
        s = {'total_area': float(A), 'sqrt_total_area': float(np.sqrt(A))}
        if parcel_areas and atl in parcel_areas:
            a = np.asarray([v for v in parcel_areas[atl].values()
                            if np.isfinite(v) and v > 0], float)
            if a.size:
                s['n_regions'] = int(a.size)
                s['mean_parcel_area'] = float(np.mean(a))
                s['median_parcel_area'] = float(np.median(a))
                s['sqrt_mean_parcel'] = float(np.sqrt(np.mean(a)))
                s['sqrt_median_parcel'] = float(np.sqrt(np.median(a)))
        per_atlas[atl] = s
    totals = [s['total_area'] for s in per_atlas.values()]
    global_scale = None
    if totals:
        gmt = float(np.mean(totals))
        global_scale = {'mean_total_area': gmt,
                        'sqrt_mean_total_area': float(np.sqrt(gmt))}
    return per_atlas, global_scale


def print_area_scales(per_atlas, global_scale):
    print(f'\n{"="*70}')
    print('PARCELLATION-AREA SCALES (coordinate space)')
    print('  sqrt_total_area = characteristic linear extent of the WHOLE map')
    print('                    (HD is reported as a fraction of this)')
    print('  per-parcel columns appear only if per-region areas were supplied')
    print(f'{"="*70}')
    tbl = pd.DataFrame(per_atlas).T
    cols = ['total_area', 'sqrt_total_area', 'n_regions',
            'median_parcel_area', 'sqrt_median_parcel',
            'mean_parcel_area', 'sqrt_mean_parcel']
    tbl = tbl[[c for c in cols if c in tbl.columns]]
    print(tbl.round(2).to_string())
    if global_scale:
        g = global_scale
        print(f'\n  GLOBAL (mean total area across atlases):')
        print(f'    mean_total_area={g["mean_total_area"]:.1f}  '
              f'sqrt={g["sqrt_mean_total_area"]:.2f}')


# === STAGE 2: CURVE CACHE ====================================================
def raw_curve_col(d, col):
    """Mean curve (vert_retain -> col) for one (atlas, algorithm) slice."""
    d = (d[['vert_retain', col]].dropna()
         .groupby('vert_retain')[col].mean().reset_index())
    return d['vert_retain'].to_numpy(), d[col].to_numpy()


class CurveStore:
    """Caches per-(atlas, algorithm, column) retention curves so each figure
    consumes precomputed curves instead of regrouping the dataframe. This is
    the one place curves are built; the spread/HD/peel code all reads here."""

    def __init__(self, df):
        self.df = df
        self._cache = {}

    def get(self, atlas, algo, col):
        key = (atlas, algo, col)
        if key not in self._cache:
            sub = self.df[(self.df.atlas == atlas) &
                          (self.df.algorithm == algo)]
            self._cache[key] = raw_curve_col(sub, col)
        return self._cache[key]

    def iou(self, atlas, algo):
        return self.get(atlas, algo, 'iou_mean')

    def curves(self, atlas, algos, col):
        """dict {algo: (x, y)} for a set of algorithms on one column."""
        return {a: self.get(atlas, a, col) for a in algos}


# === STAGE 3a: THE SINGLE IoU-SPREAD DEFINITION ==============================
def iou_spread_series(curves, lo, hi, n=GRID_N):
    """THE single IoU-spread series; every spread figure is read off this.

    curves : dict {algo: (x, y)} of IoU-vs-retention curves, ONE per reference
             algorithm. TopoVW is excluded by the caller (spread is a
             reference-agreement quantity).
    lo, hi : retention window. The caller passes the UNION of the references'
             ranges, so low-retention partial-support points (where only some
             references reached) are part of the same series and merely carry a
             smaller `count`.

    Each curve is interpolated onto a common grid with NaN outside its own
    range. At every grid point i:
        count[i]  = number of references with finite IoU at i
        spread[i] = max IoU - min IoU over the references present at i
                    (NaN where count[i] < 2, since a spread needs two curves)

    A point is FULL support where count == n_algos, PARTIAL where
    2 <= count < n_algos. Returns (grid, spread, count, n_algos).
    """
    present = [a for a in curves if len(curves[a][0])]
    n_algos = len(present)
    grid = np.linspace(lo, hi, n)
    if n_algos == 0:
        return grid, np.full(n, np.nan), np.zeros(n, int), 0
    mat = np.vstack([np.interp(grid, *curves[a], left=np.nan, right=np.nan)
                     for a in present])
    count = np.sum(np.isfinite(mat), axis=0)
    spread = np.full(n, np.nan)
    enough = count >= 2
    if enough.any():
        sub = mat[:, enough]
        spread[enough] = np.nanmax(sub, axis=0) - np.nanmin(sub, axis=0)
    return grid, spread, count, n_algos


def _support_mask(spread, count, n_algos, support):
    """Boolean mask selecting in-support, finite-spread grid points.
    support='full' -> all references present; 'any' -> at least two."""
    base = (count == n_algos) if support == 'full' else (count >= 2)
    return base & np.isfinite(spread)


def rstar_from_series(grid, spread, count, n_algos,
                      threshold=SPREAD_THRESHOLD, support=SPREAD_SUPPORT):
    """r* = highest in-support retention at which spread still reaches
    `threshold`; above it, spread < threshold for all in-support points.
    Returns (r_star, settled_everywhere). If spread is below threshold across
    the whole in-support range, r* is the lowest in-support retention and
    settled_everywhere is True."""
    mask = _support_mask(spread, count, n_algos, support)
    if not mask.any():
        return np.nan, False
    g, s = grid[mask], spread[mask]
    over = np.where(s >= threshold)[0]
    if not len(over):
        return float(g[0]), True
    return float(g[over[-1]]), False


def max_spread_from_series(grid, spread, count, n_algos, support):
    """Max spread over the chosen support, with its retention and the
    algorithm count at that point (informative for support='any', where the
    maximiser may be a partial-support, e.g. 3-of-4, point)."""
    mask = _support_mask(spread, count, n_algos, support)
    if not mask.any():
        return dict(spread=np.nan, ret=np.nan, count=0)
    idx = np.where(mask)[0]
    j = idx[int(np.argmax(spread[idx]))]
    return dict(spread=float(spread[j]), ret=float(grid[j]),
                count=int(count[j]))


def spread_settled_above(grid, spread, count, n_algos,
                         cutoff=SPREAD_CUTOFF, threshold=SPREAD_THRESHOLD,
                         support=SPREAD_SUPPORT):
    """True if spread < threshold at every in-support point with retention >
    cutoff (the per-atlas form of the '5/6' claim)."""
    mask = _support_mask(spread, count, n_algos, support) & (grid > cutoff)
    if not mask.any():
        return True
    return bool(np.all(spread[mask] < threshold))


def compute_spread_per_atlas(store, atlases, pack):
    """One spread series per atlas (references only), plus the figures derived
    from it. Returns {atlas: {grid, spread, count, n_algos, r_star, settled,
    max_full, max_any, settled_above}}."""
    out = {}
    for atl in atlases:
        curves = {a: store.iou(atl, a) for a in pack}
        xs = [x for x, _ in curves.values() if len(x)]
        if not xs:
            continue
        lo = float(min(x.min() for x in xs))   # union of reference ranges
        hi = float(max(x.max() for x in xs))
        grid, spread, count, n = iou_spread_series(curves, lo, hi)
        r_star, settled = rstar_from_series(grid, spread, count, n,
                                            support='full')
        out[atl] = dict(
            grid=grid, spread=spread, count=count, n_algos=n,
            r_star=r_star, settled=settled,
            max_full=max_spread_from_series(grid, spread, count, n, 'full'),
            max_any=max_spread_from_series(grid, spread, count, n, 'any'),
            settled_above=spread_settled_above(grid, spread, count, n),
        )
    return out


# === STAGE 3b: SHARED-SUPPORT + PEEL-OFF + HD HELPERS ========================
def shared_support_range(curves):
    """[lo, hi] over which every curve has data: lo = max of floors, hi = min of
    ceilings. (nan, nan) if they do not overlap."""
    floors = [x.min() for x, _ in curves if len(x)]
    ceils = [x.max() for x, _ in curves if len(x)]
    if not floors or not ceils:
        return np.nan, np.nan
    lo, hi = max(floors), min(ceils)
    return (lo, hi) if lo < hi else (np.nan, np.nan)


def topo_peeloff(store, atl, pack, topo, threshold=0.01, n=600):
    pack_curves = [store.iou(atl, a) for a in pack]
    tx, ty = store.iou(atl, topo)
    if not len(tx):
        return dict(peel_ret=np.nan, topo_floor=np.nan,
                    max_deficit=np.nan, max_deficit_ret=np.nan)
    t_floor, t_ceil = float(tx.min()), float(tx.max())
    grid = np.linspace(t_ceil, t_floor, n)
    pack_mat = np.vstack([np.interp(grid, x, y, left=np.nan, right=np.nan)
                          for x, y in pack_curves])
    pack_mean = np.nanmean(pack_mat, axis=0)
    topo_i = np.interp(grid, tx, ty, left=np.nan, right=np.nan)
    deficit = pack_mean - topo_i
    valid = np.isfinite(deficit)
    if not valid.any():
        return dict(peel_ret=np.nan, topo_floor=t_floor,
                    max_deficit=np.nan, max_deficit_ret=np.nan)
    sep = deficit > threshold
    peel = np.nan
    for i in range(len(grid)):
        if not valid[i]:
            continue
        if sep[i] and np.all(sep[i:][valid[i:]]):
            peel = float(grid[i])
            break
    j = int(np.nanargmax(np.where(valid, deficit, np.nan)))
    return dict(peel_ret=peel, topo_floor=t_floor,
                max_deficit=float(deficit[j]), max_deficit_ret=float(grid[j]))


def champ_lowest_over(grid, cv, champ, algos):
    """(always_lowest, compared, losers) over a grid: is champ the strict min
    everywhere both it and >=1 other have data; which algos beat it where they do."""
    always, compared, losers = True, False, set()
    for r in grid:
        vals = {a: float(np.interp(r, *cv[a], left=np.nan, right=np.nan))
                for a in algos}
        pres = [a for a in algos if np.isfinite(vals[a])]
        if champ not in pres or len(pres) < 2:
            continue
        compared = True
        champ_val = vals[champ]
        beaten_by = [a for a in pres if a != champ and vals[a] < champ_val]
        if beaten_by:
            always = False
            losers.update(beaten_by)
    return always, compared, losers


# === STAGE 1: DATA LOADING ===================================================
def load_data(csv_path):
    """Load + filter metrics, merge per-region Hausdorff aggregates, and detect
    the algorithm sets. Returns (df, pack, topo, algos, atlases)."""
    df = pd.read_csv(csv_path)
    df = df[(df['hemi'] == 'both') &
            (df['min_area'] == MIN_AREA) &
            (df['min_points'] == MIN_POINTS)].copy()

    suffix = f'_ma{MIN_AREA:g}_mp{MIN_POINTS}'
    region_rows = []
    for p in sorted(DATA_DIR.glob(f'*{suffix}.json')):
        payload = json.loads(p.read_text())
        for r in payload.get('results', []):
            if r['hemi'] != 'both':
                continue
            hpc = (r.get('evaluate') or {}).get('hausdorff_per_component')
            if not hpc or not hpc.get('region_max'):
                continue
            rmv = list(hpc['region_max'].values())
            region_rows.append({
                'atlas':            r['atlas'],
                'hemi':             r['hemi'],
                'algorithm':        r['algorithm'],
                'epsilon':          r['epsilon'],
                HD_REGION_COL:      float(np.mean(rmv)),
                HD_REGION_P95_COL:  float(np.percentile(rmv, 95)),
                HD_REGION_MAX_COL:  float(np.max(rmv)),
            })
    if region_rows:
        df = df.merge(pd.DataFrame(region_rows),
                      on=['atlas', 'hemi', 'algorithm', 'epsilon'], how='left')

    topo = next((a for a in df['algorithm'].unique()
                 if 'topo' in str(a).lower()), None)
    if topo is None:
        raise SystemExit('Could not detect topology-preserving algorithm')
    pack = sorted(a for a in df['algorithm'].unique() if a != topo)
    algos = pack + [topo]
    atlases = sorted(df['atlas'].unique())
    return df, pack, topo, algos, atlases


# === STAGE 3c: PER-ATLAS FIGURE BUNDLE =======================================
def build_per_atlas_table(df, store, spread, atlases, pack, topo, algos):
    """Assemble the per-atlas table. IoU spread + r* are read from the single
    spread series (so they match the headline by construction); floors, reach
    ratio and peel-off are computed here."""
    rows = []
    for atl in atlases:
        sub = df[df.atlas == atl]
        algo_floors = {}
        for a in algos:
            s = sub[sub.algorithm == a]['vert_retain']
            algo_floors[a] = float(s.min()) if len(s) else np.nan
        pack_floor = min((algo_floors[a] for a in pack
                          if np.isfinite(algo_floors[a])), default=np.nan)
        topo_floor = algo_floors[topo]
        reach_ratio = (topo_floor / pack_floor
                       if np.isfinite(pack_floor) and pack_floor > 0 else np.nan)
        peel = topo_peeloff(store, atl, pack, topo)
        sp = spread.get(atl)

        row = {'atlas': atl}
        for a in algos:
            row[f'floor[{_short(a)}]'] = algo_floors[a]
        row['reach_ratio'] = reach_ratio
        row['iou_spread'] = sp['max_full']['spread'] if sp else np.nan
        row['@ret'] = sp['max_full']['ret'] if sp else np.nan
        row['r_star'] = sp['r_star'] if sp else np.nan
        row['peel_ret'] = peel['peel_ret']
        row['max_deficit'] = peel['max_deficit']
        rows.append(row)
    return pd.DataFrame(rows).set_index('atlas')


# === STAGE 4: REPORT =========================================================
def print_per_atlas_table(tbl, pack, topo):
    print(f'\n{"="*70}')
    print('PER-ATLAS TABLE')
    print(f'  pack: {[_short(a) for a in pack]}   topo: {_short(topo)}')
    print('  floor[X]     = minimum vert_retain X achieves')
    print('  reach_ratio  = floor[TopoVW] / min(pack floors)')
    print('  iou_spread   = single-definition full-support max spread (pack), @ret = retention at max')
    print('  r_star       = highest full-support retention with pack spread >= 0.01')
    print('  peel_ret     = retention below which TopoVW stays >0.01 under pack mean')
    print('  max_deficit  = largest IoU gap of TopoVW below pack mean')
    print(f'{"="*70}')
    print(tbl.round(4).to_string())


def print_spread_section(spread, atlases, pack):
    """#1 + #2: the single IoU-spread definition made explicit.

    Per atlas: r* (full support), the all-support max spread, and the
    partial-support-inclusive max spread tagged with its algorithm count.
    Aggregates: max r* across atlases (and which atlas sets it), the
    'spread < threshold above the cutoff' tally (the 5/6 claim), and the
    cross-atlas partial-support max (the 3-of-4 point the all-four column can
    never show)."""
    if not spread:
        print('\n[IoU spread section skipped: no reference curves]')
        return
    n_refs = max(d['n_algos'] for d in spread.values())
    print(f'\n{"="*70}')
    print('IoU SPREAD  (single definition; reference algorithms only: '
          f'{[_short(a) for a in pack]})')
    print('  spread[r] = max IoU - min IoU over the references PRESENT at r')
    print('  count[r]  = how many references reached r '
          f'(full support = {n_refs})')
    print(f'  threshold = {SPREAD_THRESHOLD}   settled-above cutoff = '
          f'{SPREAD_CUTOFF}')
    print('  r*(full)        = highest FULL-support retention with spread >= threshold')
    print('  maxspread_full  = max spread over full-support points (@ret)')
    print('  maxspread_any   = max spread incl. PARTIAL-support points (@ret, n@)')
    print(f'{"="*70}')
    header = ('  ' + f'{"atlas":<14}{"r*(full)":>9}'
              f'{"maxspr_full":>12}{"@ret":>7}'
              f'{"maxspr_any":>12}{"@ret":>7}{"n@":>4}')
    print(header)
    for atl in atlases:
        d = spread.get(atl)
        if d is None:
            continue
        mf, ma = d['max_full'], d['max_any']
        print('  ' + f'{atl:<14}{d["r_star"]:>9.4f}'
              f'{mf["spread"]:>12.4f}{mf["ret"]:>7.3f}'
              f'{ma["spread"]:>12.4f}{ma["ret"]:>7.3f}{ma["count"]:>4d}')

    # Aggregate 1: max r* across atlases + which atlas sets it.
    rstars = [(atl, d['r_star']) for atl, d in spread.items()
              if np.isfinite(d['r_star'])]
    if rstars:
        atl_max, r_max = max(rstars, key=lambda kv: kv[1])
        print(f'\n  -> max r*(full) across atlases: {r_max:.4f}  '
              f'(set by {atl_max})')
    # Aggregate 2: the 5/6-style settled-above tally.
    n_settled = sum(1 for d in spread.values() if d['settled_above'])
    print(f'  -> full-support spread < {SPREAD_THRESHOLD} above retention '
          f'{SPREAD_CUTOFF} in {n_settled}/{len(spread)} atlases')
    # Aggregate 3: cross-atlas partial-support max (the 3-of-4 point).
    any_items = [(atl, d['max_any']) for atl, d in spread.items()
                 if np.isfinite(d['max_any']['spread'])]
    if any_items:
        atl_a, ma = max(any_items, key=lambda kv: kv[1]['spread'])
        tag = (f'{ma["count"]}/{n_refs} references present'
               + ('  <- PARTIAL-support point (not in any maxspr_full column)'
                  if ma['count'] < n_refs else '  (full support)'))
        print(f'  -> partial-support max spread: {ma["spread"]:.4f} @ ret='
              f'{ma["ret"]:.3f} in {atl_a}, {tag}')


def print_floor_ordering(tbl, atlases, pack, topo):
    dp_vw = ['Douglas-Peucker', 'Visvalingam-Whyatt']
    constrained = ['Saalfeld', 'de Berg']
    n_atl = len(atlases)

    def _floor(atl, a):
        return tbl.loc[atl, f'floor[{_short(a)}]']

    lt_constrained = sum(
        1 for atl in atlases
        if all(_floor(atl, lgt) < _floor(atl, con)
               for lgt in dp_vw for con in constrained
               if np.isfinite(_floor(atl, lgt)) and np.isfinite(_floor(atl, con))))
    lt_topo = sum(
        1 for atl in atlases
        if all(_floor(atl, p) < _floor(atl, topo)
               for p in pack
               if np.isfinite(_floor(atl, p)) and np.isfinite(_floor(atl, topo))))
    lt_topo_constrained = sum(
        1 for atl in atlases
        if all(_floor(atl, c) < _floor(atl, topo)
               for c in constrained
               if np.isfinite(_floor(atl, c)) and np.isfinite(_floor(atl, topo))))

    print(f'\n{"="*70}')
    print('FLOOR ORDERING CHECK')
    print(f'{"="*70}')
    print(
        f'  DP/VW floor < Saalfeld/deBerg floor   : {lt_constrained}/{n_atl} atlases')
    print(
        f'  Saalfeld/deBerg floor < TopoVW floor   : {lt_topo_constrained}/{n_atl} atlases')
    print(
        f'  all pack floors < TopoVW floor         : {lt_topo}/{n_atl} atlases')


def print_headline_numbers(tbl, spread):
    reach_ratio = tbl['reach_ratio']
    iou_spread = tbl['iou_spread']
    r_star_vals = tbl['r_star']
    peel_vals = tbl['peel_ret'].dropna()
    deficit_vals = tbl['max_deficit'].dropna()
    worst_s_atlas = iou_spread.idxmax()

    print(f'\n{"="*70}')
    print('HEADLINE NUMBERS')
    print(f'{"="*70}')
    print('\n  IoU agreement (pack, full support; single spread definition):')
    print(
        f'    spread range:  {iou_spread.min():.4f} - {iou_spread.max():.4f} IoU')
    print(f'    worst atlas:   {worst_s_atlas}  '
          f'({iou_spread.max():.4f} @ ret={tbl.loc[worst_s_atlas, "@ret"]:.3f})')
    print(f'    r* range:      {r_star_vals.min():.3f} - {r_star_vals.max():.3f}  '
          f'(pack agrees within {SPREAD_THRESHOLD} above {r_star_vals.max():.3f} '
          f'in all atlases)')
    n_settled = sum(1 for d in spread.values() if d['settled_above'])
    print(f'                   spread < {SPREAD_THRESHOLD} above {SPREAD_CUTOFF} '
          f'in {n_settled}/{len(spread)} atlases')
    print('\n  TopoVW reach gap (reach_ratio = topo_floor / pack_floor):')
    print(f'    ratio range:   {reach_ratio.min():.2f}x - {reach_ratio.max():.2f}x  '
          f'(median {reach_ratio.median():.2f}x)')
    print('\n  TopoVW peel-off (0.01 IoU threshold below pack mean):')
    if len(peel_vals):
        print(f'    peel_ret range:   {peel_vals.min():.3f} - {peel_vals.max():.3f}  '
              f'(TopoVW matches pack above {peel_vals.max():.3f} in worst atlas)')
        print(
            f'    max_deficit range: {deficit_vals.min():.3f} - {deficit_vals.max():.3f}')
    else:
        print('    TopoVW never separates >0.01 from the pack mean in any atlas.')


# --- Hausdorff ranking (compute, then print) ---------------------------------
def compute_hd_ranking(store, col, algos, atlases):
    """Rank over the SHARED-SUPPORT range per atlas (all algorithms present) so
    the ranking is a like-for-like N-way comparison. Returns the aggregate
    bundle reused by the calibration / full-range blocks (compute only - no
    printing - so the numbers cannot drift from what is reported)."""
    rank_sum = {a: 0.0 for a in algos}
    rank_count = {a: 0 for a in algos}
    hd_sum = {a: 0.0 for a in algos}
    hd_sum_atlas = {atl: {a: 0.0 for a in algos} for atl in atlases}
    hd_count_atlas = {atl: {a: 0 for a in algos} for atl in atlases}
    for atl in atlases:
        cv = store.curves(atl, algos, col)
        lo, hi = shared_support_range(list(cv.values()))
        if not np.isfinite(lo):
            continue
        for r in np.linspace(lo, hi, RANK_N):
            vals = {a: float(np.interp(r, *cv[a], left=np.nan, right=np.nan))
                    for a in algos}
            pres = [a for a in algos if np.isfinite(vals[a])]
            if len(pres) < 2:
                continue
            for rk, a in enumerate(sorted(pres, key=lambda a: vals[a]), 1):
                rank_sum[a] += rk
                rank_count[a] += 1
                hd_sum[a] += vals[a]
                hd_sum_atlas[atl][a] += vals[a]
                hd_count_atlas[atl][a] += 1
    champ = min(algos,
                key=lambda a: rank_sum[a] / rank_count[a]
                if rank_count[a] else float('inf'))

    shared_count = 0
    shared_losers = {}
    for atl in atlases:
        cv = store.curves(atl, algos, col)
        lo, hi = shared_support_range(list(cv.values()))
        if np.isfinite(lo):
            g = np.linspace(lo, hi, GRID_N)
            always, compared, losers = champ_lowest_over(g, cv, champ, algos)
            if compared and always:
                shared_count += 1
            if losers:
                shared_losers[atl] = losers

    def _mean(s, c):
        return s / c if c else float('nan')

    rank_tbl = pd.DataFrame(
        {_short(a): {'mean_rank': _mean(rank_sum[a], rank_count[a]),
                     'mean_hd':   _mean(hd_sum[a],   rank_count[a]),
                     'n_samples': rank_count[a]}
         for a in algos}).T.sort_values('mean_rank')
    rank_tbl['n_samples'] = rank_tbl['n_samples'].astype(int)

    return {
        'champ': champ, 'algos': algos, 'atlases': atlases, 'col': col,
        'rank_sum': rank_sum, 'rank_count': rank_count,
        'rank_tbl': rank_tbl,
        'hd_sum_atlas': hd_sum_atlas, 'hd_count_atlas': hd_count_atlas,
        'shared_count': shared_count, 'shared_losers': shared_losers,
    }


def print_hd_ranking(store, hd_out, n_atl):
    """Format the precomputed ranking bundle (no recomputation of values)."""
    if hd_out is None:
        col = HD_REGION_COL
        print(f'\n[Hausdorff ranking skipped: {col!r} not available]')
        return
    champ = hd_out['champ']
    algos = hd_out['algos']
    atlases = hd_out['atlases']
    col = hd_out['col']
    rank_tbl = hd_out['rank_tbl']
    rank_sum, rank_count = hd_out['rank_sum'], hd_out['rank_count']
    hd_sum_atlas, hd_count_atlas = hd_out['hd_sum_atlas'], hd_out['hd_count_atlas']
    shared_count, shared_losers = hd_out['shared_count'], hd_out['shared_losers']

    def _mean(s, c):
        return s / c if c else float('nan')

    def _shared_grid(atl):
        cv = store.curves(atl, algos, col)
        lo, hi = shared_support_range(list(cv.values()))
        if not np.isfinite(lo):
            return None
        return np.linspace(lo, hi, GRID_N)

    def _beater_ranges(atl, beater, grid):
        cv = store.curves(atl, algos, col)
        champ_vi = np.array([float(np.interp(r, *cv[champ], left=np.nan, right=np.nan))
                             for r in grid])
        beat_vi = np.array([float(np.interp(r, *cv[beater], left=np.nan, right=np.nan))
                            for r in grid])
        valid = np.isfinite(champ_vi) & np.isfinite(beat_vi)
        below = valid & (beat_vi < champ_vi)
        diff = np.where(valid, champ_vi - beat_vi, np.nan)
        ranges, in_run, run_idx = [], False, []
        for i, v in enumerate(below):
            if v and not in_run:
                run_start, in_run, run_idx = grid[i], True, [i]
            elif v and in_run:
                run_idx.append(i)
            elif not v and in_run:
                rd = diff[run_idx]
                rd = rd[np.isfinite(rd)]
                ranges.append((run_start, grid[i - 1],
                               float(np.max(rd)) if len(rd) else np.nan,
                               float(np.mean(rd)) if len(rd) else np.nan))
                in_run, run_idx = False, []
        if in_run:
            rd = diff[run_idx]
            rd = rd[np.isfinite(rd)]
            ranges.append((run_start, grid[-1],
                           float(np.max(rd)) if len(rd) else np.nan,
                           float(np.mean(rd)) if len(rd) else np.nan))
        return ranges

    print(f'\n{"="*70}')
    print(f'HAUSDORFF RANKING  (metric: {col})')
    print(f'{"="*70}')
    print(f'  champion (lowest mean rank over shared support): {_short(champ)}'
          f'  (mean rank {rank_sum[champ]/rank_count[champ]:.3f})')
    print(f'  lowest at ALL points over SHARED SUPPORT (like-for-like): '
          f'{shared_count}/{n_atl} atlases  <- headline claim')
    shared_beaters = sorted(
        {a for losers in shared_losers.values() for a in losers})
    if shared_beaters:
        print(f'\n  Retention sub-bands where each beater is strictly below '
              f'{_short(champ)} (shared support):')
        for beater in shared_beaters:
            print(f'    {_short(beater)}:')
            any_band = False
            for atl in atlases:
                if beater not in shared_losers.get(atl, set()):
                    continue
                grid = _shared_grid(atl)
                if grid is None:
                    continue
                ranges = _beater_ranges(atl, beater, grid)
                if ranges:
                    segs = ',  '.join(
                        f'[{lo:.3f}, {hi:.3f}] max_diff={max_d:.4f} mean_diff={mean_d:.4f}'
                        for lo, hi, max_d, mean_d in ranges)
                    print(f'      {atl:<14} {segs}')
                    any_band = True
            if not any_band:
                print(f'      (no bands found)')

    print(
        f'\n  mean rank + mean HD (over shared support, RANK_N={RANK_N}, lower = better):')
    print(rank_tbl.round(4).to_string())

    champ_hd = rank_tbl.loc[_short(champ), 'mean_hd']
    rank_tbl_pct = rank_tbl.copy()
    rank_tbl_pct['pct_vs_champ'] = (
        rank_tbl_pct['mean_hd'] / champ_hd - 1.0) * 100
    print(
        f'\n  mean HD relative to {_short(champ)} (shared support, % higher = worse):')
    print(rank_tbl_pct[['mean_hd', 'pct_vs_champ']].round(2).to_string())

    pct_rows = {}
    for atl in atlases:
        ch = _mean(hd_sum_atlas[atl][champ], hd_count_atlas[atl][champ])
        if not np.isfinite(ch) or ch == 0:
            continue
        row = {f'{_short(champ)}_mean_hd': ch}
        for a in algos:
            if a == champ:
                continue
            m = _mean(hd_sum_atlas[atl][a], hd_count_atlas[atl][a])
            row[f'{_short(a)}_%'] = (m / ch - 1.0) * \
                100 if np.isfinite(m) else np.nan
        pct_rows[atl] = row
    if pct_rows:
        pct_tbl = pd.DataFrame(pct_rows).T
        print(f'\n  Per-atlas mean HD vs {_short(champ)} (shared support):')
        print(
            f'    {_short(champ)}_mean_hd = champion mean HD; other columns = % above champion')
        print(pct_tbl.round(3).to_string())


def print_hd_calibration(hd_out, area_scales, global_scale):
    """Express mean HD raw AND divided by sqrt(area) - pooled and per-atlas.
    Pure reporting on the already-computed raw mean HD."""
    champ = hd_out['champ']
    algos = hd_out['algos']
    atlases = hd_out['atlases']
    rank_tbl = hd_out['rank_tbl']
    hd_sum_atlas = hd_out['hd_sum_atlas']
    hd_count_atlas = hd_out['hd_count_atlas']

    print(f'\n{"="*70}')
    print(f'HAUSDORFF ON THE MAP SCALE  (metric: {HD_REGION_COL})')
    print('  raw        = mean HD in coordinate units')
    print('  /sqrt(tot) = mean HD / sqrt(total parcellation area)')
    print('               = boundary excursion as a fraction of map width')
    print('  %_of_map   = the same, in percent')
    print(f'{"="*70}')

    if global_scale:
        gtot = global_scale['sqrt_mean_total_area']
        pooled = rank_tbl[['mean_hd']].copy()
        pooled['/sqrt(tot)'] = pooled['mean_hd'] / gtot
        pooled['%_of_map'] = pooled['/sqrt(tot)'] * 100
        print(
            f'\n  Pooled over shared support (global sqrt_total={gtot:.2f}):')
        print(pooled.round(4).to_string())

    has_parcel = any('sqrt_median_parcel' in s for s in area_scales.values())
    print('\n  Per-atlas (champion + worst reference):')
    extra = f'{"/sqrt(med_parcel)":>18}' if has_parcel else ''
    print(f'    {"atlas":<14}{"algo":<10}{"raw":>9}{"/sqrt(tot)":>12}'
          f'{"%_of_map":>10}{extra}')
    for atl in atlases:
        sc = area_scales.get(atl)
        if sc is None:
            continue
        tot = sc['sqrt_total_area']
        medp = sc.get('sqrt_median_parcel')
        means = {}
        for a in algos:
            c = hd_count_atlas[atl][a]
            means[a] = (hd_sum_atlas[atl][a] / c) if c else float('nan')
        present = [a for a in algos if np.isfinite(means[a])]
        if champ not in present:
            continue
        worst = max(present, key=lambda a: means[a])
        for a in (champ, worst):
            line = (f'    {atl:<14}{_short(a):<10}{means[a]:>9.4f}'
                    f'{means[a]/tot:>12.5f}{means[a]/tot*100:>10.3f}')
            if has_parcel and medp:
                line += f'{means[a]/medp:>18.4f}'
            print(line)


def print_hd_threshold_crossing(store, col, algos, atlases, area_scales):
    """For each algorithm per atlas: retention BELOW which mean-max HD still
    exceeds a threshold T (i.e. the max retention at which HD >= T)."""
    def _has(col):
        return any(len(store.get(atl, a, col)[0]) for atl in atlases for a in algos)
    if not _has(col):
        return

    def _cross_ret(x, y, T, n=GRID_N):
        if not len(x):
            return np.nan
        grid = np.linspace(float(x.min()), float(x.max()), n)
        yi = np.interp(grid, x, y)
        hit = np.where(yi >= T)[0]
        return float(grid[hit.max()]) if len(hit) else np.nan

    print(f'\n{"="*70}')
    print(f'HD THRESHOLD CROSSING  (metric: {col})')
    print('  value = max retention at which mean-max HD still >= T')
    print('          (HD only exceeds T at retentions at or below this)')
    print('          "<floor" = HD never reaches T within the sampled range')
    print(f'{"="*70}')

    threshold_sets = []
    for T in RAW_HD_THRESHOLDS:
        threshold_sets.append((f'raw>={T:g}', {atl: T for atl in atlases}))
    if area_scales is not None:
        for frac in AREA_FRAC_THRESHOLDS:
            per_atlas_T = {}
            for atl in atlases:
                sc = area_scales.get(atl)
                if sc is not None:
                    per_atlas_T[atl] = frac * sc['sqrt_total_area']
            threshold_sets.append((f'>={frac:g}*sqrt(total)', per_atlas_T))

    for label, per_atlas_T in threshold_sets:
        print(f'\n  threshold: {label}')
        header = '    ' + f'{"atlas":<14}' + \
            ''.join(f'{_short(a):>10}' for a in algos)
        print(header)
        for atl in atlases:
            if atl not in per_atlas_T:
                continue
            T = per_atlas_T[atl]
            cells = []
            for a in algos:
                x, y = store.get(atl, a, col)
                cr = _cross_ret(x, y, T)
                cells.append(f'{cr:>10.3f}' if np.isfinite(cr)
                             else f'{"<floor":>10}')
            print('    ' + f'{atl:<14}' + ''.join(cells))


def print_hd_rstar(store, col, algos, atlases, area_scales):
    """Map-scale Hausdorff r*: lowest retention above which mean-max HD stays
    below a fraction of map width, for every algorithm/atlas."""
    if area_scales is None:
        return

    def _rstar(x, y, T, n=GRID_N):
        if not len(x):
            return np.nan, 'nodata'
        grid = np.linspace(float(x.min()), float(x.max()), n)
        yi = np.interp(grid, x, y)
        hit = np.where(yi >= T)[0]
        if not len(hit):
            return float(x.min()), 'below'
        if hit.max() == len(grid) - 1:
            return float(x.max()), 'never'
        return float(grid[hit.max()]), 'ok'

    print(f'\n{"="*70}')
    print(f'HAUSDORFF r* ON THE MAP SCALE  (metric: {col})')
    print('  r* = lowest retention above which mean-max HD stays below T,')
    print('       with T a fraction of sqrt(total parcellation area).')
    print('  "<floor" = HD already below T across the whole range.')
    print('  "never"  = HD still >= T at the top of the sampled range.')
    print(f'{"="*70}')

    for frac in HD_RSTAR_FRACS:
        print(f'\n  threshold: mean-max HD < {frac*100:g}% of map width')
        header = '    ' + f'{"atlas":<14}' + \
            ''.join(f'{_short(a):>10}' for a in algos)
        print(header)
        global_rstar = 0.0
        any_never = False
        for atl in atlases:
            sc = area_scales.get(atl)
            if sc is None:
                continue
            T = frac * sc['sqrt_total_area']
            cells = []
            for a in algos:
                x, y = store.get(atl, a, col)
                rs, flag = _rstar(x, y, T)
                if flag == 'below':
                    cells.append(f'{"<floor":>10}')
                elif flag == 'never':
                    cells.append(f'{"never":>10}')
                    any_never = True
                    global_rstar = max(global_rstar, rs)
                elif flag == 'nodata':
                    cells.append(f'{"-":>10}')
                else:
                    cells.append(f'{rs:>10.3f}')
                    global_rstar = max(global_rstar, rs)
            print('    ' + f'{atl:<14}' + ''.join(cells))
        tail = ' (some algorithms never settle within range)' if any_never else ''
        print(f'    -> above retention {global_rstar:.3f}, mean-max HD < '
              f'{frac*100:g}% of map width for ALL algorithms in ALL atlases{tail}')


def print_worst_region(store, df, algos, atlases, area_scales):
    """Tail check: mean / 95th-pct / max region HD on the map scale at fixed
    operating retentions. A heavy tail shows up as max (and p95) >> mean."""
    need = [HD_REGION_COL, HD_REGION_P95_COL, HD_REGION_MAX_COL]
    missing = [c for c in need if c not in df.columns or df[c].isna().all()]
    if area_scales is None or missing:
        print('\n[worst-region diagnostic skipped: per-region max/p95 columns '
              'not available]')
        return

    def _at(x, y, r):
        if not len(x):
            return np.nan
        return float(np.interp(r, x, y, left=np.nan, right=np.nan))

    print(f'\n{"="*70}')
    print('WORST-REGION TAIL CHECK  (% of map width)')
    print('  mean = region-mean of region-max HD (the headline metric)')
    print('  p95  = 95th-pct region; max = single worst region')
    print('  heavy tail => max (and p95) sit well above mean')
    print(f'{"="*70}')

    cols = [(HD_REGION_COL, 'mean'),
            (HD_REGION_P95_COL, 'p95'),
            (HD_REGION_MAX_COL, 'max')]

    worst_overall = (0.0, None, None, None)
    for r in WORST_REGION_RETENTIONS:
        print(f'\n  retention {r:.2f}:')
        for label_col, label in cols:
            header = '    ' + f'{label:<5}{"atlas":<14}' + ''.join(
                f'{_short(a):>9}' for a in algos)
            print(header)
            for atl in atlases:
                sc = area_scales.get(atl)
                if sc is None:
                    continue
                tot = sc['sqrt_total_area']
                cells = []
                for a in algos:
                    x, y = store.get(atl, a, label_col)
                    v = _at(x, y, r)
                    if np.isfinite(v):
                        pct = v / tot * 100
                        cells.append(f'{pct:>9.3f}')
                        if label == 'max' and pct > worst_overall[0]:
                            worst_overall = (pct, atl, _short(a), r)
                    else:
                        cells.append(f'{"-":>9}')
                print('    ' + f'{"":<5}{atl:<14}' + ''.join(cells))

    pct, atl, a, r = worst_overall
    if atl is not None:
        print(f'\n  Largest single-region deviation at the sampled retentions: '
              f'{pct:.3f}% of map width  ({a} in {atl} at retention {r:.2f}).')


def print_reference_magnitude(store, col, pack, atlases, area_scales):
    """Four-reference magnitude (de Berg's lead) over windows that do NOT depend
    on TopoVW's floor. TopoVW is excluded entirely."""
    champ = 'de Berg'
    refs = list(pack)
    if champ not in refs:
        return
    others = [a for a in refs if a != champ]

    def _window(cv, mode):
        floors = [x.min() for x, _ in cv.values() if len(x)]
        ceils = [x.max() for x, _ in cv.values() if len(x)]
        if not floors or not ceils:
            return np.nan, np.nan
        if mode == 'full':
            lo, hi = min(floors), max(ceils)
        else:
            lo, hi = max(floors), min(ceils)
            if mode == 'operating':
                lo, hi = max(lo, OP_LO), min(hi, OP_HI)
        return (lo, hi) if lo < hi else (np.nan, np.nan)

    def _means(cv, lo, hi):
        grid = np.linspace(lo, hi, GRID_N)
        mat = {a: np.interp(
            grid, *cv[a], left=np.nan, right=np.nan) for a in refs}
        finite = np.all([np.isfinite(mat[a]) for a in refs], axis=0)
        if not finite.any():
            return None
        return {a: float(np.mean(mat[a][finite])) for a in refs}

    print(f'\n{"="*70}')
    print(
        f'FOUR-REFERENCE MAGNITUDE, TopoVW-INDEPENDENT WINDOWS  (metric: {col})')
    print('  de Berg = denominator; columns = % above de Berg over the window.')
    print('  Compare against the five-way "Per-atlas mean HD vs deBerg" table.')
    print(f'{"="*70}')

    for mode in ('ref_shared', 'operating', 'full'):
        win = ('[{}, {}] clipped'.format(OP_LO, OP_HI) if mode == 'operating'
               else 'min ref floor -> max ref ceil' if mode == 'full'
               else 'max ref floor -> min ref ceil')
        print(f'\n  window: {mode}  ({win})')
        header = ('    ' + f'{"atlas":<14}{"deBerg_hd":>10}'
                  + ''.join(f'{_short(a)+"_%":>12}' for a in others)
                  + (f'{"deBerg_%map":>13}' if area_scales else ''))
        print(header)
        lead = {a: [] for a in others}
        champ_lowest = 0
        n_used = 0
        for atl in atlases:
            cv = store.curves(atl, refs, col)
            lo, hi = _window(cv, mode)
            if not np.isfinite(lo):
                continue
            m = _means(cv, lo, hi)
            if m is None:
                continue
            n_used += 1
            ch = m[champ]
            if all(m[a] >= ch for a in others):
                champ_lowest += 1
            cells = f'    {atl:<14}{ch:>10.4f}'
            for a in others:
                pct = (m[a] / ch - 1.0) * 100 if ch else np.nan
                lead[a].append(pct)
                cells += f'{pct:>12.2f}'
            if area_scales:
                sc = area_scales.get(atl)
                cells += (f'{ch/sc["sqrt_total_area"]*100:>13.3f}'
                          if sc is not None else f'{"-":>13}')
            print(cells)
        for a in others:
            vals = [v for v in lead[a] if np.isfinite(v)]
            if vals:
                print(f'    {_short(a)} above de Berg: '
                      f'{min(vals):.1f}-{max(vals):.1f}%')
        print(f'    de Berg strictly lowest (mean over window) in '
              f'{champ_lowest}/{n_used} atlases')


def print_fullrange_beaters(store, col, hd_out, area_scales):
    """Over each algorithm's FULL pairwise overlap with de Berg, the retention
    bands where it is strictly below de Berg, with the HD value at each band
    edge AND (new) the band's maximum depth as %_of_map. The floor/cross tags
    let you separate the de-Berg-floor-spike artifact (floor->cross bands) from
    the genuine elbow overtakes (cross->cross bands)."""
    champ = hd_out['champ']
    algos = hd_out['algos']
    atlases = hd_out['atlases']
    others = [a for a in algos if a != champ]

    def _bands(cv, beater, n=GRID_N):
        cx, cy = cv[champ]
        bx, by = cv[beater]
        if not len(cx) or not len(bx):
            return []
        lo, hi = max(cx.min(), bx.min()), min(cx.max(), bx.max())
        if lo >= hi:
            return []
        grid = np.linspace(lo, hi, n)
        cyi = np.interp(grid, cx, cy)
        byi = np.interp(grid, bx, by)
        below = byi < cyi
        runs, in_run, s = [], False, 0
        for i in range(len(grid)):
            if below[i] and not in_run:
                s, in_run = i, True
            elif not below[i] and in_run:
                runs.append((s, i - 1))
                in_run = False
        if in_run:
            runs.append((s, len(grid) - 1))
        out = []
        for a0, b0 in runs:
            band_diff = cyi[a0:b0 + 1] - byi[a0:b0 + 1]
            out.append(dict(r_lo=float(grid[a0]), r_hi=float(grid[b0]),
                            hd_lo=float(cyi[a0]), hd_hi=float(cyi[b0]),
                            max_diff=float(np.max(band_diff)),
                            mean_diff=float(np.mean(band_diff)),
                            lo_bound=(a0 == 0), hi_bound=(b0 == len(grid) - 1)))
        return out

    print(f'\n{"="*70}')
    print(f'FULL-RANGE OVERTAKES OF {_short(champ).upper()}  (metric: {col})')
    print('  retention bands where each algorithm is strictly below de Berg over')
    print('  its full pairwise overlap; HD = de Berg HD at the band edges')
    print('  (= crossing HD where the edge is interior, tagged cross vs floor/ceil).')
    print('  depth%map = band max_diff / sqrt(total area) * 100 (the depth of the dip).')
    print('  floor->cross bands = de-Berg-floor-spike artifact; cross->cross = elbow overtakes.')
    print(f'{"="*70}')
    for beater in others:
        lines = []
        for atl in atlases:
            cv = store.curves(atl, [champ, beater], col)
            for bd in _bands(cv, beater):
                tag_lo = 'floor' if bd['lo_bound'] else 'cross'
                tag_hi = 'ceil' if bd['hi_bound'] else 'cross'
                seg = (f'ret [{bd["r_lo"]:.3f}, {bd["r_hi"]:.3f}]  '
                       f'HD [{bd["hd_lo"]:.3f}, {bd["hd_hi"]:.3f}]  '
                       f'max_diff={bd["max_diff"]:.4f} mean_diff={bd["mean_diff"]:.4f}')
                if area_scales and atl in area_scales:
                    t = area_scales[atl]['sqrt_total_area']
                    seg += (f'  %map [{bd["hd_lo"]/t*100:.3f}, '
                            f'{bd["hd_hi"]/t*100:.3f}]'
                            f'  depth%map={bd["max_diff"]/t*100:.4f}')
                seg += f'  [{tag_lo}->{tag_hi}]'
                lines.append(f'      {atl:<14} {seg}')
        if lines:
            print(f'    {_short(beater)}:')
            print('\n'.join(lines))
        else:
            print(f'    {_short(beater)}: never below de Berg over full overlap')


def print_max_hd(store, col, algos, atlases, area_scales):
    """Max over retention of the mean-max HD curve, per algorithm-atlas."""
    raw = {}
    any_data = False
    for atl in atlases:
        row = {}
        for a in algos:
            x, y = store.get(atl, a, col)
            row[_short(a)] = float(np.max(y)) if len(y) else np.nan
            any_data = any_data or len(y) > 0
        raw[atl] = row
    if not any_data:
        return
    order = [_short(a) for a in algos]
    tbl = pd.DataFrame(raw).T[order]
    print(f'\n{"="*70}')
    print(f'MAX mean-max HD per algorithm-atlas  (metric: {col})')
    print('  max over retention (occurs near the floor, aggressive regime)')
    print(f'{"="*70}')
    print(tbl.round(4).to_string())
    if area_scales:
        pct = tbl.copy()
        for atl in pct.index:
            sc = area_scales.get(atl)
            if sc is not None:
                pct.loc[atl] = pct.loc[atl] / sc['sqrt_total_area'] * 100
        print('\n  as % of map width:')
        print(pct.round(3).to_string())


def print_pair_convergence(store, df, col, algos, atlases):
    """Within-baseline-pair convergence on per-region HD.

    NOTE: superseded for the within-pair MAGNITUDE story by the gap table in
    family_structure.py. Kept for the TopoVW-below-VW sign/band sub-block only.
    """
    if col not in df.columns or df[col].isna().all():
        return
    present = set(algos)
    pairs = [(b, t) for b, t in PAIRS if b in present and t in present]
    if not pairs:
        return

    print(f'\n{"="*70}')
    print(f'BASELINE-PAIR CONVERGENCE  (metric: {col})')
    print(
        f'  window = operating range [{OP_LO}, {OP_HI}] (excludes near-floor noise)')
    print(
        f'  conv_ret[k] = lowest retention above which within-pair |Δ| stays')
    print(f'                below k * (pair median HD over the window).')
    print(f'                "none" = never converges within the window above the floors.')
    print(f'                k swept over {CONV_KS} to show robustness.')
    print(f'{"="*70}')

    def _conv_ret(bx, by, tx, ty, k):
        lo = max(bx.min(), tx.min(), OP_LO)
        hi = min(bx.max(), tx.max(), OP_HI)
        if lo >= hi:
            return np.nan
        grid = np.linspace(lo, hi, GRID_N)
        bvals = np.interp(grid, bx, by, left=np.nan, right=np.nan)
        tvals = np.interp(grid, tx, ty, left=np.nan, right=np.nan)
        ok = np.isfinite(bvals) & np.isfinite(tvals)
        if ok.sum() < 2:
            return np.nan
        diff = np.abs(bvals - tvals)
        med = np.nanmedian(np.concatenate([bvals[ok], tvals[ok]]))
        thr = k * med
        within = diff <= thr
        conv = np.nan
        for i in range(len(grid)):
            if not ok[i]:
                continue
            tail = within[i:][ok[i:]]
            if within[i] and np.all(tail):
                conv = float(grid[i])
                break
        return conv

    for b, t in pairs:
        print(f'\n  pair {_short(b)} / {_short(t)}:')
        header = '    ' + f'{"atlas":<14}' + \
            ''.join(f'k={k:<7}' for k in CONV_KS)
        print(header)
        n_conv = {k: 0 for k in CONV_KS}
        for atl in atlases:
            bx, by = store.get(atl, b, col)
            tx, ty = store.get(atl, t, col)
            cells = []
            for k in CONV_KS:
                cr = _conv_ret(bx, by, tx, ty, k)
                cells.append(f'{cr:<9.3f}' if np.isfinite(cr)
                             else f'{"none":<9}')
                if np.isfinite(cr):
                    n_conv[k] += 1
            print('    ' + f'{atl:<14}' + ''.join(cells))
        summ = '  '.join(f'k={k}: {n_conv[k]}/{len(atlases)}' for k in CONV_KS)
        print(f'    converges in window: {summ}')

    vw, tv = 'Visvalingam-Whyatt', 'TopoVW'
    if vw in present and tv in present:
        print(
            f'\n  {_short(tv)} strictly below {_short(vw)}  (sampled retention only):')
        print(f'    band = [first, last] sampled retention where TopoVW < VW;')
        print(f'    width in sampled points; sign note for context.')
        for atl in atlases:
            tx, ty = store.get(atl, tv, col)
            vx, vy = store.get(atl, vw, col)
            if not len(tx) or not len(vx):
                print(f'    {atl:<14} (insufficient data)')
                continue
            mask = (tx >= vx.min()) & (tx <= vx.max())
            xs = tx[mask]
            tv_y = ty[mask]
            vw_y = np.interp(xs, vx, vy)
            if not len(xs):
                print(f'    {atl:<14} (no overlap)')
                continue
            below = tv_y < vw_y
            if not below.any():
                print(
                    f'    {atl:<14} TopoVW never below VW (sign: TopoVW >= VW throughout)')
                continue
            xs_below = xs[below]
            print(f'    {atl:<14} band ~[{xs_below.min():.3f}, {xs_below.max():.3f}]  '
                  f'({int(below.sum())}/{int(len(xs))} sampled points below)')


# === MAIN: load -> cache -> compute -> report ================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default='results/metrics.csv')
    args = ap.parse_args()

    # STAGE 1: ingest -------------------------------------------------------
    df, pack, topo, algos, atlases = load_data(args.csv)
    n_atl = len(atlases)

    total_areas = load_total_areas(atlases)
    if total_areas:
        parcel_areas = load_parcel_areas(atlases)  # optional secondary
        area_scales, global_scale = compute_area_scales(
            total_areas, parcel_areas)
    else:
        area_scales, global_scale = None, None

    # STAGE 2: cache --------------------------------------------------------
    store = CurveStore(df)

    # STAGE 3: compute (one sweep gathers every figure's data) --------------
    spread = compute_spread_per_atlas(store, atlases, pack)
    tbl = build_per_atlas_table(df, store, spread, atlases, pack, topo, algos)
    has_hd = (HD_REGION_COL in df.columns) and (
        not df[HD_REGION_COL].isna().all())
    hd_out = compute_hd_ranking(
        store, HD_REGION_COL, algos, atlases) if has_hd else None

    # STAGE 4: report -------------------------------------------------------
    if area_scales is not None:
        print_area_scales(area_scales, global_scale)
    else:
        print('\n[area calibration skipped: load_total_areas() returns None '
              '- see TODO(area)]')

    print_per_atlas_table(tbl, pack, topo)
    print_spread_section(spread, atlases, pack)        # #1 + #2
    print_floor_ordering(tbl, atlases, pack, topo)
    print_headline_numbers(tbl, spread)

    print_hd_ranking(store, hd_out, n_atl)
    if hd_out is not None and area_scales is not None:
        print_hd_calibration(hd_out, area_scales, global_scale)
    if has_hd:
        print_hd_threshold_crossing(
            store, HD_REGION_COL, algos, atlases, area_scales)
    if hd_out is not None:
        print_fullrange_beaters(store, HD_REGION_COL,
                                hd_out, area_scales)  # #3
    if has_hd:
        print_max_hd(store, HD_REGION_COL, algos, atlases, area_scales)
    if area_scales is not None and has_hd:
        print_hd_rstar(store, HD_REGION_COL, algos, atlases, area_scales)
        print_worst_region(store, df, algos, atlases, area_scales)
    if has_hd:
        print_reference_magnitude(
            store, HD_REGION_COL, pack, atlases, area_scales)
        print_pair_convergence(store, df, HD_REGION_COL, algos, atlases)


if __name__ == '__main__':
    main()

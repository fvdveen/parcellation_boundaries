#!/usr/bin/env python3
"""
Onset-coincidence analysis for §4.2.

For each (algorithm, atlas) series (reference algorithms + TopoVW, hemi=both),
asks: do topological-check failures appear at the *same* sweep step as new face
collapses?

Data source: results/data/*_both_ma0_mp1.json
             results/meshes/*_{lh,rh}_ma0_mp1.pkl  (for A_total)

Sort order: retention DESCENDING within each series (= increasing simplification).
Step k-1 has higher retention than k.
  new_collapse(k)       = n_collapsed(k) > n_collapsed(k-1)
  error_onset(k, check) = check passed at k-1, failed at k

Outputs
-------
  1. Sweep density.
  2. Onset conditional rates per check + reverse P(col|onset), with raw counts.
  3. Error magnitude for component_count and no_overlap (violation sizes).
  4. Per-atlas breakdown of onset rates.
  5. Per-algorithm completeness failure rate.
  6. DK collapse anchor figures.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR  = Path('results/data')
MESH_DIR  = Path('results/meshes')
MIN_AREA  = 0
MIN_POINTS = 1

ALL_ALGOS = frozenset({
    'Douglas-Peucker',
    'Visvalingam-Whyatt',
    'Saalfeld',
    'de Berg',
    'TopoVW',
})

REF_ALGOS = frozenset({
    'Douglas-Peucker',
    'Visvalingam-Whyatt',
    'Saalfeld',
    'de Berg',
})

SHORT = {
    'Douglas-Peucker':    'DP',
    'Visvalingam-Whyatt': 'VW',
    'Saalfeld':           'Saalfeld',
    'de Berg':            'deBerg',
    'TopoVW':             'TopoVW',
}

TOPO_CHECKS = [
    'completeness',
    'validity',
    'component_count',
    'no_overlap',
    'adjacency_gained',
    'adjacency_lost',
    'junction_count',
    'no_gaps',
    'no_crossings',
]


# ── A_total per atlas ─────────────────────────────────────────────────────────

def load_a_total() -> dict[str, float]:
    """Sum of all region polygon areas, both hemispheres, in coordinate units."""
    suffix = f'_ma{MIN_AREA:g}_mp{MIN_POINTS}'
    result: dict[str, float] = {}
    for lh_pkl in sorted(MESH_DIR.glob(f'*_lh{suffix}.pkl')):
        atlas = lh_pkl.stem.replace(f'_lh{suffix}', '')
        rh_pkl = MESH_DIR / f'{atlas}_rh{suffix}.pkl'
        if not rh_pkl.exists():
            continue
        total = 0.0
        for pkl in (lh_pkl, rh_pkl):
            with open(pkl, 'rb') as fh:
                store = pickle.load(fh)
            orig = store.get(('original', None))
            if orig is None:
                total = float('nan')
                break
            total += sum(s.area for s in orig.to_shapes().values())
        if not np.isnan(total):
            result[atlas] = total
    return result


# ── Data loading ──────────────────────────────────────────────────────────────

def _make_error_sets(val: dict) -> dict[str, frozenset]:
    """Extract the named error instances from a validate result dict.

    Onset detection uses set deltas on these: an onset fires when new
    distinct error instances appear, not just when the boolean flips.

    completeness  — each ('M', name) or ('X', name) (missing / extra region)
    validity      — each invalid region name
    component_count — each region whose piece-count changed
    no_overlap    — each frozenset({name_a, name_b}) pair
    adjacency_graph — each ('L', edge) lost edge or ('G', edge) gained edge
    junction_count / no_gaps / no_crossings — aggregate; singleton {'fail'} or empty
    """
    sets: dict[str, frozenset] = {}

    comp = val.get('completeness', {})
    sets['completeness'] = (
        frozenset(('M', n) for n in (comp.get('missing') or []))
        | frozenset(('X', n) for n in (comp.get('extra')   or []))
    )

    vld = val.get('validity', {})
    sets['validity'] = frozenset(vld.get('invalid') or [])

    cc = val.get('component_count', {})
    sets['component_count'] = frozenset((cc.get('changed') or {}).keys())

    no = val.get('no_overlap', {})
    pairs = no.get('overlapping_pairs') or []
    sets['no_overlap'] = frozenset(
        frozenset([p[0], p[1]]) for p in pairs if len(p) >= 2
    )

    adj = val.get('adjacency_graph', {})
    sets['adjacency_gained'] = frozenset(('G', tuple(e)) for e in (adj.get('gained_edges') or []))
    sets['adjacency_lost']   = frozenset(('L', tuple(e)) for e in (adj.get('lost_edges')   or []))

    for chk in ('junction_count', 'no_gaps', 'no_crossings'):
        passed = val.get(chk, {}).get('passed', True)
        sets[chk] = frozenset() if passed else frozenset(['fail'])

    return sets


def load_series() -> list[dict]:
    """Load both-hemi ma0_mp1 data for all algorithms.

    Each step dict stores:
      retention, epsilon, n_collapsed,
      checks      — pass/fail boolean per TOPO_CHECK (for completeness-rate etc.),
      error_sets  — named error instances per check (for set-delta onset detection),
      cc_n_changed, cc_delta_pieces  (component_count violation sizes),
      no_n_pairs, no_total_area      (no_overlap violation sizes).
    """
    suffix = f'_ma{MIN_AREA:g}_mp{MIN_POINTS}'
    series_map: dict[tuple, list] = defaultdict(list)

    for json_path in sorted(DATA_DIR.glob(f'*{suffix}.json')):
        payload = json.loads(json_path.read_text())
        if payload.get('hemi') != 'both':
            continue
        atlas = payload['atlas']

        for r in payload.get('results', []):
            algo = r.get('algorithm')
            if algo not in ALL_ALGOS:
                continue
            ev  = r.get('evaluate')
            val = r.get('validate')
            if ev is None or val is None:
                continue
            retention = ev.get('vertex_retention')
            if retention is None:
                continue
            cfc = val.get('collapsed_face_components', {})
            n_collapsed = cfc.get('n_collapsed')
            if n_collapsed is None:
                continue

            error_sets = _make_error_sets(val)

            checks = {
                chk: bool(val.get(chk, {}).get('passed', True))
                for chk in TOPO_CHECKS
                if chk not in ('adjacency_gained', 'adjacency_lost')
            }
            checks['adjacency_gained'] = not error_sets['adjacency_gained']
            checks['adjacency_lost']   = not error_sets['adjacency_lost']

            # component_count magnitude
            cc_val   = val.get('component_count', {})
            changed  = cc_val.get('changed') or {}
            cc_n_changed   = len(changed)
            cc_delta_pieces = sum(
                abs(v['simplified'] - v['original']) for v in changed.values()
            )

            # no_overlap magnitude
            no_val  = val.get('no_overlap', {})
            pairs   = no_val.get('overlapping_pairs') or []
            no_n_pairs     = len(pairs)
            no_total_area  = sum(p[2] for p in pairs if len(p) >= 3)

            series_map[(algo, atlas)].append({
                'retention':       retention,
                'epsilon':         r.get('epsilon'),
                'n_collapsed':     n_collapsed,
                'checks':          checks,
                'error_sets':      error_sets,
                'cc_n_changed':    cc_n_changed,
                'cc_delta_pieces': cc_delta_pieces,
                'no_n_pairs':      no_n_pairs,
                'no_total_area':   no_total_area,
            })

    out = []
    for key, steps in series_map.items():
        steps_sorted = sorted(steps, key=lambda s: -s['retention'])
        out.append({'series_id': key, 'steps': steps_sorted})
    return out


# ── Onset analysis ────────────────────────────────────────────────────────────

def run_onset_analysis(series: list[dict]) -> dict:
    """Onset conditional rates, aggregated across all series.

    onset(k, check) = new distinct error instances appeared at step k
                    = error_sets[k] - error_sets[k-1] is non-empty.

    This fires even when the check was already failing, as long as new
    named errors (new overlapping pair, new lost edge, new region with
    changed count, etc.) are introduced at step k.
    """
    n_both             = defaultdict(int)
    n_collapse_noset   = defaultdict(int)
    n_onset_nocollapse = defaultdict(int)
    n_neither          = defaultdict(int)

    n_collapse_steps    = 0
    n_no_collapse_steps = 0
    steps_per_series: list[int] = []
    events_per_step:  list[int] = []

    n_gained_onset        = 0
    n_crossing_and_gained = 0
    n_gained_pairs              = 0   # individual (A,B) pairs that newly gained adjacency
    n_gained_pairs_with_overlap = 0   # of those, pairs also present in no_overlap at that step

    n_lost_onset              = 0   # steps where adjacency_lost onset occurred
    n_collapse_and_lost       = 0   # of those, also had new_collapse
    n_lost_pairs              = 0   # individual (A,B) pairs that newly lost adjacency
    n_lost_pairs_with_overlap = 0   # of those, pair also onset in no_overlap at that step

    # Pending tracking: pairs that gained adjacency coincident with overlap onset.
    # When their overlap ends, we check if their adjacency also ends.
    # If overlap re-onsets for a pair that still has spurious adjacency, it is re-added.
    n_tracked_overlap_ended   = 0   # pending pairs whose overlap ended (= denominator)
    n_adjacency_resolved      = 0   # of those, adjacency also ended at that step

    for s in series:
        steps = s['steps']
        steps_per_series.append(len(steps))
        pending: set[frozenset] = set()  # reset per series
        for k in range(1, len(steps)):
            prev, curr = steps[k - 1], steps[k]
            new_collapse = curr['n_collapsed'] > prev['n_collapsed']
            if new_collapse:
                n_collapse_steps += 1
            else:
                n_no_collapse_steps += 1

            n_events = int(new_collapse)
            onsets: dict[str, bool] = {}
            for chk in TOPO_CHECKS:
                onset = bool(curr['error_sets'][chk] - prev['error_sets'][chk])
                onsets[chk] = onset
                if onset:
                    n_events += 1
                if new_collapse and onset:
                    n_both[chk] += 1
                elif new_collapse and not onset:
                    n_collapse_noset[chk] += 1
                elif not new_collapse and onset:
                    n_onset_nocollapse[chk] += 1
                else:
                    n_neither[chk] += 1
            events_per_step.append(n_events)

            if onsets['adjacency_gained']:
                n_gained_onset += 1
                if onsets['no_crossings']:
                    n_crossing_and_gained += 1

            # Precompute overlap deltas and current spurious-adjacency set (used below).
            new_overlaps   = curr['error_sets']['no_overlap'] - prev['error_sets']['no_overlap']
            ended_overlaps = prev['error_sets']['no_overlap'] - curr['error_sets']['no_overlap']
            curr_gained_adj = frozenset(
                frozenset(item[1]) for item in curr['error_sets']['adjacency_gained']
            )

            # Per-pair onset check: did (A,B) newly gain adjacency AND newly start overlapping?
            new_gained_pairs = (
                curr['error_sets']['adjacency_gained']
                - prev['error_sets']['adjacency_gained']
            )
            for item in new_gained_pairs:
                _, edge = item  # item is ('G', (a, b))
                n_gained_pairs += 1
                if frozenset(edge) in new_overlaps:
                    n_gained_pairs_with_overlap += 1

            # Pending: check pairs whose overlap just ended, then (re-)add pairs that
            # newly overlap and still carry spurious adjacency.
            for pair in ended_overlaps:
                if pair in pending:
                    n_tracked_overlap_ended += 1
                    if pair not in curr_gained_adj:
                        n_adjacency_resolved += 1
                    pending.discard(pair)
            for pair in new_overlaps:
                if pair in curr_gained_adj:
                    pending.add(pair)

            # adjacency_lost: step-level collapse co-occurrence + per-pair overlap onset check
            if onsets['adjacency_lost']:
                n_lost_onset += 1
                if new_collapse:
                    n_collapse_and_lost += 1

            new_lost_pairs = (
                curr['error_sets']['adjacency_lost']
                - prev['error_sets']['adjacency_lost']
            )
            for item in new_lost_pairs:
                _, edge = item  # item is ('L', (a, b))
                n_lost_pairs += 1
                if frozenset(edge) in new_overlaps:
                    n_lost_pairs_with_overlap += 1

    check_stats: dict[str, dict] = {}
    for chk in TOPO_CHECKS:
        nb   = n_both[chk]
        nonc = n_onset_nocollapse[chk]
        p_oc  = nb   / n_collapse_steps    if n_collapse_steps    > 0 else float('nan')
        p_onc = nonc / n_no_collapse_steps if n_no_collapse_steps > 0 else float('nan')
        gap   = (p_oc - p_onc
                 if not (np.isnan(p_oc) or np.isnan(p_onc)) else float('nan'))
        check_stats[chk] = {
            'n_both': nb, 'n_onset_no_collapse': nonc,
            'P(onset|collapse)': p_oc, 'P(onset|no_collapse)': p_onc, 'gap': gap,
        }

    p_crossing_given_gained = (
        n_crossing_and_gained / n_gained_onset if n_gained_onset > 0 else float('nan')
    )
    p_overlap_given_gained_pair = (
        n_gained_pairs_with_overlap / n_gained_pairs if n_gained_pairs > 0 else float('nan')
    )
    p_collapse_given_lost = (
        n_collapse_and_lost / n_lost_onset if n_lost_onset > 0 else float('nan')
    )
    p_overlap_given_lost_pair = (
        n_lost_pairs_with_overlap / n_lost_pairs if n_lost_pairs > 0 else float('nan')
    )

    return {
        'check_stats':         check_stats,
        'n_collapse_steps':    n_collapse_steps,
        'n_no_collapse_steps': n_no_collapse_steps,
        'total_transitions':   n_collapse_steps + n_no_collapse_steps,
        'steps_per_series':    steps_per_series,
        'events_per_step':     events_per_step,
        'n_series':            len(series),
        'crossing_given_gained': {
            'n_gained_onset':        n_gained_onset,
            'n_crossing_and_gained': n_crossing_and_gained,
            'P(crossing|gained)':    p_crossing_given_gained,
        },
        'gained_adjacency_overlap': {
            'n_gained_onset':               n_gained_onset,
            'n_gained_pairs':               n_gained_pairs,
            'n_gained_pairs_with_overlap':  n_gained_pairs_with_overlap,
            'P(overlap_onset|gained_pair)': p_overlap_given_gained_pair,
            'n_tracked_overlap_ended':      n_tracked_overlap_ended,
            'n_adjacency_resolved':         n_adjacency_resolved,
            'P(adj_resolved|overlap_ended)': (
                n_adjacency_resolved / n_tracked_overlap_ended
                if n_tracked_overlap_ended > 0 else float('nan')
            ),
        },
        'lost_adjacency_collapse_overlap': {
            'n_lost_onset':               n_lost_onset,
            'n_collapse_and_lost':        n_collapse_and_lost,
            'P(collapse|lost_onset)':     p_collapse_given_lost,
            'n_lost_pairs':               n_lost_pairs,
            'n_lost_pairs_with_overlap':  n_lost_pairs_with_overlap,
            'P(overlap|lost_pair)':       p_overlap_given_lost_pair,
        },
    }


def compute_magnitude_stats(series: list[dict],
                             a_total: dict[str, float]) -> dict:
    """Violation sizes at every step where component_count or no_overlap fails."""
    cc_records: list[dict] = []  # {algo, atlas, retention, n_changed, delta_pieces}
    no_records: list[dict] = []  # {algo, atlas, retention, n_pairs, area, area_frac}

    for s in series:
        algo, atlas = s['series_id']
        a_tot = a_total.get(atlas, float('nan'))
        for step in s['steps']:
            r = step['retention']
            if not step['checks']['component_count']:
                cc_records.append({
                    'algo': algo, 'atlas': atlas, 'retention': r,
                    'n_changed':    step['cc_n_changed'],
                    'delta_pieces': step['cc_delta_pieces'],
                })
            if not step['checks']['no_overlap']:
                frac = step['no_total_area'] / a_tot if not np.isnan(a_tot) and a_tot > 0 else float('nan')
                no_records.append({
                    'algo': algo, 'atlas': atlas, 'retention': r,
                    'n_pairs':   step['no_n_pairs'],
                    'area':      step['no_total_area'],
                    'area_frac': frac,
                })

    return {'cc': cc_records, 'no': no_records}


# ── Printing ──────────────────────────────────────────────────────────────────

def print_sweep_density(result: dict) -> None:
    sps = result['steps_per_series']
    eps = np.array(result['events_per_step'])
    n_series = result['n_series']

    print(f'\n{"=" * 70}')
    print('SWEEP DENSITY')
    print('  Sort order: retention DESCENDING (increasing simplification).')
    print(f'{"=" * 70}')
    print(f'  Series:               {n_series}')
    print(f'  Steps per series:     min={min(sps)}  '
          f'median={np.median(sps):.1f}  max={max(sps)}')
    print(f'  Total transitions:    {result["total_transitions"]}')
    print(f'    with new_collapse:  {result["n_collapse_steps"]}')
    print(f'    no new_collapse:    {result["n_no_collapse_steps"]}')

    print(f'\n  Events per step (new_collapse=1 + each check onset=1):')
    max_shown = int(eps.max()) if len(eps) else 0
    for n in range(0, min(max_shown + 2, 8)):
        count = int(np.sum(eps == n))
        pct = count / len(eps) * 100 if len(eps) else 0
        print(f'    {n} events: {count:5d}  ({pct:.1f}%)')
    if max_shown >= 8:
        count = int(np.sum(eps >= 8))
        pct = count / len(eps) * 100
        print(f'    ≥8 events: {count:5d}  ({pct:.1f}%)')

    multi = int(np.sum(eps > 1))
    pct_multi = multi / len(eps) * 100 if len(eps) else 0
    print(f'\n  Steps bundling >1 event: {multi}/{len(eps)}  ({pct_multi:.1f}%)')
    if pct_multi > 20:
        print('  ** >20% of steps bundle multiple events — '
              'coincidence rates may be inflated. **')


def print_onset_table(result: dict) -> None:
    cs  = result['check_stats']
    nc  = result['n_collapse_steps']
    noc = result['n_no_collapse_steps']
    n_s = result['n_series']

    print(f'\n{"=" * 70}')
    print('ONSET-COINCIDENCE RATES PER TOPOLOGICAL CHECK')
    print(f'  {n_s} series (5 algorithms × 6 atlases, hemi=both).')
    print(f'  new_collapse(k)       : n_collapsed(k) > n_collapsed(k-1)')
    print(f'  error_onset(k, check) : new named error instances appeared at k'
          f' (error_sets[k] - error_sets[k-1] ≠ ∅)')
    print(f'  Steps with new_collapse:    {nc}')
    print(f'  Steps without new_collapse: {noc}')
    print(f'{"=" * 70}')

    w = 20
    print(f'  {"check":<{w}}  {"P(onset|col)":>14}  {"P(onset|no-col)":>15}'
          f'  {"gap":>8}  {"n_both/n_col":>14}  {"n_onset_nc/n_nc":>16}'
          f'  {"P(col|onset)":>14}  {"n_both/n_onset":>15}')
    print('  ' + '-' * 123)

    for chk in TOPO_CHECKS:
        st = cs[chk]
        poc  = f'{st["P(onset|collapse)"]:.4f}'    if not np.isnan(st['P(onset|collapse)'])   else 'N/A'
        ponc = f'{st["P(onset|no_collapse)"]:.4f}' if not np.isnan(st['P(onset|no_collapse)']) else 'N/A'
        gap  = f'{st["gap"]:+.4f}'                 if not np.isnan(st['gap'])                 else 'N/A'
        dc   = f'{st["n_both"]}/{nc}'
        dnc  = f'{st["n_onset_no_collapse"]}/{noc}'
        n_onset_total = st['n_both'] + st['n_onset_no_collapse']
        pco  = f'{st["n_both"] / n_onset_total:.4f}' if n_onset_total > 0 else 'N/A'
        do   = f'{st["n_both"]}/{n_onset_total}'
        print(f'  {chk:<{w}}  {poc:>14}  {ponc:>15}  {gap:>8}'
              f'  {dc:>14}  {dnc:>16}  {pco:>14}  {do:>15}')

    print()
    thin = [c for c in TOPO_CHECKS
            if cs[c]['n_both'] < 5 or cs[c]['n_onset_no_collapse'] < 5]
    if thin:
        print(f'  NOTE — thinly supported (raw count < 5 in at least one cell):')
        print(f'    {", ".join(thin)}')
        print('  Treat those conditional rates as noise, not evidence.')


_ALGO_ORDER = ['Douglas-Peucker', 'Visvalingam-Whyatt', 'Saalfeld', 'de Berg', 'TopoVW']


def print_onset_per_atlas(series: list[dict]) -> None:
    """One table per atlas; algorithms as columns.

    Cell format: P(onset|col) nb/nc  |  P(col|onset) nb/n_onset
    Only checks that ever fire in any cell are shown.
    """
    atlases = sorted(set(s['series_id'][1] for s in series))

    cell_res: dict[tuple, dict | None] = {}
    for atl in atlases:
        for algo in _ALGO_ORDER:
            sub = [s for s in series
                   if s['series_id'][1] == atl and s['series_id'][0] == algo]
            cell_res[(atl, algo)] = run_onset_analysis(sub) if sub else None

    active_checks = [chk for chk in TOPO_CHECKS
                     if any(r is not None and r['check_stats'][chk]['n_both'] > 0
                            for r in cell_res.values())]

    cw = 20   # check name width
    aw = 26   # algo cell width

    def _cell(r: dict | None, chk: str) -> str:
        if r is None or r['n_collapse_steps'] == 0:
            return '---'
        st      = r['check_stats'][chk]
        nc      = r['n_collapse_steps']
        nb      = st['n_both']
        poc     = st['P(onset|collapse)']
        n_onset = nb + st['n_onset_no_collapse']
        pco     = nb / n_onset if n_onset > 0 else float('nan')
        poc_s   = f'{poc:.3f}' if not np.isnan(poc) else ' N/A '
        pco_s   = f'{pco:.3f}' if not np.isnan(pco) else ' N/A '
        return f'{poc_s} {nb}/{nc}  {pco_s} {nb}/{n_onset}'

    algo_lbls = [SHORT[a] for a in _ALGO_ORDER]
    row_width = cw + 2 + len(_ALGO_ORDER) * (aw + 2)

    print(f'\n{"=" * 70}')
    print('ONSET RATES BY ATLAS × ALGORITHM')
    print('  Cell: P(onset|col) nb/nc  |  P(col|onset) nb/n_onset')
    print('  Only checks with ≥1 onset in any cell are shown.')

    for atl in atlases:
        print(f'\n  {"─" * row_width}')
        print(f'  {atl}')
        nc_parts = [f'{SHORT[a]}={cell_res[(atl, a)]["n_collapse_steps"] if cell_res[(atl, a)] else 0}'
                    for a in _ALGO_ORDER]
        print(f'  collapse steps: {", ".join(nc_parts)}')
        print(f'  {"─" * row_width}')

        hdr = f'  {"check":<{cw}}'
        for lbl in algo_lbls:
            hdr += f'  {lbl:^{aw}}'
        print(hdr)
        print('  ' + '-' * row_width)

        for chk in active_checks:
            row = f'  {chk:<{cw}}'
            for algo in _ALGO_ORDER:
                row += f'  {_cell(cell_res[(atl, algo)], chk):<{aw}}'
            print(row)

    print()
    print('  nb = steps where both new_collapse AND this check onset occurred')


def print_magnitude_section(mag: dict, a_total: dict[str, float]) -> None:
    cc = mag['cc']
    no = mag['no']

    print(f'\n{"=" * 70}')
    print('ERROR MAGNITUDE  (violation sizes at every failing step)')
    print(f'{"=" * 70}')

    # ── component_count ────────────────────────────────────────────────────────
    print('\n  component_count  (wrong piece-count across simplification sweep)')
    print(f'  Failing steps: {len(cc)}')
    if cc:
        nc_vals  = np.array([r['n_changed']    for r in cc])
        dp_vals  = np.array([r['delta_pieces'] for r in cc])
        print(f'  n_regions_changed   : mean={nc_vals.mean():.2f}  '
              f'p50={np.median(nc_vals):.1f}  p95={np.percentile(nc_vals, 95):.1f}  '
              f'max={nc_vals.max():.0f}')
        print(f'  |Δ pieces| total    : mean={dp_vals.mean():.2f}  '
              f'p50={np.median(dp_vals):.1f}  p95={np.percentile(dp_vals, 95):.1f}  '
              f'max={dp_vals.max():.0f}')

        print(f'\n  Per algorithm (mean n_changed, mean |Δ pieces|):')
        by_algo: dict[str, list] = defaultdict(list)
        for r in cc:
            by_algo[r['algo']].append(r)
        for algo in sorted(ALL_ALGOS, key=lambda a: SHORT[a]):
            recs = by_algo.get(algo, [])
            if not recs:
                print(f'    {SHORT[algo]:<10}: (no failures)')
                continue
            nc_a = np.array([r['n_changed']    for r in recs])
            dp_a = np.array([r['delta_pieces'] for r in recs])
            print(f'    {SHORT[algo]:<10}: {len(recs):3d} failing steps  '
                  f'mean_n_changed={nc_a.mean():.2f}  mean_|Δ|={dp_a.mean():.2f}  '
                  f'max_|Δ|={dp_a.max():.0f}')

    # ── no_overlap ─────────────────────────────────────────────────────────────
    print(f'\n  no_overlap  (total overlapping area across simplification sweep)')
    print(f'  Failing steps: {len(no)}')
    if no:
        np_vals   = np.array([r['n_pairs']   for r in no])
        area_vals = np.array([r['area']      for r in no])
        frac_vals = np.array([r['area_frac'] for r in no if not np.isnan(r['area_frac'])])

        print(f'  n_overlapping_pairs : mean={np_vals.mean():.2f}  '
              f'p50={np.median(np_vals):.1f}  p95={np.percentile(np_vals, 95):.1f}  '
              f'max={np_vals.max():.0f}')
        print(f'  total_area (coord)  : mean={area_vals.mean():.5f}  '
              f'p50={np.median(area_vals):.5f}  '
              f'p95={np.percentile(area_vals, 95):.5f}  '
              f'max={area_vals.max():.5f}')
        if len(frac_vals):
            print(f'  area / A_total      : mean={frac_vals.mean():.2e}  '
                  f'p50={np.median(frac_vals):.2e}  '
                  f'p95={np.percentile(frac_vals, 95):.2e}  '
                  f'max={frac_vals.max():.2e}')
        else:
            print('  area / A_total      : (A_total unavailable)')

        if a_total:
            atl_line = '  A_total per atlas (coord²): ' + \
                '  '.join(f'{k}={v:.1f}' for k, v in sorted(a_total.items()))
            print(f'\n{atl_line}')

        print(f'\n  Per algorithm (mean n_pairs, mean area/A_total):')
        by_algo_no: dict[str, list] = defaultdict(list)
        for r in no:
            by_algo_no[r['algo']].append(r)
        for algo in sorted(ALL_ALGOS, key=lambda a: SHORT[a]):
            recs = by_algo_no.get(algo, [])
            if not recs:
                print(f'    {SHORT[algo]:<10}: (no failures)')
                continue
            np_a = np.array([r['n_pairs'] for r in recs])
            fr_a = np.array([r['area_frac'] for r in recs
                             if not np.isnan(r['area_frac'])])
            frac_str = f'mean_frac={fr_a.mean():.2e}  max_frac={fr_a.max():.2e}' \
                       if len(fr_a) else '(no A_total)'
            print(f'    {SHORT[algo]:<10}: {len(recs):3d} failing steps  '
                  f'mean_n_pairs={np_a.mean():.2f}  {frac_str}')


def print_per_atlas_breakdown(series: list[dict]) -> None:
    """Per-atlas onset counts for the two dominant checks + collapse rate."""

    # atlas → algo → {n_transitions, n_new_collapse, cc_onset, no_onset}
    tbl: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for s in series:
        algo, atlas = s['series_id']
        steps = s['steps']
        for k in range(1, len(steps)):
            prev, curr = steps[k - 1], steps[k]
            new_col = curr['n_collapsed'] > prev['n_collapsed']
            tbl[atlas][algo]['n_trans'] += 1
            if new_col:
                tbl[atlas][algo]['n_col'] += 1
            for chk in ('component_count', 'no_overlap', 'completeness'):
                if prev['checks'][chk] and not curr['checks'][chk]:
                    tbl[atlas][algo][f'{chk}_onset'] += 1

    atlases = sorted(tbl)
    algos   = sorted(ALL_ALGOS, key=lambda a: SHORT[a])

    print(f'\n{"=" * 70}')
    print('PER-ATLAS BREAKDOWN')
    print('  Values: onset_count/n_transitions  (new failure events / sweep steps)')
    print('  Checks shown: new_collapse, component_count, no_overlap, completeness')
    print(f'{"=" * 70}')

    for atl in atlases:
        print(f'\n  {atl}:')
        header = f'    {"algo":<10}' + ''.join(
            f'{"new_col":>12}{"cc_onset":>12}{"no_onset":>12}{"comp_onset":>12}'
        )
        print(header)
        print('    ' + '-' * (10 + 48))
        for algo in algos:
            d = tbl[atl].get(algo, {})
            nt = d.get('n_trans', 0)
            if nt == 0:
                continue
            nc  = d.get('n_col', 0)
            cc  = d.get('component_count_onset', 0)
            no  = d.get('no_overlap_onset', 0)
            com = d.get('completeness_onset', 0)

            def _f(n: int) -> str:
                return f'{n}/{nt}'

            print(f'    {SHORT[algo]:<10}'
                  f'{_f(nc):>12}{_f(cc):>12}{_f(no):>12}{_f(com):>12}')


def print_crossing_given_gained(result: dict) -> None:
    """P(crossing_onset | adjacency_gained_onset): crossing risk when new adjacency appears."""
    cgg = result['crossing_given_gained']
    n_gained  = cgg['n_gained_onset']
    n_both    = cgg['n_crossing_and_gained']
    p         = cgg['P(crossing|gained)']

    print(f'\n{"=" * 70}')
    print('CROSSING RISK GIVEN ADJACENCY GAIN')
    print('  Question: when new adjacency edges are gained at a step, how often')
    print('  does a crossing onset also occur at that same step?')
    print(f'{"=" * 70}')
    print(f'  Steps with adjacency_gained onset:          {n_gained}')
    print(f'  Of those, also had no_crossings onset:      {n_both}')
    p_str = f'{p:.4f}' if not np.isnan(p) else 'N/A'
    print(f'  P(crossing_onset | adjacency_gained_onset): {p_str}')
    if n_gained < 5:
        print('  NOTE — fewer than 5 adjacency_gained onset steps; treat as noise.')


def print_gained_adjacency_overlap(result: dict) -> None:
    """Is adjacency ever gained, and when so, do the gained pairs also overlap?"""
    gao = result['gained_adjacency_overlap']
    n_onset  = gao['n_gained_onset']
    n_pairs  = gao['n_gained_pairs']
    n_with   = gao['n_gained_pairs_with_overlap']
    p        = gao['P(overlap_onset|gained_pair)']
    n_tracked = gao['n_tracked_overlap_ended']
    n_resolved = gao['n_adjacency_resolved']
    p_resolved = gao['P(adj_resolved|overlap_ended)']

    print(f'\n{"=" * 70}')
    print('ADJACENCY GAINED — EXISTENCE AND OVERLAP CHECK')
    print('  Question 1: is adjacency ever gained at all?')
    print('  Question 2: for each newly gained pair (A,B), does overlap between A and B also newly onset at that step?')
    print('  Question 3: when a tracked pair\'s overlap ends, does its adjacency also end?')
    print('    (pairs with spurious adjacency re-entering overlap are re-tracked;')
    print('     pending set resets per series)')
    print(f'{"=" * 70}')
    print(f'  Steps where adjacency_gained onset occurred:  {n_onset}')
    if n_onset == 0:
        print('  -> Adjacency is NEVER gained; all checks below are vacuous.')
        return
    print(f'  -> Adjacency IS gained ({n_onset} step(s) across all series).')
    print()
    print(f'  Individual (A,B) pairs newly gaining adjacency:       {n_pairs}')
    print(f'  Of those, pair also newly onset in no_overlap:        {n_with}')
    p_str = f'{p:.4f}' if not np.isnan(p) else 'N/A'
    print(f'  P(overlap_onset | gained_pair):                       {p_str}')
    if n_pairs < 5:
        print('  NOTE — fewer than 5 gained pairs total; treat as noise.')
    elif n_with == 0:
        print('  NOTE — no gained pair has a simultaneous overlap onset; the gained '
              'adjacency does not appear to be artefact-overlap-driven.')
    elif p > 0.5:
        print('  NOTE — majority of gained pairs also newly overlap; gained adjacency '
              'likely reflects overlap artefacts rather than true topological adjacency.')
    print()
    print(f'  Tracked pairs whose overlap subsequently ended:        {n_tracked}')
    print(f'  Of those, adjacency also ended at that step:           {n_resolved}')
    p_res_str = f'{p_resolved:.4f}' if not np.isnan(p_resolved) else 'N/A'
    print(f'  P(adj_resolved | overlap_ended):                       {p_res_str}')
    if n_tracked == 0:
        print('  NOTE — no tracked pairs had their overlap end; resolution rate is vacuous.')
    elif n_resolved == n_tracked:
        print('  NOTE — all tracked pairs resolved: adjacency ended with its overlap in every case.')
    elif p_resolved < 0.5:
        print('  NOTE — fewer than half of tracked pairs resolved; spurious adjacency '
              'tends to outlast the overlap that introduced it.')


def print_lost_adjacency_collapse_overlap(result: dict) -> None:
    """Is adjacency ever lost, and when so: does collapse co-occur + do lost pairs overlap?"""
    lac = result['lost_adjacency_collapse_overlap']
    n_onset    = lac['n_lost_onset']
    n_col_lost = lac['n_collapse_and_lost']
    p_col      = lac['P(collapse|lost_onset)']
    n_pairs    = lac['n_lost_pairs']
    n_with     = lac['n_lost_pairs_with_overlap']
    p_ovl      = lac['P(overlap|lost_pair)']

    print(f'\n{"=" * 70}')
    print('ADJACENCY LOST — EXISTENCE, COLLAPSE ONSET, AND OVERLAP ONSET CHECK')
    print('  Question 1: is adjacency ever lost at all?')
    print('  Question 2: when adjacency is lost, does a collapse also onset at that step?')
    print('  Question 3: for each newly lost pair (A,B), does overlap between A and B also newly onset at that step?')
    print(f'{"=" * 70}')
    print(f'  Steps where adjacency_lost onset occurred:    {n_onset}')
    if n_onset == 0:
        print('  -> Adjacency is NEVER lost; all checks below are vacuous.')
        return
    print(f'  -> Adjacency IS lost ({n_onset} step(s) across all series).')
    print()
    p_col_str = f'{p_col:.4f}' if not np.isnan(p_col) else 'N/A'
    print(f'  Of those steps, also had new_collapse:        {n_col_lost}')
    print(f'  P(collapse_onset | lost_onset):               {p_col_str}')
    print()
    print(f'  Individual (A,B) pairs newly losing adjacency:        {n_pairs}')
    print(f'  Of those, pair also newly onset in no_overlap:        {n_with}')
    p_ovl_str = f'{p_ovl:.4f}' if not np.isnan(p_ovl) else 'N/A'
    print(f'  P(overlap_onset | lost_pair):                         {p_ovl_str}')
    if n_pairs < 5:
        print('  NOTE — fewer than 5 lost pairs total; treat as noise.')
    elif n_with == 0:
        print('  NOTE — no lost pair has a simultaneous overlap onset; lost adjacency '
              'does not appear to be artefact-overlap-driven.')
    elif p_ovl > 0.5:
        print('  NOTE — majority of lost pairs also newly overlap; lost adjacency '
              'likely reflects overlap artefacts rather than true boundary separation.')


def print_completeness_rates(series: list[dict]) -> None:
    algo_total: dict[str, int] = defaultdict(int)
    algo_fail:  dict[str, int] = defaultdict(int)
    for s in series:
        algo = s['series_id'][0]
        for step in s['steps']:
            algo_total[algo] += 1
            if not step['checks'].get('completeness', True):
                algo_fail[algo] += 1

    print(f'\n{"=" * 70}')
    print('COMPLETENESS FAILURE RATE  (% of sampled outputs, per algorithm)')
    print(f'{"=" * 70}')
    for algo in sorted(ALL_ALGOS, key=lambda a: SHORT[a]):
        total = algo_total[algo]
        fail  = algo_fail[algo]
        rate  = fail / total * 100 if total > 0 else float('nan')
        print(f'  {SHORT[algo]:<10} ({algo:<24}): {fail:4d}/{total:4d}  ({rate:.2f}%)')


def print_component_count_rates(series: list[dict]) -> None:
    algo_total: dict[str, int] = defaultdict(int)
    algo_fail:  dict[str, int] = defaultdict(int)
    for s in series:
        algo = s['series_id'][0]
        for step in s['steps']:
            algo_total[algo] += 1
            if not step['checks'].get('component_count', True):
                algo_fail[algo] += 1

    print(f'\n{"=" * 70}')
    print('COMPONENT_COUNT FAILURE RATE  (% of sampled outputs, per algorithm)')
    print(f'{"=" * 70}')
    for algo in sorted(ALL_ALGOS, key=lambda a: SHORT[a]):
        total = algo_total[algo]
        fail  = algo_fail[algo]
        rate  = fail / total * 100 if total > 0 else float('nan')
        print(f'  {SHORT[algo]:<10} ({algo:<24}): {fail:4d}/{total:4d}  ({rate:.2f}%)')


def print_dk_anchors(series: list[dict]) -> None:
    dk = [s for s in series if s['series_id'][1] == 'DK']
    if not dk:
        print('\n[DK atlas not found — skipping collapse anchors]')
        return

    anchors = [0.10, 0.20, 0.30]
    print(f'\n{"=" * 70}')
    print('DK COLLAPSE ANCHORS  (raw n_collapsed at stated retention)')
    print('  Nearest sampled step to each anchor retention is used.')
    print(f'{"=" * 70}')

    for s in sorted(dk, key=lambda x: SHORT[x['series_id'][0]]):
        algo = s['series_id'][0]
        steps = s['steps']
        if not steps:
            continue
        rets = np.array([st['retention'] for st in steps])
        cols = np.array([st['n_collapsed'] for st in steps])
        print(f'\n  {SHORT[algo]} ({algo}):')
        for r in anchors:
            idx = int(np.argmin(np.abs(rets - r)))
            print(f'    retention ≈ {r:.2f}  '
                  f'(actual {rets[idx]:.4f}):  n_collapsed = {cols[idx]}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f'Loading data from {DATA_DIR}  (ma={MIN_AREA}, mp={MIN_POINTS})...')
    series = load_series()
    if not series:
        print('ERROR: No series found.', file=sys.stderr)
        sys.exit(1)
    total_steps = sum(len(s['steps']) for s in series)
    print(f'Loaded {len(series)} series, {total_steps} total steps.')

    print('Loading A_total from mesh pickles...')
    a_total = load_a_total()
    print(f'  A_total available for: {sorted(a_total)}')

    result   = run_onset_analysis(series)
    mag      = compute_magnitude_stats(series, a_total)

    print_sweep_density(result)
    print_onset_table(result)
    print_crossing_given_gained(result)
    print_gained_adjacency_overlap(result)
    print_lost_adjacency_collapse_overlap(result)
    print_onset_per_atlas(series)
    print_magnitude_section(mag, a_total)
    print_per_atlas_breakdown(series)
    print_completeness_rates(series)
    print_component_count_rates(series)
    print_dk_anchors(series)


if __name__ == '__main__':
    main()

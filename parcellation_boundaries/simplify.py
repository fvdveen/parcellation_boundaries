"""Arc simplification algorithms and mesh-level simplification."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint as MP, Point, Polygon

from .half_edge import Arc, HalfEdge, HalfEdgeMesh


# ── Shared geometry primitives ────────────────────────────────────────────────

def _farthest(coords: np.ndarray, i: int, j: int) -> tuple[int, float]:
    """Index and distance of farthest intermediate vertex from chord [i, j]."""
    if j <= i + 1:
        return i, 0.0
    vi, vj = coords[i], coords[j]
    seg = vj - vi
    norm = float(np.linalg.norm(seg))
    sub = coords[i + 1:j]
    d = sub - vi
    cross = seg[0] * d[:, 1] - seg[1] * d[:, 0]
    dists = (np.abs(cross) / norm
             if norm > 0 else np.linalg.norm(d, axis=1))
    k_rel = int(np.argmax(dists))
    return i + 1 + k_rel, float(dists[k_rel])


def _dp_indices(coords: np.ndarray, epsilon: float) -> set[int]:
    """Douglas-Peucker: return set of kept indices."""
    n = len(coords)
    keep: set[int] = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        k, dist = _farthest(coords, i, j)
        if dist > epsilon:
            keep.add(k)
            stack.append((i, k))
            stack.append((k, j))
    return keep


def _make_segs(coords: np.ndarray) -> np.ndarray:
    """(N-1, 2, 2) segment array from (N, 2) coords."""
    return np.stack([coords[:-1], coords[1:]], axis=1)


# ── Per-arc algorithms ────────────────────────────────────────────────────────

def dp(coords: np.ndarray, epsilon: float) -> np.ndarray:
    """Douglas-Peucker simplification. Always keeps endpoints."""
    if len(coords) <= 2:
        return coords
    return coords[sorted(_dp_indices(coords, epsilon))]


def vw(coords: np.ndarray, epsilon: float) -> np.ndarray:
    """Visvalingam-Whyatt simplification. Always keeps endpoints."""
    import heapq
    if len(coords) <= 2:
        return coords
    n = len(coords)
    prev_idx = list(range(-1, n - 1))
    next_idx = list(range(1, n + 1))

    def triangle_area(i):
        p, q, r = coords[prev_idx[i]], coords[i], coords[next_idx[i]]
        return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]))

    effective = [0.0] * n
    version   = [0] * n
    heap: list = []
    for i in range(1, n - 1):
        effective[i] = triangle_area(i)
        heapq.heappush(heap, (effective[i], version[i], i))

    removed = [False] * n
    while heap:
        a, v, i = heapq.heappop(heap)
        if removed[i] or v != version[i]:
            continue
        if a > epsilon:
            break
        removed[i] = True
        p, nx = prev_idx[i], next_idx[i]
        if nx < n:
            prev_idx[nx] = p
        if p >= 0:
            next_idx[p] = nx
        for nb in (p, nx):
            if 0 < nb < n - 1:
                version[nb] += 1
                effective[nb] = max(triangle_area(nb), a)
                heapq.heappush(heap, (effective[nb], version[nb], nb))

    return coords[~np.array(removed)]


# ── Topology-aware VW ────────────────────────────────────────────────────────

def topovw_modified(coords: np.ndarray, epsilon: float,
                    features: np.ndarray) -> np.ndarray:
    """Topology-aware Visvalingam-Whyatt (modified, per-arc formulation).

    Identical to VW except that a vertex is only removed when its ear
    triangle contains no foreign feature point.  Blocked vertices are
    reconsidered when a neighbour's removal changes their ear.  Features
    are treated as fixed — correct for the per-arc formulation in
    simplify_topovw_modified.
    """
    import heapq

    if len(coords) <= 2:
        return coords
    n = len(coords)

    prev_idx = list(range(-1, n - 1))
    next_idx = list(range(1, n + 1))

    def tri_area(i: int) -> float:
        p, q, r = coords[prev_idx[i]], coords[i], coords[next_idx[i]]
        return 0.5 * abs((q[0]-p[0])*(r[1]-p[1]) - (r[0]-p[0])*(q[1]-p[1]))

    # Spatial hash-grid over features for fast bbox queries
    has_features = len(features) > 0
    if has_features:
        span = float(np.max(features.max(0) - features.min(0)))
        cell_size = max(span / 200.0, 1e-12)
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, pt in enumerate(features):
            grid[(int(np.floor(pt[0] / cell_size)),
                  int(np.floor(pt[1] / cell_size)))].append(idx)

    def ear_is_safe(i: int) -> bool:
        if not has_features:
            return True
        a, v, b = coords[prev_idx[i]], coords[i], coords[next_idx[i]]
        xmin = min(a[0], v[0], b[0]);  xmax = max(a[0], v[0], b[0])
        ymin = min(a[1], v[1], b[1]);  ymax = max(a[1], v[1], b[1])
        cs   = cell_size
        cx0  = int(np.floor(xmin / cs));  cx1 = int(np.floor(xmax / cs))
        cy0  = int(np.floor(ymin / cs));  cy1 = int(np.floor(ymax / cs))
        cands: list[int] = []
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                cands.extend(grid.get((cx, cy), []))
        if not cands:
            return True
        pts = features[cands]
        return not bool(np.any(_pts_in_triangle(pts, a, v, b)))

    effective = [0.0] * n
    version   = [0] * n
    heap: list = []
    for i in range(1, n - 1):
        effective[i] = tri_area(i)
        heapq.heappush(heap, (effective[i], version[i], i))

    removed = [False] * n
    while heap:
        area, ver, i = heapq.heappop(heap)
        if removed[i] or ver != version[i]:
            continue
        if area > epsilon:
            break
        if not ear_is_safe(i):
            continue                     # feature inside ear — skip, re-test if ear changes

        removed[i] = True
        p, nx = prev_idx[i], next_idx[i]
        if nx < n:
            prev_idx[nx] = p
        if p >= 0:
            next_idx[p] = nx
        for nb in (p, nx):
            if 0 < nb < n - 1:
                version[nb] += 1
                effective[nb] = max(tri_area(nb), area)
                heapq.heappush(heap, (effective[nb], version[nb], nb))

    return coords[~np.array(removed)]


# ── de Berg, van Kreveld & Schirra (1998) ─────────────────────────────────────

def _pts_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray-casting point-in-polygon. pts:(N,2) poly:(M,2)."""
    if len(poly) < 3:
        return np.zeros(len(pts), dtype=bool)
    px    = pts[:, 0:1]                          # (N, 1)
    py    = pts[:, 1:2]
    x1    = poly[:, 0];  y1 = poly[:, 1]         # (M,)
    x2    = np.roll(poly[:, 0], -1)
    y2    = np.roll(poly[:, 1], -1)
    cond  = (y1 > py) != (y2 > py)              # (N, M)
    denom = np.where(np.abs(y2 - y1) < 1e-300, 1e-300, y2 - y1)
    x_int = (x2 - x1) * (py - y1) / denom + x1  # (N, M)
    hits  = np.sum(cond & (px < x_int), axis=1)   # (N,)
    return (hits % 2) == 1


def _half_line_wedge_graph(coords: np.ndarray, eps: float) -> set:
    """(i,j) iff the half-line from vi toward vj passes within eps of every vk, i<k≤j."""
    n = len(coords)
    arcs: set[tuple[int, int]] = set()
    for i in range(n - 1):
        lo, hi = -np.pi, np.pi
        for j in range(i + 1, n):
            if lo > hi:
                break
            d     = coords[j] - coords[i]
            theta = np.arctan2(d[1], d[0])
            # Normalise into [mid-π, mid+π] to handle angular wrap-around
            mid   = 0.5 * (lo + hi)
            theta = mid + ((theta - mid + np.pi) % (2.0 * np.pi)) - np.pi
            if lo <= theta <= hi:
                arcs.add((i, j))
            dist = float(np.hypot(d[0], d[1]))
            if dist > eps:
                hs = np.arcsin(min(eps / dist, 1.0))
                lo = max(lo, theta - hs)
                hi = min(hi, theta + hs)
    return arcs


def _allowed_shortcut_graph(coords: np.ndarray, eps: float) -> set:
    """(i,j) iff bandwidth of shortcut vi→vj over the subchain is ≤ eps.

    Runs the wedge sweep forward (half-line from vi) and backward on the
    reversed chain (half-line from vj), returns their intersection.
    """
    n   = len(coords)
    g1  = _half_line_wedge_graph(coords, eps)
    g2r = _half_line_wedge_graph(coords[::-1], eps)
    g2  = {(n - 1 - j, n - 1 - i) for i, j in g2r}
    return g1 & g2


def _consistent_shortcut_graph(coords: np.ndarray,
                                features: np.ndarray) -> set:
    """(i,j) iff no feature point lies inside the polygon formed by the
    subchain vi…vj closed by the shortcut vj→vi."""
    n    = len(coords)
    arcs: set[tuple[int, int]] = set()

    if len(features) == 0:
        return {(i, j) for i in range(n) for j in range(i + 1, n)}

    span      = float(np.max(features.max(0) - features.min(0)))
    cell_size = max(span / 200.0, 1e-12)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, pt in enumerate(features):
        grid[(int(np.floor(pt[0] / cell_size)),
              int(np.floor(pt[1] / cell_size)))].append(idx)

    for i in range(n - 1):
        for j in range(i + 1, n):
            sub  = coords[i:j + 1]
            xmin, ymin = sub.min(0)
            xmax, ymax = sub.max(0)
            cs   = cell_size
            cx0  = int(np.floor(xmin / cs));  cx1 = int(np.floor(xmax / cs))
            cy0  = int(np.floor(ymin / cs));  cy1 = int(np.floor(ymax / cs))
            cands: list[int] = []
            for cx in range(cx0, cx1 + 1):
                for cy in range(cy0, cy1 + 1):
                    cands.extend(grid.get((cx, cy), []))
            if not cands:
                arcs.add((i, j))
                continue
            if not np.any(_pts_in_polygon(features[cands], sub)):
                arcs.add((i, j))

    return arcs


def _min_link_path(n: int, arcs: set) -> list:
    """Fewest-arc path from vertex 0 to vertex n-1 on a DAG (all arcs i<j)."""
    INF  = n
    dp   = [INF] * n
    prev = [-1]  * n
    dp[0] = 0
    for j in range(1, n):
        for i in range(j):
            if dp[i] < INF and (i, j) in arcs and dp[i] + 1 < dp[j]:
                dp[j] = dp[i] + 1
                prev[j] = i
    if dp[n - 1] == INF:
        return list(range(n))           # no valid path — keep all vertices
    path = []
    k    = n - 1
    while k >= 0:
        path.append(k)
        k = prev[k]
    return path[::-1]


def deberg(coords: np.ndarray, epsilon: float,
           features: np.ndarray) -> np.ndarray:
    """de Berg et al. (1998) minimum-link topology-preserving simplification.

    Finds the fewest-segment polyline from coords[0] to coords[-1] such that:
    - every original vertex is within epsilon of its covering segment
      (bandwidth / allowed-shortcut test via angular wedge sweep); and
    - no feature point changes sides of the chain (consistent-shortcut test
      via polygon containment).
    """
    if len(coords) <= 2:
        return coords
    n = len(coords)

    g_allowed    = _allowed_shortcut_graph(coords, epsilon)
    g_consistent = _consistent_shortcut_graph(coords, features)
    # Trivial edges are always valid; guarantees the DAG is connected
    g = (g_allowed & g_consistent) | {(i, i + 1) for i in range(n - 1)}

    return coords[_min_link_path(n, g)]


# ── Saalfeld (1999) helpers ───────────────────────────────────────────────────

def _init_parity(features: np.ndarray, coords: np.ndarray,
                 kept: set[int]) -> np.ndarray:
    """Initial parity of each feature against the closed curve:
    (original polyline forward) + (DP simplified path reversed).

    True = inside closed curve = wrong parity (odd ray crossings).
    Vectorised over all features simultaneously.
    """
    fwd = _make_segs(coords)
    bwd = _make_segs(coords[sorted(kept)][::-1])
    segs = np.concatenate([fwd, bwd])          # (S, 2, 2)

    xs = features[:, 0, None]                  # (M, 1)
    ys = features[:, 1, None]                  # (M, 1)
    x0, y0 = segs[:, 0, 0], segs[:, 0, 1]     # (S,)
    x1, y1 = segs[:, 1, 0], segs[:, 1, 1]     # (S,)

    straddle = (y0 > ys) != (y1 > ys)          # (M, S)
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = x0 + (ys - y0) * (x1 - x0) / (y1 - y0)   # (M, S)

    crossings = np.sum(straddle & (xi > xs), axis=1)    # (M,)
    return (crossings % 2).astype(bool)


def _pts_in_triangle(pts: np.ndarray, a: np.ndarray,
                     b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vectorised barycentric point-in-triangle test."""
    v0, v1 = c - a, b - a
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:                     # degenerate triangle
        return np.zeros(len(pts), dtype=bool)
    v2  = pts - a
    d02 = v2 @ v0
    d12 = v2 @ v1
    inv = 1.0 / denom
    u   = (d11 * d02 - d01 * d12) * inv
    v   = (d00 * d12 - d01 * d02) * inv
    return (u >= 0) & (v >= 0) & (u + v <= 1)


def _hull_verts(coords: np.ndarray, i: int, j: int) -> np.ndarray | None:
    """CCW convex hull vertices of coords[i..j], or None if degenerate."""
    pts = coords[i:j + 1]
    if len(pts) < 3:
        return None
    try:
        ch = ConvexHull(pts)
        return pts[ch.vertices]         # scipy gives CCW order for 2D
    except Exception:
        return None


def _in_hull(pts: np.ndarray, hull_verts: np.ndarray) -> np.ndarray:
    """Vectorised half-plane containment test against a CCW convex polygon."""
    n = len(hull_verts)
    mask = np.ones(len(pts), dtype=bool)
    for k in range(n):
        a = hull_verts[k]
        b = hull_verts[(k + 1) % n]
        e = b - a
        cross = e[0] * (pts[:, 1] - a[1]) - e[1] * (pts[:, 0] - a[0])
        mask &= (cross >= -1e-10)
    return mask


def _sub_hulls_interfere(coords: np.ndarray, i: int, k: int, j: int) -> bool:
    """True if the convex hulls of coords[i..k] and coords[k..j] overlap in area.

    Saalfeld §3.2: a sub-hull overlap implies that recursive simplification of
    the two halves could produce crossing edges.
    """
    left  = coords[i:k + 1]
    right = coords[k:j + 1]
    if len(left) < 3 or len(right) < 3:
        return False
    hl = MP(left.tolist()).convex_hull
    hr = MP(right.tolist()).convex_hull
    return hl.intersection(hr).area > 0


def _refine(coords: np.ndarray, features: np.ndarray,
            parity: np.ndarray, keep: set[int],
            i: int, j: int, epsilon: float) -> None:
    """Saalfeld Refine step for the current simplified edge [coords[i], coords[j]].

    Inserts the farthest intermediate vertex whenever the epsilon test fails,
    any feature inside the convex hull of P[i..j] has wrong parity, or the
    two sub-hulls interfere.  Updates parity by triangle-inversion for every
    inserted vertex, then recurses on both halves.
    """
    if j <= i + 1:
        return

    k, dist = _farthest(coords, i, j)
    vi, vk, vj = coords[i], coords[k], coords[j]

    # Bad features: inside hull of P[i..j] and currently wrong-sided (Lemma 1)
    hv = _hull_verts(coords, i, j)
    if hv is not None and len(features):
        bad = bool(np.any(_in_hull(features, hv) & parity))
    else:
        bad = False

    if dist > epsilon or bad or _sub_hulls_interfere(coords, i, k, j):
        keep.add(k)

        # Triangle inversion: only features inside Δ(vi, vk, vj) change side
        parity[_pts_in_triangle(features, vi, vk, vj)] ^= True

        _refine(coords, features, parity, keep, i, k, epsilon)
        _refine(coords, features, parity, keep, k, j, epsilon)


# ── Saalfeld public API ───────────────────────────────────────────────────────

def saalfeld(coords: np.ndarray, features: np.ndarray,
             epsilon: float) -> np.ndarray:
    """Saalfeld (1999) topologically-consistent line simplification.

    coords:   (N, 2) polyline; first and last vertices always kept.
    features: (M, 2) vertices of adjacent arcs — none may change side.
    epsilon:  perpendicular-distance tolerance (same units as coords).

    Returns simplified (K, 2) array.  Falls back to plain DP when M = 0.
    """
    if len(coords) <= 2:
        return coords
    if not len(features):
        return dp(coords, epsilon)

    # Step 1: ordinary Douglas-Peucker
    keep = _dp_indices(coords, epsilon)

    # Step 2: initialise parity once against (original poly + DP path)
    parity = _init_parity(features, coords, keep)

    # Step 3: refine every DP edge
    sk = sorted(keep)
    for i, j in zip(sk, sk[1:]):
        _refine(coords, features, parity, keep, i, j, epsilon)

    return coords[sorted(keep)]


# ── Mesh-level simplification ─────────────────────────────────────────────────

def simplify(mesh: HalfEdgeMesh, epsilon: float,
             algorithm: Callable[[np.ndarray, float], np.ndarray] = dp) -> HalfEdgeMesh:
    """Return a new mesh with simplified arc coordinates.

    Topology (next/prev/twin/face pointers) is preserved exactly.
    Only arc.coords changes; junction endpoints are always kept.
    """
    seen: set[int] = set()
    arc_map: dict[int, np.ndarray] = {}
    for h in mesh.half_edges:
        if id(h.arc) in seen:
            continue
        seen.add(id(h.arc))
        arc_map[id(h.arc)] = algorithm(h.arc.coords, epsilon)
    return mesh.copy_with_new_coords(arc_map)


def simplify_topovw_modified(mesh: HalfEdgeMesh, epsilon: float,
                             face_seeds: dict[str, np.ndarray] | None = None) -> HalfEdgeMesh:
    """TopoVW (modified)-simplify every arc, using all other arcs' vertices as features.

    Each arc is simplified in turn against the original (pre-simplification)
    vertex positions of every other arc, preventing topological crossings.

    face_seeds: {face_name: (N, 2) seed-point array} from the Voronoi diagram.
    When provided, each closed arc (island) gets a single interior anchor added
    to its feature set: the seed point of the enclosed face that is closest to
    the arc's junction vertex.  That seed is the Voronoi cell generator incident
    to the arc, so it is guaranteed to lie inside the island polygon.  Without
    it the in-triangle test has nothing to block and collapses the loop to nothing.

    Topology (next/prev/twin/face) is preserved exactly.
    """
    seen:    set[int]                              = set()
    entries: list[tuple[Arc, HalfEdge, HalfEdge]] = []
    for h in mesh.half_edges:
        if id(h.arc) in seen:
            continue
        seen.add(id(h.arc))
        h_fwd = h if h.forward else h.twin
        entries.append((h.arc, h_fwd, h_fwd.twin))

    all_arcs = [arc for arc, _, _ in entries]

    simplified: dict[int, np.ndarray] = {}
    for arc, h_fwd, h_bwd in entries:
        others   = [a.coords for a in all_arcs if a is not arc]
        features = np.concatenate(others) if others else np.empty((0, 2))

        if face_seeds is not None and np.allclose(arc.coords[0], arc.coords[-1]):
            # CCW winding (positive signed area) → h_fwd is on the interior side.
            a, b = arc.coords[:-1], arc.coords[1:]
            signed_area = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()) * 0.5
            island_fid  = h_fwd.face if signed_area > 0 else h_bwd.face
            island_face = mesh.face_region[island_fid]
            seeds = face_seeds.get(island_face)
            if seeds is not None and len(seeds):
                # Multiple disjoint faces can share the same label, so face_seeds
                # may contain seeds from other components. Filter to seeds that
                # actually lie inside this island polygon before picking the anchor.
                island_poly = Polygon(arc.coords)
                mask = np.array([island_poly.contains(Point(s)) for s in seeds])
                candidates = seeds[mask] if mask.any() else seeds
                anchor = candidates[np.argmin(
                    np.linalg.norm(candidates - arc.coords[0], axis=1))]
                features = np.concatenate([features, anchor.reshape(1, 2)])

        simplified[id(arc)] = topovw_modified(arc.coords, epsilon, features)

    arc_map = {id(arc): simplified[id(arc)] for arc, _, _ in entries}
    return mesh.copy_with_new_coords(arc_map)


def simplify_deberg(mesh: HalfEdgeMesh, epsilon: float) -> HalfEdgeMesh:
    """de Berg et al. (1998)-simplify every arc using all other arcs' vertices
    as features (same per-arc formulation as simplify_topovw_modified / simplify_saalfeld).
    """
    seen:    set[int]                              = set()
    entries: list[tuple[Arc, HalfEdge, HalfEdge]] = []
    for h in mesh.half_edges:
        if id(h.arc) in seen:
            continue
        seen.add(id(h.arc))
        h_fwd = h if h.forward else h.twin
        entries.append((h.arc, h_fwd, h_fwd.twin))

    all_arcs = [arc for arc, _, _ in entries]

    simplified: dict[int, np.ndarray] = {}
    for arc, _, _ in entries:
        others   = [a.coords for a in all_arcs if a is not arc]
        features = np.concatenate(others) if others else np.empty((0, 2))
        simplified[id(arc)] = deberg(arc.coords, epsilon, features)

    arc_map = {id(arc): simplified[id(arc)] for arc, _, _ in entries}
    return mesh.copy_with_new_coords(arc_map)


def count_interior_points(mesh: HalfEdgeMesh) -> int:
    """Total interior (non-endpoint) vertices across all unique arcs."""
    seen: set[int] = set()
    total = 0
    for h in mesh.half_edges:
        if id(h.arc) in seen:
            continue
        seen.add(id(h.arc))
        total += max(0, len(h.arc.coords) - 2)
    return total


def simplify_topovw(
    mesh: HalfEdgeMesh,
    n_remove: int,
    dummy_eps: float = 1e-8,
) -> HalfEdgeMesh:
    """Paper-faithful TopoVW (global queue, count budget, dynamic obstacles).

    This is the true TopoVW algorithm as described in the paper. The per-arc
    variant simplify_topovw_modified is a modified formulation; see the README.

    Differences from simplify_topovw_modified:
      - Single global priority queue across all arcs, not per-arc heaps
      - Terminates on a point-count budget (n_remove), not an epsilon threshold
      - Obstacle set is dynamic: removed vertices leave the grid each step
      - Blocked vertices are re-tested when a remote removal may unblock them
      - Plain triangle_area recompute for neighbours (no monotone max() guard)
      - Dummy control points for closed loops, lens pairs, and two-point arcs
      - Edge-inclusive triangle containment (boundary counts as inside)
      - Collinear (zero-area) vertices removed first outside the heap

    n_remove: number of interior vertices to remove. Use
        count_interior_points(mesh) // 2  for the paper's 50% target.
    dummy_eps: perpendicular offset for dummy control points.
    """
    import heapq

    # ── collect unique arcs ───────────────────────────────────────────────────
    seen: set[int] = set()
    entries: list[tuple[Arc, HalfEdge, HalfEdge]] = []
    for h in mesh.half_edges:
        if id(h.arc) in seen:
            continue
        seen.add(id(h.arc))
        h_fwd = h if h.forward else h.twin
        entries.append((h.arc, h_fwd, h_fwd.twin))

    all_arcs = [arc for arc, _, _ in entries]
    na = len(all_arcs)

    # ── per-arc linked lists ──────────────────────────────────────────────────
    nv       = [len(a.coords) for a in all_arcs]
    prev_idx = [list(range(-1, n - 1)) for n in nv]
    next_idx = [list(range(1,  n + 1)) for n in nv]
    removed  = [[False] * n for n in nv]
    version  = [[0]     * n for n in nv]

    def tri_area(ai: int, vi: int) -> float:
        c = all_arcs[ai].coords
        p, q, r = c[prev_idx[ai][vi]], c[vi], c[next_idx[ai][vi]]
        return 0.5 * abs((q[0]-p[0])*(r[1]-p[1]) - (r[0]-p[0])*(q[1]-p[1]))

    # ── dynamic spatial grid (obstacles leave as vertices are removed) ────────
    all_interior = [
        all_arcs[ai].coords[vi]
        for ai in range(na)
        for vi in range(1, nv[ai] - 1)
    ]
    if not all_interior:
        return _rebuild_mesh_from_coords(
            entries, {id(arc): arc.coords for arc in all_arcs}, mesh)

    pts_all  = np.stack(all_interior)
    span     = float(np.max(pts_all.max(0) - pts_all.min(0)))
    cs       = max(span / 200.0, 1e-12)   # cell size

    def _cell(x: float, y: float) -> tuple[int, int]:
        return int(np.floor(x / cs)), int(np.floor(y / cs))

    dyn_grid: dict[tuple[int, int], list]           = defaultdict(list)
    v_cell:   list[list[tuple[int, int] | None]]    = [[None] * n for n in nv]

    def _dyn_insert(ai: int, vi: int) -> None:
        x, y = all_arcs[ai].coords[vi]
        key  = _cell(x, y)
        dyn_grid[key].append((x, y, ai, vi))
        v_cell[ai][vi] = key

    def _dyn_remove(ai: int, vi: int) -> None:
        key = v_cell[ai][vi]
        dyn_grid[key] = [(x, y, a, v) for x, y, a, v in dyn_grid[key]
                         if not (a == ai and v == vi)]
        v_cell[ai][vi] = None

    for ai in range(na):
        _dyn_insert(ai, 0)               # endpoints are pinned but still obstacles
        _dyn_insert(ai, nv[ai] - 1)
        for vi in range(1, nv[ai] - 1):
            _dyn_insert(ai, vi)

    # ── fixed dummy-point grid (never removed) ────────────────────────────────
    fixed_grid: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)

    def _add_dummy_around_segment(a_pt: np.ndarray, b_pt: np.ndarray) -> None:
        mid  = 0.5 * (a_pt + b_pt)
        d    = b_pt - a_pt
        perp = np.array([-d[1], d[0]])
        norm = float(np.linalg.norm(perp))
        if norm < 1e-30:
            return
        perp = perp / norm * dummy_eps
        for pt in (mid + perp, mid - perp):
            fixed_grid[_cell(pt[0], pt[1])].append(pt)

    def _add_dummy_for_arc(ai: int) -> None:
        """Proactive: add dummies around the original first segment."""
        coords = all_arcs[ai].coords
        if len(coords) < 2:
            return
        _add_dummy_around_segment(coords[0], coords[1])

    def _add_dummy_for_chord(ai: int) -> None:
        """Reactive: add dummies around the current chord coords[0]→coords[-1]."""
        coords = all_arcs[ai].coords
        if len(coords) < 2:
            return
        _add_dummy_around_segment(coords[0], coords[-1])

    # (a) closed loops
    for ai, arc in enumerate(all_arcs):
        if np.allclose(arc.coords[0], arc.coords[-1]) and len(arc.coords) > 2:
            _add_dummy_for_arc(ai)

    # (b) lens pairs — arcs sharing both endpoints
    ep_key_to_arcs: dict[tuple, list[int]] = defaultdict(list)
    for ai, arc in enumerate(all_arcs):
        p0 = tuple(np.round(arc.coords[0],  8))
        p1 = tuple(np.round(arc.coords[-1], 8))
        ep_key_to_arcs[(min(p0, p1), max(p0, p1))].append(ai)

    for ais in ep_key_to_arcs.values():
        if len(ais) >= 2:
            for ai in ais:
                _add_dummy_for_arc(ai)

    # (c) two-point arcs (no interior vertices)
    for ai, arc in enumerate(all_arcs):
        if len(arc.coords) == 2:
            _add_dummy_for_arc(ai)

    # ── edge-inclusive point-in-triangle ──────────────────────────────────────
    def _in_tri(pts: np.ndarray, a: np.ndarray, v: np.ndarray,
                b: np.ndarray) -> np.ndarray:
        v0, v1 = b - a, v - a
        d00 = float(v0 @ v0); d01 = float(v0 @ v1); d11 = float(v1 @ v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            # degenerate: point on segment a-b counts as inside
            ab = b - a; ab2 = float(ab @ ab)
            if ab2 < 1e-24:
                return np.zeros(len(pts), dtype=bool)
            t    = ((pts - a) @ ab) / ab2
            proj = a + np.outer(t, ab)
            return (t >= -1e-10) & (t <= 1.0 + 1e-10) & (
                np.sum((pts - proj) ** 2, axis=1) < 1e-10)
        w  = pts - a
        u  = (d11 * (w @ v0) - d01 * (w @ v1)) / denom
        vv = (d00 * (w @ v1) - d01 * (w @ v0)) / denom
        return (u >= -1e-10) & (vv >= -1e-10) & (u + vv <= 1.0 + 1e-10)

    def ear_is_safe(ai: int, vi: int) -> bool:
        c = all_arcs[ai].coords
        a, v, b = c[prev_idx[ai][vi]], c[vi], c[next_idx[ai][vi]]
        xmin = min(a[0], v[0], b[0]); xmax = max(a[0], v[0], b[0])
        ymin = min(a[1], v[1], b[1]); ymax = max(a[1], v[1], b[1])
        cx0 = int(np.floor(xmin / cs)); cx1 = int(np.floor(xmax / cs))
        cy0 = int(np.floor(ymin / cs)); cy1 = int(np.floor(ymax / cs))
        # exclude only the three ear corners, not the whole arc —
        # non-adjacent vertices of arc ai that fall inside the ear must still block
        ear_corners = frozenset({
            (ai, prev_idx[ai][vi]),
            (ai, vi),
            (ai, next_idx[ai][vi]),
        })
        pts_list: list[np.ndarray] = []
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                for x, y, a2, vi2 in dyn_grid.get((cx, cy), []):
                    if (a2, vi2) not in ear_corners:
                        pts_list.append(np.array([x, y]))
                pts_list.extend(fixed_grid.get((cx, cy), []))
        if not pts_list:
            return True
        return not bool(np.any(_in_tri(np.stack(pts_list), a, v, b)))

    # ── phase 1: remove collinear (zero-area) vertices ────────────────────────
    zero_removed = 0
    for ai in range(na):
        for vi in range(1, nv[ai] - 1):
            if not removed[ai][vi] and tri_area(ai, vi) == 0.0 and ear_is_safe(ai, vi):
                removed[ai][vi] = True
                _dyn_remove(ai, vi)
                p, nx = prev_idx[ai][vi], next_idx[ai][vi]
                if nx < nv[ai]:    prev_idx[ai][nx] = p
                if p  >= 0:        next_idx[ai][p]  = nx
                zero_removed += 1
                # reactive dummy: arc may have just collapsed to 2 points
                if next_idx[ai][0] == nv[ai] - 1:
                    _add_dummy_for_chord(ai)

    # ── phase 2: global heap, count budget ───────────────────────────────────
    heap: list = []
    for ai in range(na):
        for vi in range(1, nv[ai] - 1):
            if not removed[ai][vi]:
                heapq.heappush(heap, (tri_area(ai, vi), version[ai][vi], ai, vi))

    blocked: set[tuple[int, int]] = set()
    budget  = max(0, n_remove - zero_removed)
    done    = 0

    while done < budget and heap:
        area, ver, ai, vi = heapq.heappop(heap)

        if removed[ai][vi] or ver != version[ai][vi]:
            continue

        if not ear_is_safe(ai, vi):
            blocked.add((ai, vi))
            continue

        # ── remove ──
        removed[ai][vi] = True
        rem_pos = all_arcs[ai].coords[vi].copy()
        _dyn_remove(ai, vi)
        done += 1

        p, nx = prev_idx[ai][vi], next_idx[ai][vi]
        if nx < nv[ai]:  prev_idx[ai][nx] = p
        if p  >= 0:      next_idx[ai][p]  = nx

        # reactive dummy: if arc just became a 2-point arc, protect it
        if next_idx[ai][0] == nv[ai] - 1:
            _add_dummy_for_chord(ai)

        # update neighbours — plain recompute, no max() guard
        for nb in (p, nx):
            if 0 < nb < nv[ai] - 1 and not removed[ai][nb]:
                version[ai][nb] += 1
                heapq.heappush(heap, (tri_area(ai, nb), version[ai][nb], ai, nb))

        # re-test blocked vertices whose triangle contained rem_pos
        newly_unblocked: list[tuple[int, int]] = []
        for (bai, bvi) in blocked:
            if removed[bai][bvi]:
                newly_unblocked.append((bai, bvi))
                continue
            bc = all_arcs[bai].coords
            ba, bv, bb = bc[prev_idx[bai][bvi]], bc[bvi], bc[next_idx[bai][bvi]]
            xmin = min(ba[0], bv[0], bb[0]); xmax = max(ba[0], bv[0], bb[0])
            ymin = min(ba[1], bv[1], bb[1]); ymax = max(ba[1], bv[1], bb[1])
            if xmin <= rem_pos[0] <= xmax and ymin <= rem_pos[1] <= ymax:
                if _in_tri(rem_pos.reshape(1, 2), ba, bv, bb)[0]:
                    version[bai][bvi] += 1
                    heapq.heappush(heap,
                                   (tri_area(bai, bvi), version[bai][bvi], bai, bvi))
                    newly_unblocked.append((bai, bvi))
        for key in newly_unblocked:
            blocked.discard(key)

    # ── rebuild mesh ──────────────────────────────────────────────────────────
    simplified = {
        id(arc): arc.coords[~np.array(removed[ai], dtype=bool)]
        for ai, (arc, _, _) in enumerate(entries)
    }
    return _rebuild_mesh_from_coords(entries, simplified, mesh)


def _rebuild_mesh_from_coords(
    entries: list[tuple[Arc, HalfEdge, HalfEdge]],
    simplified: dict[int, np.ndarray],
    mesh: HalfEdgeMesh,
) -> HalfEdgeMesh:
    """Shared mesh-rebuild helper: delegates to copy_with_new_coords."""
    arc_map = {id(arc): simplified[id(arc)] for arc, _, _ in entries}
    return mesh.copy_with_new_coords(arc_map)


def simplify_saalfeld(mesh: HalfEdgeMesh, epsilon: float) -> HalfEdgeMesh:
    """Saalfeld-simplify every arc, using all other arcs' vertices as features.

    Each arc is simplified in turn against the original (pre-simplification)
    vertex positions of every other arc, preventing topological crossings.
    Topology (next/prev/twin/face) is preserved exactly.
    """
    # Collect unique arcs and their canonical (forward) half-edges
    seen:    set[int]                          = set()
    entries: list[tuple[Arc, HalfEdge, HalfEdge]] = []
    for h in mesh.half_edges:
        if id(h.arc) in seen:
            continue
        seen.add(id(h.arc))
        h_fwd = h if h.forward else h.twin
        entries.append((h.arc, h_fwd, h_fwd.twin))

    all_arcs = [arc for arc, _, _ in entries]

    # Simplify each arc; features = all vertices of every other arc
    simplified: dict[int, np.ndarray] = {}
    for arc, _, _ in entries:
        others   = [a.coords for a in all_arcs if a is not arc]
        features = np.concatenate(others) if others else np.empty((0, 2))
        simplified[id(arc)] = saalfeld(arc.coords, features, epsilon)

    arc_map = {id(arc): simplified[id(arc)] for arc, _, _ in entries}
    return mesh.copy_with_new_coords(arc_map)

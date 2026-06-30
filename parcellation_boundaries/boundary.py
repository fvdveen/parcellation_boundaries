"""Point-cloud loading and Voronoi boundary extraction for brain parcellations."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial import Delaunay, Voronoi, cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely.ops import unary_union, polygonize, substring, voronoi_diagram
import shapely.geometry as geometry

from .half_edge import HalfEdgeMesh


# ── Data loading ──────────────────────────────────────────────────────────────

def load_hemisphere_data(path: Path) -> dict:
    """Load and normalise a hemisphere .mat file.

    Returns {region_name: {'x': ndarray, 'y': ndarray, 'color': (r,g,b)}}.
    Region names already carry a '-lh' or '-rh' suffix.
    """
    patch_data = loadmat(path)
    colors = patch_data['color']
    labels = patch_data['labels']
    x = patch_data['x'][0]
    y = patch_data['y'][0]
    vno = patch_data['vno'][0]
    labels = labels[vno]
    region_names = [r[0][0] for r in patch_data['region_names']]

    hemi = 'lh' if 'lh' in path.name else 'rh'
    coordinates_max = 100
    gap = 0.06

    if hemi == 'lh':
        x, y = -x, -y
        x = (x - x.min()) / (x.max() - x.min()) * coordinates_max \
            - coordinates_max - (gap / 2) * coordinates_max
        y = (y - y.min()) / (y.max() - y.min()) * coordinates_max
    else:
        x = (x - x.min()) / (x.max() - x.min()) * coordinates_max \
            + (gap / 2) * coordinates_max
        y = (y - y.min()) / (y.max() - y.min()) * coordinates_max

    regions = {}
    for i in range(colors.shape[0]):
        if i < 1:  # skip medial wall
            continue
        name = region_names[i] + '-' + hemi
        sel = np.where(labels == colors[i, 4])[0]
        if len(sel) == 0:
            continue
        regions[name] = {
            'x':     x[sel],
            'y':     y[sel],
            'color': tuple(colors[i, :3] / 255),
        }
    return regions


# ── Point cloud processing ────────────────────────────────────────────────────

def _voronoi_cell_areas(pts: np.ndarray, hull_poly) -> np.ndarray:
    """Per-point Voronoi cell area, clipped to hull_poly."""
    mp = geometry.MultiPoint(pts.tolist())
    regions = voronoi_diagram(mp, envelope=hull_poly)
    tree = cKDTree(pts)
    areas = np.zeros(len(pts))
    for poly in regions.geoms:
        centroid = np.array(poly.centroid.coords[0])
        _, idx = tree.query(centroid)
        areas[idx] += poly.intersection(hull_poly).area
    return areas


def filter_components(pts: np.ndarray, labels: np.ndarray,
                      tri: Delaunay, hull_poly,
                      min_points: int = 2,
                      min_area: float = 1.0) -> tuple:
    """Drop same-label connected components that are too small.

    A component is kept only if it meets BOTH criteria:
      - at least min_points points  (set to 1 to disable)
      - total Voronoi cell area >= min_area  (set to 0 to disable)

    Area is the sum of per-point Voronoi cell areas clipped to hull_poly,
    not the crude hull.area/N approximation.
    Returns (pts_filtered, labels_filtered, tri_filtered).
    """
    s = tri.simplices
    edges = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    same = labels[edges[:, 0]] == labels[edges[:, 1]]
    e = edges[same]
    n = len(pts)
    adj = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, comp_id = connected_components(adj + adj.T, directed=False)

    n_comps = int(comp_id.max()) + 1
    comp_sizes = np.bincount(comp_id, minlength=n_comps)
    size_ok = comp_sizes[comp_id] >= min_points

    if min_area > 0:
        pt_areas = _voronoi_cell_areas(pts, hull_poly)
        comp_areas = np.bincount(comp_id, weights=pt_areas, minlength=n_comps)
        area_ok = comp_areas[comp_id] >= min_area
        keep = size_ok & area_ok
    else:
        keep = size_ok

    pts_f = pts[keep]
    labs_f = labels[keep]
    return pts_f, labs_f, Delaunay(pts_f)


# ── Hull ──────────────────────────────────────────────────────────────────────

def alpha_shape(tri: Delaunay, coords: np.ndarray, alpha: float):
    """Concave hull (alpha shape) of a point set."""
    if len(coords) < 4:
        return geometry.MultiPoint(coords.tolist()).convex_hull

    ia, ib, ic = tri.simplices[:, 0], tri.simplices[:, 1], tri.simplices[:, 2]
    pa, pb, pc = coords[ia], coords[ib], coords[ic]
    ab, ac = pb - pa, pc - pa
    area = 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    a_len = np.linalg.norm(pa - pb, axis=1)
    b_len = np.linalg.norm(pb - pc, axis=1)
    c_len = np.linalg.norm(pc - pa, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        circum_r = a_len * b_len * c_len / (4.0 * area)

    filtered = tri.simplices[circum_r < 1.0 / alpha]
    edge_idx = np.sort(np.concatenate([
        filtered[:, [0, 1]], filtered[:, [1, 2]], filtered[:, [2, 0]],
    ]), axis=1)
    unique_edges = np.unique(edge_idx, axis=0)
    m = geometry.MultiLineString(coords[unique_edges].tolist())
    return unary_union(list(polygonize(m)))


# ── Arc extraction ────────────────────────────────────────────────────────────

def _largest_polygon(geom):
    if geom.geom_type == 'MultiPolygon':
        return max(geom.geoms, key=lambda p: p.area)
    return geom


def _extract_brain_arcs(vor: Voronoi, labels: np.ndarray,
                        region_names: list[str], hull_poly) -> list:
    """Clip cross-region Voronoi arcs to hull_poly and walk them into arcs.

    Returns list of (coords, left_face_name, right_face_name).
    left_face is the region to the LEFT of the forward direction.
    """
    hull_poly = _largest_polygon(hull_poly)

    center = vor.points.mean(axis=0)
    ext = max(hull_poly.bounds[2] - hull_poly.bounds[0],
              hull_poly.bounds[3] - hull_poly.bounds[1])
    far = ext * 5

    ridge_pts_arr = np.array(vor.ridge_points)
    ridge_verts_arr = np.array(vor.ridge_vertices)
    l0s = labels[ridge_pts_arr[:, 0]]
    l1s = labels[ridge_pts_arr[:, 1]]
    cross_idx = np.where(l0s != l1s)[0]

    PREC = 6

    def rkey(xy):
        return (round(float(xy[0]), PREC), round(float(xy[1]), PREC))

    ridge_info = {}          # ri → (ka, kb, l0, l1, site0_xy)
    adj = defaultdict(list)
    coord_xy = {}

    for ri in cross_idx:
        v0, v1 = ridge_verts_arr[ri]
        l0, l1 = int(l0s[ri]), int(l1s[ri])
        s0 = vor.points[ridge_pts_arr[ri, 0]]
        s1 = vor.points[ridge_pts_arr[ri, 1]]

        if v0 >= 0 and v1 >= 0:
            line = geometry.LineString([vor.vertices[v0], vor.vertices[v1]])
        elif v0 == -1 and v1 == -1:
            continue
        else:
            fv = vor.vertices[v1 if v0 == -1 else v0]
            tang = s1 - s0
            norm = np.array([-tang[1], tang[0]])
            if np.dot(norm, fv - center) < 0:
                norm = -norm
            norm /= np.linalg.norm(norm)
            line = geometry.LineString([fv, fv + norm * far])

        clipped = line.intersection(hull_poly)
        if clipped.is_empty or clipped.geom_type == 'Point':
            continue
        if clipped.geom_type == 'MultiLineString':
            clipped = max(clipped.geoms, key=lambda g: g.length)
        if clipped.geom_type != 'LineString':
            continue

        rcoords = np.array(clipped.coords)
        if len(rcoords) < 2:
            continue

        A, B = rcoords[0], rcoords[-1]
        ka, kb = rkey(A), rkey(B)
        coord_xy[ka] = A
        coord_xy[kb] = B
        ridge_info[ri] = (ka, kb, l0, l1, s0)
        adj[ka].append((kb, ri))
        adj[kb].append((ka, ri))

    # Junctions: degree != 2 (hull boundary endpoints have degree 1)
    junctions = {k for k, nbrs in adj.items() if len(nbrs) != 2}

    visited = set()

    def walk(start_k, next_k, start_ri):
        keys = [start_k, next_k]
        visited.add(start_ri)
        prev_k, cur_k = start_k, next_k
        while cur_k not in junctions:
            nexts = [(nk, ri) for nk, ri in adj[cur_k]
                     if nk != prev_k and ri not in visited]
            if not nexts:
                break
            nk, ri = nexts[0]
            visited.add(ri)
            keys.append(nk)
            prev_k, cur_k = cur_k, nk

        coords = np.array([coord_xy[k] for k in keys])
        _, _, l0, l1, s0 = ridge_info[start_ri]
        A = coord_xy[start_k]
        d = coord_xy[next_k] - A
        left_is_0 = d[0] * (s0[1] - A[1]) - d[1] * (s0[0] - A[0]) > 0
        fl = l0 if left_is_0 else l1
        fr = l1 if left_is_0 else l0
        return coords, region_names[fl], region_names[fr]

    arcs = []
    for jk in sorted(junctions):
        for nk, ri in adj[jk]:
            if ri not in visited:
                arcs.append(walk(jk, nk, ri))
    for ri in ridge_info:
        if ri not in visited:
            ka, kb, *_ = ridge_info[ri]
            arcs.append(walk(ka, kb, ri))

    return arcs


def _hull_arcs(hull_poly, brain_arcs: list,
               pts: np.ndarray, labels: np.ndarray,
               region_names: list[str]) -> list:
    """Generate arcs along the hull exterior between brain-arc junction points.

    Returns list of (coords, inner_face_name, '__hull__').
    Hull exterior is CCW so the brain-side region is to the LEFT.
    """
    hull_poly = _largest_polygon(hull_poly)
    hull_ext = hull_poly.exterior
    hull_len = hull_ext.length
    tol = 1e-4
    PREC = 6

    def rkey(xy):
        return (round(float(xy[0]), PREC), round(float(xy[1]), PREC))

    seen_keys = {}
    junc_xys = []
    for coords, _, _ in brain_arcs:
        for ep in (coords[0], coords[-1]):
            if hull_ext.distance(geometry.Point(ep)) < tol:
                k = rkey(ep)
                if k not in seen_keys:
                    seen_keys[k] = True
                    junc_xys.append(ep)

    if len(junc_xys) < 2:
        return []

    params = [hull_ext.project(geometry.Point(xy)) for xy in junc_xys]
    order = np.argsort(params)
    s_params = [params[i] for i in order]
    s_xys = [junc_xys[i] for i in order]

    kd = cKDTree(pts)
    n = len(s_params)
    result = []

    for i in range(n):
        t0 = s_params[i]
        t1 = s_params[(i + 1) % n]
        xy0 = s_xys[i]
        xy1 = s_xys[(i + 1) % n]

        if t1 > t0:
            seg = substring(hull_ext, t0, t1)
        else:
            seg = geometry.LineString(
                list(substring(hull_ext, t0, hull_len).coords)
                + list(substring(hull_ext, 0.0, t1).coords)[1:]
            )

        seg_coords = np.array(seg.coords)
        if len(seg_coords) < 2:
            continue

        seg_coords[0] = xy0
        seg_coords[-1] = xy1

        mid_t = (t0 + t1) / \
            2 if t1 > t0 else ((t0 + hull_len + t1) / 2) % hull_len
        mid_pt = hull_ext.interpolate(mid_t)
        _, idx = kd.query([mid_pt.x, mid_pt.y])
        inner = int(labels[idx])

        result.append((seg_coords, '__hull__', region_names[inner]))

    return result


# ── Top-level builder ─────────────────────────────────────────────────────────

def build_mesh(data: dict,
               alpha: float = 2.8,
               min_points: int = 2,
               min_area: float = 1.0) -> tuple[HalfEdgeMesh, dict]:
    """Build a half-edge mesh from a regions dict (as returned by load_hemisphere).

    data:       {region_name: {'x': ndarray, 'y': ndarray, 'color': (r,g,b)}}
    min_points: drop components with fewer points than this (1 = keep all)
    min_area:   drop components whose total Voronoi cell area is below this
                threshold in coord units²  (0 = keep all)
    """
    region_names = list(data.keys())

    pts = np.column_stack([
        np.concatenate([r['x'] for r in data.values()]),
        np.concatenate([r['y'] for r in data.values()]),
    ])
    labels = np.repeat(np.arange(len(data)),
                       [len(r['x']) for r in data.values()])
    tri = Delaunay(pts)
    hull = alpha_shape(tri, pts, alpha=alpha)
    pts, labels, tri = filter_components(pts, labels, tri, hull,
                                         min_points=min_points, min_area=min_area)

    vor = Voronoi(pts)
    brain_arcs = _extract_brain_arcs(vor, labels, region_names, hull)
    h_arcs = _hull_arcs(hull, brain_arcs, pts, labels, region_names)

    mesh = HalfEdgeMesh.from_arcs(brain_arcs + h_arcs)
    return mesh, data


# ── Arc serialisation helpers ─────────────────────────────────────────────────

def _save_arcs(path: Path, arcs: list) -> None:
    arrays = {f'coords_{i}': a[0] for i, a in enumerate(arcs)}
    meta = [(a[1], a[2]) for a in arcs]
    np.savez_compressed(path, **arrays)
    (path.parent / (path.stem + '_meta.json')).write_text(
        json.dumps(meta), encoding='utf-8')


def _load_arcs(path: Path) -> list:
    npz = np.load(path)
    meta = json.loads((path.parent / (path.stem + '_meta.json')).read_text())
    return [(npz[f'coords_{i}'], lf, rf) for i, (lf, rf) in enumerate(meta)]


def _save_regions(path: Path, data: dict) -> None:
    arrays = {}
    meta = {}
    for name, r in data.items():
        key = name.replace('/', '__')
        arrays[f'{key}__x'] = r['x']
        arrays[f'{key}__y'] = r['y']
        meta[name] = list(r['color'])
    np.savez_compressed(path, **arrays)
    (path.parent / (path.stem + '_meta.json')).write_text(
        json.dumps(meta), encoding='utf-8')


def _load_regions(path: Path) -> dict:
    npz = np.load(path)
    meta = json.loads((path.parent / (path.stem + '_meta.json')).read_text())
    regions = {}
    for name, color in meta.items():
        key = name.replace('/', '__')
        regions[name] = {
            'x':     npz[f'{key}__x'],
            'y':     npz[f'{key}__y'],
            'color': tuple(color),
        }
    return regions


# ── Cached build ──────────────────────────────────────────────────────────────

def build_mesh_cached(atlas_or_annot, hemi: str,
                      alpha: float = 2.8,
                      min_points: int = 2,
                      min_area: float = 1.0,
                      cache_dir: Path = Path('data/cache'),
                      force: bool = False) -> tuple[HalfEdgeMesh, dict]:
    """Load and build a half-edge mesh, caching arc data to disk.

    Cache layout:
        cache/{atlas}/alpha{alpha}_mp{min_points}_ma{min_area}/
            lh_arcs.npz + lh_arcs_meta.json
            lh_regions.npz + lh_regions_meta.json
            rh_arcs.npz + ...

    Both hemispheres of the same atlas+params share a folder.
    A cache hit requires the same atlas, hemi, alpha, min_points, and min_area.
    """
    from .load_freesurfer_data import Atlas, load_hemisphere

    atlas_key = (atlas_or_annot.name
                 if isinstance(atlas_or_annot, Atlas)
                 else Path(atlas_or_annot).stem)
    param_key = f'alpha{alpha:g}_mp{min_points}_ma{min_area:g}'
    slot = cache_dir / atlas_key / param_key
    arcs_path = slot / f'{hemi}_arcs.npz'
    regions_path = slot / f'{hemi}_regions.npz'

    if not force and arcs_path.exists() and regions_path.exists():
        arcs = _load_arcs(arcs_path)
        data = _load_regions(regions_path)
        mesh = HalfEdgeMesh.from_arcs(arcs)
        return mesh, data

    data = load_hemisphere(atlas_or_annot, hemi)
    mesh, data = build_mesh(data, alpha=alpha,
                            min_points=min_points, min_area=min_area)

    slot.mkdir(parents=True, exist_ok=True)
    all_arcs = [(h.arc.coords,
                 mesh.face_region[h.face],
                 mesh.face_region[h.twin.face])
                for h in mesh.half_edges
                if h.forward]
    _save_arcs(arcs_path, all_arcs)
    _save_regions(regions_path, data)

    return mesh, data

import numpy as np
import shapely
from shapely import hausdorff_distance
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


def _polygon_vertex_count(geom: BaseGeometry) -> int:
    polys = geom.geoms if hasattr(geom, 'geoms') else [geom]
    return sum(
        len(p.exterior.coords) + sum(len(i.coords) for i in p.interiors)
        for p in polys
        if p.geom_type == 'Polygon'
    )


def _geom_exterior_coords(geom: BaseGeometry) -> np.ndarray:
    polys = geom.geoms if hasattr(geom, 'geoms') else [geom]
    parts = [np.array(p.exterior.coords)
             for p in polys if p.geom_type == 'Polygon']
    return np.concatenate(parts) if parts else np.empty((0, 2))


# Sample interval along simplified edges for the directed Hausdorff distance,
# in world (coordinate) units.
HD1_SAMPLE_SPACING = 0.025


def _sample_boundary(geom: BaseGeometry, spacing: float) -> np.ndarray:
    """Points sampled every `spacing` world units along each exterior edge.

    Both endpoints of every edge are always included; edges shorter than
    `spacing` contribute just their two endpoints. Rings are sampled
    independently so no spurious edge is introduced between polygons.
    """
    polys = geom.geoms if hasattr(geom, 'geoms') else [geom]
    pts_list = []
    for p in polys:
        if p.geom_type != 'Polygon':
            continue
        coords = np.asarray(p.exterior.coords)
        for i in range(len(coords) - 1):
            a, b = coords[i], coords[i + 1]
            edge_len = float(np.linalg.norm(b - a))
            if edge_len <= 0:
                continue
            ds = np.concatenate(
                [[0.0], np.arange(spacing, edge_len, spacing), [edge_len]])
            pts_list.append(a + np.outer(ds, (b - a) / edge_len))
    return np.concatenate(pts_list) if pts_list else np.empty((0, 2))


# ── Metrics ───────────────────────────────────────────────────────────────────

def vertex_count(shapes: dict) -> dict:
    """Total vertices across all region polygons, and per-region breakdown."""
    per_region = {name: _polygon_vertex_count(
        geom) for name, geom in shapes.items()}
    return {'total': sum(per_region.values()), 'per_region': per_region}


def trace_count(shapes: dict) -> int:
    """Total polygon components (Plotly traces) across all regions."""
    total = 0
    for geom in shapes.values():
        total += len(geom.geoms) if hasattr(geom, 'geoms') else 1
    return total


def vertex_retention(original: dict, simplified: dict) -> float:
    """Fraction of original vertices retained after simplification (0–1)."""
    orig_total = vertex_count(original)['total']
    simp_total = vertex_count(simplified)['total']
    return simp_total / orig_total if orig_total > 0 else 1.0


def hausdorff(original: dict, simplified: dict) -> dict:
    """Symmetric Hausdorff distance per region, in world units."""
    common = [n for n in original if n in simplified]
    distances = np.array(
        [hausdorff_distance(original[n], simplified[n]) for n in common])
    return {
        'max':  float(distances.max())  if len(distances) else float('nan'),
        'mean': float(distances.mean()) if len(distances) else float('nan'),
        'per_region': dict(zip(common, distances.tolist())),
    }


def hausdorff_one_sided(original: dict, simplified: dict,
                        spacing: float = HD1_SAMPLE_SPACING) -> dict:
    """Directed Hausdorff distance from the simplified boundary to the original boundary.

    The simplified boundary is sampled every `spacing` world units along each
    edge interior (not just at vertices), so the metric captures how far the
    straightened edges drift from the original outline, not only the corners.
    """
    common = [n for n in original if n in simplified]
    per_region = {}
    for name in common:
        coords = _sample_boundary(simplified[name], spacing)
        if len(coords) == 0:
            per_region[name] = 0.0
            continue
        pts = shapely.points(coords)
        dists = shapely.distance(pts, original[name].boundary)
        finite = dists[np.isfinite(dists)]
        per_region[name] = float(finite.max()) if len(finite) else 0.0
    values = np.array(list(per_region.values()))
    return {
        'max':  float(values.max())  if len(values) else float('nan'),
        'mean': float(values.mean()) if len(values) else float('nan'),
        'per_region': per_region,
    }


def iou(original: dict, simplified: dict) -> dict:
    """Intersection-over-union per region. Reports min and mean across regions."""
    common = [n for n in original if n in simplified]
    per_region = {}
    for name in common:
        orig, simp = original[name], simplified[name]
        intersection = orig.intersection(simp).area
        union = orig.union(simp).area
        per_region[name] = intersection / union if union > 0 else 1.0
    values = np.array(list(per_region.values()))
    return {
        'min':  float(values.min())  if len(values) else float('nan'),
        'mean': float(values.mean()) if len(values) else float('nan'),
        'per_region': per_region,
    }


def relative_area_error(original: dict, simplified: dict) -> dict:
    """Relative area error per region: |simplified - original| / original."""
    common = [n for n in original if n in simplified]
    per_region = {}
    for name in common:
        orig_area = original[name].area
        simp_area = simplified[name].area
        per_region[name] = abs(simp_area - orig_area) / \
            orig_area if orig_area > 0 else 0.0
    values = np.array(list(per_region.values()))
    return {
        'max':  float(values.max())  if len(values) else float('nan'),
        'mean': float(values.mean()) if len(values) else float('nan'),
        'per_region': per_region,
    }


def pixel_comparison(original: dict, simplified: dict, resolution: int = 1000) -> dict:
    """Fraction of pixels with matching region label at target rendering resolution.

    Rasterizes both coverages at `resolution` pixels wide (height scaled to match
    aspect ratio). Only pixels covered by at least one region in the original are
    counted. Uses PIL since rasterio/skimage are not in the environment.
    """
    from PIL import Image, ImageDraw

    minx = min(g.bounds[0] for g in original.values())
    miny = min(g.bounds[1] for g in original.values())
    maxx = max(g.bounds[2] for g in original.values())
    maxy = max(g.bounds[3] for g in original.values())

    width = resolution
    scale = resolution / (maxx - minx)
    height = max(1, int((maxy - miny) * scale))

    def to_pixels(coords):
        return [((x - minx) * scale, (maxy - y) * scale) for x, y in coords]

    names = sorted(original.keys())
    label_index = {name: i + 1 for i, name in enumerate(names)}

    def rasterize(shapes):
        img = Image.new('I', (width, height), 0)
        draw = ImageDraw.Draw(img)
        for name in names:
            if name not in shapes:
                continue
            geom = shapes[name]
            polys = geom.geoms if hasattr(geom, 'geoms') else [geom]
            for poly in polys:
                if poly.geom_type != 'Polygon' or poly.is_empty:
                    continue
                draw.polygon(to_pixels(poly.exterior.coords),
                             fill=label_index[name])
        return np.array(img)

    orig_img = rasterize(original)
    simp_img = rasterize(simplified)

    mask = orig_img > 0
    total = int(mask.sum())
    matching = int((orig_img[mask] == simp_img[mask]).sum())
    return {
        'accuracy': matching / total if total > 0 else 1.0,
        'resolution': (width, height),
        'pixels_compared': total,
    }


def label_accuracy(simplified: dict, points: np.ndarray, labels: np.ndarray) -> dict:
    """Fraction of original data points that fall inside their correct simplified region.

    points: (N, 2) original point cloud coordinates.
    labels: (N,) ground-truth region name for each point (string array).

    A point is correct when the simplified region that contains it matches
    its ground-truth label.  Points not covered by any simplified region
    count as wrong.
    """
    if len(points) == 0:
        return {'accuracy': 1.0, 'correct': 0, 'total': 0}

    pts = shapely.points(points)
    assigned = np.full(len(points), '', dtype=object)
    for name, geom in simplified.items():
        inside = shapely.contains(geom, pts)
        assigned[inside] = name

    correct = int(np.sum(assigned == np.asarray(labels, dtype=object)))
    total   = len(points)
    return {
        'accuracy': correct / total,
        'correct':  correct,
        'total':    total,
    }


def _face_component_polys(mesh) -> dict:
    """Build {face_id: Shapely Polygon} for every non-hull face component."""
    from .half_edge import _loop_coords
    out = {}
    for fid, loops in enumerate(mesh.faces):
        if mesh.face_region[fid] == '__hull__' or not loops:
            continue
        ext_c = _loop_coords(loops[0])
        if len(ext_c) < 4:
            continue
        hole_cs = [_loop_coords(h) for h in loops[1:] if len(_loop_coords(h)) >= 4]
        p = Polygon(ext_c, hole_cs)
        if not p.is_valid:
            p = shapely.make_valid(p)
        if not p.is_empty:
            out[fid] = p
    return out


def hausdorff_per_component_both(orig_lh_mesh, simp_lh_mesh,
                                  orig_rh_mesh, simp_rh_mesh,
                                  min_area: float = 0.0) -> dict:
    """hausdorff_per_component for lh+rh combined.

    Runs each hemisphere independently and merges the results.  rh face IDs
    are offset by 10_000_000 so they cannot collide with lh face IDs.
    """
    OFFSET = 10_000_000
    lh = hausdorff_per_component(orig_lh_mesh, simp_lh_mesh, min_area)
    rh = hausdorff_per_component(orig_rh_mesh, simp_rh_mesh, min_area)
    pc = {**lh['per_component'],
          **{fid + OFFSET: d for fid, d in rh['per_component'].items()}}
    values = np.array(list(pc.values())) if pc else np.array([])
    return {
        'max':            float(values.max())  if len(values) else 0.0,
        'mean':           float(values.mean()) if len(values) else 0.0,
        'per_component':  pc,
        'collapsed_fids': lh['collapsed_fids'] + [fid + OFFSET for fid in rh['collapsed_fids']],
        'region_max':     {**lh['region_max'],  **rh['region_max']},
        'region_mean':    {**lh['region_mean'], **rh['region_mean']},
    }


def hausdorff_per_component(original_mesh, simplified_mesh,
                             min_area: float = 0.0) -> dict:
    """Symmetric Hausdorff distance per surviving face component.

    Uses stable face-component IDs from copy_with_new_coords so each component
    in the simplified mesh is matched directly to its original counterpart.

    Components whose simplified polygon has area <= min_area are counted as
    collapsed and excluded from the aggregates.

    Returns:
        max, mean           — over surviving components only
        per_component       — {face_id: distance}
        collapsed_fids      — face IDs that did not survive
        region_max, region_mean — worst/mean Hausdorff per region name
                                  (max over that region's surviving components)
    """
    orig_fp = _face_component_polys(original_mesh)
    simp_fp = _face_component_polys(simplified_mesh)

    per_component: dict[int, float] = {}
    collapsed_fids: list[int] = []

    for fid, orig_p in orig_fp.items():
        simp_p = simp_fp.get(fid)
        if simp_p is None or simp_p.area <= min_area:
            collapsed_fids.append(fid)
            continue
        per_component[fid] = float(hausdorff_distance(orig_p, simp_p))

    values = np.array(list(per_component.values())) if per_component else np.array([])

    # Aggregate per region (max of its surviving components)
    region_vals: dict[str, list[float]] = {}
    for fid, d in per_component.items():
        name = original_mesh.face_region[fid]
        region_vals.setdefault(name, []).append(d)
    region_max  = {n: float(max(vs))  for n, vs in region_vals.items()}
    region_mean = {n: float(sum(vs) / len(vs)) for n, vs in region_vals.items()}

    return {
        'max':            float(values.max())  if len(values) else 0.0,
        'mean':           float(values.mean()) if len(values) else 0.0,
        'per_component':  per_component,
        'collapsed_fids': collapsed_fids,
        'region_max':     region_max,
        'region_mean':    region_mean,
    }

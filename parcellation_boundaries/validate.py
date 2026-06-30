from collections import Counter

import shapely.geometry as geometry
from shapely.ops import unary_union, polygonize


# ── Helpers ───────────────────────────────────────────────────────────────────

def _component_count(geom) -> int:
    return len(geom.geoms) if hasattr(geom, 'geoms') else 1


def _adjacency_set(shapes: dict, tol: float = 1e-6) -> tuple:
    """Returns (edge_set, error_list). Errors are (pair_tuple, exception_type_name)."""
    names = list(shapes.keys())
    edges = set()
    errors = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            try:
                if shapes[na].intersection(shapes[nb]).length > tol:
                    edges.add((na, nb) if na < nb else (nb, na))
            except Exception as e:
                errors.append(
                    ((na, nb) if na < nb else (nb, na), type(e).__name__))
    return edges, errors


def _junction_count_from_arcs(arcs: list) -> int:
    endpoint_counts = Counter()
    for arc in arcs:
        for pt in (arc[0], arc[-1]):
            key = (round(float(pt[0]), 6), round(float(pt[1]), 6))
            endpoint_counts[key] += 1
    return sum(1 for c in endpoint_counts.values() if c >= 3)


def _component_count_above(geom, min_area: float) -> int:
    if min_area <= 0:
        return _component_count(geom)
    polys = geom.geoms if hasattr(geom, 'geoms') else [geom]
    return sum(1 for p in polys if p.geom_type == 'Polygon' and p.area >= min_area)


def _hull_from_outer_arcs(outer_arcs: list):
    outer_lines = [geometry.LineString(oa) for oa in outer_arcs]
    hull_candidates = sorted(polygonize(unary_union(outer_lines)),
                             key=lambda p: p.area, reverse=True)
    return hull_candidates[0] if hull_candidates else None


# ── Checks ────────────────────────────────────────────────────────────────────

def check_completeness(original: dict, simplified: dict) -> dict:
    """All region names present in original are present in simplified, and no extras."""
    missing = sorted(set(original) - set(simplified))
    extra = sorted(set(simplified) - set(original))
    return {'passed': not missing and not extra, 'missing': missing, 'extra': extra}


def check_validity(simplified: dict) -> dict:
    """All simplified polygons are valid Shapely geometries (no self-intersections)."""
    invalid = [n for n, g in simplified.items() if not g.is_valid]
    return {'passed': not invalid, 'invalid': invalid}


def check_component_count(original: dict, simplified: dict,
                          min_component_area: float = 0.0) -> dict:
    """No region gains or loses components larger than min_component_area.

    For regions that lose components, 'removed_components' maps region name to
    a list of (centroid_xy, area) tuples for each dropped polygon.
    """
    changed = {}
    removed_components = {}

    for n in original:
        if n not in simplified:
            continue
        orig_c = _component_count_above(original[n], min_component_area)
        simp_c = _component_count_above(simplified[n], min_component_area)
        if orig_c == simp_c:
            continue
        changed[n] = {'original': orig_c, 'simplified': simp_c}
        if simp_c < orig_c:
            orig_polys = original[n].geoms if hasattr(
                original[n], 'geoms') else [original[n]]
            dropped = [
                p for p in orig_polys
                if p.geom_type == 'Polygon'
                and (min_component_area <= 0 or p.area >= min_component_area)
                and simplified[n].intersection(p).area < 0.5 * p.area
            ]
            if dropped:
                removed_components[n] = [
                    (list(p.centroid.coords[0]), p.area) for p in dropped
                ]

    return {'passed': not changed, 'changed': changed,
            'removed_components': removed_components}


def check_arc_local_overlap(mesh, tol: float = 1e-6) -> dict:
    """For each boundary arc separating regions A and B, check if A and B overlap
    specifically at that arc.

    Overlap is considered local to an arc if the arc itself forms part of the
    boundary of the A∩B intersection polygon — meaning the overlap is right at the
    shared boundary, not a side-effect of a distant arc crossing deforming a polygon.
    """
    from shapely.geometry import Polygon, LineString
    from shapely.ops import unary_union
    from .half_edge import _loop_coords
    import shapely

    def _region_poly(name):
        polys = []
        for fid in mesh.regions.get(name, []):
            loops = mesh.faces[fid]
            if not loops:
                continue
            ext_c = _loop_coords(loops[0])
            if len(ext_c) < 4:
                continue
            hole_cs = [_loop_coords(h) for h in loops[1:] if len(_loop_coords(h)) >= 4]
            p = Polygon(ext_c, hole_cs)
            if not p.is_valid:
                p = shapely.make_valid(p)
            if not p.is_empty:
                polys.append(p)
        if not polys:
            return None
        return unary_union(polys) if len(polys) > 1 else polys[0]

    poly_cache = {}
    def get_poly(name):
        if name not in poly_cache:
            poly_cache[name] = _region_poly(name)
        return poly_cache[name]

    seen = set()
    local_overlaps = []
    errors = []

    for h in mesh.half_edges:
        if id(h.arc) in seen:
            continue
        seen.add(id(h.arc))

        h_fwd = h if h.forward else h.twin
        h_bwd = h_fwd.twin
        na = mesh.face_region[h_fwd.face]
        nb = mesh.face_region[h_bwd.face]

        if na == nb or '__hull__' in (na, nb):
            continue

        PA = get_poly(na)
        PB = get_poly(nb)
        if PA is None or PB is None:
            continue

        shared_line = LineString(h_fwd.arc.coords)

        try:
            overlap = PA.intersection(PB)
            if overlap.area <= tol:
                continue
            if shared_line.intersection(overlap.boundary).length > tol:
                local_overlaps.append((na, nb, overlap.area))
        except Exception as e:
            errors.append((na, nb, type(e).__name__))

    return {'passed': not local_overlaps and not errors,
            'local_overlaps': local_overlaps, 'errors': errors}


def check_adjacent_polygon_overlap(mesh, tol: float = 1e-6) -> dict:
    """Check whether DCEL-adjacent face polygons overlap in area.

    Walks every twin pair once and checks that the reconstructed Shapely polygons
    of neighbouring regions have no interior intersection.  Overlap can be caused
    by an arc crossing anywhere in the mesh — not necessarily the shared arc —
    so use check_no_crossings_mesh to identify the root cause.
    """
    from shapely.ops import unary_union
    from shapely.geometry import Polygon

    def _face_poly(fid):
        loops = mesh.faces[fid]
        if not loops:
            return None
        from .half_edge import _loop_coords
        ext_c = _loop_coords(loops[0])
        if len(ext_c) < 4:
            return None
        hole_cs = [_loop_coords(h) for h in loops[1:] if len(_loop_coords(h)) >= 4]
        return Polygon(ext_c, hole_cs)

    poly_cache = {}
    def get_poly(fid):
        if fid not in poly_cache:
            poly_cache[fid] = _face_poly(fid)
        return poly_cache[fid]

    invalid_faces = []
    seen_invalid = set()
    seen_pairs = set()
    overlapping = []
    errors = []

    for h in mesh.half_edges:
        if h.twin is None:
            continue
        fa, fb = h.face, h.twin.face
        if fa == fb:
            continue

        na = mesh.face_region[fa]
        nb = mesh.face_region[fb]
        if na == nb or '__hull__' in (na, nb):
            continue

        pa = get_poly(fa)
        pb = get_poly(fb)

        for fid, name, p in [(fa, na, pa), (fb, nb, pb)]:
            if fid not in seen_invalid and p is not None and not p.is_valid:
                seen_invalid.add(fid)
                invalid_faces.append((name, fid))

        key = (min(fa, fb), max(fa, fb))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        if pa is None or pb is None or not pa.is_valid or not pb.is_valid:
            continue

        try:
            area = pa.intersection(pb).area
        except Exception as e:
            errors.append((na, nb, type(e).__name__))
            continue

        if area > tol:
            overlapping.append((na, nb, area))

    passed = not invalid_faces and not overlapping and not errors
    return {'passed': passed, 'invalid_faces': invalid_faces,
            'overlapping_pairs': overlapping, 'errors': errors}


def check_no_overlap(simplified: dict, tol: float = 1e-6) -> dict:
    """No two simplified regions share interior area."""
    names = list(simplified.keys())
    overlapping = []
    errors = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            try:
                area = simplified[na].intersection(simplified[nb]).area
            except Exception as e:
                errors.append(f"{na}&{nb}: {type(e).__name__}")
                overlapping.append((na, nb, float('nan')))
                continue
            if area > tol:
                overlapping.append((na, nb, area))
    return {'passed': not overlapping, 'overlapping_pairs': overlapping, 'errors': errors}


def check_adjacency_graph(original: dict, simplified: dict, tol: float = 1e-6) -> dict:
    """The adjacency graph is exactly preserved — no edges lost or gained.

    Combines the former check_adjacency_preserved (no lost edges) and
    check_no_overlap (no gained edges) into one check with full detail.
    """
    common = [n for n in original if n in simplified]
    orig_adj, _ = _adjacency_set({n: original[n] for n in common}, tol)
    simp_adj, errors = _adjacency_set({n: simplified[n] for n in common}, tol)
    lost = sorted(orig_adj - simp_adj)
    gained = sorted(simp_adj - orig_adj)
    return {
        'passed': not lost and not gained and not errors,
        'lost_edges': lost,
        'gained_edges': gained,
        'errors': errors,  # list of (pair_tuple, exception_type_name)
    }


def check_junction_count(original_arcs: list, simplified_arcs: list) -> dict:
    """Same number of degree-3+ arc endpoints in the simplified network as in the original."""
    orig_j = _junction_count_from_arcs(original_arcs)
    simp_j = _junction_count_from_arcs(simplified_arcs)
    return {'passed': orig_j == simp_j, 'original': orig_j, 'simplified': simp_j}


def check_no_gaps(simplified: dict, simplified_outer_arcs: list, tol: float = 1e-6) -> dict:
    """No area inside the simplified hull is left uncovered after simplification.

    A gap appears when a simplified arc creates a polygon fragment containing no
    seed points, so no region claims it.
    """
    hull_poly = _hull_from_outer_arcs(simplified_outer_arcs)
    if hull_poly is None:
        return {'passed': True, 'gap_fraction': 0.0}
    simplified_union = unary_union(list(simplified.values()))
    gap_fraction = hull_poly.difference(simplified_union).area / hull_poly.area
    return {'passed': gap_fraction <= tol, 'gap_fraction': gap_fraction}


def check_no_crossings_mesh(mesh) -> dict:
    """Check for arc crossings in the DCEL, reporting crossings by region pair.

    For each crossing, reports the two arcs involved as (left_region, right_region)
    tuples so results can be correlated with check_shared_boundary_no_overlap.
    """
    seen = set()
    arcs = []
    for h in mesh.half_edges:
        if id(h.arc) in seen:
            continue
        seen.add(id(h.arc))
        h_fwd = h if h.forward else h.twin
        h_bwd = h_fwd.twin
        la = mesh.face_region[h_fwd.face]
        lb = mesh.face_region[h_bwd.face]
        line = geometry.LineString(h_fwd.arc.coords)
        arcs.append((line, (la, lb)))

    crossings = []
    for i, (li, pi) in enumerate(arcs):
        for j in range(i + 1, len(arcs)):
            lj, pj = arcs[j]
            if li.crosses(lj):
                crossings.append((pi, pj))

    return {'passed': not crossings, 'crossings': crossings}


def check_no_crossings(inner_arcs: list, outer_arcs: list) -> dict:
    """No two segments of the boundary network cross in their interiors."""
    outer_lines = [geometry.LineString(oa) for oa in outer_arcs]
    hull_poly = _hull_from_outer_arcs(outer_arcs)
    if hull_poly is None:
        return {'passed': True}

    lines = list(outer_lines)
    for arc in inner_arcs:
        clipped = geometry.LineString(arc).intersection(hull_poly)
        if clipped.is_empty:
            continue
        if clipped.geom_type == 'LineString':
            lines.append(clipped)
        elif clipped.geom_type == 'MultiLineString':
            lines.extend(clipped.geoms)

    for i, la in enumerate(lines):
        for j in range(i + 1, len(lines)):
            lb = lines[j]
            if not la.crosses(lb):
                continue
            inter = la.intersection(lb)
            endpoints = [
                geometry.Point(la.coords[0]), geometry.Point(la.coords[-1]),
                geometry.Point(lb.coords[0]), geometry.Point(lb.coords[-1]),
            ]
            if min(inter.distance(ep) for ep in endpoints) < 1e-9:
                continue
            return {'passed': False}
    return {'passed': True}


def check_collapsed_face_components(original_mesh, simplified_mesh,
                                    min_area: float = 0.0) -> dict:
    """Count face components that collapsed after simplification.

    Uses stable face-component IDs (from copy_with_new_coords) to match each
    component in the original mesh to its counterpart in the simplified mesh.
    A component is considered collapsed when its simplified polygon is absent or
    has area <= min_area.

    Returns:
        passed            — True iff no components collapsed
        n_collapsed       — number of collapsed face components
        n_total           — total non-hull face components in original
        collapsed_fraction
        collapsed_fids    — list of face IDs that collapsed
    """
    import shapely
    from shapely.geometry import Polygon
    from .half_edge import _loop_coords

    def _face_polys(mesh):
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

    orig_fp = _face_polys(original_mesh)
    simp_fp = _face_polys(simplified_mesh)

    collapsed_fids = [
        fid for fid in orig_fp
        if simp_fp.get(fid) is None or simp_fp[fid].area <= min_area
    ]
    n_total = len(orig_fp)
    n_collapsed = len(collapsed_fids)

    return {
        'passed':             n_collapsed == 0,
        'n_collapsed':        n_collapsed,
        'n_total':            n_total,
        'collapsed_fraction': n_collapsed / n_total if n_total else 0.0,
        'collapsed_fids':     collapsed_fids,
    }


# ── Combined lh+rh variants ───────────────────────────────────────────────────

def check_collapsed_face_components_both(orig_lh, simp_lh,
                                          orig_rh, simp_rh,
                                          min_area: float = 0.0) -> dict:
    """check_collapsed_face_components for lh+rh combined."""
    lh = check_collapsed_face_components(orig_lh, simp_lh, min_area)
    rh = check_collapsed_face_components(orig_rh, simp_rh, min_area)
    n_collapsed = lh['n_collapsed'] + rh['n_collapsed']
    n_total     = lh['n_total']     + rh['n_total']
    return {
        'passed':             n_collapsed == 0,
        'n_collapsed':        n_collapsed,
        'n_total':            n_total,
        'collapsed_fraction': n_collapsed / n_total if n_total else 0.0,
        'collapsed_fids':     lh['collapsed_fids'] + rh['collapsed_fids'],
    }


def check_junction_count_both(orig_inner_lh: list, simp_inner_lh: list,
                               orig_inner_rh: list, simp_inner_rh: list) -> dict:
    """check_junction_count for lh+rh combined.

    Concatenates arc lists from both hemispheres — safe because lh and rh
    live in separate flat-map coordinate spaces so no endpoints coincide.
    """
    return check_junction_count(orig_inner_lh + orig_inner_rh,
                                simp_inner_lh + simp_inner_rh)


def check_no_gaps_both(simp_lh: dict, outer_lh: list,
                        simp_rh: dict, outer_rh: list,
                        tol: float = 1e-6) -> dict:
    """check_no_gaps for lh+rh combined; gap fraction weighted by hull area."""
    lh = check_no_gaps(simp_lh, outer_lh, tol)
    rh = check_no_gaps(simp_rh, outer_rh, tol)
    hull_lh = _hull_from_outer_arcs(outer_lh)
    hull_rh = _hull_from_outer_arcs(outer_rh)
    a_lh = hull_lh.area if hull_lh else 0.0
    a_rh = hull_rh.area if hull_rh else 0.0
    total = a_lh + a_rh
    gap = (lh['gap_fraction'] * a_lh + rh['gap_fraction'] * a_rh) / total if total else 0.0
    return {'passed': lh['passed'] and rh['passed'], 'gap_fraction': gap}


def check_no_crossings_both(inner_lh: list, outer_lh: list,
                             inner_rh: list, outer_rh: list) -> dict:
    """check_no_crossings for lh+rh — run independently per hemisphere."""
    lh = check_no_crossings(inner_lh, outer_lh)
    rh = check_no_crossings(inner_rh, outer_rh)
    return {'passed': lh['passed'] and rh['passed']}


# ── Diagnostics ───────────────────────────────────────────────────────────────

def diagnose_crossings(inner_arcs: list, outer_arcs: list):
    """Print all crossing pairs in the boundary network and distance to nearest endpoint."""
    outer_lines = [geometry.LineString(oa) for oa in outer_arcs]
    hull_poly = _hull_from_outer_arcs(outer_arcs)

    entries = []
    for i, oa in enumerate(outer_arcs):
        eps = [geometry.Point(oa[0]), geometry.Point(oa[-1])]
        entries.append((f'outer {i}', outer_lines[i], eps))
    for i, arc in enumerate(inner_arcs):
        line = geometry.LineString(arc)
        if hull_poly is not None:
            line = line.intersection(hull_poly)
            if line.is_empty:
                continue
        eps = [geometry.Point(arc[0]), geometry.Point(arc[-1])]
        entries.append((f'inner {i}', line, eps))

    found = 0
    for i, (la_name, la, la_eps) in enumerate(entries):
        for lb_name, lb, lb_eps in entries[i + 1:]:
            if la.crosses(lb):
                inter = la.intersection(lb)
                dist = min(inter.distance(ep) for ep in la_eps + lb_eps)
                if dist < 1e-9:
                    continue
                print(
                    f"  {la_name} × {lb_name}  —  dist to nearest endpoint: {dist:.6g}")
                found += 1
    print(f"{found} crossing(s) found")


# ── Aggregated evaluation ─────────────────────────────────────────────────────

def evaluate_topology(original: dict, simplified: dict,
                      original_inner_arcs: list = None,
                      simplified_inner_arcs: list = None,
                      simplified_outer_arcs: list = None,
                      min_component_area: float = 0.0) -> bool:
    """Run all topology checks, print individual results, return True if all pass."""
    checks = {
        'completeness':           check_completeness(original, simplified),
        'validity':               check_validity(simplified),
        'component_count':        check_component_count(original, simplified, min_component_area),
        'no_overlap':             check_no_overlap(simplified),
        'adjacency_graph':        check_adjacency_graph(original, simplified),
        'junction_count':         check_junction_count(original_inner_arcs, simplified_inner_arcs) if original_inner_arcs is not None else {'passed': True},
        'no_gaps':                check_no_gaps(simplified, simplified_outer_arcs) if simplified_outer_arcs is not None else {'passed': True},
        'no_crossings':           check_no_crossings(simplified_inner_arcs, simplified_outer_arcs) if simplified_inner_arcs is not None else {'passed': True},
    }
    for name, result in checks.items():
        print(f"  {'PASS' if result['passed'] else 'FAIL'}  {name}")
    return all(r['passed'] for r in checks.values())

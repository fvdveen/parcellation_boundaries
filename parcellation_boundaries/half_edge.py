"""Planar half-edge mesh for region boundary graphs."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import shapely
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


class Arc:
    """Geometric data shared by a twin pair of half-edges."""
    __slots__ = ('coords',)

    def __init__(self, coords: np.ndarray):
        self.coords = coords  # (N, 2), indexed start-junction → end-junction


class HalfEdge:
    """One directed side of a boundary arc.

    The face to the LEFT of this half-edge's direction is `self.face`.
    `next` walks the boundary of that face component counter-clockwise.
    face is an int face-component ID (index into HalfEdgeMesh.faces).
    During construction it temporarily holds a string region name until
    HalfEdgeMesh._finalize() assigns the stable int ID.
    """
    __slots__ = ('arc', 'forward', 'face', 'twin', 'next', 'prev')

    def __init__(self, arc: Arc, forward: bool, face: Any):
        self.arc = arc
        self.forward = forward
        self.face: int = face  # int after _finalize; str only during construction
        self.twin:   HalfEdge | None = None
        self.next:   HalfEdge | None = None
        self.prev:   HalfEdge | None = None

    @property
    def coords(self) -> np.ndarray:
        """Arc vertices in this half-edge's traversal direction."""
        return self.arc.coords if self.forward else self.arc.coords[::-1]


# ── Module-level helpers ───────────────────────────────────────────────────────

def _loop_coords(rep: HalfEdge) -> np.ndarray:
    """Closed coordinate array for the face loop starting at rep."""
    pts = []
    h = rep
    while True:
        pts.append(h.coords[:-1])
        h = h.next
        if h is rep:
            break
    arr = np.concatenate(pts)
    return np.vstack([arr, arr[:1]])


def _loop_signed_area(coords: np.ndarray) -> float:
    """Signed area of a closed ring; positive = CCW (exterior), negative = CW (hole)."""
    a, b = coords[:-1], coords[1:]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return float(cross.sum()) * 0.5


def _build_face_components(
    reps: list[HalfEdge],
) -> list[tuple[HalfEdge, list[HalfEdge]]]:
    """Group loops of a single region into face components.

    Each component = (exterior_rep, [hole_rep, ...]).

    Builds a containment tree: CCW loops are face exteriors, CW loops are holes.
    Island-in-island nesting: CCW loops inside CW holes become new face components.
    """
    if not reps:
        return []

    coords_list = [_loop_coords(r) for r in reps]
    areas = [_loop_signed_area(c) for c in coords_list]
    polys: list[Any] = []
    for c in coords_list:
        if len(c) < 4:
            polys.append(None)
            continue
        p = Polygon(c)
        if not p.is_valid:
            p = shapely.make_valid(p)
        polys.append(p)

    n = len(reps)

    # For each loop find parent = smallest loop whose polygon contains this loop.
    # Use the midpoint of loop i's first segment as the test point.  A boundary
    # point of loop i is guaranteed strictly inside loop j's simple polygon if and
    # only if loop i is nested inside loop j (DCEL loops never cross, so no point
    # of loop i lies on loop j's boundary).  This avoids the representative_point()
    # pitfall where a CCW exterior's interior point falls inside a large CW hole.
    parent = [-1] * n
    for i in range(n):
        if polys[i] is None or len(coords_list[i]) < 2:
            continue
        best_j, best_area = -1, float('inf')
        try:
            # Midpoint of a segment (never coincides with another loop's boundary
            # because DCEL loops don't share arc interiors, only junction vertices)
            k   = len(coords_list[i]) // 2
            seg = (coords_list[i][k - 1] + coords_list[i][k]) * 0.5
            rp  = Point(float(seg[0]), float(seg[1]))
        except Exception:
            continue
        for j in range(n):
            if i == j or polys[j] is None:
                continue
            try:
                if polys[j].contains(rp) and abs(areas[j]) < best_area:
                    best_j, best_area = j, abs(areas[j])
            except Exception:
                pass
        parent[i] = best_j

    children: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        if parent[i] != -1:
            children[parent[i]].append(i)

    result: list[tuple[HalfEdge, list[HalfEdge]]] = []

    def _process_exterior(i: int) -> None:
        hole_reps: list[HalfEdge] = []
        for child in children[i]:
            if areas[child] < 0:  # CW = hole of this face component
                hole_reps.append(reps[child])
                for grandchild in children[child]:
                    if areas[grandchild] > 0:  # CCW sub-island = new face component
                        _process_exterior(grandchild)
        result.append((reps[i], hole_reps))

    for i in range(n):
        if parent[i] == -1 and polys[i] is not None:
            if areas[i] > 0:
                _process_exterior(i)
            else:
                # CW root-level loop: outer boundary face (e.g. __hull__).
                # No sub-structure; register as a single-loop component so all
                # half-edges receive int face IDs.
                result.append((reps[i], []))

    return result


# ── Mesh ──────────────────────────────────────────────────────────────────────

class HalfEdgeMesh:
    """Planar half-edge mesh of a Voronoi region boundary graph.

    faces[i]         = [ext_rep, hole_rep, ...]  — per-component loop representatives
    regions[name]    = [face_id, ...]             — region name → face component IDs
    face_region[i]   = name                       — backpointer from face ID to region
    """

    def __init__(self, half_edges: list[HalfEdge],
                 faces: list[list[HalfEdge]],
                 regions: dict[str, list[int]],
                 face_region: list[str]):
        self.half_edges  = half_edges
        self.faces       = faces        # faces[i] = [ext_rep, hole_rep, ...]
        self.regions     = regions      # name → [face_id, ...]
        self.face_region = face_region  # face_region[i] = region name

    # ── Geometry ──────────────────────────────────────────────────────────────

    def face_loop_coords(self, name: str) -> list[np.ndarray]:
        """Closed coordinate arrays for all loops of a named region (all components)."""
        result = []
        for fid in self.regions.get(name, []):
            for rep in self.faces[fid]:
                result.append(_loop_coords(rep))
        return result

    def to_shapes(self, exclude: frozenset[str] = frozenset({'__hull__'})) \
            -> dict[str, object]:
        """Convert each region to a Shapely geometry, skipping excluded regions.

        Each face component becomes Polygon(exterior, [holes]).
        Multiple components per region are unioned into a MultiPolygon.
        No difference/union heuristics — topology drives the structure directly.
        """
        shapes = {}
        for name, fids in self.regions.items():
            if name in exclude:
                continue
            polys = []
            for fid in fids:
                loops = self.faces[fid]
                if not loops:
                    continue
                ext_c = _loop_coords(loops[0])
                if len(ext_c) < 4:
                    continue
                hole_cs = [_loop_coords(h) for h in loops[1:]]
                hole_cs = [c for c in hole_cs if len(c) >= 4]
                p = Polygon(ext_c, hole_cs)
                if not p.is_valid:
                    p = shapely.make_valid(p)
                if not p.is_empty:
                    polys.append(p)
            if not polys:
                continue
            shapes[name] = unary_union(polys) if len(polys) > 1 else polys[0]
        return shapes

    def copy_with_new_coords(self, arc_map: dict[int, np.ndarray]) -> 'HalfEdgeMesh':
        """Topology-preserving shallow copy with updated arc coordinates.

        arc_map: {id(arc): new_coords}.  Arcs absent from arc_map keep their coords.
        face/region/face_region metadata is shared (not copied) since it is stable
        under coordinate-only changes.
        """
        old_to_new: dict[int, HalfEdge] = {}
        new_hes: list[HalfEdge] = []
        seen: set[int] = set()

        for h in self.half_edges:
            if id(h.arc) in seen:
                continue
            seen.add(id(h.arc))
            h_fwd = h if h.forward else h.twin
            h_bwd = h_fwd.twin

            new_arc = Arc(arc_map.get(id(h_fwd.arc), h_fwd.arc.coords))
            new_fwd = HalfEdge(new_arc, True,  h_fwd.face)
            new_bwd = HalfEdge(new_arc, False, h_bwd.face)
            new_fwd.twin = new_bwd
            new_bwd.twin = new_fwd
            old_to_new[id(h_fwd)] = new_fwd
            old_to_new[id(h_bwd)] = new_bwd
            new_hes.extend([new_fwd, new_bwd])

        for h in self.half_edges:
            nh      = old_to_new[id(h)]
            nh.next = old_to_new[id(h.next)]
            nh.prev = old_to_new[id(h.prev)]

        new_faces = [
            [old_to_new[id(rep)] for rep in face_loops]
            for face_loops in self.faces
        ]
        return HalfEdgeMesh(new_hes, new_faces, self.regions, self.face_region)

    # ── Internal: phase-2 face-ID assignment ──────────────────────────────────

    @staticmethod
    def _finalize(all_hes: list[HalfEdge],
                  face_reps_str: dict[str, list[HalfEdge]]) \
            -> tuple[list, list, dict, list]:
        """Convert temporary string face names → stable int face-component IDs.

        Builds containment tree per region, assigns HalfEdge.face in-place.
        Returns (all_hes, faces, regions, face_region).
        """
        faces:       list[list[HalfEdge]] = []
        regions:     dict[str, list[int]] = {}
        face_region: list[str]            = []

        for name, reps in face_reps_str.items():
            components = _build_face_components(reps)
            fids: list[int] = []
            for ext_rep, hole_reps in components:
                fid = len(faces)
                faces.append([ext_rep] + hole_reps)
                face_region.append(name)
                fids.append(fid)
            regions[name] = fids

        # Walk every loop and stamp each half-edge with its int face ID
        for fid, face_loops in enumerate(faces):
            for rep in face_loops:
                h = rep
                while True:
                    h.face = fid
                    h = h.next
                    if h is rep:
                        break

        return all_hes, faces, regions, face_region

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_voronoi(cls, vor: Voronoi,
                     point_labels: np.ndarray,
                     region_names: list[str]) -> 'HalfEdgeMesh':
        """Build a half-edge mesh from a scipy Voronoi diagram.

        point_labels[i]  — integer region index for Voronoi site i.
        region_names[j]  — name string for label j (include '__hull__' if used).
        """
        ridge_pts   = np.array(vor.ridge_points)
        ridge_verts = np.array(vor.ridge_vertices)

        cross  = point_labels[ridge_pts[:, 0]] != point_labels[ridge_pts[:, 1]]
        finite = np.all(ridge_verts >= 0, axis=1)
        mask   = cross & finite

        b_ridges   = ridge_verts[mask]
        b_labels   = np.column_stack([point_labels[ridge_pts[mask, 0]],
                                      point_labels[ridge_pts[mask, 1]]])
        b_site_pos = np.stack([vor.points[ridge_pts[mask, 0]],
                               vor.points[ridge_pts[mask, 1]]], axis=1)

        # ── Arc walking ───────────────────────────────────────────────────────
        adj: dict[int, list] = defaultdict(list)
        for i, (v0, v1) in enumerate(b_ridges):
            adj[v0].append((v1, i))
            adj[v1].append((v0, i))

        junctions = {v for v, nbrs in adj.items() if len(nbrs) != 2}
        visited: set[int] = set()

        def _left_right(ri: int, fv: int, tv: int) -> tuple[int, int]:
            vf = vor.vertices[fv];  vt = vor.vertices[tv]
            s0 = b_site_pos[ri, 0]
            d  = vt - vf
            left0 = d[0] * (s0[1] - vf[1]) - d[1] * (s0[0] - vf[0]) > 0
            l0, l1 = int(b_labels[ri, 0]), int(b_labels[ri, 1])
            return (l0, l1) if left0 else (l1, l0)

        def _walk(start: int, first_nb: int, first_ri: int):
            verts = [start, first_nb]
            visited.add(first_ri)
            prev, cur = start, first_nb
            while cur not in junctions:
                nxts = [(nv, ri) for nv, ri in adj[cur]
                        if nv != prev and ri not in visited]
                if not nxts:
                    break
                nv, ri = nxts[0]
                visited.add(ri)
                verts.append(nv)
                prev, cur = cur, nv
            coords = vor.vertices[verts]
            fl, fr = _left_right(first_ri, start, first_nb)
            return coords, fl, fr

        arc_data: list = []
        for v in sorted(junctions):
            for nb, ri in adj[v]:
                if ri not in visited:
                    arc_data.append(_walk(v, nb, ri))
        for i in range(len(b_ridges)):
            if i not in visited:
                v0, v1 = b_ridges[i]
                arc_data.append(_walk(v0, v1, i))

        # ── Phase 1: HalfEdge objects with string region names ────────────────
        all_hes: list[HalfEdge] = []
        for coords, fl, fr in arc_data:
            arc   = Arc(coords)
            h_fwd = HalfEdge(arc, True,  region_names[fl])
            h_bwd = HalfEdge(arc, False, region_names[fr])
            h_fwd.twin = h_bwd;  h_bwd.twin = h_fwd
            all_hes.extend([h_fwd, h_bwd])

        # Angular ordering at junctions → next/prev
        outgoing: dict[tuple, list[HalfEdge]] = defaultdict(list)
        for h in all_hes:
            outgoing[tuple(h.coords[0].tolist())].append(h)

        for _, out_hes in outgoing.items():
            n_out = len(out_hes)
            if n_out == 1:
                h = out_hes[0]; arr = h.twin
                arr.next = h;  h.prev = arr
                continue
            angles = [np.arctan2(h.coords[1, 1] - h.coords[0, 1],
                                 h.coords[1, 0] - h.coords[0, 0])
                      for h in out_hes]
            order     = np.argsort(angles)
            sorted_hs = [out_hes[i] for i in order]
            for i, h_dep in enumerate(sorted_hs):
                arriving = h_dep.twin
                nxt = sorted_hs[(i - 1) % n_out]
                arriving.next = nxt;  nxt.prev = arriving

        # Collect face loops by region name
        seen_loops: set[int] = set()
        face_reps_str: dict[str, list[HalfEdge]] = defaultdict(list)
        for h in all_hes:
            if id(h) in seen_loops:
                continue
            cur = h
            while True:
                seen_loops.add(id(cur))
                cur = cur.next
                if cur is h:
                    break
            face_reps_str[h.face].append(h)  # type: ignore[index]

        # ── Phase 2: containment tree → int face IDs ──────────────────────────
        all_hes, faces, regions, face_region = cls._finalize(all_hes, face_reps_str)
        return cls(all_hes, faces, regions, face_region)

    @classmethod
    def from_arcs(cls, arc_data: list) -> 'HalfEdgeMesh':
        """Build a half-edge mesh from (coords, left_face, right_face) arc tuples.

        coords[0] is the start junction; coords[-1] is the end junction.
        left_face / right_face are string region names.
        """
        all_hes: list[HalfEdge] = []
        for coords, left_face, right_face in arc_data:
            arc   = Arc(np.asarray(coords))
            h_fwd = HalfEdge(arc, True,  left_face)
            h_bwd = HalfEdge(arc, False, right_face)
            h_fwd.twin = h_bwd;  h_bwd.twin = h_fwd
            all_hes.extend([h_fwd, h_bwd])

        # Angular ordering at each junction → next/prev
        PREC = 6
        outgoing: dict[tuple, list[HalfEdge]] = defaultdict(list)
        for h in all_hes:
            sk = tuple(np.round(h.coords[0],  PREC).tolist())
            ek = tuple(np.round(h.coords[-1], PREC).tolist())
            if sk == ek:
                # Closed-loop arc: self-link, skip angular ordering
                h.next = h;  h.prev = h
            else:
                outgoing[sk].append(h)

        for _, out_hes in outgoing.items():
            n_out = len(out_hes)
            if n_out == 1:
                h = out_hes[0]
                h.twin.next = h;  h.prev = h.twin
                continue
            angles = [np.arctan2(h.coords[1, 1] - h.coords[0, 1],
                                 h.coords[1, 0] - h.coords[0, 0])
                      for h in out_hes]
            order     = np.argsort(angles)
            sorted_hs = [out_hes[i] for i in order]
            for i, h_dep in enumerate(sorted_hs):
                arriving = h_dep.twin
                nxt = sorted_hs[(i - 1) % n_out]
                arriving.next = nxt;  nxt.prev = arriving

        # Collect face loops by region name
        seen_loops: set[int] = set()
        face_reps_str: dict[str, list[HalfEdge]] = defaultdict(list)
        for h in all_hes:
            if id(h) in seen_loops:
                continue
            cur = h
            while True:
                seen_loops.add(id(cur))
                cur = cur.next
                if cur is h:
                    break
            face_reps_str[h.face].append(h)  # type: ignore[index]

        # Phase 2: containment tree → int face IDs
        all_hes, faces, regions, face_region = cls._finalize(all_hes, face_reps_str)
        return cls(all_hes, faces, regions, face_region)

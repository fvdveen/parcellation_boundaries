"""
SVG flatmap renderer: bypasses Plotly entirely.

__init__  — extracts coordinates once, pre-builds SVG path strings with
            colour placeholder positions baked in as split string pairs
render_svg() — only computes colours + string concatenation, no library overhead
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


class SVGRenderer:
    def __init__(self, region_labels: list, region_to_polygon_map: dict) -> None:
        label_to_idx = {label: i for i, label in enumerate(region_labels)}
        self._trace_label_indices: list[int] = []

        # Each entry is the fixed part of the path tag up to where colour goes.
        # render_svg does: prefix + color + MIDDLE + color + SUFFIX
        self._path_prefixes: list[str] = []

        all_x: list[float] = []
        all_y: list[float] = []

        for label in region_labels:
            geom = region_to_polygon_map.get(label)
            if geom is None:
                continue
            idx = label_to_idx[label]
            for poly in (geom.geoms if hasattr(geom, 'geoms') else [geom]):
                if len(poly.exterior.coords) < 4:
                    continue
                xs, ys = poly.exterior.coords.xy
                all_x.extend(xs)
                all_y.extend(ys)
                # Close ring by dropping the repeated last point, SVG Z closes it
                coords = list(zip(xs, ys))
                d = 'M ' + ' L '.join(f'{x:.2f} {y:.2f}' for x, y in coords[:-1]) + ' Z'
                self._path_prefixes.append(f'<path d="{d}" fill="')
                self._trace_label_indices.append(idx)

        # Pre-build header with viewBox computed from all coordinates
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        cy = min_y + max_y  # translate after scale(1,-1) to keep content in view
        self._header = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{min_x:.2f} {min_y:.2f} '
            f'{max_x - min_x:.2f} {max_y - min_y:.2f}" '
            f'style="width:100%;height:100%">'
            f'<g transform="translate(0 {cy:.2f}) scale(1 -1)">'
        )

    # Constant string fragments shared across all render calls
    _BETWEEN = '" stroke="'
    _TAIL    = '" stroke-width="0.5"/>'
    _CLOSE   = '</g></svg>'

    def render_svg(
        self,
        biomarker_data,
        biomarker_title: str,  # noqa: ARG002 — kept for interface parity
        cmap_name: str = 'RdBu_r',
        cmin_val: float = 0.0,
        cmax_val: float = 1.0,
    ) -> str:
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=cmin_val, vmax=cmax_val)

        parts = [self._header]
        for i, label_idx in enumerate(self._trace_label_indices):
            r, g, b, _ = cmap(norm(float(biomarker_data[label_idx])))
            color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            parts.append(self._path_prefixes[i] + color + self._BETWEEN + color + self._TAIL)
        parts.append(self._CLOSE)
        return ''.join(parts)

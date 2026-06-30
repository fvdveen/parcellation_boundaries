"""
Fast flatmap renderer v2: same as fast_renderer.py but stores numpy arrays
instead of Python lists in the trace templates.

Plotly's DataArrayValidator copies Python lists into numpy arrays on every
go.Figure() call (O(total vertices) per render). Storing numpy arrays lets
Plotly make a read-only view instead, removing the per-render vertex cost.
"""
from __future__ import annotations

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class FlatmapRenderer:
    def __init__(self, region_labels: list, region_to_polygon_map: dict) -> None:
        label_to_idx = {label: i for i, label in enumerate(region_labels)}
        self._trace_label_indices: list[int] = []
        # Trace dicts without colour fields — x/y stored as numpy arrays so
        # Plotly can reference them without copying on each render() call.
        self._trace_templates: list[dict] = []

        for label in region_labels:
            geom = region_to_polygon_map.get(label)
            if geom is None:
                continue
            idx = label_to_idx[label]
            for poly in (geom.geoms if hasattr(geom, 'geoms') else [geom]):
                if len(poly.exterior.coords) < 4:
                    continue
                xs, ys = poly.exterior.coords.xy
                self._trace_templates.append({
                    'type': 'scatter',
                    'x': np.asarray(xs),
                    'y': np.asarray(ys),
                    'fill': 'toself',
                    'mode': 'lines',
                    'showlegend': False,
                    'hoverinfo': 'skip',
                })
                self._trace_label_indices.append(idx)

    def render(
        self,
        biomarker_data,
        biomarker_title: str,
        cmap_name: str = 'RdBu_r',
        cmin_val: float = 0.0,
        cmax_val: float = 1.0,
    ) -> go.Figure:
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=cmin_val, vmax=cmax_val)

        traces = []
        for i, label_idx in enumerate(self._trace_label_indices):
            r, g, b, _ = cmap(norm(float(biomarker_data[label_idx])))
            color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            d = dict(self._trace_templates[i])  # shallow copy — x/y refs reused
            d['fillcolor'] = color
            d['line'] = {'color': color, 'width': 2}
            traces.append(d)

        return go.Figure(
            data=traces,
            layout=go.Layout(
                title=biomarker_title,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False, scaleanchor='x'),
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor='white',
            ),
        )

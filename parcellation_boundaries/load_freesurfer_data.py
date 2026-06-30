from enum import Enum
from pathlib import Path

import numpy as np
import nibabel.freesurfer.io as fsio
import scipy.io

ANNOT_DIR = Path('data/freesurfer_annot')
PATCH_DIR = Path('data/freesurfer')


class Atlas(Enum):
    DK = 'aparc'             # Desikan-Killiany (34 regions)
    DKT = 'aparc.a2005s'     # Desikan-Killiany-Tourville (2005)
    DESTRIEUX = 'aparc.a2009s'    # Destrieux (74 regions)
    YEO7 = 'Yeo2011_7Networks_N1000'
    YEO17 = 'Yeo2011_17Networks_N1000'
    SCHEAFER100 = 'Schaefer2018_100Parcels_7Networks_order'

    def annot_path(self, hemi: str) -> Path:
        return ANNOT_DIR / f'{hemi}.{self.value}.annot'


def read_patch(fname):
    with open(fname, 'rb') as f:
        ver = np.frombuffer(f.read(4), dtype='>i4')[0]
        if ver != -1:
            raise ValueError(
                f'Unexpected patch file version {ver} (expected -1)')
        npts = np.frombuffer(f.read(4), dtype='>i4')[0]
        data = np.frombuffer(f.read(npts * 16), dtype=np.dtype([
            ('vno', '>i4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')
        ]))
    raw = data['vno'].copy()
    border = raw < 0
    vno = np.where(border, -(raw + 1), raw - 1)
    return {'x': data['x'], 'y': data['y'], 'z': data['z'], 'vno': vno}


def load_hemisphere(atlas_or_annot, hemi: str,
                    coordinates_max: float = 100.0,
                    gap: float = 0.06) -> dict:
    """Load a FreeSurfer hemisphere.

    atlas_or_annot: Atlas enum member, or a path to a .annot file.
    hemi:           'lh' or 'rh'.

    Returns {region_name: {'x': ndarray, 'y': ndarray, 'color': (r,g,b)}}.
    """
    annot_path = (atlas_or_annot.annot_path(hemi)
                  if isinstance(atlas_or_annot, Atlas)
                  else atlas_or_annot)
    patch_path = PATCH_DIR / f'{hemi}.cortex.patch.flat'

    ann_labels, ctab, names = fsio.read_annot(annot_path, orig_ids=True)
    patch = read_patch(patch_path)
    x = patch['x'].astype(float)
    y = patch['y'].astype(float)
    vno = patch['vno']
    labels = ann_labels[vno]
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
    for i in range(ctab.shape[0]):
        if i < 1:  # skip medial wall
            continue
        name = names[i].decode() if isinstance(names[i], bytes) else names[i]
        name = name + '-' + hemi
        sel = np.where(labels == ctab[i, 4])[0]
        if len(sel) == 0:
            continue
        regions[name] = {
            'x':     x[sel],
            'y':     y[sel],
            'color': tuple(ctab[i, :3] / 255),
        }
    return regions


def _names_to_cell(names):
    cell = np.empty((len(names), 1), dtype=object)
    for i, name in enumerate(names):
        s = name.decode() if isinstance(name, bytes) else name
        cell[i, 0] = np.array([s])
    return cell


if __name__ == '__main__':
    lh_labels, lh_ctab, lh_names = fsio.read_annot(
        'data/freesurfer_annot/lh.aparc.a2009s.annot', orig_ids=True)
    lh_patch = read_patch('data/freesurfer/lh.cortex.patch.flat')

    scipy.io.savemat('data/freesurfer/patch_data_dk_lh.mat', {
        'x': lh_patch['x'],
        'y': lh_patch['y'],
        'color': lh_ctab,
        'labels': lh_labels.reshape(-1, 1),
        'vno': lh_patch['vno'],
        'region_names': _names_to_cell(lh_names),
    })

    rh_labels, rh_ctab, rh_names = fsio.read_annot(
        'data/freesurfer_annot/rh.aparc.a2009s.annot', orig_ids=True)
    rh_patch = read_patch('data/freesurfer/rh.cortex.patch.flat')

    scipy.io.savemat('data/freesurfer/patch_data_dk_rh.mat', {
        'x': rh_patch['x'],
        'y': rh_patch['y'],
        'color': rh_ctab,
        'labels': rh_labels.reshape(-1, 1),
        'vno': rh_patch['vno'],
        'region_names': _names_to_cell(rh_names),
    })

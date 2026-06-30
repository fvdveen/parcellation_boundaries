"""Warm the mesh cache for all atlases and hemispheres in parallel."""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from parcellation_boundaries.boundary import build_mesh_cached
from parcellation_boundaries.load_freesurfer_data import Atlas

HEMIS = ('lh', 'rh')

# MIN_AREAS = [0, 0.25, 0.5, 1, 2, 5, 10, 20, 50]
MIN_AREAS = [0]
# MIN_POINTS = [1, 2, 3, 5, 10]
MIN_POINTS = [1]

COMBOS = [
    (atlas, hemi, min_area, min_points)
    for atlas in Atlas
    for hemi in HEMIS
    for min_area in MIN_AREAS
    for min_points in MIN_POINTS
]


def _warm(atlas, hemi, min_area, min_points):
    print(
        f'{atlas.name} {hemi}  mca={min_area}  mcc={min_points} started')
    t0 = time.time()
    build_mesh_cached(atlas, hemi, min_area=min_area, min_points=min_points)
    return atlas.name, hemi, min_area, min_points, time.time() - t0


if __name__ == '__main__':
    print(f'Warming {len(COMBOS)} combinations ...')
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_warm, atlas, hemi, min_area, min_points): combo
            for combo in COMBOS
            for atlas, hemi, min_area, min_points in [combo]
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            name, hemi, ma, mp, elapsed = fut.result()
            print(
                f'  [{done}/{len(COMBOS)}] {name} {hemi}  mca={ma}  mcc={mp}  {elapsed:.1f}s')

    print(f'Done in {time.time() - t_start:.1f}s total')

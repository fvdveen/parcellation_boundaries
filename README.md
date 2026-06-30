# Parcellation Boundaries

![DK parcellation generated using this pipeline](/assets/DK_parcellation.png)

Code accompanying my Bachelor's thesis on boundary extraction and simplification
for brain surface parcellations.

📄 [Read the thesis](https://repository.tudelft.nl/record/uuid:9b5cdd9f-fec0-42cd-ad02-dd1293010d0e)

Given a FreeSurfer surface and an atlas annotation, it extracts the boundary
network between regions, builds a planar half-edge mesh, simplifies the boundary
arcs with several line-simplification algorithms, and evaluates fidelity and
topology preservation.

## Layout

- `parcellation_boundaries/` — core library (boundaries, half-edge mesh,
  simplification, evaluation)
- `scripts/` — batch experiment drivers (see each file's docstring)
- `benchmark/` — rendering benchmark suite
- `simplification_explorer.ipynb` — interactive notebook for viewing a
  parcellation at custom simplification levels

## Setup

Requires Python ≥ 3.13; managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

Depends on `nbt-core`, an internal package that is not yet publicly available
(local path `../NBT-core`). Without it the rendering benchmarks can't run; the
core library does not need it.

## Data

The library reads FreeSurfer files from two directories (relative to the working
directory):

- `data/freesurfer_annot/` — atlas annotations, named `{hemi}.{atlas}.annot`
  (e.g. `lh.aparc.annot`, `rh.aparc.a2009s.annot`)
- `data/freesurfer/` — flattened cortical patches (`{hemi}.cortex.patch.flat`)

The Desikan-Killiany (`aparc`), DKT (`aparc.a2005s`), Destrieux
(`aparc.a2009s`), and Yeo (`Yeo2011_7Networks_N1000`,
`Yeo2011_17Networks_N1000`) annotations ship with FreeSurfer's `fsaverage`
subject (`$FREESURFER_HOME/subjects/fsaverage/label/`). The Schaefer
(`Schaefer2018_100Parcels_7Networks_order`) annotation is mapped to `fsaverage`
and distributed [separately](https://github.com/ThomasYeoLab/CBIG). Copy the `.annot` files into
`data/freesurfer_annot/` and the flat patches into `data/freesurfer/`.

## Usage

1. **Warm the mesh cache** (optional) — pre-builds and caches the meshes for all
   atlases and hemispheres so later steps are fast:

   ```sh
   uv run python scripts/warm_cache.py
   ```

   Caching is on by default (`build_mesh_cached`), so this is only a speed-up —
   any step builds and caches meshes on demand if the cache is cold.

2. **Evaluate simplification quality** across all algorithms and settings:

   ```sh
   uv run python scripts/evaluate_all.py
   ```

3. **Run the rendering benchmarks** (requires `nbt-core`):

   ```sh
   uv run python benchmark/run_benchmark_full.py
   ```

Results are written under `results/`. See each script's module docstring for
its specific inputs and outputs.

`scripts/` also holds additional analysis drivers (filter sweeps, onset
coincidence, claim summaries); each is documented in its own module docstring.

To export geometries as NBT-core `.patches` files (one per atlas / algorithm /
filter configuration), edit the `CONFIGS` list in `scripts/export_patches.py`
and run:

```sh
uv run python scripts/export_patches.py
```

Files are written to `results/patches/` and load directly in NBT-core.

### Interactive explorer

To view a parcellation at custom simplification levels, open
`simplification_explorer.ipynb`:

```sh
uv run jupyter lab simplification_explorer.ipynb
```

Pick an atlas, hemisphere, and simplification algorithm, then drag the ε slider
to redraw the boundaries live with fidelity and topology metrics. It reads from
the same mesh cache as the scripts above.

## Notes

- **TopoVW** (`simplify_topovw`) is the true algorithm from the paper; **TopoVW
  (modified)** (`simplify_topovw_modified`) is an edited per-arc variant that
  diverges from it, kept for comparison.
- The `fast` and `fast_v2` benchmark renderers are not equivalent — `fast_v2`
  fixes a bug in `fast`, so prefer `fast_v2`.

## Status

Thesis complete — provided for reference and reproducibility.

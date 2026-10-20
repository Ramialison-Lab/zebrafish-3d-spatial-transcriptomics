# Zebrafish 3D spatial transcriptomics

Computational pipeline for reconstructing MERFISH / MERSCOPE data from serial zebrafish sections into a 3D spatial transcriptomic volume, assigning anatomical regions, and running differential expression analysis.

The workflow was developed for a cohort of six fish (three wild-type and three *fmr1* mutants). Paths in the scripts currently point to the lab cluster and should be updated before running on a new machine.

## Pipeline overview

```
MERFISH raw data
        │
        ▼
[00] Extract DAPI histology images (VPT)
        │
        ▼
[01] Crop histology to individual sections
[02] Crop transcriptomics to the same polygons
        │
        ▼
     In-house aligner  ──►  per-section rotation / translation
        │
        ▼
[03] Apply transforms to DAPI  →  per-fish z-stack (NRRD)
[04] Apply the same transforms to cells  →  3D AnnData
        │
        ▼
     Alignment evaluation (NCC + edge similarity)
        │
        ▼
     3D Slicer  ──►  image-to-atlas warp
        │
        ▼
[05] Register cells onto the shared mesh / atlas
[06] Assign anatomical regions
        │
        ▼
[07] Pseudobulk DEG (all cells and by region)
```

External tools used between scripts:

- **Vizgen Post-processing Tool (VPT)** extracts DAPI image patches.
- **In-house section aligner** estimates a 2D transformation (rotation and translation) that aligns consecutive DAPI sections. Those parameters are applied to both images and cell coordinates.
- **3D Slicer** produces the 3D registration (thin-plate spline warp) from each reconstructed fish volume onto a shared atlas mesh.

## Repository structure

```
scripts/
  00_generate_vpt_extraction_commands.ipynb   Generate VPT extract-image-patch commands
  01_crop_histology.py                        Crop DAPI images by polygon ROI
  02_get_individual_transcripts.py            Crop cell-by-gene data by the same ROIs
  03_align_DAPI_zstack_byfish.py              Build aligned DAPI z-stacks (NRRD)
  04_align_transcripts_3d.py                  Apply the same 2D alignment to cells
  alignment_eval_wstats.py                    Quantitative alignment QC
  05_transcripts_to_mesh.py                   Apply 3D Slicer warp to cell coordinates
  06_1_transcripts_label_regions.py           Assign cells to atlas region masks
  06_2_major_regions.py                       Collapse labels to major brain regions
  07_0_export_pseudobulk_counts_all_cells.py  Whole-fish pseudobulk counts
  07_1_DEG_all_cells.py                       DEG: mutant vs WT, all cells
  07_2_DEG_by_region.py                       DEG: mutant vs WT, per region
```

Each numbered script is meant to be run in order. Most scripts loop over all fish once paths and parameters are set at the top of the file.

## Requirements

Python 3, with:

```
pip install numpy pandas scanpy anndata scipy scikit-learn scikit-image \
    matplotlib seaborn pillow squidpy trimesh h5py pynrrd tifffile pydeseq2
```

Also required outside this repo:

- Vizgen VPT, for image extraction
- The lab’s in-house section aligner, which writes per-fish alignment CSVs
- [3D Slicer](https://www.slicer.org/), to generate the image-to-mesh warp (ITK HDF5 transform)

## Inputs you need before running

| Input | Used by | Notes |
| --- | --- | --- |
| MERSCOPE experiment folders (`images/`, `cell_by_gene.csv`, `cell_metadata.csv`, `micron_to_mosaic_pixel_transform.csv`) | 00, 01, 02, 03, 04 | One folder per slide / imaging region |
| Polygon ROIs (GeoJSON) outlining each physical section | 01, 02 | Same polygons crop both DAPI and cells |
| Per-fish alignment CSV from the in-house aligner | 03, 04 | Rotation and translation per section |
| 3D Slicer warp (`*_image_to_mesh.h5`) and atlas mesh (`mz_DAPI.stl`) | 05 | Shared atlas space across fish |
| 3D region masks (TIF) | 06 | One mask per anatomical region |

## Step-by-step

### 00. Extract histology

`scripts/00_generate_vpt_extraction_commands.ipynb` reads each region’s `manifest.json` bounding box and prints `vpt extract-image-patch` commands for DAPI. The notebook does not run VPT; copy the commands into an environment where VPT is installed.

### 01. Crop histology images

`scripts/01_crop_histology.py` loads each extracted DAPI mosaic, converts polygon vertices from microns to pixels with `micron_to_mosaic_pixel_transform.csv`, downsamples (default 10× so images are small enough for the aligner), and writes one grayscale PNG per section, organised by fish.

### 02. Crop transcriptomics data

`scripts/02_get_individual_transcripts.py` loads Vizgen cell-by-gene tables with Squidpy, keeps cells inside the same GeoJSON polygons, recentres coordinates on the polygon, and writes one `.h5ad` per section.

### 03–04. Align DAPI and transcripts

The in-house aligner is run on the cropped DAPI images and writes a CSV of rotation / translation per section.

- `scripts/03_align_DAPI_zstack_byfish.py` applies those 2D transforms to the DAPI images, stacks sections at 7 µm z-spacing, drops known bad / blurry sections, and writes a per-fish NRRD z-stack plus JSON metadata (including pre- and post-alignment origins).
- `scripts/04_align_transcripts_3d.py` applies the **same** rotation and translation to cell `(x, y)` coordinates (converting pixel translations back to microns with the inverse mosaic transform), assigns a z coordinate, and concatenates sections into `{fish}_full.h5ad`.

### Alignment evaluation

`scripts/alignment_eval_wstats.py` compares adjacent-section similarity **before** vs **after** alignment using:

1. Normalised cross-correlation (NCC)
2. Gradient / edge similarity (NCC of Sobel magnitude images)

Metrics are computed on tissue-masked slices. Each fish contributes one median per metric; paired t-tests and 95% CIs are reported across fish. Higher scores mean more similar neighbouring images; they do not by themselves prove anatomical correctness.

### 05. 3D registration (3D Slicer)

Load the reconstructed DAPI volume and the atlas mesh in 3D Slicer, estimate the warp, and export an ITK HDF5 transform.

`scripts/05_transcripts_to_mesh.py` then:

1. Offsets cell coordinates into image space using z-stack metadata
2. Applies a thin-plate spline interpolant built from the Slicer landmarks
3. Stores registered coordinates in `adata.obsm["spatial_registered"]`

### 06. Assign regions

- `scripts/06_1_transcripts_label_regions.py` tests each registered cell against 3D atlas masks. Overlapping labels are resolved into `region_specific` (smallest matching mask) and `region_broad` (largest matching mask).
- `scripts/06_2_major_regions.py` maps broad labels to major categories: Forebrain, Midbrain, Hindbrain, PNS, or Unassigned.

### 07. Differential expression

Three *fmr1* mutants vs three wild-type fish; each fish is one biological replicate.

1. `scripts/07_0_export_pseudobulk_counts_all_cells.py` sums raw counts across all cells in a fish.
2. `scripts/07_1_DEG_all_cells.py` runs PyDESeq2 on that whole-fish matrix (`~genotype`, WT as reference), plus volcano plots and PCA / correlation QC.
3. `scripts/07_2_DEG_by_region.py` builds region-level pseudobulks (minimum 10 cells per sample) and runs the same mutant vs WT test independently in each `region_broad` category.

DESeq2 requires raw integer counts. Confirm that `layers["counts"]` (or `adata.X`) is unnormalised before interpreting results.

## Outputs (typical)

| Stage | Output |
| --- | --- |
| 01 | Cropped DAPI PNGs per section / fish |
| 02 | Per-section `.h5ad` files |
| 03 | `{fish}_zstack.nrrd` and z-stack metadata JSON |
| 04 | `{fish}_full.h5ad` with `obsm["spatial_3d"]` |
| Eval | Per-fish and combined NCC / edge-similarity tables and plots |
| 05 | `{fish}_in_mesh.h5ad` with `obsm["spatial_registered"]` |
| 06 | `{fish}_full_labelled.h5ad` with region columns |
| 07 | Pseudobulk count tables, DESeq2 results, volcano plots, QC plots |

## Notes

- Scripts are configured by editing the path and parameter blocks at the top of each file; they are not a single CLI.
- `BAD_SECTIONS_BY_FISH` in scripts 03 and 04 must stay in sync so dropped DAPI slices are also omitted from the transcript volume.
- Alignment evaluation assumes previously identified bad sections were already excluded from the z-stacks.
- Do not commit large image, NRRD, or `.h5ad` files to this repository.

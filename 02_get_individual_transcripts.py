from pathlib import Path
from matplotlib.path import Path as MplPath
import json
import numpy as np
import scanpy as sc
import squidpy as sq

"""
crop the transcriptomics data per region using polygons to individual sections
"""
# ----------------------------
# Parameters
# ----------------------------
slides = [1, 2]
regions = [0, 1]

base_data_folder = "/group/tran3/3dImaging/zebrafishBatch2_2025Q1"
roi_base_folder = Path("/group/tran3/swei/MERSCOPE/data/individual_polygons_for_Hieu")
base_output_dir = Path("/group/tran3/swei/MERSCOPE/data/h5ad-individual-coord")

# ----------------------------
# Loop over slides and regions
# ----------------------------
for slide in slides:
    for region in regions:
        slide_name = f"slide{slide}"
        region_name = f"region_{region}"
        vizgen_dir = Path(base_data_folder) / slide_name / region_name

        print(f"\nProcessing {slide_name}, {region_name}")
        if not vizgen_dir.exists():
            print(f"Directory not found: {vizgen_dir}, skipping")
            continue

        # Load Vizgen data
        adata = sq.read.vizgen(
            path=vizgen_dir,
            counts_file="cell_by_gene.csv",
            meta_file="cell_metadata.csv",
            transformation_file="micron_to_mosaic_pixel_transform.csv",
        )
        print(f"Loaded data: {adata.n_obs} cells")

        # Polygon ROI folder
        roi_folder = roi_base_folder / f"{slide_name}_region{region}"
        if not roi_folder.exists():
            print(f"ROI folder not found: {roi_folder}, skipping")
            continue

        # Process each ROI
        for roi_file in sorted(roi_folder.glob("*.geojson")):
            print("Processing ROI:", roi_file)
            with open(roi_file, "r") as f:
                roi_data = json.load(f)

            # Extract identifiers from filename
            parts = roi_file.stem.split("_")
            fish_id = parts[2]                     # e.g. "WT1"
            section_id = "_".join(parts[3:])       # e.g. "z10_s2"

            # Create output folder for this fish
            output_dir = base_output_dir / slide_name / region_name / fish_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Polygon vertices in micron coordinates
            verts_micron = np.array(roi_data["features"][0]["geometry"]["coordinates"][0])
            polygon_path = MplPath(verts_micron)

            # Select cells inside polygon
            points = adata.obsm["spatial"]
            mask = polygon_path.contains_points(points)

            # Subset AnnData for this ROI
            adata_section = adata[mask].copy()

            # Shift coordinates so bounding-box center is at (0,0)
            coords = adata_section.obsm["spatial"]
            min_coords_poly = verts_micron.min(axis=0)
            max_coords_poly = verts_micron.max(axis=0)
            center_poly = (min_coords_poly + max_coords_poly) / 2.0
            coords_centered = coords - center_poly
            adata_section.obsm["spatial"] = coords_centered
            adata_section.obsm["spatial_orig"] = coords

            # Save ROI AnnData
            out_path = output_dir / f"{section_id}.h5ad"
            adata_section.write_h5ad(out_path, compression="gzip")
            print(f"Saved {section_id} with {adata_section.n_obs} cells to {out_path}")
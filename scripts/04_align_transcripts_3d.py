from pathlib import Path
import re

import numpy as np
import pandas as pd

import scanpy as sc

# ============================================================
# CONFIGURATION (shared across all fish)
# ============================================================
downsample_factor = 10
z_spacing = 7.0  # µm
z_gap_missing = 7  # µm gap for missing sections

base_data_folder = Path("/group/tran3/3dImaging/zebrafishBatch2_2025Q1")
alignment_base_path = Path("/group/tran3/swei/MERFISH/MERSCOPE/data/alignment_data")
fish_data_root = Path("/group/tran3/swei/MERFISH/MERSCOPE/data")
output_dir = fish_data_root / "merged_3d"
output_dir.mkdir(exist_ok=True)

BAD_SECTIONS_BY_FISH = {
    "WT1": [
        "slide2_region0_WT1_z47_s39",
        "slide2_region0_WT1_z46_s38",
        "slide1_region1_WT1_z27_s18",
        "slide1_region1_WT1_z26_s17",
        "slide1_region1_WT1_z25_s16",
        "slide2_region1_WT1_z35_s27",
        "slide1_region1_WT1_z27_s19",
        "slide1_region1_WT1_z26_s18",
        "slide1_region1_WT1_z25_s17",
        "slide1_region1_WT1_z24_s16",
    ],
    "WT2": [
        "slide2_region0_WT2_z48_s46",
        "slide2_region1_WT2_z39_s37",
        "slide2_region1_WT2_z33_s31",
        "slide2_region1_WT2_z29_s27",
        "slide1_region1_WT2_z24_s22",
        "slide1_region1_WT2_z27_s25",
        "slide1_region0_WT2_z2_s0",
    ],
    "WT3": [
        "slide2_region0_WT3_z48_s48",
        "slide2_region0_WT3_z47_s47",
        "slide2_region1_WT3_z39_s39",
        "slide2_region1_WT3_z28_s28",
        "slide1_region1_WT3_z24_s24",
        "slide1_region1_WT3_z23_s23",
    ],
    "mut1": [
        "slide2_region0_mut1_z48_s49",
        "slide2_region0_mut1_z47_s48",
        "slide2_region0_mut1_z39_s40",
        "slide2_region1_mut1_z28_s29",
        "slide2_region1_mut1_z27_s28",
        "slide2_region1_mut1_z26_s27",
        "slide1_region1_mut1_z26_s27",
    ],
    "mut2": [
        "slide2_region1_mut2_z27_s20",
        "slide2_region1_mut2_z26_s19",
        "slide2_region0_mut2_z48_s41",
        "slide2_region0_mut2_z47_s40",
        "slide1_region1_mut2_z24_s17",
    ],
    "mut3": [
        "slide2_region0_mut3_z48_s37",
        "slide2_region0_mut3_z47_s36",
        "slide2_region1_mut3_z39_s28",
        "slide2_region1_mut3_z27_s16",
        "slide2_region1_mut3_z26_s15",
    ],
}


# ============================================================
# HELPER FUNCTIONS (fish-agnostic)
# ============================================================
def load_transform_matrix(slide_num, region_num):
    transform_path = (
        base_data_folder
        / f"slide{slide_num}"
        / f"region_{region_num}"
        / "images"
        / "micron_to_mosaic_pixel_transform.csv"
    )
    if not transform_path.exists():
        raise FileNotFoundError(f"Missing transform matrix: {transform_path}")
    M = np.loadtxt(transform_path)
    M_inv = np.linalg.inv(M)
    M_inv = M_inv[:2, :2]  # upper-left block
    return M_inv


def transform_coords_centered(coords, param_row, M_inv, downsample_factor):
    x, y = coords[:, 0], coords[:, 1]
    theta = -np.radians(param_row["Rotation"])

    # Rotate
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)

    # Apply translation (in pixels, scaled by downsample)
    dx_px = param_row["Translation X (px)"] * downsample_factor
    dy_px = param_row["Translation Y (px)"] * downsample_factor

    # Convert pixel translation → microns
    dx_microns, dy_microns = M_inv @ np.array([dx_px, dy_px])

    x_rot += dx_microns
    y_rot -= dy_microns

    return np.column_stack([x_rot, y_rot])


# ============================================================
# PER-FISH PIPELINE
# ============================================================
def process_fish(fish_name, bad_sections):
    print(f"\n{'='*60}\nProcessing fish: {fish_name}\n{'='*60}")

    # --- Load alignment CSV ---
    align_csv = alignment_base_path / f"{fish_name}_alignment-data.csv"
    if not align_csv.exists():
        raise FileNotFoundError(f"Alignment CSV not found: {align_csv}")
    align_df = pd.read_csv(align_csv)
    print(f"Loaded alignment data for {len(align_df)} entries")
    print(f"Example filenames: {align_df['Filename'].head().tolist()}")

    # --- Find all h5ad files for this fish ---
    h5ad_files = sorted(
        fish_data_root.glob(f"h5ad-individual-coord/slide*/region_*/{fish_name}/*.h5ad"),
        key=lambda p: int(re.search(r"z(\d+)", p.name).group(1)) if re.search(r"z(\d+)", p.name) else 0,
        reverse=True,
    )

    if not h5ad_files:
        print(f"* No h5ad files found for {fish_name}, skipping.")
        return

    print(f"Found {len(h5ad_files)} .h5ad files")
    for f in h5ad_files[:5]:
        print(f)

    # Process from lowest z to highest
    h5ad_files = list(reversed(h5ad_files))

    adata_list = []
    current_z = 0  # track cumulative z for stacking

    for idx, h5ad_file in enumerate(h5ad_files):
        # Parse slide/region/section
        slide_match = re.search(r"slide(\d+)", str(h5ad_file))
        region_match = re.search(r"region_(\d+)", str(h5ad_file))
        z_match = re.search(r"z(\d+)_s(\d+)", str(h5ad_file))

        slide_num = int(slide_match.group(1)) if slide_match else 1
        region_num = int(region_match.group(1)) if region_match else 0
        section_id = (
            f"slide{slide_num}_region{region_num}_{fish_name}_z{z_match.group(1)}_s{z_match.group(2)}"
            if z_match
            else ""
        )

        # Check if this section is blurry / missing
        if section_id in bad_sections:
            print(f"* {section_id} is blurry / missing. Adding z-gap only.")
            current_z += z_gap_missing
            continue  # skip adding data for this section

        # Load adata
        adata = sc.read_h5ad(h5ad_file)

        # Load transform matrix
        M_inv = load_transform_matrix(slide_num, region_num)

        # Get coordinates
        coords = adata.obsm.get("spatial", np.random.rand(adata.n_obs, 2) * 1000)

        # Apply rotation + translation if available
        filename_png = f"slide{slide_num}_region{region_num}_{fish_name}_{h5ad_file.stem}.png"
        row = align_df.loc[align_df["Filename"] == filename_png]

        if not row.empty:
            coords_shifted = transform_coords_centered(coords, row.iloc[0], M_inv, downsample_factor)
            print(
                f"Applied transform: Rotation={row.iloc[0]['Rotation']:.2f}°, "
                f"Tx={row.iloc[0]['Translation X (px)']:.1f}, Ty={row.iloc[0]['Translation Y (px)']:.1f}"
            )
        else:
            print(f"* No alignment entry for {filename_png}, using raw coords.")
            coords_shifted = coords

        # Add z coordinate
        coords_3d = np.hstack([coords_shifted, np.full((adata.n_obs, 1), current_z)])
        adata.obsm["spatial_3d"] = coords_3d

        # Update cumulative z
        current_z += z_spacing

        # Metadata
        adata.obs["slide"] = slide_num
        adata.obs["region"] = region_num
        adata.obs["section"] = h5ad_file.stem
        adata.obs["z_index"] = idx
        adata.obs["fish"] = fish_name

        # Convert object columns to string
        for col in adata.obs.columns:
            if adata.obs[col].dtype == "object":
                adata.obs[col] = adata.obs[col].astype(str)

        adata_list.append(adata)

    if not adata_list:
        print(f"*No valid sections for {fish_name}, skipping merge.")
        return

    # --- Concatenate all sections for this fish ---
    print("\nConcatenating all sections...")
    merged_adata = sc.concat(adata_list, axis=0, join="outer", index_unique="-", fill_value=0)

    # Add batch info if missing
    if "batch" not in merged_adata.obs.columns:
        merged_adata.obs["batch"] = [
            f"section_{i}" for i, adata in enumerate(adata_list) for _ in range(adata.n_obs)
        ]

    # --- Save output ---
    output_file = output_dir / f"{fish_name}_full.h5ad"
    merged_adata.write(output_file)

    print(f"\n Saved merged 3D h5ad: {output_file}")
    print(f"Total cells: {merged_adata.n_obs}, Total genes: {merged_adata.n_vars}")
    print(f"Sections processed: {len(adata_list)}")

    return merged_adata


# ============================================================
# LOOP OVER ALL FISH
# ============================================================
results = {}
for fish_name, bad_sections in BAD_SECTIONS_BY_FISH.items():
    try:
        results[fish_name] = process_fish(fish_name, bad_sections)
    except FileNotFoundError as e:
        print(f"* Skipping {fish_name}: {e}")
        continue

print("\n\nAll fish processed:", list(results.keys()))
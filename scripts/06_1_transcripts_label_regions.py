import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import tifffile as tiff

# ---- paths ----
in_dir = "/group/tran3/swei/MERFISH/MERSCOPE/data/merged_3d/full_in_mesh"
out_dir = "/group/tran3/swei/MERFISH/MERSCOPE/data/merged_3d/full_region_label"
mask_dir = Path("/group/tran3/swei/MERFISH/MERSCOPE/data/brain_region_masks")
os.makedirs(out_dir, exist_ok=True)

# ---- load all region masks ONCE (shared across all fish) ----
print("Loading region masks...")
masks = {}
mask_sizes = {}
for mask_file in mask_dir.glob("*.tif"):
    region = mask_file.stem
    mask = tiff.imread(mask_file)
    masks[region] = mask
    mask_sizes[region] = (mask > 0).sum()
print(f"Loaded {len(masks)} region masks")

# ---- discover all fish from the registered h5ad files present ----
h5ad_files = sorted(glob.glob(f"{in_dir}/*_in_mesh.h5ad"))
fish_list = [os.path.basename(f).replace("_in_mesh.h5ad", "") for f in h5ad_files]
print(f"Found {len(fish_list)} fish: {fish_list}")


def most_specific(region_string):
    """Choose the smallest mask among all matched regions."""
    if region_string == "unassigned":
        return "unassigned"
    regs = region_string.split(";")
    return min(regs, key=lambda r: mask_sizes[r])


def broad_region(region_string):
    """Choose the largest mask among all matched regions."""
    if region_string == "unassigned":
        return "unassigned"
    regs = region_string.split(";")
    return max(regs, key=lambda r: mask_sizes[r])


for fish in fish_list:
    print(f"\n=== Processing {fish} ===")

    out_path = f"{out_dir}/{fish}_full_labelled.h5ad"
    if os.path.exists(out_path):
        print(f"  already exists, skipping: {out_path}")
        continue

    try:
        h5ad_p = f"{in_dir}/{fish}_in_mesh.h5ad"
        adata = sc.read_h5ad(h5ad_p)

        coords = adata.obsm["spatial_registered"]
        x = np.round(coords[:, 0]).astype(int)
        y = np.round(coords[:, 1]).astype(int)
        z = np.round(coords[:, 2]).astype(int)

        adata.obs["regions"] = ""

        for region, mask in masks.items():
            valid = (
                (x >= 0) & (x < mask.shape[2]) &
                (y >= 0) & (y < mask.shape[1]) &
                (z >= 0) & (z < mask.shape[0])
            )
            inside = np.zeros(len(adata), dtype=bool)
            inside[valid] = mask[z[valid], y[valid], x[valid]] > 0

            adata.obs.loc[inside, "regions"] += region + ";"
            print(f"  {region}: {inside.sum()}")

        adata.obs["regions"] = adata.obs["regions"].str.rstrip(";")
        adata.obs.loc[adata.obs["regions"] == "", "regions"] = "unassigned"

        adata.obs["region_specific"] = adata.obs["regions"].apply(most_specific)
        adata.obs["region_broad"] = adata.obs["regions"].apply(broad_region)

        adata.write_h5ad(out_path)
        print(f"  saved: {out_path}")

    except Exception as e:
        print(f"  FAILED for {fish}: {e}")
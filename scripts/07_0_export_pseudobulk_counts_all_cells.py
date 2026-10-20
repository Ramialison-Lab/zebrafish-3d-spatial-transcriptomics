import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp


# ============================================================
# SETTINGS
# ============================================================

# Directory containing the merged 3D AnnData file for each fish.
DATA_DIR = (
    "/group/tran3/swei/MERFISH/MERSCOPE/"
    "data/merged_3d/full"
)

# Fish included in the comparison, with three biological
# replicates per genotype (WT and fmr1 mutant).
FISH_LIST = [
    "WT1",
    "WT2",
    "WT3",
    "mut1",
    "mut2",
    "mut3",
]

# Output directory for pseudobulk count tables
OUT_DIR = (
    "/group/tran3/swei/MERFISH/MERSCOPE/"
    "data/DEG/degust_counts_all"
)

os.makedirs(OUT_DIR, exist_ok=True)

# Minimum number of cells required to generate a pseudobulk sample.
MIN_CELLS_PER_PSEUDOBULK = 10


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def genotype_from_fish(fish_id):
    """Assign genotype based on the fish ID."""
    if fish_id.startswith("mut"):
        return "mut"
    elif fish_id.startswith("WT"):
        return "WT"
    else:
        raise ValueError(f"Unknown fish ID: {fish_id}")


def get_raw_counts(adata):
    """
    Select the available expression matrix containing raw counts.

    Priority:
        1. layers['counts']
        2. raw.X
        3. X

    The selected matrix is used to calculate summed gene counts
    for each fish. This assumes the chosen source contains raw,
    unnormalised counts.
    """

    if "counts" in adata.layers:
        X = adata.layers["counts"]
        source = "layers['counts']"

    elif adata.raw is not None:
        X = adata.raw.X
        source = "raw.X"

    else:
        X = adata.X
        source = "X"

    print(f"\nUsing count source: {source}")

    return X


# ============================================================
# 1. LOAD FISH
# ============================================================

adatas = []

for fish in FISH_LIST:

    path = os.path.join(
        DATA_DIR,
        f"{fish}_full.h5ad"
    )

    print(f"Loading {fish}")
    print(path)

    a = sc.read_h5ad(path)

    a.obs["fish_id"] = fish
    a.obs["genotype"] = genotype_from_fish(fish)

    adatas.append(a)

print("\nConcatenating fish...")

adata = ad.concat(
    adatas,
    join="outer",
    index_unique="-",
    fill_value=0,
)

print("\nCombined dataset:")
print(adata)

print(
    f"\nCells: {adata.n_obs:,}"
    f"\nGenes: {adata.n_vars:,}"
)


# ============================================================
# 2. GET RAW COUNTS
# ============================================================
# Retrieve the selected count matrix and the metadata required
# to group cells by their biological replicate.

X = get_raw_counts(adata)

obs = adata.obs[
    ["fish_id", "genotype"]
].copy()

print("\nBuilding pseudobulk counts...")


# ============================================================
# 3. PSEUDOBULK
# ============================================================
# Sum gene counts across all cells belonging to each fish.
# Each fish produces one pseudobulk sample, preserving the fish
# as the biological replicate for downstream differential
# expression analysis.

pseudobulk_counts = {}
metadata_rows = []

for fish in FISH_LIST:

    mask = (
        obs["fish_id"].to_numpy() == fish
    )

    n_cells = int(mask.sum())

    if n_cells < MIN_CELLS_PER_PSEUDOBULK:

        print(
            f"[SKIP] {fish}: "
            f"{n_cells} cells"
        )

        continue

    positions = np.flatnonzero(mask)

    if sp.issparse(X):
        summed = np.asarray(
            X[positions].sum(axis=0)
        ).ravel()

    else:
        summed = np.asarray(
            X[positions]
        ).sum(axis=0)

    pseudobulk_counts[fish] = summed

    metadata_rows.append({
        "sample": fish,
        "fish_id": fish,
        "genotype": genotype_from_fish(fish),
        "n_cells": n_cells,
        "library_size": int(summed.sum()),
    })

    print(
        f"{fish}: "
        f"{n_cells:,} cells, "
        f"{summed.sum():,.0f} counts"
    )


# ============================================================
# 4. BUILD OUTPUT TABLES
# ============================================================
# Arrange the summed counts into a gene-by-sample matrix.
# Rows represent genes; columns represent individual fish.

pb_counts = pd.DataFrame(
    pseudobulk_counts,
    index=adata.var_names
)

pb_counts = pb_counts[FISH_LIST]

pb_metadata = pd.DataFrame(
    metadata_rows
).set_index("sample")

print("\nPseudobulk count matrix:")
print(pb_counts.shape)

print("\nMetadata:")
print(pb_metadata)


# ============================================================
# 5. EXPORT COUNTS
# ============================================================

degust_df = pb_counts.reset_index()

degust_df = degust_df.rename(
    columns={"index": "Gene_ID"}
)

count_path = os.path.join(
    OUT_DIR,
    "all_cells_mut_vs_WT_counts.csv"
)

degust_df.to_csv(
    count_path,
    index=False
)

print(
    f"\n[SAVED]\n{count_path}"
)

print
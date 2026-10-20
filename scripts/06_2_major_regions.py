import glob
import os
import scanpy as sc

# ---- paths ----
# Directory containing the previously labelled AnnData files.
# Each file contains the broad anatomical region assigned to each transcript.
in_dir = "/group/tran3/swei/MERFISH/MERSCOPE/data/merged_3d/full_region_label"

# ---- define major anatomical region mapping ----
# Map the broad anatomical region labels to higher-level anatomical categories
# for downstream analyses.
#
# The forebrain includes the prosencephalon and dorsal thalamus proper.
# The midbrain and hindbrain retain their respective anatomical classifications.
# The peripheral nervous system (PNS) is kept as a separate category.
#
# GABAergic domains, retinal arborization field 10, and transcripts outside
# the defined anatomical regions are grouped as "Unassigned".
major_region_map = {
    "prosencephalon_(forebrain)": "Forebrain",
    "dorsal_thalamus_proper": "Forebrain",

    "mesencephalon_(midbrain)": "Midbrain",

    "rhombencephalon_(hindbrain)": "Hindbrain",

    "peripheral_nervous_system": "PNS",

    "gabaergic_domain": "Unassigned",
    "retinal_arborization_field_10": "Unassigned",
    "unassigned": "Unassigned",
}

files = glob.glob(f"{in_dir}/*_labelled.h5ad")

# ---- assign major anatomical regions ----
for file in sorted(files):

    fish = os.path.basename(file)
    print(f"\nProcessing {fish}")

    adata = sc.read_h5ad(file)

    adata.obs["major_region"] = (
        adata.obs["region_broad"]
        .map(major_region_map)
        .fillna("Unassigned")
    )

    print(adata.obs["major_region"].value_counts())
    
    # Save the updated major_region annotation to the existing AnnData file.
    adata.write_h5ad(file)

    print("Saved")
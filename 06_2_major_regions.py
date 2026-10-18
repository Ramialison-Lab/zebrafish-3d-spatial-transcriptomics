import glob
import os
import scanpy as sc

in_dir = "/group/tran3/swei/MERFISH/MERSCOPE/data/merged_3d/full_region_label"

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

    adata.write_h5ad(file)

    print("Saved")
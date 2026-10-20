import os
import glob
import json
import h5py
import nrrd
import numpy as np
import trimesh
import scanpy as sc
from scipy.interpolate import RBFInterpolator

# ---- paths ----
base = "/group/tran3/swei/MERFISH/MERSCOPE/data"
mesh_p = f"{base}/mesh/mz_DAPI.stl"
full_dir = f"{base}/merged_3d/full"
out_dir = f"{base}/merged_3d/full_in_mesh"
os.makedirs(out_dir, exist_ok=True)

# ---- shared mesh (same for all fish, load once) ----
mesh = trimesh.load(mesh_p)
print("Mesh bounds:", mesh.bounds)

# ---- discover all fish from the h5ad files present ----
h5ad_files = sorted(glob.glob(f"{full_dir}/*_full.h5ad"))
fish_list = [os.path.basename(f).replace("_full.h5ad", "") for f in h5ad_files]
print(f"Found {len(fish_list)} fish: {fish_list}")

def get_fish_offset(section_metadata):
    """Find the first section with valid center_x_um/center_y_um."""
    for i, sm in enumerate(section_metadata):
        cx, cy = sm.get("center_x_um"), sm.get("center_y_um")
        if cx is not None and cy is not None:
            return float(cx), float(cy), i
    raise ValueError("No section with valid center_x_um/center_y_um found")

for fish in fish_list:
    print(f"\n=== Processing {fish} ===")

    out_path = f"{out_dir}/{fish}_in_mesh.h5ad"
    if os.path.exists(out_path):
        print(f"  already exists, skipping: {out_path}")
        continue

    try:
        meta_p = f"{base}/zstack/{fish}/{fish}_zstack_metadata.json"
        with open(meta_p) as f:
            meta = json.load(f)

        cx, cy, used_idx = get_fish_offset(meta["section_metadata"])
        print(f"  using section {used_idx} for offset: cx={cx}, cy={cy}")

        # ---- load expression data ----
        h5ad_p = f"{full_dir}/{fish}_full.h5ad"
        adata = sc.read_h5ad(h5ad_p)
        coords = np.asarray(adata.obsm["spatial_3d"])

        # apply constant per-fish offset (once, not per-section)
        coords_image = coords.copy()
        coords_image[:, 0] += cx
        coords_image[:, 1] += cy

        # ---- load TPS landmarks (per-fish warp file) ----
        warp_p = f"{base}/3d_warp/{fish}_image_to_mesh.h5"
        with h5py.File(warp_p, "r") as f:
            image_pts = f["TransformGroup/0/TransformFixedParameters"][:].reshape(-1, 3)
            mesh_pts = f["TransformGroup/0/TransformParameters"][:].reshape(-1, 3)

        # build interpolators once per fish
        rbf_x = RBFInterpolator(image_pts, mesh_pts[:, 0], kernel="thin_plate_spline")
        rbf_y = RBFInterpolator(image_pts, mesh_pts[:, 1], kernel="thin_plate_spline")
        rbf_z = RBFInterpolator(image_pts, mesh_pts[:, 2], kernel="thin_plate_spline")

        # ---- warp all cells for this fish ----
        coords_mesh = np.column_stack([
            rbf_x(coords_image),
            rbf_y(coords_image),
            rbf_z(coords_image),
        ])

        adata.obsm["spatial_registered"] = coords_mesh
        adata.write_h5ad(out_path)
        print(f"  saved: {out_path}  ({coords_mesh.shape[0]} cells)")

    except Exception as e:
        print(f"  FAILED for {fish}: {e}")
import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageOps

"""
crop the extracted image (histology image per region) using polygons to individual sections, and store by fish
"""

# ----------------------------
# Base directories
# ----------------------------
base_image_dir = "/group/tran3/swei/MERFISH/MERSCOPE/data/extracted-image"
roi_base_dir = "/group/tran3/swei/MERFISH/MERSCOPE/data/individual_polygons_for_Hieu"
output_base_dir = "/group/tran3/swei/MERFISH/MERSCOPE/data/cropped_images/cropped_DAPI_gray_fish"
os.makedirs(output_base_dir, exist_ok=True)

# ----------------------------
# Parameters
# ----------------------------
# shrink factor, so the cropped image won't break aligner tool
downsample_factor = 10  

# ----------------------------
# Helper functions
# ----------------------------
def micron_to_pixel(x, y, transform):
    coords = np.vstack([x, y, np.ones_like(x)])
    pixel_coords = transform @ coords
    return pixel_coords[0], pixel_coords[1]


def process_dapi_image(dapi_path, transform_path, roi_folder, output_base_dir, slide_name, region_number):
    # Load DAPI image
    Image.MAX_IMAGE_PIXELS = None
    print("loading images")
    img = Image.open(dapi_path).convert("L")  # Convert to grayscale

    # Enhance contrast
    img = ImageOps.autocontrast(img, cutoff=0)

    # Downsample
    img_small = img.resize(
        (img.width // downsample_factor, img.height // downsample_factor),
        resample=Image.BILINEAR
    )
    img_array = np.array(img_small)

    # Load transform
    transform_matrix = np.loadtxt(transform_path)

    # Loop through ROI files
    roi_files = [f for f in os.listdir(roi_folder) if f.endswith(".geojson")]
    if not roi_files:
        print(f" No ROI files found in {roi_folder}")
        return

    for roi_file in roi_files:
        print(f"\n Processing {roi_file} in {os.path.basename(dapi_path)}")
        try:
            roi_path = os.path.join(roi_folder, roi_file)
            with open(roi_path, "r") as f:
                roi_data = json.load(f)

            verts_micron = np.array(roi_data["features"][0]["geometry"]["coordinates"][0])
            x_micron, y_micron = verts_micron[:, 0], verts_micron[:, 1]

            x_pixel, y_pixel = micron_to_pixel(x_micron, y_micron, transform_matrix)
            x_pixel_ds, y_pixel_ds = x_pixel / downsample_factor, y_pixel / downsample_factor
            polygon_pixel = list(zip(x_pixel_ds, y_pixel_ds))

            # Create mask
            mask_img = Image.new("L", img_small.size, 0)
            ImageDraw.Draw(mask_img).polygon(polygon_pixel, outline=1, fill=1)
            mask = np.array(mask_img)

            masked_img = img_array * mask

            # Crop bounding box
            coords = np.column_stack(np.where(mask > 0))
            if coords.size == 0:
                print(f"*Skipping {roi_file}: empty mask")
                continue

            min_row, min_col = coords.min(axis=0)
            max_row, max_col = coords.max(axis=0)

            cropped = masked_img[min_row:max_row+1, min_col:max_col+1]
            out_img = Image.fromarray(cropped)

            # Extract fishname from filename
            parts = roi_file.split("_")
            if len(parts) < 3:
                print(f"*Could not extract fishname from {roi_file}")
                fishname = "unknown"
            else:
                fishname = parts[2]  # e.g., mut1

            # Create fish-specific directory
            output_dir = os.path.join(output_base_dir, fishname)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save using original ROI filename
            base_name = os.path.splitext(roi_file)[0]
            out_path = os.path.join(output_dir, f"{base_name}.png")
            
            out_img.save(out_path)
            print(f"Saved: {out_path}")

        except Exception as e:
            print(f"*Error processing {roi_file}: {e}")


# ----------------------------
# Main loop
# ----------------------------
for img_file in sorted(os.listdir(base_image_dir)):
    if not img_file.endswith("_DAPI.png"):
        continue

    dapi_path = os.path.join(base_image_dir, img_file)

    # Extract slide and region correctly
    parts = os.path.splitext(img_file)[0].split("_")
    slide_name = parts[0]          # e.g. 'slide1'
    region_number = parts[2]       # e.g. '0'

    roi_folder = os.path.join(roi_base_dir, f"{slide_name}_region{region_number}")

    # Find transform CSV
    transform_dir = f"/group/tran3/3dImaging/zebrafishBatch2_2025Q1/{slide_name}/region_{region_number}/images"
    transform_files = [f for f in os.listdir(transform_dir) if f.endswith("micron_to_mosaic_pixel_transform.csv")]
    if not transform_files:
        print(f"*Transform file not found for {slide_name} region_{region_number}")
        continue
    transform_path = os.path.join(transform_dir, transform_files[0])

    process_dapi_image(dapi_path, transform_path, roi_folder, output_base_dir, slide_name, region_number)
    print("done")
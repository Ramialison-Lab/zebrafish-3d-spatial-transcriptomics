"""
Build per-fish 3-D DAPI z-stacks (NRRD) with tracked physical origins AND
pre-alignment centers.

Pipeline (transform math / embed / crop) is unchanged from the version that
was already verified to work — the only addition is that a second point,
`center` (the section's own image centre BEFORE rotation/translation is
applied), is now tracked through the exact same embed + crop offsets that
`origin` (the point AFTER rotation/translation) goes through. That gives us,
per slice:
    - origin_row_px / origin_col_px / origin_y_um / origin_x_um   (post-align)
    - center_row_px / center_col_px / center_y_um / center_x_um  (pre-align)
    - shift_row_px  / shift_col_px  / shift_y_um  / shift_x_um   (origin - center)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import rotate, shift
import json
import nrrd

# ══════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════
Z_SPACING_UM = 7.0
DOWNSAMPLE_FACTOR = 10  # NOTE: working single-fish example used 20 for mut1 — verify per fish

IMAGE_ROOT = Path(
    "/group/tran3/swei/MERFISH/MERSCOPE/data/cropped_images/cropped_DAPI_gray_fish"
)
ALIGNMENT_FOLDER = Path(
    "/group/tran3/swei/MERFISH/MERSCOPE/data/alignment_data"
)
BASE_DATA_FOLDER = Path(
    "/group/tran3/3dImaging/zebrafishBatch2_2025Q1"
)
OUTPUT_DIR = Path(
    "/group/tran3/swei/MERFISH/MERSCOPE/data/zstack"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        "slide1_region1_WT1_z24_s16"
    ],
    "WT2": [
        "slide2_region0_WT2_z48_s46",
        "slide2_region1_WT2_z39_s37", 
        "slide2_region1_WT2_z33_s31", 
        "slide2_region1_WT2_z29_s27", 
        "slide1_region1_WT2_z24_s22", 
        "slide1_region1_WT2_z27_s25",
        "slide1_region0_WT2_z2_s0"
    ],
    "WT3": [
        "slide2_region0_WT3_z48_s48", 
        "slide2_region0_WT3_z47_s47", 
        "slide2_region1_WT3_z39_s39", 
        "slide2_region1_WT3_z28_s28", 
        "slide1_region1_WT3_z24_s24", 
        "slide1_region1_WT3_z23_s23"
    ],
    "mut1": [
        "slide2_region0_mut1_z48_s49",
        "slide2_region0_mut1_z47_s48", 
        "slide2_region0_mut1_z39_s40",
        "slide2_region1_mut1_z28_s29",
        "slide2_region1_mut1_z27_s28",
        "slide2_region1_mut1_z26_s27",
        "slide1_region1_mut1_z26_s27"
    ],
    "mut2": [
        "slide2_region1_mut2_z27_s20",
        "slide2_region1_mut2_z26_s19",
        "slide2_region0_mut2_z48_s41",
        "slide2_region0_mut2_z47_s40",
        "slide1_region1_mut2_z24_s17"
    ],
    "mut3": [
        "slide2_region0_mut3_z48_s37", 
        "slide2_region0_mut3_z47_s36", 
        "slide2_region1_mut3_z39_s28", 
        "slide2_region1_mut3_z27_s16", 
        "slide2_region1_mut3_z26_s15"  
    ]    
}


# ══════════════════════════════════════════════════════════════════════════
# Small utilities (path/name parsing, geometry)
# ══════════════════════════════════════════════════════════════════════════
def build_image_path(fname: str) -> Path:
    """Reconstruct image path from the fish-first directory structure."""
    parts = Path(fname).stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected section filename: {fname}")

    fish_name = parts[2]  # e.g. WT1, WT2, mut1

    return IMAGE_ROOT / fish_name / fname


def get_section_id(filename: str) -> str:
    return Path(filename).stem


def parse_slide_region(filename: str):
    """Return (slide_num, region_num) from a filename string."""
    stem = Path(filename).stem
    parts = stem.split("_")
    return int(parts[0].replace("slide", "")), int(parts[1].replace("region", ""))


def alignment_csv_path(fish_name: str) -> Path:
    """Path to the alignment CSV for a given fish (confirmed-working pattern)."""
    return ALIGNMENT_FOLDER / f"{fish_name}_alignment-data.csv"


def load_transform_matrix(slide_num: int, region_num: int):
    """Load micron->mosaic-pixel affine matrix (3x3) and its inverse."""
    path = (BASE_DATA_FOLDER / f"slide{slide_num}"
            / f"region_{region_num}" / "images"
            / "micron_to_mosaic_pixel_transform.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing transform matrix: {path}")
    M = np.loadtxt(path)
    M_inv = np.linalg.inv(M)
    return M, M_inv


def transform_with_padding(img: np.ndarray, rotation: float, tx: float, ty: float):
    """
    Pad, rotate, then translate a 2-D image.

    Returns
    -------
    shifted : np.ndarray
        Transformed image.
    origin_in_output : np.ndarray, shape (2,)
        Pixel coordinate (row, col) of the ORIGINAL IMAGE CENTER, AFTER
        rotation + translation, inside the returned array. This is the
        (0,0) origin of the section's transcript coordinates.
    orig_center : np.ndarray, shape (2,)
        Pixel coordinate (row, col) of the ORIGINAL IMAGE CENTER, BEFORE
        rotation/translation, inside the returned array (i.e. just from
        padding/centering). This is the "pre-alignment center" reference
        point used to compute center_x_um / center_y_um and the alignment
        shift.
    """
    h, w = img.shape

    # ── Padding so rotation never clips content ───────────────────────────
    diag = int(np.ceil(np.hypot(h, w)))
    pad_y = diag + int(np.ceil(abs(ty) / 4))
    pad_x = diag + int(np.ceil(abs(tx) / 4))

    padded = np.pad(
        img.astype(np.float32),
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="constant",
        constant_values=0,
    )
    Ph, Pw = padded.shape

    # ── Center of the original image inside the padded array ─────────────
    #    (this is the pre-alignment "center" reference point)
    orig_center = np.array([pad_y + h / 2.0, pad_x + w / 2.0])  # (row, col)

    # ── Rotate around the padded-image center ────────────────────────────
    pad_center = np.array([Ph / 2.0, Pw / 2.0])
    rotated = rotate(padded, rotation, reshape=False, order=1)

    angle_rad = np.deg2rad(rotation)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a],
                  [sin_a, cos_a]])
    delta = orig_center - pad_center
    origin_after_rot = pad_center + R @ delta

    # ── Translate ─────────────────────────────────────────────────────────
    shifted = shift(rotated, shift=(ty, tx), order=1)
    origin_in_output = origin_after_rot + np.array([ty, tx])

    return shifted, origin_in_output, orig_center


def tight_bbox(arrays: list, H: int, W: int, threshold: float = 1e-3):
    """Bounding box that contains all non-zero content across a list of 2-D arrays."""
    mask = np.zeros((H, W), dtype=bool)
    for arr in arrays:
        mask |= arr > threshold

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    print(f"  BBox — y:[{y_min},{y_max}]  x:[{x_min},{x_max}]  "
          f"→ cropped size {y_max - y_min + 1}×{x_max - x_min + 1}  (was {H}×{W})")
    return y_min, y_max, x_min, x_max


def embed_to_canvas(img: np.ndarray, H: int, W: int):
    """Centre-pad img onto an H×W canvas. Returns canvas and (y0, x0) offset."""
    h, w = img.shape
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    canvas = np.zeros((H, W), dtype=np.float32)
    canvas[y0: y0 + h, x0: x0 + w] = img
    return canvas, (y0, x0)


# ══════════════════════════════════════════════════════════════════════════
# Pipeline steps (each is self-contained: explicit inputs -> explicit outputs)
# ══════════════════════════════════════════════════════════════════════════
def step1_load_alignment(csv: Path) -> pd.DataFrame:
    """Load & normalise the alignment CSV (column names, numeric coercion)."""
    df = pd.read_csv(csv)

    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[\s\-\/µ]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    fn_col = next(c for c in df.columns if "filename" in c.lower())
    df = df.rename(columns={fn_col: "Filename"})

    for key in ["Rotation", "Translation_X", "Translation_Y"]:
        col = next(
            (c for c in df.columns
             if key.lower().replace("_", "") in c.lower().replace("_", "")),
            None,
        )
        df[key] = (
            pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if col else 0.0
        )

    print(f"Loaded {len(df)} alignment rows")
    return df


def step2_load_images(df: pd.DataFrame, bad_sections: list):
    """Load each section's image, marking bad/missing sections as None."""
    imgs = []
    for _, row in df.iterrows():
        path = build_image_path(row["Filename"])
        section_id = get_section_id(row["Filename"])

        if section_id in bad_sections or not path.exists():
            print(f"  [blank] {section_id}")
            imgs.append((None, row))
        else:
            img = np.array(Image.open(path).convert("L"))
            imgs.append((img, row))

    n_blank = sum(1 for img, _ in imgs if img is None)
    return imgs, n_blank


def step3_transform_sections(imgs: list):
    """
    Rotate/translate each loaded image; track BOTH:
      - origin: physical origin AFTER rotation/translation
      - center: physical center BEFORE rotation/translation (pre-alignment)
    Returns a list of (transformed_img, origin, center) or (None, None, None).
    """
    transformed = []
    for img, row in imgs:
        if img is None:
            transformed.append((None, None, None))
        else:
            t, origin, center = transform_with_padding(
                img,
                rotation=row["Rotation"],
                tx=row["Translation_X"],
                ty=-row["Translation_Y"],
            )
            transformed.append((t, origin, center))
    return transformed


def step4_compute_canvas_size(transformed: list):
    """Determine the global H×W canvas that fits every transformed slice."""
    H = max(t.shape[0] for t, _, _ in transformed if t is not None)
    W = max(t.shape[1] for t, _, _ in transformed if t is not None)
    print(f"  Global canvas: {H} × {W}")
    return H, W


def step5_embed_on_canvas(transformed: list, H: int, W: int):
    """Centre-embed every transformed slice onto a common H×W canvas.

    The same (y0, x0) embed offset is applied to both `origin` and `center`
    since they are two points within the same transformed array.
    """
    embedded = []
    origins_embedded = []
    centers_embedded = []

    for t, origin, center in transformed:
        if t is None:
            embedded.append(np.zeros((H, W), dtype=np.float32))
            origins_embedded.append(None)
            centers_embedded.append(None)
        else:
            canvas, (y0, x0) = embed_to_canvas(t, H, W)
            offset = np.array([y0, x0])
            embedded.append(canvas)
            origins_embedded.append(origin + offset)
            centers_embedded.append(center + offset)

    print(f"  Embedded {len(embedded)} slices.")
    return embedded, origins_embedded, centers_embedded


def step6_crop_to_bbox(embedded: list, origins_embedded: list, centers_embedded: list,
                        H: int, W: int):
    """Crop the whole stack to the tight bounding box of non-zero content.

    Same crop offset is subtracted from both `origin` and `center`.
    """
    y_min, y_max, x_min, x_max = tight_bbox(embedded, H, W)

    cropped = [arr[y_min: y_max + 1, x_min: x_max + 1] for arr in embedded]
    crop_offset = np.array([y_min, x_min])

    origins_final = []
    centers_final = []
    for origin, center in zip(origins_embedded, centers_embedded):
        if origin is None:
            origins_final.append(None)
            centers_final.append(None)
        else:
            origins_final.append(origin - crop_offset)
            centers_final.append(center - crop_offset)

    print(f"  Cropped stack: {len(cropped)} slices, each {cropped[0].shape}")

    # Quick sanity check on how much the centers spread across slices
    good_centers = np.array([c for c in centers_final if c is not None])
    if len(good_centers) > 0:
        print(f"  Center spread — row: {good_centers[:, 0].max() - good_centers[:, 0].min():.1f} px, "
              f"col: {good_centers[:, 1].max() - good_centers[:, 1].min():.1f} px")

    return cropped, origins_final, centers_final, int(y_min), int(x_min)


def step7_compute_pixel_size(imgs: list):
    """Compute physical pixel size (µm/px) from each region's transform matrix."""
    region_pixel_sizes = {}

    for img, row in imgs:
        if img is None:
            continue
        slide_num, region_num = parse_slide_region(row["Filename"])
        key = (slide_num, region_num)
        if key not in region_pixel_sizes:
            M, _ = load_transform_matrix(slide_num, region_num)
            px_x = (1.0 / M[0, 0]) * DOWNSAMPLE_FACTOR
            px_y = (1.0 / M[1, 1]) * DOWNSAMPLE_FACTOR
            region_pixel_sizes[key] = (px_x, px_y)
            print(f"  slide{slide_num} region{region_num}: {px_x:.4f} × {px_y:.4f} µm/px")

    all_px_x = [v[0] for v in region_pixel_sizes.values()]
    all_px_y = [v[1] for v in region_pixel_sizes.values()]

    pixel_size_x_um = float(np.mean(all_px_x))
    pixel_size_y_um = float(np.mean(all_px_y))
    print(f"\n  Using voxel size: {pixel_size_y_um:.4f} × {pixel_size_x_um:.4f} × {Z_SPACING_UM} µm")

    dx = max(all_px_x) - min(all_px_x)
    dy = max(all_px_y) - min(all_px_y)
    if dx > 0.01 or dy > 0.01:
        print(f"  WARNING: scale varies across regions (Δx={dx:.6f}, Δy={dy:.6f} µm/px)")

    return pixel_size_x_um, pixel_size_y_um, region_pixel_sizes


def step8_build_stack(cropped: list):
    """Stack cropped 2-D slices into a normalised uint16 3-D volume."""
    stack = np.stack(cropped, axis=0).astype(np.float32)  # (Z, H_crop, W_crop)
    s_min, s_max = stack.min(), stack.max()

    if s_max > s_min:
        stack_u16 = ((stack - s_min) / (s_max - s_min) * 65535).astype(np.uint16)
    else:
        stack_u16 = stack.astype(np.uint16)

    print(f"  Stack shape : {stack_u16.shape}  dtype: {stack_u16.dtype}")
    print(f"  Raw range   : [{s_min:.1f}, {s_max:.1f}]")
    return stack_u16, float(s_min), float(s_max)


def step9_build_origin_records(imgs: list, origins_final: list, centers_final: list,
                                pixel_size_x_um: float, pixel_size_y_um: float):
    """
    Build a per-z-index table with:
      - origin_*: physical origin AFTER alignment (rotation/translation)
      - center_*: physical center BEFORE alignment (pre-alignment reference)
      - shift_* : origin - center (the alignment displacement)
    Both px and um are recorded for origin and center.
    """
    origin_records = []

    for z_idx, ((img, row), origin, center) in enumerate(
        zip(imgs, origins_final, centers_final)
    ):
        filename = row["Filename"]
        is_blank = img is None or origin is None or center is None

        if is_blank:
            record = {
                "z_index": z_idx,
                "origin_row_px": None,
                "origin_col_px": None,
                "origin_y_um": None,
                "origin_x_um": None,
                "center_row_px": None,
                "center_col_px": None,
                "center_y_um": None,
                "center_x_um": None,
                "shift_row_px": None,
                "shift_col_px": None,
                "shift_y_um": None,
                "shift_x_um": None,
                "origin_z_um": z_idx * Z_SPACING_UM,
                "filename": filename,
                "is_blank": True,
            }
        else:
            origin_row_px = float(origin[0])
            origin_col_px = float(origin[1])
            origin_y_um = origin_row_px * pixel_size_y_um
            origin_x_um = origin_col_px * pixel_size_x_um

            center_row_px = float(center[0])
            center_col_px = float(center[1])
            center_y_um = center_row_px * pixel_size_y_um
            center_x_um = center_col_px * pixel_size_x_um

            shift_row_px = origin_row_px - center_row_px
            shift_col_px = origin_col_px - center_col_px
            shift_y_um = shift_row_px * pixel_size_y_um
            shift_x_um = shift_col_px * pixel_size_x_um

            record = {
                "z_index": z_idx,
                "origin_row_px": origin_row_px,
                "origin_col_px": origin_col_px,
                "origin_y_um": origin_y_um,
                "origin_x_um": origin_x_um,
                "center_row_px": center_row_px,
                "center_col_px": center_col_px,
                "center_y_um": center_y_um,
                "center_x_um": center_x_um,
                "shift_row_px": shift_row_px,
                "shift_col_px": shift_col_px,
                "shift_y_um": shift_y_um,
                "shift_x_um": shift_x_um,
                "origin_z_um": z_idx * Z_SPACING_UM,
                "filename": filename,
                "is_blank": False,
            }
            print(f"  z={z_idx:03d}  origin=({origin_row_px:.1f}, {origin_col_px:.1f}) px  "
                  f"center=({center_row_px:.1f}, {center_col_px:.1f}) px  "
                  f"shift=({shift_row_px:.1f}, {shift_col_px:.1f}) px  "
                  f"| {get_section_id(filename)}")

        origin_records.append(record)

    print(f"\n  {sum(1 for r in origin_records if not r['is_blank'])} good records recorded.")
    return origin_records


def step10_save_nrrd(fish_output_dir: Path, fish_name: str, stack_u16: np.ndarray,
                      pixel_size_x_um: float, pixel_size_y_um: float,
                      origin_records: list) -> Path:
    """Save the volume as NRRD (X, Y, Z axis order) with metadata in the header."""
    stack_nrrd = stack_u16.transpose(2, 1, 0)  # (Z, H, W) -> (W, H, Z) = (X, Y, Z)

    header = {
        "type": "uint16",
        "dimension": 3,
        "space": "left-posterior-superior",
        "sizes": list(stack_nrrd.shape),
        "space directions": [
            [pixel_size_x_um, 0, 0],
            [0, pixel_size_y_um, 0],
            [0, 0, Z_SPACING_UM],
        ],
        "space origin": [0.0, 0.0, 0.0],
        "kinds": ["domain", "domain", "domain"],
        "endian": "little",
        "encoding": "gzip",
        "section_metadata_json": json.dumps(origin_records),
    }

    nrrd_path = fish_output_dir / f"{fish_name}_zstack.nrrd"
    nrrd.write(str(nrrd_path), stack_nrrd, header=header)
    print(f"  Saved NRRD : {nrrd_path}  shape={stack_nrrd.shape}")
    return nrrd_path


def step11_save_sidecar_json(fish_output_dir: Path, fish_name: str,
                              pixel_size_x_um: float, pixel_size_y_um: float,
                              stack_nrrd_shape, y_min: int, x_min: int,
                              origin_records: list) -> Path:
    """Save the canonical, human-readable origin/center/shift sidecar JSON."""
    sidecar = {
        "fish": fish_name,
        "description": (
            "Per-slice metadata for transcript and image coordinate mapping. "
            "center_* fields represent the section's own image centre BEFORE "
            "SlideAligner rotation/translation is applied (a shared, "
            "pre-alignment reference point). origin_* fields represent the "
            "position of transcript-space (0,0) AFTER alignment. shift_* "
            "fields are the alignment displacement (origin - center). All "
            "three are already expressed in the final cropped NRRD coordinate "
            "system (padding, rotation, translation, embed, and crop offsets "
            "are all baked in)."
        ),
        "voxel_size_um": {
            "x": pixel_size_x_um,
            "y": pixel_size_y_um,
            "z": Z_SPACING_UM,
        },
        "nrrd_shape_xyz": list(stack_nrrd_shape),
        "pipeline_offsets": {
            "bbox_crop_y_min": y_min,
            "bbox_crop_x_min": x_min,
        },
        "section_metadata": origin_records,
    }

    json_path = fish_output_dir / f"{fish_name}_zstack_metadata.json"
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"  Saved JSON : {json_path}")
    return json_path


# ══════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════
def process_fish(fish_name: str, bad_sections: list) -> dict:
    print("\n" + "=" * 100)
    print(f"PROCESSING {fish_name}")
    print("=" * 100)

    alignment_csv = alignment_csv_path(fish_name)
    fish_output_dir = OUTPUT_DIR / fish_name 
    fish_output_dir.mkdir(parents=True, exist_ok=True)

    if not alignment_csv.exists():
        raise FileNotFoundError(f"Missing alignment CSV: {alignment_csv}")

    print("\n── STEP 1: Load alignment CSV ──────────────────────────────────────────")
    df = step1_load_alignment(alignment_csv)

    print("\n── STEP 2: Load images ─────────────────────────────────────────────────")
    imgs, n_blank = step2_load_images(df, bad_sections)

    print("\n── STEP 3: Transform sections & track origin + center ──────────────────")
    transformed = step3_transform_sections(imgs)

    print("\n── STEP 4: Compute global canvas size ──────────────────────────────────")
    H, W = step4_compute_canvas_size(transformed)

    print("\n── STEP 5: Embed onto global canvas ────────────────────────────────────")
    embedded, origins_embedded, centers_embedded = step5_embed_on_canvas(transformed, H, W)

    print("\n── STEP 6: Tight bounding box crop ─────────────────────────────────────")
    cropped, origins_final, centers_final, y_min, x_min = step6_crop_to_bbox(
        embedded, origins_embedded, centers_embedded, H, W
    )

    print("\n── STEP 7: Compute physical pixel size (µm/px) ─────────────────────────")
    pixel_size_x_um, pixel_size_y_um, _ = step7_compute_pixel_size(imgs)

    print("\n── STEP 8: Build & normalise z-stack ───────────────────────────────────")
    stack_u16, s_min, s_max = step8_build_stack(cropped)

    print("\n── STEP 9: Build origin & center metadata ──────────────────────────────")
    origin_records = step9_build_origin_records(
        imgs, origins_final, centers_final, pixel_size_x_um, pixel_size_y_um
    )

    print("\n── STEP 10: Save NRRD ──────────────────────────────────────────────────")
    nrrd_path = step10_save_nrrd(
        fish_output_dir, fish_name, stack_u16,
        pixel_size_x_um, pixel_size_y_um, origin_records,
    )
    stack_nrrd_shape = stack_u16.transpose(2, 1, 0).shape

    print("\n── STEP 11: Save sidecar JSON ──────────────────────────────────────────")
    step11_save_sidecar_json(
        fish_output_dir, fish_name, pixel_size_x_um, pixel_size_y_um,
        stack_nrrd_shape, y_min, x_min, origin_records,
    )

    return {
        "fish": fish_name,
        "n_slices": len(cropped),
        "n_blank": n_blank,
        "shape": str(stack_nrrd_shape),
        "nrrd_path": str(nrrd_path),
    }


def main():
    results = []
    for fish_name, bad_sections in BAD_SECTIONS_BY_FISH.items():
        try:
            result = process_fish(fish_name=fish_name, bad_sections=bad_sections)
            results.append(result)
        except Exception as e:
            print("\n" + "!" * 100)
            print(f"FAILED: {fish_name}")
            print(e)
            print("!" * 100)

    summary = pd.DataFrame(results)

    print("\n")
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    try:
        display(summary)  # only defined inside Jupyter/IPython
    except NameError:
        print(summary)

    return summary


if __name__ == "__main__":
    main()
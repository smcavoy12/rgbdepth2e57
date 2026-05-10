#!/usr/bin/env python3
"""
depth_to_e57.py
---------------
Convert matched RGB (JPG) + depth (TIFF) pairs from DepthCam on iPhone
into E57 point clouds for import into RealityCapture / RealityScan.

REALITYSCAN WORKFLOW
--------------------
1. Import the E57 files via  Workflow → LiDAR Scan → Mobile LiDAR
   (Registration: Unregistered, let RC generate virtual cameras automatically)
2. Import the RGB JPGs separately via  Workflow → Add Images
3. Align everything together — RC fuses the geometry and texture

The images are NOT embedded inside the E57. RealityScan's Mobile LiDAR path
does not read pinholeRepresentation from E57; images must be imported as
separate photos alongside the point cloud.

COORDINATE CONVENTION
---------------------
Points are written in right-handed, Y-up world coordinates:
  X = right,  Y = up,  Z = towards camera (out of screen)
This matches what RealityCapture and Metashape expect from a camera-space
point cloud, and means the RGB image and point cloud are correctly oriented
relative to each other when imported together.

USAGE
-----
Edit the CONFIG section below, then run:
    python depth_to_e57.py
"""

# =============================================================================
# CONFIG
# =============================================================================

# --- Mode --------------------------------------------------------------------
MODE = "batch"    # "single" or "batch"

# --- Single mode inputs ------------------------------------------------------
SINGLE_RGB    = "/path/to/frame_001.jpg"
SINGLE_DEPTH  = "/path/to/frame_001.tif"
SINGLE_OUTPUT = None    # None = auto (same folder/stem as RGB, .e57 extension)

# --- Batch mode inputs -------------------------------------------------------
# RGB folder: JPG files.  Depth folder: TIFF files.  Matched by filename stem.
BATCH_RGB_DIR    = "M:/smcavoy/depthcam_as_e57/iphone_portrait_rgb/"
BATCH_DEPTH_DIR  = "M:/smcavoy/depthcam_as_e57/depthcam_depth/"
BATCH_OUTPUT_DIR = "M:/smcavoy/depthcam_as_e57/e57_outputs_rc_tls/"

# --- Depth units -------------------------------------------------------------
DEPTH_UNIT = "m"    # "m" = float32 metres  |  "mm" = uint16 millimetres

# --- Depth filtering ---------------------------------------------------------
MIN_DEPTH = 0.1    # metres — pixels closer than this become no-returns
MAX_DEPTH = 4.0    # metres — pixels farther than this become no-returns

# --- Camera intrinsics -------------------------------------------------------
# Leave FX = None to extract from RGB EXIF automatically.
# If set manually, provide values at RGB image resolution;
# the script scales them to depth resolution automatically.
#
# Approx values at 4032×3024:
#   iPhone 12 Pro  → FX ≈ 2416,  CX = 2016,  CY = 1512
#   iPhone 13 Pro  → FX ≈ 3279,  CX = 2016,  CY = 1512
FX = None    # float or None (auto from EXIF)
FY = None    # float or None (defaults to FX)
CX = None    # float or None (defaults to image centre)
CY = None    # float or None (defaults to image centre)

SCALE_INTRINSICS_TO_DEPTH = True    # True = intrinsics given at RGB resolution

# --- iPhone pixel pitch (for embedded image focalLength field) --------------
# Physical pixel size in metres, used only for the E57 pinholeRepresentation
# metadata that records the embedded image's focalLength. Does not affect
# point geometry.
# iPhone 12 Pro wide: 1.74 µm.  iPhone 13/14/15/16 Pro wide: ~2.19 µm.
PIXEL_PITCH_M = 1.74e-6

# =============================================================================
# END OF CONFIG
# =============================================================================

import math
import shutil
import sys
import uuid
import warnings
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

try:
    import pye57
    import pye57.libe57 as libe57
except ImportError:
    sys.exit("pye57 is required:  pip install pye57")

try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False


# ---------------------------------------------------------------------------
# iPhone sensor lookup for EXIF model fallback
# (sensor_width_mm, sensor_height_mm, wide_focal_length_mm)
# ---------------------------------------------------------------------------
IPHONE_SENSORS = {
    "iPhone 16 Pro Max": (8.83, 6.62, 6.765),
    "iPhone 16 Pro":     (8.83, 6.62, 6.765),
    "iPhone 16 Plus":    (7.91, 5.93, 5.77),
    "iPhone 16":         (7.91, 5.93, 5.77),
    "iPhone 15 Pro Max": (8.83, 6.62, 6.765),
    "iPhone 15 Pro":     (8.83, 6.62, 6.765),
    "iPhone 15 Plus":    (7.91, 5.93, 5.77),
    "iPhone 15":         (7.91, 5.93, 5.77),
    "iPhone 14 Pro Max": (8.83, 6.62, 6.765),
    "iPhone 14 Pro":     (8.83, 6.62, 6.765),
    "iPhone 14 Plus":    (7.91, 5.93, 5.77),
    "iPhone 14":         (7.91, 5.93, 5.77),
    "iPhone 13 Pro Max": (7.01, 5.27, 5.7),
    "iPhone 13 Pro":     (7.01, 5.27, 5.7),
    "iPhone 13 mini":    (7.01, 5.27, 5.1),
    "iPhone 13":         (7.01, 5.27, 5.1),
    "iPhone 12 Pro Max": (7.01, 5.27, 5.0),
    "iPhone 12 Pro":     (7.01, 5.27, 4.2),
    "iPhone 12 mini":    (5.64, 4.23, 4.2),
    "iPhone 12":         (6.17, 4.63, 4.2),
}


# ---------------------------------------------------------------------------
# EXIF intrinsics extraction
# ---------------------------------------------------------------------------

def _rational(val):
    if isinstance(val, tuple) and len(val) == 2:
        return val[0] / val[1] if val[1] != 0 else 0.0
    return float(val)


def extract_intrinsics_from_exif(rgb_path: Path):
    """Return {fx, fy, cx, cy, img_w, img_h, source} or None."""
    try:
        img = Image.open(str(rgb_path))
        img_w, img_h = img.size
    except Exception as e:
        print(f"  Cannot open RGB for EXIF: {e}"); return None

    exif_dict = None
    if HAS_PIEXIF:
        try: exif_dict = piexif.load(str(rgb_path))
        except Exception: pass
    if exif_dict is None:
        try:
            raw = img._getexif()
            if raw: exif_dict = {"Exif": raw, "0th": {}}
        except Exception: pass
    if not exif_dict:
        print("  No EXIF found."); return None

    exif = exif_dict.get("Exif", {})
    ifd0 = exif_dict.get("0th", {})
    T_FL   = piexif.ExifIFD.FocalLength if HAS_PIEXIF else 37386
    T_FPXR = piexif.ExifIFD.FocalPlaneXResolution if HAS_PIEXIF else 41486
    T_FPYR = piexif.ExifIFD.FocalPlaneYResolution if HAS_PIEXIF else 41487
    T_FPRU = piexif.ExifIFD.FocalPlaneResolutionUnit if HAS_PIEXIF else 41488
    T_FL35 = piexif.ExifIFD.FocalLengthIn35mmFilm if HAS_PIEXIF else 41989
    T_MOD  = piexif.ImageIFD.Model if HAS_PIEXIF else 272

    focal_mm  = _rational(exif[T_FL]) if T_FL in exif else None
    raw_model = ifd0.get(T_MOD)
    model_str = (raw_model.decode() if isinstance(raw_model, bytes)
                 else str(raw_model)) if raw_model else None
    cx0, cy0  = img_w / 2.0, img_h / 2.0

    # Method 1 — FocalLength + FocalPlaneXResolution
    if focal_mm and T_FPXR in exif:
        xres = _rational(exif[T_FPXR])
        yres = _rational(exif.get(T_FPYR, exif[T_FPXR]))
        unit = exif.get(T_FPRU, 2)
        div  = 25.4 if unit == 2 else (10.0 if unit == 3 else 1.0)
        xres /= div; yres /= div
        if xres > 0 and yres > 0:
            return dict(fx=focal_mm*xres, fy=focal_mm*yres, cx=cx0, cy=cy0,
                        img_w=img_w, img_h=img_h,
                        source=f"FocalLength({focal_mm}mm)+FocalPlaneXRes({xres:.1f}px/mm)")

    # Method 2 — FocalLength + model lookup
    if focal_mm and model_str:
        for key, (sw, sh, _) in IPHONE_SENSORS.items():
            if key in model_str:
                return dict(fx=focal_mm/(sw/img_w), fy=focal_mm/(sh/img_h),
                            cx=cx0, cy=cy0, img_w=img_w, img_h=img_h,
                            source=f"FocalLength({focal_mm}mm)+model({key})")

    # Method 3 — FocalLengthIn35mmFilm (rough)
    if T_FL35 in exif:
        fl35 = _rational(exif[T_FL35]) if isinstance(exif[T_FL35], tuple) \
               else float(exif[T_FL35])
        if fl35 > 0:
            fx = fl35 * (img_w / 36.0)
            warnings.warn(f"Using approximate 35mm-equiv fx≈{fx:.0f}. Set FX in CONFIG.")
            return dict(fx=fx, fy=fx, cx=cx0, cy=cy0, img_w=img_w, img_h=img_h,
                        source=f"FocalLengthIn35mmFilm({fl35}mm)[approx]")

    print("  EXIF present but missing needed tags.")
    return None


def resolve_intrinsics(rgb_path: Path, depth_h: int, depth_w: int):
    """Return (fx, fy, cx, cy) at depth map resolution."""
    if FX is not None:
        fx = float(FX); fy = float(FY) if FY is not None else fx
        cx = float(CX) if CX is not None else depth_w / 2.0
        cy = float(CY) if CY is not None else depth_h / 2.0
        if SCALE_INTRINSICS_TO_DEPTH:
            img = Image.open(str(rgb_path)); rw, rh = img.size
            if rw != depth_w or rh != depth_h:
                sx, sy = depth_w/rw, depth_h/rh
                fx*=sx; fy*=sy; cx*=sx; cy*=sy
                print(f"  Intrinsics (config scaled {rw}x{rh}→{depth_w}x{depth_h}): "
                      f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
                return fx, fy, cx, cy
        print(f"  Intrinsics (config): fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
        return fx, fy, cx, cy

    print(f"  FX not set — extracting from EXIF of {rgb_path.name} ...")
    info = extract_intrinsics_from_exif(rgb_path)
    if info is None:
        sys.exit("\nERROR: Cannot determine intrinsics. Set FX in CONFIG.")

    fx_r, fy_r, cx_r, cy_r = info["fx"], info["fy"], info["cx"], info["cy"]
    rw, rh = info["img_w"], info["img_h"]
    print(f"  Source: {info['source']}")
    print(f"  At RGB ({rw}x{rh}): fx={fx_r:.1f} fy={fy_r:.1f} cx={cx_r:.1f} cy={cy_r:.1f}")

    if SCALE_INTRINSICS_TO_DEPTH and (rw != depth_w or rh != depth_h):
        sx, sy = depth_w/rw, depth_h/rh
        fx = fx_r*sx; fy = fy_r*sy; cx = cx_r*sx; cy = cy_r*sy
        print(f"  Scaled to depth ({depth_w}x{depth_h}): "
              f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    else:
        fx, fy, cx, cy = fx_r, fy_r, cx_r, cy_r

    return fx, fy, cx, cy


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def _make_cube_face(n_rows: int, n_cols: int, row_offset: int) -> dict:
    """
    Build one empty cube face: a structured grid of NaN points marked invalid.

    A real TLS scan covers a full sphere; RC's cube-map projection expects
    six 90° faces. Our depth camera only fills the real face, so we pad the
    other five with placeholder no-return cells.

    Two safeguards make sure these dummy points don't render anywhere:
      - cartesianInvalidState = 2 (E57 standard "invalid" flag, honoured by RC)
      - X/Y/Z = NaN (tools that ignore cartesianInvalidState, like Metashape,
                     drop NaN points from rendering)

    The grid still has valid row/column indices so the structured-scan
    invariants are preserved.
    """
    n = n_rows * n_cols
    rows = (np.tile(np.arange(n_rows, dtype=np.uint16).reshape(-1, 1),
                    (1, n_cols)) + row_offset).ravel()
    cols =  np.tile(np.arange(n_cols, dtype=np.uint16), (n_rows, 1)).ravel()
    nan_pts = np.full(n, np.nan, dtype=np.float64)

    return {
        "cartesianX":            nan_pts.copy(),
        "cartesianY":            nan_pts.copy(),
        "cartesianZ":            nan_pts.copy(),
        "colorRed":              np.zeros(n, dtype=np.uint8),
        "colorGreen":            np.zeros(n, dtype=np.uint8),
        "colorBlue":             np.zeros(n, dtype=np.uint8),
        "rowIndex":              rows,
        "columnIndex":           cols,
        "cartesianInvalidState": np.full(n, 2, dtype=np.int8),
    }


def pair_to_e57(rgb_path: Path, depth_path: Path, output_path: Path) -> None:
    # --- Load depth ---
    raw = tifffile.imread(str(depth_path)).astype(np.float32)
    if DEPTH_UNIT == "mm":
        raw /= 1000.0
    raw[raw <= 0] = 0.0
    depth = raw
    h, w  = depth.shape

    # --- Intrinsics ---
    fx, fy, cx, cy = resolve_intrinsics(rgb_path, h, w)

    # --- RGB at depth resolution for point colouring ---
    rgb_small = np.array(
        Image.open(str(rgb_path)).convert("RGB").resize((w, h), Image.BILINEAR),
        dtype=np.uint8
    )

    # --- Back-project to camera space (X right, Y down, Z forward positive) ---
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    valid = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)

    X_cam = (uu - cx) * depth / fx     # camera right
    Y_cam = (vv - cy) * depth / fy     # camera down
    Z_cam = depth                      # camera forward

    # Rotate camera-space points into TLS scanner convention:
    #   world +X = right (scanner left/right axis)
    #   world +Y = forward (scanner looks horizontally along +Y)
    #   world +Z = up (scanner zenith — TLS convention)
    #
    # Mapping: world_X = camera_X, world_Y = camera_Z, world_Z = -camera_Y
    # This puts the real photographic face on the +Y side of the cube
    # (horizontal, where TLS cameras belong) instead of on top (+Z, the zenith).
    X_real = np.where(valid,  X_cam,  0.0)   # right
    Y_real = np.where(valid,  Z_cam,  0.0)   # forward
    Z_real = np.where(valid, -Y_cam,  0.0)   # up

    n_real = h * w
    rows_real = np.repeat(np.arange(h, dtype=np.uint16), w)
    cols_real = np.tile(np.arange(w, dtype=np.uint16), h)

    # cartesianInvalidState: 0 for valid points, 2 for no-return / out of range
    invalid_real = np.where(valid.ravel(), 0, 2).astype(np.int8)

    real_face = {
        "cartesianX":            X_real.ravel().astype(np.float64),
        "cartesianY":            Y_real.ravel().astype(np.float64),
        "cartesianZ":            Z_real.ravel().astype(np.float64),
        "colorRed":              rgb_small[:, :, 0].ravel(),
        "colorGreen":            rgb_small[:, :, 1].ravel(),
        "colorBlue":             rgb_small[:, :, 2].ravel(),
        "rowIndex":              rows_real,
        "columnIndex":           cols_real,
        "cartesianInvalidState": invalid_real,
    }

    n_valid = int(valid.sum())
    print(f"  {depth_path.name}: face 0 (forward) — {h}×{w} grid "
          f"({n_valid:,} valid, {n_real-n_valid:,} no-return)")

    # --- Build five dummy faces to complete the TLS cube ---
    # Faces tile vertically: real face occupies rows 0..h-1, dummy faces
    # rows h..6h-1. Each dummy face is a NaN no-return grid that tools will
    # drop from rendering but that preserves the structured-scan invariants.
    faces = [real_face]
    for i in range(1, 6):
        faces.append(_make_cube_face(h, w, row_offset=i * h))

    # Concatenate all faces
    keys = list(real_face.keys())
    scan_data = {k: np.concatenate([f[k] for f in faces]) for k in keys}

    total_rows = h * 6
    total_pts  = scan_data["cartesianX"].size
    print(f"  Cube assembled: 6 faces × {h}×{w} = {total_pts:,} pts "
          f"(grid rows 0..{total_rows-1})")

    # --- Write E57 ---
    e57 = pye57.E57(str(output_path), mode="w")
    imf = e57.image_file

    e57.write_scan_raw(
        scan_data,
        name=rgb_path.stem,
        rotation=np.array([1.0, 0.0, 0.0, 0.0]),
        translation=np.array([0.0, 0.0, 0.0]),
    )

    # pointGroupingSchemes for the full 6-face grid
    data3d    = libe57.VectorNode(e57.root.get("data3D"))
    scan_node = libe57.StructureNode(data3d.get(data3d.childCount() - 1))
    pgs  = libe57.StructureNode(imf)
    grid = libe57.StructureNode(imf)
    grid.set("rowMinimum",    libe57.IntegerNode(imf, 0))
    grid.set("rowMaximum",    libe57.IntegerNode(imf, total_rows - 1))
    grid.set("columnMinimum", libe57.IntegerNode(imf, 0))
    grid.set("columnMaximum", libe57.IntegerNode(imf, w - 1))
    pgs.set("gridIndex", grid)
    scan_node.set("pointGroupingSchemes", pgs)

    # --- Embed the full-resolution RGB as a pinholeRepresentation ---
    scan_guid = libe57.StringNode(scan_node.get("guid")).value()

    rgb_full = Image.open(str(rgb_path)).convert("RGB")
    rgb_w, rgb_h = rgb_full.size
    sx_back = rgb_w / w
    sy_back = rgb_h / h
    fx_rgb  = fx * sx_back
    cx_rgb  = cx * sx_back
    cy_rgb  = cy * sy_back
    focal_m = fx_rgb * PIXEL_PITCH_M

    # Embed the original (unflipped) JPEG. The image's pose rotation aligns its
    # local frame with the rotated scanner convention so the photographic content
    # ends up correctly oriented relative to the points.
    with open(str(rgb_path), "rb") as f:
        jpeg_bytes = f.read()

    images2d = libe57.VectorNode(e57.root.get("images2D"))
    img_node = libe57.StructureNode(imf)
    img_node.set("guid",                 libe57.StringNode(imf, "{%s}" % uuid.uuid4()))
    img_node.set("name",                 libe57.StringNode(imf, rgb_path.stem))
    img_node.set("description",          libe57.StringNode(imf, "RGB image from DepthCam"))
    img_node.set("associatedData3DGuid", libe57.StringNode(imf, scan_guid))

    pose  = libe57.StructureNode(imf)
    rot   = libe57.StructureNode(imf)
    # Image pose rotation (image frame → scanner frame).
    # The image's local axes per E57 spec: +X right, +Y down, +Z into scene.
    # The scanner (after our point rotation) uses: +X right, +Y forward, +Z up.
    # Mapping image axes to scanner axes:
    #   image +X (right)    -> scanner +X
    #   image +Y (down)     -> scanner -Z (down in TLS world)
    #   image +Z (forward)  -> scanner +Y (forward in TLS world)
    # As a quaternion this is a +90° rotation around scanner +X (the world
    # right axis): (w, x, y, z) = (√2/2, √2/2, 0, 0).
    sqrt2 = math.sqrt(2.0) / 2.0
    rot.set("w", libe57.FloatNode(imf, sqrt2))
    rot.set("x", libe57.FloatNode(imf, sqrt2))
    rot.set("y", libe57.FloatNode(imf, 0.0))
    rot.set("z", libe57.FloatNode(imf, 0.0))
    trans = libe57.StructureNode(imf)
    trans.set("x", libe57.FloatNode(imf, 0.0))
    trans.set("y", libe57.FloatNode(imf, 0.0))
    trans.set("z", libe57.FloatNode(imf, 0.0))
    pose.set("rotation",    rot)
    pose.set("translation", trans)
    img_node.set("pose", pose)

    pinhole = libe57.StructureNode(imf)
    blob    = libe57.BlobNode(imf, len(jpeg_bytes))
    pinhole.set("jpegImage",       blob)
    pinhole.set("imageWidth",      libe57.IntegerNode(imf, rgb_w))
    pinhole.set("imageHeight",     libe57.IntegerNode(imf, rgb_h))
    pinhole.set("imageSize",       libe57.IntegerNode(imf, len(jpeg_bytes)))
    pinhole.set("focalLength",     libe57.FloatNode(imf, focal_m))
    pinhole.set("pixelWidth",      libe57.FloatNode(imf, PIXEL_PITCH_M))
    pinhole.set("pixelHeight",     libe57.FloatNode(imf, PIXEL_PITCH_M))
    pinhole.set("principalPointX", libe57.FloatNode(imf, cx_rgb))
    pinhole.set("principalPointY", libe57.FloatNode(imf, cy_rgb))
    img_node.set("pinholeRepresentation", pinhole)
    images2d.append(img_node)
    blob.write(jpeg_bytes, 0, len(jpeg_bytes))

    e57.close()

    size_mb = output_path.stat().st_size / 1e6
    img_kb  = len(jpeg_bytes) / 1024
    print(f"  Embedded image: {rgb_path.name} ({img_kb:.0f} KB, "
          f"{rgb_w}×{rgb_h}px, focal={focal_m*1000:.2f}mm)")
    print(f"  → {output_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_single():
    rgb_p   = Path(SINGLE_RGB)
    depth_p = Path(SINGLE_DEPTH)
    out_p   = Path(SINGLE_OUTPUT) if SINGLE_OUTPUT else rgb_p.with_suffix(".e57")

    for p, label in [(rgb_p, "SINGLE_RGB"), (depth_p, "SINGLE_DEPTH")]:
        if not p.exists():
            sys.exit(f"ERROR: {label} not found: {p}")

    print("Single mode")
    print(f"  RGB   : {rgb_p}")
    print(f"  Depth : {depth_p}")
    print(f"  Output: {out_p}")
    pair_to_e57(rgb_p, depth_p, out_p)
    # Copy RGB alongside the E57 (terrestrial scans have the image embedded
    # in the E57 itself; the loose copy is just a convenience for inspection)
    jpg_out = out_p.with_suffix(rgb_p.suffix)
    if jpg_out != rgb_p:
        shutil.copy2(str(rgb_p), str(jpg_out))
        print(f"  RGB copy → {jpg_out}")


def run_batch():
    rgb_dir   = Path(BATCH_RGB_DIR)
    depth_dir = Path(BATCH_DEPTH_DIR)
    out_dir   = Path(BATCH_OUTPUT_DIR) if BATCH_OUTPUT_DIR \
                else rgb_dir / "e57_output"

    for d, label in [(rgb_dir, "BATCH_RGB_DIR"), (depth_dir, "BATCH_DEPTH_DIR")]:
        if not d.is_dir():
            sys.exit(f"ERROR: {label} not found: {d}")

    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_exts   = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
    depth_exts = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    rgb_files   = [f for e in rgb_exts   for f in sorted(rgb_dir.glob(e))]
    depth_files = [f for e in depth_exts for f in sorted(depth_dir.glob(e))]
    rgb_map     = {f.stem: f for f in rgb_files}
    depth_map   = {f.stem: f for f in depth_files}
    stems       = sorted(set(rgb_map) & set(depth_map))

    if not stems:
        sys.exit(
            "ERROR: No matching pairs found.\n"
            "RGB folder needs JPG files, depth folder needs TIFF files,\n"
            "with matching stems (e.g. frame_001.jpg + frame_001.tif)."
        )

    print(f"Batch mode — {len(stems)} pairs → {out_dir}")
    for stem in stems:
        rgb_p   = rgb_map[stem]
        depth_p = depth_map[stem]
        out_p   = out_dir / (stem + ".e57")
        pair_to_e57(rgb_p, depth_p, out_p)
        # Copy RGB alongside the E57 for convenience
        jpg_dest = out_dir / rgb_p.name
        shutil.copy2(str(rgb_p), str(jpg_dest))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if MODE == "single":
        run_single()
    elif MODE == "batch":
        run_batch()
    else:
        sys.exit(f"ERROR: MODE must be 'single' or 'batch', got '{MODE}'")

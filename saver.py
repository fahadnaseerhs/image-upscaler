"""
saver.py — Pipeline Exit Point

Responsibility: Receive three fully interpolated float64 channel arrays from
interpolation.py, convert them back to integer pixel values, merge the
channels into a single RGB image, and write the final image to disk.

Also owns output directory management — creation, filename generation, and
overwrite protection via timestamp suffixes.

Dependencies:
    NumPy                 : denormalization, clipping, rounding, stacking
    Pillow (PIL)          : writing the final image + optional sharpening
    pathlib               : directory creation and path handling
    datetime              : timestamp suffix for overwrite protection
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Public orchestrator — the only function main.py needs to call
# ---------------------------------------------------------------------------

def save_all_channels(
    r_filled: np.ndarray,
    g_filled: np.ndarray,
    b_filled: np.ndarray,
    input_path: str | Path,
    output_dir: str | Path,
    method: str,
    scale_factor: int,
    save_channels: bool = False,
    sharpen: bool = False,
    sharpen_radius: float = 1.5,
    sharpen_percent: int = 120,
    sharpen_threshold: int = 3,
) -> str:
    """
    Full saver pipeline in one call.

    Steps (in order):
        1. denormalize()        → r_uint8, g_uint8, b_uint8  (uint8, 0–255)
        2. merge_channels()     → image_array  (H*scale, W*scale, 3) uint8
        3. [sharpen_image()]    → optional UnsharpMask post-process
        4. generate_filename()  → e.g. "photo_bicubic_4x.png"
        5. save_image()         → writes PNG, returns absolute path string

    Optionally saves each channel as a separate grayscale PNG when
    *save_channels* is True (debug mode, --save-channels CLI flag).

    Args:
        r_filled, g_filled, b_filled: float64 arrays (H*scale, W*scale),
            values ~0.0–1.0 (Lanczos may slightly overshoot).
        input_path:        Original image path — used for filename stem.
        output_dir:        Destination directory (created if missing).
        method:            "bicubic" or "lanczos" — embedded in filename.
        scale_factor:      Integer upscale factor — embedded in filename.
        save_channels:     If True, also saves R/G/B grayscale debug images.
        sharpen:           If True, applies UnsharpMask after interpolation.
        sharpen_radius:    UnsharpMask blur radius (pixels). Default 1.5.
        sharpen_percent:   Strength of sharpening, 0–200+. Default 120.
        sharpen_threshold: Minimum edge difference to sharpen. Default 3.

    Returns:
        Absolute path string of the saved image.
    """
    # Step 1 — denormalize all three channels
    r_uint8 = denormalize(r_filled)
    g_uint8 = denormalize(g_filled)
    b_uint8 = denormalize(b_filled)

    # Step 2 — merge into one (H*scale, W*scale, 3) array
    image_array = merge_channels(r_uint8, g_uint8, b_uint8)

    # Step 3 — optional sharpening (counteracts interpolation smoothing)
    if sharpen:
        image_array = sharpen_image(
            image_array,
            radius=sharpen_radius,
            percent=sharpen_percent,
            threshold=sharpen_threshold,
        )

    # Step 4 — build a descriptive, collision-safe output filename
    filename = generate_filename(input_path, output_dir, method, scale_factor)

    # Step 5 — write to disk
    saved_path = save_image(image_array, output_dir, filename)

    # Optional debug channel saves
    if save_channels:
        base = Path(filename).stem   # e.g. "photo_bicubic_4x"
        for ch_array, ch_name in ((r_uint8, "R"), (g_uint8, "G"), (b_uint8, "B")):
            save_channel_debug(ch_array, output_dir, base, ch_name)

    return saved_path


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------

def denormalize(channel_array: np.ndarray) -> np.ndarray:
    """
    Convert a float64 channel (0.0–1.0) back to uint8 (0–255).

    This is the exact inverse of loader.py's ``normalize()``, with an extra
    clip step to handle Lanczos ringing (values that slightly exceed [0, 1]).

    Operation order:
        1. Multiply by 255.0   — scale to float [0.0, 255.0]
        2. Clip to [0.0, 255.0] — eliminates Lanczos overshoot *before* rounding
        3. Round               — 254.9 → 255, not 254 (prevents systematic bias)
        4. Cast to uint8

    Args:
        channel_array: (H*scale, W*scale) float64, values nominally 0.0–1.0.

    Returns:
        denormalized — same shape, dtype uint8, values 0–255.
    """
    scaled = channel_array * 255.0
    clipped = np.clip(scaled, 0.0, 255.0)   # handles Lanczos ringing overshoot
    rounded = np.round(clipped)              # prevents systematic darkening bias
    denormalized: np.ndarray = rounded.astype(np.uint8)
    return denormalized


def merge_channels(
    r_uint8: np.ndarray,
    g_uint8: np.ndarray,
    b_uint8: np.ndarray,
) -> np.ndarray:
    """
    Stack three separate 2D uint8 channels into one (H, W, 3) RGB array.

    Uses ``np.stack(..., axis=2)`` (explicit axis) rather than ``np.dstack``
    for readability — it makes the new axis position unambiguous.

    Args:
        r_uint8, g_uint8, b_uint8: uint8 arrays, each (H*scale, W*scale).

    Returns:
        image_array — shape (H*scale, W*scale, 3), dtype uint8.

    Raises:
        ValueError: If the three channel arrays do not all share the same shape.
    """
    if not (r_uint8.shape == g_uint8.shape == b_uint8.shape):
        raise ValueError(
            f"Channel shape mismatch — cannot merge:\n"
            f"  R: {r_uint8.shape}\n"
            f"  G: {g_uint8.shape}\n"
            f"  B: {b_uint8.shape}\n"
            "All three channels must have identical (H, W) shape."
        )

    image_array: np.ndarray = np.stack([r_uint8, g_uint8, b_uint8], axis=2)

    expected_shape = (*r_uint8.shape, 3)
    assert image_array.shape == expected_shape, (
        f"Merge produced unexpected shape {image_array.shape}, "
        f"expected {expected_shape}."
    )

    return image_array


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def sharpen_image(
    image_array: np.ndarray,
    radius: float = 1.5,
    percent: int = 120,
    threshold: int = 3,
) -> np.ndarray:
    """
    Apply an UnsharpMask to the merged RGB array.

    Bicubic and Lanczos are smoothing operations — they eliminate pixelation
    but also soften edges.  UnsharpMask is the standard post-process to
    recover perceived sharpness without introducing compression artefacts.

    How UnsharpMask works:
        1. Blur the image by *radius* pixels
        2. Subtract the blur from the original — this extracts the edges
        3. Add *percent*% of the extracted edges back to the original
        4. Only apply where the difference exceeds *threshold* (avoids
           sharpening smooth areas / noise)

    Args:
        image_array: (H, W, 3) uint8 array — fully merged RGB.
        radius:      Blur radius in pixels. Larger = broader sharpening halo.
                     Default 1.5 suits most upscaled photos.
        percent:     Sharpening strength 0–200+. 100 = moderate, 150 = strong.
                     Default 120.
        threshold:   Minimum pixel difference to sharpen. Default 3.

    Returns:
        Sharpened (H, W, 3) uint8 array — same shape and dtype as input.
    """
    pil_image = Image.fromarray(image_array, mode="RGB")
    sharpened = pil_image.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )
    return np.array(sharpened, dtype=np.uint8)


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def generate_filename(
    input_path: str | Path,
    output_dir: str | Path,
    method: str,
    scale_factor: int,
) -> str:
    """
    Build a descriptive, collision-safe output filename.

    Pattern:
        <stem>_<method>_<scale>x.png           (no conflict)
        <stem>_<method>_<scale>x_<HHMMSS>.png  (file already exists)

    Always outputs as PNG — the pipeline must not produce lossy output.

    Args:
        input_path:   Original image path; only the stem is used.
        output_dir:   Destination directory, checked for existing files.
        method:       "bicubic" or "lanczos".
        scale_factor: Integer upscale factor.

    Returns:
        Filename string (basename only, not a full path).
    """
    stem = Path(input_path).stem
    base_name = f"{stem}_{method}_{scale_factor}x.png"

    output_path = Path(output_dir) / base_name
    if output_path.exists():
        timestamp = datetime.now().strftime("%H%M%S")
        base_name = f"{stem}_{method}_{scale_factor}x_{timestamp}.png"

    return base_name


def ensure_output_directory(output_dir: str | Path) -> Path:
    """
    Create the output directory (and any parents) if it does not exist.

    Args:
        output_dir: Target directory path (relative or absolute).

    Returns:
        Resolved absolute Path — eliminates ambiguity for confirmation messages.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)   # safe to call even if dir exists
    return path.resolve()


def save_image(
    image_array: np.ndarray,
    output_dir: str | Path,
    filename: str,
) -> str:
    """
    Convert the merged uint8 array to a PIL Image and write it to disk as PNG.

    Always specifies ``mode='RGB'`` explicitly rather than letting PIL guess
    from the array shape — guessing occasionally fails for unusual array shapes.

    Args:
        image_array: (H*scale, W*scale, 3) uint8 array.
        output_dir:  Destination directory — created if missing.
        filename:    Basename of the output file (e.g. "photo_bicubic_4x.png").

    Returns:
        Absolute path string of the written file.

    Raises:
        RuntimeError: If the file is absent after saving (disk full, permission
                      issue, etc.).
    """
    resolved_dir = ensure_output_directory(output_dir)
    output_path = resolved_dir / filename

    image = Image.fromarray(image_array, mode="RGB")  # explicit mode — never guess
    image.save(str(output_path), format="PNG")

    # Post-save verification — catches disk-full / permission failures that PIL
    # might surface as generic IO errors without a clear message.
    if not output_path.exists():
        raise RuntimeError(
            f"Image was not found at '{output_path}' after saving. "
            "Check available disk space and write permissions."
        )

    return str(output_path)


# ---------------------------------------------------------------------------
# Debug channel export (--save-channels flag)
# ---------------------------------------------------------------------------

def save_channel_debug(
    channel_uint8: np.ndarray,
    output_dir: str | Path,
    filename_base: str,
    channel_name: str,
) -> str:
    """
    Save a single channel as a grayscale PNG for visual inspection.

    Useful when diagnosing color casts — compare the three channel files to
    identify which channel has the wrong values.

    Args:
        channel_uint8: 2D uint8 array, shape (H*scale, W*scale).
        output_dir:    Destination directory.
        filename_base: Base name without extension (e.g. "photo_bicubic_4x").
        channel_name:  "R", "G", or "B".

    Returns:
        Absolute path string of the written grayscale PNG.
    """
    resolved_dir = ensure_output_directory(output_dir)
    filename = f"{filename_base}_{channel_name}.png"
    output_path = resolved_dir / filename

    image = Image.fromarray(channel_uint8, mode="L")   # mode='L' = 8-bit grayscale
    image.save(str(output_path), format="PNG")

    return str(output_path)

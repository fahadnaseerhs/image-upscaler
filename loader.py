"""
loader.py — Image Decoding Pipeline Entry Point

Responsibility: Take a raw image file from disk and hand off clean,
normalized, channel-split data to the rest of the pipeline.

Nothing downstream ever touches the raw file — they only see what
prepare_image() returns.

Dependencies:
    - Pillow (PIL) : reading image files, format normalization
    - NumPy        : all array operations after loading
"""

from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Public orchestrator — the only function main.py needs to call
# ---------------------------------------------------------------------------

def prepare_image(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Full loader pipeline in one call.

    Steps (in order):
        1. load_image()      → raw (H, W, 3) uint8 array
        2. validate_image()  → raises on bad data, silent on good
        3. get_image_info()  → extracts height / width / channels
        4. split_channels()  → r, g, b each (H, W) uint8
        5. normalize()       → r, g, b each (H, W) float64, 0.0–1.0

    Args:
        path: Path to the image file (str or pathlib.Path).

    Returns:
        Tuple of (r_norm, g_norm, b_norm, image_info) where:
            r_norm, g_norm, b_norm — (H, W) float64 arrays, values 0.0–1.0
            image_info             — dict with 'height', 'width', 'channels'
    """
    image_array = load_image(path)
    validate_image(image_array)
    image_info = get_image_info(image_array)
    r, g, b = split_channels(image_array)
    r_norm = normalize(r)
    g_norm = normalize(g)
    b_norm = normalize(b)
    return r_norm, g_norm, b_norm, image_info


# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """
    Open an image file from disk and return it as a raw NumPy array.

    Handles all common formats (JPEG, PNG, BMP, TIFF, …) transparently
    by forcing a conversion to RGB, which guarantees exactly 3 channels
    regardless of the source format (RGBA, grayscale, palette-mode, etc.).

    Args:
        path: Path to the image file.

    Returns:
        image_array — shape (H, W, 3), dtype uint8, values 0–255.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    path = Path(path)  # normalise: accept both str and Path

    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: '{path}'. "
            "Please check the path and try again."
        )

    with Image.open(path) as image:
        # Force exactly 3 channels — handles RGBA, L (greyscale), P (palette), etc.
        image = image.convert("RGB")
        image_array: np.ndarray = np.array(image)  # shape (H, W, 3), uint8

    return image_array


def validate_image(image_array: np.ndarray) -> bool:
    """
    Sanity-check the loaded array before anything else in the pipeline sees it.

    Keeping validation out of load_image lets you reuse this function with
    alternative loaders (e.g. load from URL, load from .npy file) without
    duplicating the checks.

    Args:
        image_array: The raw array produced by load_image().

    Returns:
        True when all checks pass.

    Raises:
        ValueError: On shape, dtype, size, or value-range violations.
    """
    # --- Shape check ---
    if image_array.ndim != 3:
        raise ValueError(
            f"Expected a 3-D array (H, W, 3), got {image_array.ndim}-D array "
            f"with shape {image_array.shape}."
        )

    if image_array.shape[2] != 3:
        raise ValueError(
            f"Expected exactly 3 channels (RGB), got {image_array.shape[2]} "
            f"channels. Shape: {image_array.shape}."
        )

    # --- dtype check ---
    if image_array.dtype != np.uint8:
        raise ValueError(
            f"Expected dtype uint8 (0–255 integers), got '{image_array.dtype}'. "
            "Normalization math will produce incorrect results on other dtypes."
        )

    # --- Size check ---
    h, w = image_array.shape[:2]
    if h <= 1 or w <= 1:
        raise ValueError(
            f"Image is too small for interpolation. "
            f"Minimum size is 2×2 pixels; got {h}×{w}."
        )

    # --- Value range check ---
    pixel_min = int(image_array.min())
    pixel_max = int(image_array.max())
    if pixel_min < 0 or pixel_max > 255:
        raise ValueError(
            f"Pixel values out of expected range [0, 255]. "
            f"Got min={pixel_min}, max={pixel_max}."
        )

    return True


def split_channels(image_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate a (H, W, 3) array into three independent 2-D arrays.

    NumPy slices are *views* that share memory with the original array.
    Calling .copy() on each slice ensures that downstream modifications
    to r/g/b never silently corrupt image_array — a subtle bug that
    would be very hard to trace without this guard.

    Args:
        image_array: Validated (H, W, 3) uint8 array.

    Returns:
        Tuple (r, g, b) — each shape (H, W), dtype uint8.
    """
    r: np.ndarray = image_array[:, :, 0].copy()
    g: np.ndarray = image_array[:, :, 1].copy()
    b: np.ndarray = image_array[:, :, 2].copy()
    return r, g, b


def normalize(channel_array: np.ndarray) -> np.ndarray:
    """
    Convert a uint8 channel (0–255) into float64 (0.0–1.0).

    This "analog representation" step moves pixel values from discrete
    integer space into the continuous float space where interpolation
    kernels compute weighted sums of neighbours.

    Explicit cast to float64 before division is mandatory — dividing a
    uint8 array without casting can produce integer-division artefacts
    (values < 255 divided by 255 in integer space → 0).

    Args:
        channel_array: 2-D uint8 array, shape (H, W).

    Returns:
        normalized — shape (H, W), dtype float64, values clipped to [0.0, 1.0].
    """
    normalized: np.ndarray = channel_array.astype(np.float64) / 255.0

    # Defensive clip — should be a no-op for clean uint8 input, but protects
    # against any unexpected floating-point edge cases.
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized


def get_image_info(image_array: np.ndarray) -> dict:
    """
    Extract and return the key dimensional metadata of the image.

    Returns a named dictionary instead of raw .shape so callers never
    need to remember whether index 0 is height or width.

    Args:
        image_array: The (H, W, 3) array.

    Returns:
        dict with keys:
            'height'   — number of rows    (int)
            'width'    — number of columns (int)
            'channels' — always 3 for RGB  (int)
    """
    h, w, c = image_array.shape
    return {
        "height": h,
        "width": w,
        "channels": c,
    }

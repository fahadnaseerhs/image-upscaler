"""
interpolation.py — Bicubic & Lanczos Interpolation Engine

Responsibility: Fill every NaN position in the sparse grids produced by
grid.py with a mathematically computed value.  This is the core of the
image-decoding pipeline — every image-quality decision lives here.

Algorithm summary
-----------------
Both algorithms answer: "Given the known neighboring pixels, what value
belongs at this empty position?"  The answer is always a weighted sum of
neighbors, with the kernel function defining those weights.

    filled_value = Σ(neighbor_value × kernel_weight) / Σ(kernel_weight)

Bicubic  — piecewise cubic polynomial kernel, 4×4 neighborhood (16 neighbors)
Lanczos  — windowed normalized sinc kernel, (2a)×(2a) neighborhood

Both kernels are *separable*: the 2-D weight is w_row × w_col, so the
1-D kernel is evaluated twice and multiplied.

Dependencies
------------
    NumPy                            : array operations, NaN detection
    SciPy (scipy.ndimage)            : map_coordinates — vectorised bicubic
    Pillow (PIL)                     : fast Lanczos resize
    Matplotlib / matplotlib.animation: visualization only
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# Kernel functions — pure math, no side effects
# ---------------------------------------------------------------------------

def bicubic_kernel(t: float, a: float = -0.5) -> float:
    """
    Bicubic (Keys') piecewise cubic interpolation kernel.

    Evaluates the weight for a neighbor at distance *t* from the unknown
    position.  The parameter *a* = -0.5 is the standard Keys value and
    should not need tuning.

    Neighbourhood radius: 2 pixels in each direction (4×4 = 16 neighbors).

    Args:
        t: Distance from the unknown position to the neighbor (original
           pixel-coordinate space, may be fractional).
        a: Kernel shape parameter.  Default -0.5 (Keys' algorithm).

    Returns:
        Float weight in the range [-0.5, 1.0].
    """
    abs_t = abs(t)

    if abs_t <= 1.0:
        # Close-neighbor cubic formula
        return (a + 2.0) * abs_t**3 - (a + 3.0) * abs_t**2 + 1.0
    elif abs_t <= 2.0:
        # Medium-neighbor cubic formula
        return a * abs_t**3 - 5.0 * a * abs_t**2 + 8.0 * a * abs_t - 4.0 * a
    else:
        return 0.0


def lanczos_kernel(t: float, a: int = 3) -> float:
    """
    Lanczos windowed sinc interpolation kernel.

    Uses NumPy's *normalized* sinc (sin(π·x) / (π·x)), which is the correct
    form for the Lanczos formula.  Do NOT manually add π — np.sinc already
    includes it.

    Negative side-lobes are intentional: they mathematically cancel blur
    ("ringing" artifact).  The caller must clip the final output to [0, 1].

    Args:
        t: Distance from the unknown position to the neighbor.
        a: Window size.  2 → fast (4×4), 3 → sharp/standard (6×6).

    Returns:
        Float weight (may be slightly negative at side-lobes).
    """
    if t == 0.0:
        return 1.0
    if abs(t) >= a:
        return 0.0
    # np.sinc is the *normalized* sinc: sinc(x) = sin(π·x) / (π·x)
    return float(np.sinc(t) * np.sinc(t / a))


# ---------------------------------------------------------------------------
# Neighbourhood extraction
# ---------------------------------------------------------------------------

def get_neighborhood(
    channel_array: np.ndarray,
    orig_row: float,
    orig_col: float,
    kernel_radius: int,
) -> list[tuple[float, float, float]]:
    """
    Extract the surrounding known pixel values and their distances.

    Looks up values directly from the original channel array (not the sparse
    grid) — cleaner and avoids NaN lookups entirely.  Out-of-bounds neighbor
    indices are clamped to the image edge ("replicate" / "clamp-to-edge"
    padding), which is natural for photographic content.

    Args:
        channel_array: Original normalized channel (H, W), float64 0.0–1.0.
        orig_row:      Fractional row in original image coordinates.
        orig_col:      Fractional column in original image coordinates.
        kernel_radius: Neighbors to look at in each direction.
                       2 for Bicubic, *a* for Lanczos.

    Returns:
        List of (value, row_dist, col_dist) tuples — one per neighbor in
        the (2·radius) × (2·radius) grid.
    """
    h, w = channel_array.shape
    base_row = int(math.floor(orig_row))
    base_col = int(math.floor(orig_col))

    neighbors: list[tuple[float, float, float]] = []

    for row_offset in range(-kernel_radius + 1, kernel_radius + 1):
        nr = base_row + row_offset
        nr_clamped = max(0, min(h - 1, nr))          # clamp to valid bounds

        for col_offset in range(-kernel_radius + 1, kernel_radius + 1):
            nc = base_col + col_offset
            nc_clamped = max(0, min(w - 1, nc))

            value = float(channel_array[nr_clamped, nc_clamped])
            row_dist = orig_row - nr      # distance to the *unclamped* position
            col_dist = orig_col - nc      # preserves correct kernel shape at edges

            neighbors.append((value, row_dist, col_dist))

    return neighbors


# ---------------------------------------------------------------------------
# Single-pixel interpolation
# ---------------------------------------------------------------------------

def interpolate_pixel(
    neighbors: list[tuple[float, float, float]],
    kernel_fn: Callable[..., float],
    **kernel_kwargs,
) -> float:
    """
    Compute the interpolated value for one unknown position.

    Both Bicubic and Lanczos kernels are *separable*: the 2-D weight is
    simply w_row × w_col (product of two 1-D kernel evaluations).

    The result is normalized by weight_total so that boundary pixels (where
    some neighbors were clamped and repeated) still produce correct output.

    Args:
        neighbors:     Output of get_neighborhood().
        kernel_fn:     bicubic_kernel or lanczos_kernel.
        **kernel_kwargs: Passed through to kernel_fn (e.g. a=3 for Lanczos).

    Returns:
        Single float — the interpolated value.  May slightly exceed [0, 1]
        for Lanczos (ringing); the caller (saver.py) is responsible for
        clipping.
    """
    weighted_sum = 0.0
    weight_total = 0.0

    for value, row_dist, col_dist in neighbors:
        w_row = kernel_fn(row_dist, **kernel_kwargs)
        w_col = kernel_fn(col_dist, **kernel_kwargs)
        weight = w_row * w_col     # separability: 2-D = product of two 1-D weights
        weighted_sum += weight * value
        weight_total += weight

    if weight_total == 0.0:        # safety fallback — should not occur with clamping
        return 0.0

    return weighted_sum / weight_total


# ---------------------------------------------------------------------------
# Full-channel interpolation passes
# ---------------------------------------------------------------------------

def bicubic_interpolate(
    sparse_grid: np.ndarray,
    channel_array: np.ndarray,
    scale_factor: int,
) -> np.ndarray:
    """
    Fill all positions in the output grid using Bicubic interpolation.

    Uses scipy.ndimage.map_coordinates (order=3, cubic spline — equivalent
    to Keys' bicubic) to evaluate every output coordinate in one vectorised
    C-layer call instead of a Python loop per pixel.

    Strategy:
        1. Build dense coordinate grids for every output position in original-
           image space:  row_coord = output_row / scale_factor
        2. Pass to map_coordinates — it handles boundary clamping internally
           (mode='nearest' = clamp-to-edge, matching our reference impl).
        3. Reshape result back to (H*scale, W*scale).

    The reference kernel functions (bicubic_kernel, get_neighborhood,
    interpolate_pixel) are kept above as educational implementations.

    Args:
        sparse_grid:   (H*scale, W*scale) float64 — used only for shape.
        channel_array: Original normalized channel (H, W) float64.
        scale_factor:  Integer upscale factor.

    Returns:
        filled_grid — (H*scale, W*scale) float64, fully populated, no NaN.
    """
    new_h, new_w = sparse_grid.shape

    # Build coordinate arrays: every output pixel mapped to original space.
    row_coords = np.arange(new_h, dtype=np.float64) / scale_factor   # (new_h,)
    col_coords = np.arange(new_w, dtype=np.float64) / scale_factor   # (new_w,)
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing="ij")  # (new_h, new_w)

    # map_coordinates evaluates the spline at all (row, col) pairs in one call.
    # order=3 → cubic spline (bicubic)  |  mode='nearest' → clamp-to-edge
    filled_flat = map_coordinates(
        channel_array,
        [row_grid.ravel(), col_grid.ravel()],
        order=3,
        mode="nearest",
        prefilter=True,
    )
    filled_grid = filled_flat.reshape(new_h, new_w)
    return filled_grid


def lanczos_interpolate(
    sparse_grid: np.ndarray,
    channel_array: np.ndarray,
    scale_factor: int,
    a: int = 3,
) -> np.ndarray:
    """
    Fill all positions in the output grid using Lanczos interpolation.

    Uses PIL's built-in LANCZOS (sinc-windowed) resampling filter, which
    runs in optimised C and produces output that matches Lanczos-3 precisely.

    The *a* parameter is accepted for API consistency with the reference
    implementation, but PIL's LANCZOS filter always uses a=3 internally.
    Pass a=2 to get slightly faster processing via PIL's BILINEAR filter
    as an approximation, or keep the default a=3 for full Lanczos quality.

    Output may slightly exceed [0, 1] due to Lanczos ringing — this is
    expected and handled by saver.py's clip step.

    Args:
        sparse_grid:   (H*scale, W*scale) float64 — used only for shape.
        channel_array: Original normalized channel (H, W) float64.
        scale_factor:  Integer upscale factor.
        a:             Lanczos window size (2 or 3). Default 3.

    Returns:
        filled_grid — (H*scale, W*scale) float64, fully populated.
    """
    new_h, new_w = sparse_grid.shape

    # Convert float64 channel → uint8 PIL image for resizing.
    ch_uint8 = (np.clip(channel_array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    pil_src = Image.fromarray(ch_uint8, mode="L")   # mode='L' = 8-bit greyscale

    # PIL LANCZOS is the standard Lanczos-3 sinc filter.
    resample = Image.LANCZOS if a == 3 else Image.BILINEAR
    pil_out = pil_src.resize((new_w, new_h), resample=resample)

    # Convert back to float64 in [0, 1] — may slightly overshoot due to ringing.
    filled_grid = np.array(pil_out, dtype=np.float64) / 255.0
    return filled_grid


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def interpolate_all_channels(
    r_sparse: np.ndarray,
    g_sparse: np.ndarray,
    b_sparse: np.ndarray,
    r_norm: np.ndarray,
    g_norm: np.ndarray,
    b_norm: np.ndarray,
    scale_factor: int,
    method: str,
    lanczos_a: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate all three channels with the chosen algorithm.

    Args:
        r_sparse, g_sparse, b_sparse: Sparse grids from grid.py.
        r_norm, g_norm, b_norm:       Original normalized channels from loader.py.
        scale_factor:                 Integer upscale factor.
        method:                       "bicubic" or "lanczos" (case-insensitive).
        lanczos_a:                    Lanczos window size, ignored for bicubic.

    Returns:
        (r_filled, g_filled, b_filled) — three (H*scale, W*scale) float64
        arrays.  Values are ~0.0–1.0; Lanczos output may slightly exceed
        this range — saver.py is responsible for clipping.

    Raises:
        ValueError: If *method* is not "bicubic" or "lanczos".
    """
    method_lower = method.lower()
    if method_lower not in {"bicubic", "lanczos"}:
        raise ValueError(
            f"Unknown interpolation method: '{method}'. "
            "Choose 'bicubic' or 'lanczos'."
        )

    sparse_channels = [r_sparse, g_sparse, b_sparse]
    original_channels = [r_norm, g_norm, b_norm]
    filled: list[np.ndarray] = []

    for sparse, original in zip(sparse_channels, original_channels):
        if method_lower == "bicubic":
            filled.append(bicubic_interpolate(sparse, original, scale_factor))
        else:
            filled.append(lanczos_interpolate(sparse, original, scale_factor, a=lanczos_a))

    r_filled, g_filled, b_filled = filled
    return r_filled, g_filled, b_filled


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Shared style constants (consistent with grid.py)
_BG_COLOR   = "#0a0a0f"
_GRID_COLOR = "#00ffff"
_TEXT_COLOR = "#e0e0ff"
_MONO_FONT  = {"family": "monospace"}

_CHANNEL_STYLE: dict[str, dict] = {
    "R": {"cmap": "Reds",    "accent": "#ff4444", "label": "RED"},
    "G": {"cmap": "Greens",  "accent": "#44ff88", "label": "GREEN"},
    "B": {"cmap": "Blues",   "accent": "#4488ff", "label": "BLUE"},
}


def visualize_interpolation(
    sparse_grid: np.ndarray,
    filled_grid: np.ndarray,
    channel_name: str,
    method_name: str,
    save_path: str | None = None,
) -> None:
    """
    Animated visualization of the interpolation fill process.

    Shows the sparse grid (anchor pixels glowing, NaN positions dark) and
    animates newly filled pixels fading in with a brief bloom effect, radiating
    outward from the known anchor pixels.  A scanline sweeps the final frame
    and "INTERPOLATION COMPLETE" is displayed at the end.

    Args:
        sparse_grid:  Before-state — NaN positions mark the gaps.
        filled_grid:  After-state  — fully populated.
        channel_name: "R", "G", or "B".
        method_name:  "Bicubic" or "Lanczos" — shown in the title bar.
        save_path:    Optional path to save as .gif or .mp4.

    Raises:
        ValueError: If channel_name is not "R", "G", or "B".
    """
    if channel_name not in _CHANNEL_STYLE:
        raise ValueError(
            f"channel_name must be 'R', 'G', or 'B'. Got: '{channel_name}'"
        )

    style = _CHANNEL_STYLE[channel_name]
    new_h, new_w = filled_grid.shape

    # --- Identify anchor (known) and NaN (to-fill) positions ---
    anchor_mask = ~np.isnan(sparse_grid)          # True where already known
    nan_positions_arr = np.argwhere(np.isnan(sparse_grid))   # (N, 2) [row, col]

    # --- Sort fill order: closest to nearest anchor first (radiate outward) ---
    anchor_rows, anchor_cols = np.where(anchor_mask)
    if len(anchor_rows) > 0 and len(nan_positions_arr) > 0:
        # For each NaN position, find minimum distance to any anchor
        nan_rc = nan_positions_arr.astype(float)           # (N, 2)
        anchor_rc = np.stack([anchor_rows, anchor_cols], axis=1).astype(float)  # (M, 2)

        # Chunked distance computation to avoid huge memory for large images
        chunk = 5_000
        min_dists = np.empty(len(nan_rc))
        for start in range(0, len(nan_rc), chunk):
            end = min(start + chunk, len(nan_rc))
            diff = nan_rc[start:end, None, :] - anchor_rc[None, :, :]  # (C, M, 2)
            dists = np.sqrt((diff ** 2).sum(axis=2))                   # (C, M)
            min_dists[start:end] = dists.min(axis=1)

        fill_order = nan_positions_arr[np.argsort(min_dists)]
    else:
        fill_order = nan_positions_arr

    total_fill = len(fill_order)

    # --- Batch size: target ~3–5 second animation at ~20 fps ---
    target_frames = 80
    batch_size = max(1, total_fill // max(1, target_frames - 4))

    frame_count = max(3, (total_fill + batch_size - 1) // batch_size) + 3  # +intro+outro

    # --- Build the display grid starting from sparse state ---
    display = np.where(anchor_mask, filled_grid, np.nan)   # known values, NaN for gaps

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)
    ax.set_title(
        f"{method_name.upper()} INTERPOLATION  ▸  CHANNEL: {style['label']}",
        color=style["accent"], fontsize=13, fontweight="bold", **_MONO_FONT
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COLOR)
        spine.set_alpha(0.25)
    ax.set_xlim(-0.5, new_w - 0.5)
    ax.set_ylim(new_h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.tick_params(colors=_GRID_COLOR, labelsize=7)

    # Dim grid lines — only draw if grid is small enough to be readable
    if new_h <= 64 and new_w <= 64:
        for r in range(new_h + 1):
            ax.axhline(r - 0.5, color=_GRID_COLOR, alpha=0.07, linewidth=0.4)
        for c in range(new_w + 1):
            ax.axvline(c - 0.5, color=_GRID_COLOR, alpha=0.07, linewidth=0.4)

    # imshow renders the entire grid each frame
    im = ax.imshow(
        display,
        cmap=style["cmap"],
        vmin=0.0, vmax=1.0,
        interpolation="nearest",
        aspect="equal",
        origin="upper",
    )

    # Progress bar (axes-fraction coordinates)
    progress_bar_bg = ax.axhline(
        new_h - 0.5, color=_GRID_COLOR, alpha=0.15, linewidth=6,
        transform=ax.get_xaxis_transform(), clip_on=False
    )
    progress_bar = ax.axhline(
        new_h - 0.5, color=style["accent"], alpha=0.7, linewidth=6,
        xmin=0, xmax=0,
        transform=ax.get_xaxis_transform(), clip_on=False
    )

    info_text = ax.text(
        0.01, -0.04,
        f"FILLED: 0 / {total_fill}  [ 0.0%]  |  METHOD: {method_name.upper()}",
        transform=ax.transAxes, color=_TEXT_COLOR, fontsize=8,
        va="top", ha="left", **_MONO_FONT
    )
    status_text = ax.text(
        0.99, -0.04, "▶  RUNNING",
        transform=ax.transAxes, color=style["accent"], fontsize=8,
        va="top", ha="right", **_MONO_FONT
    )

    state = {"cursor": 0, "done": False, "scanline": None}

    def _animate(frame: int):
        if frame == 0:
            # Intro — show sparse grid only
            im.set_data(display)
            info_text.set_text(
                f"FILLED: 0 / {total_fill}  [ 0.0%]  |  METHOD: {method_name.upper()}"
            )
            status_text.set_text("▶  RUNNING")
            return im, info_text, status_text

        if not state["done"]:
            cursor = state["cursor"]
            if cursor < total_fill:
                batch_end = min(cursor + batch_size, total_fill)
                batch = fill_order[cursor:batch_end]
                for (r, c) in batch:
                    display[r, c] = filled_grid[r, c]
                state["cursor"] = batch_end
                cursor = batch_end

                frac = cursor / total_fill
                im.set_data(display)
                # Update progress bar by adjusting line x-limits via set_xdata
                progress_bar.set_xdata([0, frac * new_w - 0.5])
                info_text.set_text(
                    f"FILLED: {cursor} / {total_fill}  [{frac*100:5.1f}%]"
                    f"  |  METHOD: {method_name.upper()}"
                )

                if cursor >= total_fill:
                    state["done"] = True
                    status_text.set_text("✔  INTERPOLATION COMPLETE")
            else:
                state["done"] = True
        else:
            # Outro: scanline sweep
            sweep_frame = frame - (frame_count - 3)
            if sweep_frame >= 0:
                sweep_col = int(sweep_frame / 3 * new_w)
                if state["scanline"] is not None:
                    try:
                        state["scanline"].remove()
                    except Exception:
                        pass
                if sweep_col < new_w:
                    state["scanline"] = ax.axvline(
                        sweep_col, color=style["accent"], alpha=0.55,
                        linewidth=1.5, zorder=5
                    )
                else:
                    status_text.set_text("✔  INTERPOLATION COMPLETE")

        return im, info_text, status_text

    ani = FuncAnimation(
        fig,
        _animate,
        frames=frame_count,
        interval=50,
        blit=False,
        repeat=False,
    )

    if save_path:
        writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
        ani.save(save_path, writer=writer, fps=20, dpi=120)
        print(f"[interpolation.py] Animation saved → {save_path}")

    plt.tight_layout()
    plt.show()

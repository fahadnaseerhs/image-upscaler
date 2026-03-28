"""
grid.py — Sparse Grid Creation & Pixel Mapping

Responsibility: Take normalized channel arrays from loader.py, allocate a
larger empty grid at the target resolution, and place each known pixel into
its correct mapped position. The result is a *sparse* grid — real values at
regular (scale_factor-spaced) intervals, NaN everywhere else.

That sparse grid is the input interpolation.py will fill.

Dependencies:
    - NumPy                  : all grid creation and mapping
    - Matplotlib             : visualization panels
    - matplotlib.animation   : FuncAnimation for the animated sequence
    - matplotlib.colors      : color mapping per channel
    - matplotlib.patches     : FancyArrowPatch for vector arrows
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Public orchestrator — the only function main.py needs to call
# ---------------------------------------------------------------------------

def prepare_all_channels(
    r_norm: np.ndarray,
    g_norm: np.ndarray,
    b_norm: np.ndarray,
    scale_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Map all three normalized channels into sparse grids.

    Args:
        r_norm, g_norm, b_norm: float64 arrays (H, W), values 0.0–1.0.
        scale_factor:           Integer upscale factor (2, 4, 8 …).

    Returns:
        Tuple of (r_sparse, g_sparse, b_sparse, grid_info) where each
        sparse array has shape (H*scale, W*scale) and grid_info is a
        metadata dict from get_grid_info().
    """
    sparse_grids = []
    for channel in (r_norm, g_norm, b_norm):
        sparse_grids.append(map_pixels(channel, scale_factor))

    r_sparse, g_sparse, b_sparse = sparse_grids
    grid_info = get_grid_info(r_sparse, scale_factor)

    return r_sparse, g_sparse, b_sparse, grid_info


# ---------------------------------------------------------------------------
# Core grid functions
# ---------------------------------------------------------------------------

def create_empty_grid(height: int, width: int, scale_factor: int) -> np.ndarray:
    """
    Allocate a new larger grid filled entirely with NaN.

    NaN is used — never zero — because 0.0 is a valid pixel value (black).
    Using zero as a placeholder would silently corrupt interpolation math.

    Args:
        height:       Original image row count.
        width:        Original image column count.
        scale_factor: Integer upscale factor.

    Returns:
        empty_grid — shape (height*scale_factor, width*scale_factor),
                     dtype float64, all np.nan.
    """
    new_height = height * scale_factor
    new_width = width * scale_factor
    empty_grid: np.ndarray = np.full(
        (new_height, new_width), np.nan, dtype=np.float64
    )
    return empty_grid


def map_pixels(channel_array: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    Place each pixel from a normalized channel into the correct position
    in a larger sparse grid.

    The mapping rule:
        old position (row, col) → new position (row*scale, col*scale)

    This is achieved in one vectorized step with NumPy slice stepping:
        sparse_grid[::scale_factor, ::scale_factor] = channel_array

    No Python loop is needed — the slice assignment runs in NumPy's C layer
    and is orders of magnitude faster for large images.

    Args:
        channel_array: float64 array (H, W), values 0.0–1.0.
        scale_factor:  Integer upscale factor.

    Returns:
        sparse_grid — shape (H*scale, W*scale), float64.
                      Known values at every scale_factor-th position,
                      np.nan everywhere else.
    """
    h, w = channel_array.shape
    sparse_grid = create_empty_grid(h, w, scale_factor)

    # Vectorized placement — no Python loop required.
    # grid[::s, ::s] selects every s-th row and every s-th column,
    # starting at index 0 (top-left corner of each block, intentionally).
    sparse_grid[::scale_factor, ::scale_factor] = channel_array

    return sparse_grid


def get_grid_info(sparse_grid: np.ndarray, scale_factor: int) -> dict:
    """
    Return metadata about a sparse grid.

    The fill ratio should always equal 1 / scale_factor² for a correctly
    mapped grid — use it as a sanity-check assertion if needed.

    Args:
        sparse_grid:  Output of map_pixels().
        scale_factor: Integer upscale factor (stored for reference).

    Returns:
        dict with keys:
            'new_height'    — total rows
            'new_width'     — total columns
            'total_pixels'  — new_height * new_width
            'filled_pixels' — count of non-NaN positions
            'empty_pixels'  — count of NaN positions
            'fill_ratio'    — filled / total  (≈ 1/scale²)
            'scale_factor'  — the scale factor used
    """
    new_height, new_width = sparse_grid.shape
    total_pixels = new_height * new_width
    filled_pixels = int(np.count_nonzero(~np.isnan(sparse_grid)))
    empty_pixels = total_pixels - filled_pixels
    fill_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0.0

    return {
        "new_height": new_height,
        "new_width": new_width,
        "total_pixels": total_pixels,
        "filled_pixels": filled_pixels,
        "empty_pixels": empty_pixels,
        "fill_ratio": fill_ratio,
        "scale_factor": scale_factor,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Color configuration per channel — (scatter color, arrow color, colormap name)
_CHANNEL_STYLE: dict[str, dict] = {
    "R": {"scatter": "#ff4444", "arrow": "#ff8800", "cmap": "Reds",     "label": "RED"},
    "G": {"scatter": "#44ff88", "arrow": "#00ffcc", "cmap": "Greens",   "label": "GREEN"},
    "B": {"scatter": "#4488ff", "arrow": "#8844ff", "cmap": "Purples",  "label": "BLUE"},
}

_BG_COLOR      = "#0a0a0f"
_GRID_COLOR    = "#00ffff"
_EMPTY_COLOR   = "#1a1a2e"
_TEXT_COLOR    = "#e0e0ff"
_MONO_FONT     = {"family": "monospace"}


def visualize_grid_mapping(
    channel_array: np.ndarray,
    sparse_grid: np.ndarray,
    scale_factor: int,
    channel_name: str,
    save_path: str | None = None,
) -> None:
    """
    Render an animated visualization of the pixel mapping for one channel.

    Three panels:
        Left  — original small grid (zoomed tile of up to 12×12 pixels)
        Centre — info / stats panel
        Right  — sparse large grid with mapped pixels lighting up frame by frame

    Vectors arc from each pixel's old position to its new mapped position.
    Animation completes in roughly 3–5 seconds regardless of image size
    by dynamically scaling the batch size.

    Args:
        channel_array: Original normalized channel (H, W), float64 0.0–1.0.
        sparse_grid:   Mapped sparse grid (H*scale, W*scale), float64.
        scale_factor:  Integer upscale factor.
        channel_name:  Exactly "R", "G", or "B" (case-sensitive).
        save_path:     Optional path to save the animation as .gif or .mp4.
                       If None, the animation is only displayed.

    Raises:
        ValueError: If channel_name is not one of "R", "G", "B".
    """
    if channel_name not in _CHANNEL_STYLE:
        raise ValueError(
            f"channel_name must be 'R', 'G', or 'B'. Got: '{channel_name}'"
        )

    style = _CHANNEL_STYLE[channel_name]
    h, w = channel_array.shape
    grid_info = get_grid_info(sparse_grid, scale_factor)
 
    # --- Tile the small grid to a viewable size (max 12×12 pixels shown) ---
    tile_rows = min(h, 12)
    tile_cols = min(w, 12)
    tile = channel_array[:tile_rows, :tile_cols]

    # --- Determine animation batch size ---
    total_pixels = h * w
    if total_pixels < 100:
        batch_size = 1
    elif total_pixels < 10_000:
        batch_size = max(1, total_pixels // 60)
    else:
        batch_size = max(50, total_pixels // 120)

    # Pixel coordinates in original grid (row, col pairs) — only the tile
    rows_orig, cols_orig = np.meshgrid(
        np.arange(tile_rows), np.arange(tile_cols), indexing="ij"
    )
    orig_coords = list(zip(rows_orig.ravel(), cols_orig.ravel()))
    # Mapped coordinates in the large grid
    mapped_coords = [(r * scale_factor, c * scale_factor) for r, c in orig_coords]

    num_pixels = len(orig_coords)
    frame_count = max(2, (num_pixels + batch_size - 1) // batch_size) + 2  # +2 for intro/outro

    # -----------------------------------------------------------------------
    # Figure & axes setup
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 7), facecolor=_BG_COLOR)
    fig.suptitle(
        f"GRID MAPPING  ▸  CHANNEL: {style['label']}",
        color=style["scatter"], fontsize=14, fontweight="bold", **_MONO_FONT, y=0.97
    )

    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.55, 2], wspace=0.35,
                          left=0.06, right=0.97, top=0.88, bottom=0.1)
    ax_small = fig.add_subplot(gs[0])
    ax_info  = fig.add_subplot(gs[1])
    ax_large = fig.add_subplot(gs[2])

    for ax in (ax_small, ax_info, ax_large):
        ax.set_facecolor(_BG_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID_COLOR)
            spine.set_alpha(0.3)

    # -----------------------------------------------------------------------
    # Left panel — original tile
    # -----------------------------------------------------------------------
    ax_small.set_title("ORIGINAL GRID", color=_TEXT_COLOR, fontsize=9, **_MONO_FONT, pad=6)
    ax_small.set_xlim(-0.5, tile_cols - 0.5)
    ax_small.set_ylim(tile_rows - 0.5, -0.5)
    ax_small.set_aspect("equal")
    ax_small.tick_params(colors=_GRID_COLOR, labelsize=7)

    # Draw dim grid lines
    for r in range(tile_rows + 1):
        ax_small.axhline(r - 0.5, color=_GRID_COLOR, alpha=0.15, linewidth=0.5)
    for c in range(tile_cols + 1):
        ax_small.axvline(c - 0.5, color=_GRID_COLOR, alpha=0.15, linewidth=0.5)

    # Filled squares for each original pixel
    norm_map = Normalize(vmin=0.0, vmax=1.0)
    cmap_obj  = plt.get_cmap(style["cmap"])
    for r in range(tile_rows):
        for c in range(tile_cols):
            val = float(tile[r, c])
            color = cmap_obj(norm_map(val))
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                  color=color, alpha=0.85, zorder=2)
            ax_small.add_patch(rect)

    # -----------------------------------------------------------------------
    # Right panel — large sparse grid
    # -----------------------------------------------------------------------
    new_h = tile_rows * scale_factor
    new_w = tile_cols * scale_factor

    ax_large.set_title("SPARSE GRID  (NaN = awaiting interpolation)",
                        color=_TEXT_COLOR, fontsize=9, **_MONO_FONT, pad=6)
    ax_large.set_xlim(-0.5, new_w - 0.5)
    ax_large.set_ylim(new_h - 0.5, -0.5)
    ax_large.set_aspect("equal")
    ax_large.tick_params(colors=_GRID_COLOR, labelsize=7)

    # Dim NaN positions
    nan_r, nan_c = np.indices((new_h, new_w))
    ax_large.scatter(nan_c.ravel(), nan_r.ravel(),
                     color=_EMPTY_COLOR, s=12, zorder=1, alpha=0.5)

    # Dim grid lines
    for r in range(new_h + 1):
        ax_large.axhline(r - 0.5, color=_GRID_COLOR, alpha=0.08, linewidth=0.4)
    for c in range(new_w + 1):
        ax_large.axvline(c - 0.5, color=_GRID_COLOR, alpha=0.08, linewidth=0.4)

    # Scatter for mapped (filled) pixels — starts empty, grows each frame
    scatter_filled = ax_large.scatter(
        [], [], color=style["scatter"], s=40,
        zorder=3, alpha=0.95,
        edgecolors=style["arrow"], linewidths=0.8
    )

    # -----------------------------------------------------------------------
    # Centre info panel
    # -----------------------------------------------------------------------
    ax_info.axis("off")
    info_text = ax_info.text(
        0.5, 0.55, "", transform=ax_info.transAxes,
        color=_TEXT_COLOR, fontsize=8, va="center", ha="center",
        **_MONO_FONT,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#0d0d1a",
                  edgecolor=_GRID_COLOR, alpha=0.6)
    )
    status_text = ax_info.text(
        0.5, 0.12, "INITIALISING...", transform=ax_info.transAxes,
        color=style["arrow"], fontsize=8, va="center", ha="center",
        **_MONO_FONT
    )

    # -----------------------------------------------------------------------
    # Shared mutable state for FuncAnimation closure
    # -----------------------------------------------------------------------
    state = {
        "mapped_cols": [],
        "mapped_rows": [],
        "arrows": [],          # FancyArrowPatch objects to remove each frame
        "pixel_cursor": 0,     # index into orig_coords
    }

    def _update_info(mapped_count: int, status: str) -> None:
        info_text.set_text(
            f"CHANNEL   : {channel_name}\n"
            f"SCALE     : ×{scale_factor}\n"
            f"ORIG SIZE : {h}×{w}\n"
            f"NEW SIZE  : {h*scale_factor}×{w*scale_factor}\n"
            f"MAPPED    : {mapped_count:>6} / {num_pixels}\n"
            f"FILL RATIO: {mapped_count/num_pixels*100:5.1f}%"
        )
        status_text.set_text(status)

    def _animate(frame: int):
        # Remove previous frame's arrows
        for arrow in state["arrows"]:
            try:
                arrow.remove()
            except Exception:
                pass
        state["arrows"].clear()

        if frame == 0:
            # Intro frame
            _update_info(0, "▶  MAPPING IN PROGRESS")
            return scatter_filled, info_text, status_text

        pixel_cursor = state["pixel_cursor"]

        if pixel_cursor < num_pixels:
            # Map the next batch
            batch_end = min(pixel_cursor + batch_size, num_pixels)
            batch = orig_coords[pixel_cursor:batch_end]
            mbatch = mapped_coords[pixel_cursor:batch_end]

            for (orig_r, orig_c), (map_r, map_c) in zip(batch, mbatch):
                state["mapped_rows"].append(map_r)
                state["mapped_cols"].append(map_c)

                # Draw arc arrow from small-grid position to large-grid position
                # We use annotation arrows directly on ax_large (mapped positions only)
                arrow = mpatches.FancyArrowPatch(
                    posA=(orig_c / tile_cols * new_w * 0.12 + map_c * 0.88,
                          orig_r / tile_rows * new_h * 0.12 + map_r * 0.88),
                    posB=(map_c, map_r),
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.25",
                    color=style["arrow"],
                    linewidth=0.6,
                    alpha=0.5,
                    mutation_scale=8,
                    zorder=4,
                )
                ax_large.add_patch(arrow)
                state["arrows"].append(arrow)

            state["pixel_cursor"] = batch_end
            mapped_count = batch_end
            _update_info(mapped_count,
                         "▶  MAPPING IN PROGRESS" if batch_end < num_pixels
                         else "✔  SPARSE GRID READY — AWAITING INTERPOLATION")
        else:
            # Outro / done frame
            _update_info(num_pixels, "✔  SPARSE GRID READY — AWAITING INTERPOLATION")

        scatter_filled.set_offsets(
            np.column_stack([state["mapped_cols"], state["mapped_rows"]])
            if state["mapped_cols"] else np.empty((0, 2))
        )
        return scatter_filled, info_text, status_text

    ani = FuncAnimation(
        fig,
        _animate,
        frames=frame_count,
        interval=50,        # ms per frame → ~20 fps
        blit=False,
        repeat=False,
    )

    if save_path:
        writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
        ani.save(save_path, writer=writer, fps=20, dpi=120)
        print(f"[grid.py] Animation saved → {save_path}")

    plt.tight_layout()
    plt.show()

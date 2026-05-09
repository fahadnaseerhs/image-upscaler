"""
spatial_domain.py — Visualizes the spatial domain changes.
Shows the before and after images, and a difference overlay highlighting the enhanced pixels.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .style import apply_dark_theme, MONO_FONT, TEXT_COLOR

def plot(r_orig: np.ndarray, g_orig: np.ndarray, b_orig: np.ndarray,
         r_filled: np.ndarray, g_filled: np.ndarray, b_filled: np.ndarray,
         scale_factor: int, output_dir: str | Path) -> str:
    """
    Plots the spatial domain with nearest baseline, ESRGAN output, absolute
    difference heatmap, and a center zoom comparison.
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 6))
    fig.patch.set_facecolor("#0a0a0f")
    fig.suptitle("SPATIAL DOMAIN: Nearest Baseline vs ESRGAN + Difference + Zoom", 
                 color=TEXT_COLOR, fontsize=14, fontweight="bold", **MONO_FONT)

    # Stack channels to RGB
    orig_rgb = np.stack([r_orig, g_orig, b_orig], axis=2)
    filled_rgb = np.stack([r_filled, g_filled, b_filled], axis=2)
    
    # Fast nearest-neighbor upscale baseline.
    orig_upscaled = np.kron(
        orig_rgb,
        np.ones((scale_factor, scale_factor, 1), dtype=orig_rgb.dtype),
    )

    # Absolute per-pixel difference (0-255 style scale for readability).
    diff = np.abs(filled_rgb - orig_upscaled).mean(axis=2) * 255.0

    for ax in (ax1, ax2, ax3, ax4):
        apply_dark_theme(fig, ax, "")
        for spine in ax.spines.values():
            spine.set_edgecolor("#00ffff")

    out_h, out_w, _ = filled_rgb.shape

    ax1.set_title("Original (Nearest Upscaled)", color=TEXT_COLOR, fontsize=10, **MONO_FONT)
    ax2.set_title("ESRGAN Output", color=TEXT_COLOR, fontsize=10, **MONO_FONT)
    ax3.set_title("Absolute Difference Heatmap", color="#00ffff", fontsize=10, fontweight="bold", **MONO_FONT)
    ax4.set_title("Center Crop: Baseline | ESRGAN | Diff", color=TEXT_COLOR, fontsize=10, **MONO_FONT)

    ax1.set_xlabel(f"{out_w} px", color=TEXT_COLOR, fontsize=9, **MONO_FONT)
    ax1.set_ylabel(f"{out_h} px", color=TEXT_COLOR, fontsize=9, **MONO_FONT)
    ax2.set_xlabel(f"{out_w} px", color=TEXT_COLOR, fontsize=9, **MONO_FONT)
    ax2.set_ylabel(f"{out_h} px", color=TEXT_COLOR, fontsize=9, **MONO_FONT)
    ax3.set_xlabel(f"{out_w} px", color=TEXT_COLOR, fontsize=9, **MONO_FONT)
    ax3.set_ylabel(f"{out_h} px", color=TEXT_COLOR, fontsize=9, **MONO_FONT)
    ax4.set_xlabel("Center 128x128 comparison strip", color=TEXT_COLOR, fontsize=9, **MONO_FONT)
    ax4.set_ylabel("128 px", color=TEXT_COLOR, fontsize=9, **MONO_FONT)

    ax1.imshow(orig_upscaled)
    ax2.imshow(filled_rgb)
    im = ax3.imshow(diff, cmap="inferno", vmin=0, vmax=255)
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.02)
    cbar.set_label("Pixel difference (0-255)", color=TEXT_COLOR, **MONO_FONT)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_COLOR, **MONO_FONT)

    # Center crop panel (most important for 4x vs 8x texture comparison).
    crop = 128
    cy, cx = out_h // 2, out_w // 2
    y0 = max(cy - crop // 2, 0)
    x0 = max(cx - crop // 2, 0)
    y1 = min(y0 + crop, out_h)
    x1 = min(x0 + crop, out_w)

    crop_base = orig_upscaled[y0:y1, x0:x1]
    crop_out = filled_rgb[y0:y1, x0:x1]
    crop_diff = diff[y0:y1, x0:x1] / 255.0
    crop_diff_rgb = np.stack([crop_diff, crop_diff, crop_diff], axis=2)
    zoom_strip = np.concatenate([crop_base, crop_out, crop_diff_rgb], axis=1)
    ax4.imshow(zoom_strip)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = Path(output_dir) / "dsp_spatial_domain.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()
    
    return str(output_path)

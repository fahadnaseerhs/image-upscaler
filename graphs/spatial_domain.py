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
    Plots the spatial domain before/after and an overlay of enhanced pixels.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0a0a0f")
    fig.suptitle("SPATIAL DOMAIN: Enhanced Pixels Highlighted", 
                 color=TEXT_COLOR, fontsize=14, fontweight="bold", **MONO_FONT)

    # Stack channels to RGB
    orig_rgb = np.stack([r_orig, g_orig, b_orig], axis=2)
    filled_rgb = np.stack([r_filled, g_filled, b_filled], axis=2)
    
    # To compare sizes, we need to upsample the original image (nearest neighbor)
    # just for visual alignment.
    h, w, _ = orig_rgb.shape
    orig_upscaled = np.zeros_like(filled_rgb)
    for r in range(h):
        for c in range(w):
            orig_upscaled[r*scale_factor:(r+1)*scale_factor, c*scale_factor:(c+1)*scale_factor] = orig_rgb[r, c]

    # Calculate difference (highlight)
    diff = np.abs(filled_rgb - orig_upscaled).mean(axis=2) # mean across RGB
    
    # Create glow overlay (cyan where difference is high)
    glow_rgb = np.zeros_like(filled_rgb)
    glow_rgb[:, :, 0] = diff * 0.0 # Red
    glow_rgb[:, :, 1] = diff * 1.0 # Green
    glow_rgb[:, :, 2] = diff * 1.0 # Blue (Cyan overall)
    
    overlay = np.clip(filled_rgb * 0.4 + glow_rgb * 0.6, 0.0, 1.0)

    for ax in (ax1, ax2, ax3):
        apply_dark_theme(fig, ax, "")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#00ffff")

    ax1.set_title("BEFORE (Nearest Neighbor)", color=TEXT_COLOR, fontsize=10, **MONO_FONT)
    ax2.set_title("AFTER (Interpolated)", color=TEXT_COLOR, fontsize=10, **MONO_FONT)
    ax3.set_title("ENHANCED PIXELS (Cyan Glow)", color="#00ffff", fontsize=10, fontweight="bold", **MONO_FONT)

    ax1.imshow(orig_upscaled)
    ax2.imshow(filled_rgb)
    ax3.imshow(overlay)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = Path(output_dir) / "dsp_spatial_domain.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()
    
    return str(output_path)

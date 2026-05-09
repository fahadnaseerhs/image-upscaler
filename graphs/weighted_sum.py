"""
weighted_sum.py — Visualizes the 2D separable kernel weights.
Shows a heatmap of how neighboring pixels contribute to one interpolated pixel.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from interpolation import bicubic_kernel, lanczos_kernel
from .style import apply_dark_theme, MONO_FONT

def plot(method: str, output_dir: str | Path, lanczos_a: int = 3) -> str:
    """
    Plots the 2D weights for a sample sub-pixel offset (e.g. 0.5, 0.5)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sub-pixel offset for interpolation
    dx, dy = 0.5, 0.5 
    
    if method == "lanczos":
        apply_dark_theme(fig, ax, f"2D WEIGHT MATRIX (Lanczos, a={lanczos_a})")
        radius = lanczos_a
        kernel_fn = lambda t: lanczos_kernel(t, a=lanczos_a)
        cmap = sns.diverging_palette(10, 240, as_cmap=True) # Red-Blue for neg/pos
    else:
        apply_dark_theme(fig, ax, "2D WEIGHT MATRIX (Bicubic)")
        radius = 2
        kernel_fn = bicubic_kernel
        cmap = "mako" # Nice dark theme heatmap

    size = radius * 2
    weights = np.zeros((size, size))
    
    for r_idx, row_offset in enumerate(range(-radius + 1, radius + 1)):
        for c_idx, col_offset in enumerate(range(-radius + 1, radius + 1)):
            # Distance from the unknown position (which is at dx, dy from the top-left neighbor)
            row_dist = dx - row_offset
            col_dist = dy - col_offset
            
            w_row = kernel_fn(row_dist)
            w_col = kernel_fn(col_dist)
            weights[r_idx, c_idx] = w_row * w_col
            
    sns.heatmap(weights, annot=True, fmt=".3f", cmap=cmap, center=0, 
                cbar_kws={'label': 'Weight Contribution'},
                ax=ax, annot_kws={"size": 10, **MONO_FONT})
                
    ax.set_xlabel("Column Offset")
    ax.set_ylabel("Row Offset")
    
    # Set tick labels to the offsets
    ax.set_xticklabels(range(-radius + 1, radius + 1))
    ax.set_yticklabels(range(-radius + 1, radius + 1))
    
    # Make cbar text readable
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color="#00ffff")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#e0e0ff")
    cbar.set_label('Weight Contribution', color="#e0e0ff", weight='bold')

    plt.tight_layout()
    output_path = Path(output_dir) / f"dsp_weights_{method}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()
    
    return str(output_path)

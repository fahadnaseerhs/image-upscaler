"""
kernel_plot.py — Visualizes the 1D interpolation kernels.
Plots the continuous function for Bicubic or Lanczos, highlighting sidelobes if Lanczos.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from interpolation import bicubic_kernel, lanczos_kernel
from .style import apply_dark_theme, ACCENT_LANCZOS, ACCENT_BICUBIC, TEXT_COLOR

def plot(method: str, output_dir: str | Path, lanczos_a: int = 3) -> str:
    """
    Plots the 1D kernel shape.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.linspace(-4, 4, 1000)
    
    if method == "lanczos":
        apply_dark_theme(fig, ax, f"1D KERNEL SHAPE (Lanczos, a={lanczos_a})")
        y = np.array([lanczos_kernel(t, a=lanczos_a) for t in x])
        
        # Plot the main curve
        ax.plot(x, y, color=ACCENT_LANCZOS, linewidth=2, label=f"Lanczos-{lanczos_a}")
        
        # Highlight negative sidelobes
        ax.fill_between(x, y, 0, where=(y < 0), color="red", alpha=0.3, label="Negative Sidelobes (Anti-aliasing)")
        ax.fill_between(x, y, 0, where=(y >= 0), color=ACCENT_LANCZOS, alpha=0.1)
        
    else:
        apply_dark_theme(fig, ax, "1D KERNEL SHAPE (Bicubic, a=-0.5)")
        y = np.array([bicubic_kernel(t) for t in x])
        
        ax.plot(x, y, color=ACCENT_BICUBIC, linewidth=2, label="Bicubic (Keys')")
        ax.fill_between(x, y, 0, color=ACCENT_BICUBIC, alpha=0.1)
        
    ax.axhline(0, color=TEXT_COLOR, linewidth=1, alpha=0.5)
    ax.axvline(0, color=TEXT_COLOR, linewidth=1, alpha=0.5)
    
    # Mark pixel grid
    for i in range(-4, 5):
        ax.axvline(i, color="#00ffff", linestyle="--", alpha=0.2)
        
    ax.set_xlabel("Distance from center (pixels)")
    ax.set_ylabel("Weight")
    ax.legend(facecolor="#0a0a0f", edgecolor="#00ffff", labelcolor="#e0e0ff")
    
    plt.tight_layout()
    output_path = Path(output_dir) / f"dsp_kernel_{method}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()
    
    return str(output_path)

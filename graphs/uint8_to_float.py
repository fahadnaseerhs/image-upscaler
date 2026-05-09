"""
uint8_to_float.py — Visualizes the normalization process.
Shows a bar chart comparing raw 0-255 integers to their 0.0-1.0 float equivalents.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .style import apply_dark_theme, ACCENT_BLUE, ACCENT_LANCZOS

def plot(channel_uint8: np.ndarray, channel_float: np.ndarray, output_dir: str | Path) -> str:
    """
    Plots a sample of pixel values before and after normalization.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    apply_dark_theme(fig, ax1, "DATA TYPE CONVERSION (Normalization)")
    
    # Take a 1D slice of 20 pixels from the center
    h, w = channel_uint8.shape
    sample_uint8 = channel_uint8[h//2, w//2 - 10 : w//2 + 10]
    sample_float = channel_float[h//2, w//2 - 10 : w//2 + 10]
    
    x = np.arange(len(sample_uint8))
    width = 0.35
    
    # Primary axis for uint8
    bars1 = ax1.bar(x - width/2, sample_uint8, width, label='Raw (uint8, 0-255)', color=ACCENT_BLUE, alpha=0.8)
    ax1.set_ylabel('Integer Value', color=ACCENT_BLUE, fontweight="bold")
    ax1.tick_params(axis='y', labelcolor=ACCENT_BLUE)
    ax1.set_ylim(0, 260)
    
    # Secondary axis for float64
    ax2 = ax1.twinx()
    
    # Need to apply some dark theme properties to the secondary axis
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_edgecolor("#00ffff")
    ax2.spines['right'].set_alpha(0.25)
    
    bars2 = ax2.bar(x + width/2, sample_float, width, label='Normalized (float64, 0.0-1.0)', color=ACCENT_LANCZOS, alpha=0.8)
    ax2.set_ylabel('Float Value', color=ACCENT_LANCZOS, fontweight="bold")
    ax2.tick_params(axis='y', labelcolor=ACCENT_LANCZOS, colors="#00ffff")
    ax2.set_ylim(0, 1.05)
    
    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', facecolor="#0a0a0f", edgecolor="#00ffff", labelcolor="#e0e0ff")
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "dsp_uint8_to_float.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2.0) # Show for 2 seconds then continue
    plt.close()
    return str(output_path)

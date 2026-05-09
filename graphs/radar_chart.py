"""
radar_chart.py — Comparison radar chart for Bicubic vs Lanczos.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .style import BG_COLOR, TEXT_COLOR, GRID_COLOR, ACCENT_LANCZOS, ACCENT_BICUBIC, MONO_FONT

def plot(method: str, output_dir: str | Path) -> str:
    """
    Plots a radar chart comparing Bicubic and Lanczos.
    """
    # Define metrics and dummy scores (0 to 10 scale)
    categories = ['Sharpness', 'Anti-aliasing', 'Computational\nSpeed', 'Artifact\nFree (No Ringing)', 'Edge\nPreservation']
    N = len(categories)
    
    # Values based on DSP theory
    values_lanczos = [9, 9, 4, 3, 8]
    values_bicubic = [6, 5, 8, 8, 5]
    
    # Close the loop
    values_lanczos += values_lanczos[:1]
    values_bicubic += values_bicubic[:1]
    
    # Angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color=TEXT_COLOR, size=10, **MONO_FONT)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color=GRID_COLOR, size=8)
    plt.ylim(0, 10)
    
    # Configure grid lines
    ax.grid(color=GRID_COLOR, alpha=0.2, linestyle='--')
    ax.spines['polar'].set_color(GRID_COLOR)
    ax.spines['polar'].set_alpha(0.3)
    
    # Plot Lanczos
    ax.plot(angles, values_lanczos, linewidth=2, linestyle='solid', color=ACCENT_LANCZOS, label='Lanczos')
    if method == "lanczos":
        ax.fill(angles, values_lanczos, color=ACCENT_LANCZOS, alpha=0.25)
    
    # Plot Bicubic
    ax.plot(angles, values_bicubic, linewidth=2, linestyle='solid', color=ACCENT_BICUBIC, label='Bicubic')
    if method == "bicubic":
        ax.fill(angles, values_bicubic, color=ACCENT_BICUBIC, alpha=0.25)
    
    plt.title(f"ALGORITHM PERFORMANCE RADAR\n(Highlighting {method.capitalize()})", 
              size=14, color=TEXT_COLOR, y=1.1, fontweight='bold', **MONO_FONT)
              
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor="#0a0a0f", edgecolor="#00ffff", labelcolor="#e0e0ff")
    
    plt.tight_layout()
    output_path = Path(output_dir) / "dsp_radar_comparison.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()
    
    return str(output_path)

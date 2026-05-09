"""
Common styling for DSP analysis graphs.
Maintains the dark, high-tech aesthetic from the main visualization scripts.
"""

import matplotlib.pyplot as plt

BG_COLOR = "#0a0a0f"
GRID_COLOR = "#00ffff"
TEXT_COLOR = "#e0e0ff"
ACCENT_LANCZOS = "#ff4444"  # Red
ACCENT_BICUBIC = "#44ff88"  # Green
ACCENT_BLUE = "#4488ff"     # Blue
MONO_FONT = {"family": "monospace"}

def apply_dark_theme(fig, ax, title):
    """Applies the common dark theme to a figure and axis."""
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight="bold", **MONO_FONT, pad=15)
    
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
        spine.set_alpha(0.25)
        
    ax.tick_params(colors=GRID_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    
    ax.grid(color=GRID_COLOR, alpha=0.1, linestyle="--", linewidth=0.5)

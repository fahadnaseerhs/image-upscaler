"""
frequency_domain.py — Visualizes the frequency spectrum before and after upscaling.
Uses Fast Fourier Transform (FFT) to show how high-frequency content changes.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .style import apply_dark_theme, MONO_FONT, TEXT_COLOR

def plot(r_orig: np.ndarray, g_orig: np.ndarray, b_orig: np.ndarray,
         r_filled: np.ndarray, g_filled: np.ndarray, b_filled: np.ndarray,
         output_dir: str | Path) -> str:
    """
    Plots the 2D FFT magnitude spectrum before and after.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("#0a0a0f")
    fig.suptitle("FREQUENCY DOMAIN ANALYSIS (FFT Magnitude Spectrum)", 
                 color=TEXT_COLOR, fontsize=14, fontweight="bold", **MONO_FONT)

    # Convert to grayscale for simpler FFT
    gray_orig = 0.2989 * r_orig + 0.5870 * g_orig + 0.1140 * b_orig
    gray_filled = 0.2989 * r_filled + 0.5870 * g_filled + 0.1140 * b_filled
    
    # Apply 2D FFT and shift zero frequency to center
    f_orig = np.fft.fftshift(np.fft.fft2(gray_orig))
    f_filled = np.fft.fftshift(np.fft.fft2(gray_filled))
    
    # Calculate magnitude spectrum (log scale for visibility)
    mag_orig = np.abs(f_orig)
    mag_filled = np.abs(f_filled)
    
    # Avoid log(0)
    mag_orig[mag_orig == 0] = 1e-10
    mag_filled[mag_filled == 0] = 1e-10

    # Apply dark theme to axes
    for ax in (ax1, ax2):
        apply_dark_theme(fig, ax, "")
        ax.set_xticks([])
        ax.set_yticks([])

    ax1.set_title("BEFORE (Low Resolution)", color=TEXT_COLOR, fontsize=10, **MONO_FONT)
    ax2.set_title("AFTER (Interpolated)", color=TEXT_COLOR, fontsize=10, **MONO_FONT)

    # Use 'inferno' colormap which fits dark themes well
    im1 = ax1.imshow(mag_orig, norm=LogNorm(vmin=np.percentile(mag_orig, 5), vmax=np.percentile(mag_orig, 99.9)), cmap="inferno")
    im2 = ax2.imshow(mag_filled, norm=LogNorm(vmin=np.percentile(mag_filled, 5), vmax=np.percentile(mag_filled, 99.9)), cmap="inferno")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = Path(output_dir) / "dsp_frequency_domain.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()
    
    return str(output_path)

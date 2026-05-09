"""
antialiasing.py — Visualizes the frequency response of the kernels.
Shows why Lanczos is better at anti-aliasing than Bicubic by plotting their
Fourier transforms (frequency response), approaching the ideal brick-wall filter.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

from interpolation import bicubic_kernel, lanczos_kernel
from .style import apply_dark_theme, ACCENT_LANCZOS, ACCENT_BICUBIC, TEXT_COLOR

def plot(output_dir: str | Path, lanczos_a: int = 3) -> str:
    """
    Plots the frequency response of Lanczos vs Bicubic vs Ideal filter.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    apply_dark_theme(fig, ax, "FREQUENCY RESPONSE (Anti-aliasing properties)")
    
    # Generate spatial kernels
    x = np.linspace(-4, 4, 1024)
    dx = x[1] - x[0]
    
    y_lanczos = np.array([lanczos_kernel(t, a=lanczos_a) for t in x])
    y_bicubic = np.array([bicubic_kernel(t) for t in x])
    
    # Compute Frequency Response
    freqs = fftfreq(len(x), d=dx)
    
    # We want to plot the magnitude spectrum
    H_lanczos = np.abs(fft(y_lanczos))
    H_bicubic = np.abs(fft(y_bicubic))
    
    # Normalize
    H_lanczos /= np.max(H_lanczos)
    H_bicubic /= np.max(H_bicubic)
    
    # Sort for plotting
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    H_lanczos = H_lanczos[idx]
    H_bicubic = H_bicubic[idx]
    
    # Plot only positive frequencies up to Nyquist-ish
    mask = (freqs >= 0) & (freqs <= 2.0)
    freqs = freqs[mask]
    H_lanczos = H_lanczos[mask]
    H_bicubic = H_bicubic[mask]
    
    # Plot Ideal Low-pass (Brick Wall)
    ideal_freqs = np.linspace(0, 2.0, 500)
    ideal_H = np.where(ideal_freqs <= 0.5, 1.0, 0.0)
    ax.plot(ideal_freqs, ideal_H, color=TEXT_COLOR, linestyle=":", linewidth=2, label="Ideal Low-pass (Brick Wall)")
    
    # Plot real responses
    ax.plot(freqs, H_lanczos, color=ACCENT_LANCZOS, linewidth=2, label=f"Lanczos-{lanczos_a} (Sharper cutoff, less aliasing)")
    ax.plot(freqs, H_bicubic, color=ACCENT_BICUBIC, linewidth=2, label="Bicubic (Softer cutoff, more aliasing)")
    
    # Fill the 'aliasing' leak area for bicubic (just illustrative)
    ax.fill_between(freqs, 0, H_bicubic, where=(freqs > 0.5), color=ACCENT_BICUBIC, alpha=0.1)
    ax.fill_between(freqs, 0, H_lanczos, where=(freqs > 0.5), color=ACCENT_LANCZOS, alpha=0.2)
    
    ax.set_xlabel("Normalized Spatial Frequency (Cycles/Pixel)")
    ax.set_ylabel("Magnitude Response")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.05, 1.1)
    ax.axvline(0.5, color="#00ffff", linestyle="--", alpha=0.3, label="Nyquist Frequency")
    
    ax.legend(facecolor="#0a0a0f", edgecolor="#00ffff", labelcolor="#e0e0ff")
    
    plt.tight_layout()
    output_path = Path(output_dir) / "dsp_antialiasing.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()
    
    return str(output_path)

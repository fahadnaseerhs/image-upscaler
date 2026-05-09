"""
graphs/realesrgan_viz.py — Real-ESRGAN Deep Feature Visualization Suite

Generates 5 dark-themed plot files:
  01_filter_responses_64.png      — 64 first-layer feature maps (8×8 grid)
  02_block_progression_23.png     — 23 RRDB mean activations + energy bar
  03_frequency_before_after.png   — FFT magnitude before / after / diff
  04_new_frequencies_generated.png— Radial profile + generated content
  05_radar_summary.png            — 8-band radar chart

Usage:
    from graphs.realesrgan_viz import run_visualization_suite
    run_visualization_suite(model, input_np, output_np, out_dir)
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Dark theme
# ---------------------------------------------------------------------------
BG    = "#07080f"
BG2   = "#0d0f1a"
BG3   = "#111526"
TEXT  = "#e0e4ff"
TEXT2 = "#8890b8"
ACCENT  = "#6c8bff"
ACCENT2 = "#a259ff"
CYAN    = "#00e5ff"
GREEN   = "#39ff8a"
RED     = "#ff4d6a"
GOLD    = "#ffd166"

_RCPARAMS = {
    "figure.facecolor": BG,
    "axes.facecolor":   BG2,
    "axes.edgecolor":   BG3,
    "text.color":       TEXT,
    "axes.labelcolor":  TEXT2,
    "xtick.color":      TEXT2,
    "ytick.color":      TEXT2,
    "grid.color":       BG3,
    "font.family":      "monospace",
    "savefig.facecolor": BG,
}


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG2)
    for sp in ax.spines.values():
        sp.set_edgecolor(BG3)
    if title:   ax.set_title(title,  color=TEXT,  fontsize=9,  pad=6)
    if xlabel:  ax.set_xlabel(xlabel, color=TEXT2, fontsize=8)
    if ylabel:  ax.set_ylabel(ylabel, color=TEXT2, fontsize=8)
    ax.tick_params(colors=TEXT2, labelsize=7)
    ax.grid(True, alpha=0.12, color=BG3)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_visualization_suite(
    model,              # RealESRGANer instance (.model = RRDBNet)
    input_np: np.ndarray,   # H×W×3 uint8 RGB
    output_np: np.ndarray,  # H×W×3 uint8 RGB (enhanced)
    out_dir: Path,
    progress_cb=None,   # optional callback(step, total, label)
) -> None:
    """Run all 5 visualization steps and save results to out_dir."""
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _log(step, total, label):
        if progress_cb:
            progress_cb(step, total, label)
        print(f"  [ESRGAN-VIZ {step}/{total}] {label}")

    # ---- collect forward hooks ----
    rrdbnet = model.model
    device  = next(rrdbnet.parameters()).device

    conv_first_feat: dict = {}
    block_feats: dict     = {}
    hooks = []

    def _hook_first(m, inp, out):
        conv_first_feat["data"] = out.detach().cpu()

    hooks.append(rrdbnet.conv_first.register_forward_hook(_hook_first))

    for idx, rrdb in enumerate(rrdbnet.body):
        def _make(i):
            def _h(m, inp, out):
                block_feats[i] = out.detach().cpu()
            return _h
        hooks.append(rrdb.register_forward_hook(_make(idx)))

    # prepare tensor
    img_f   = input_np.astype(np.float32) / 255.0
    img_bgr = img_f[:, :, ::-1]
    img_t   = (
        torch.from_numpy(np.ascontiguousarray(img_bgr.transpose(2, 0, 1)))
        .float().unsqueeze(0).to(device)
    )

    _log(1, 6, "Running forward pass with hooks …")
    rrdbnet.eval()
    with torch.no_grad():
        rrdbnet(img_t)

    for h in hooks:
        h.remove()

    # ---- run each plot ----
    _log(2, 6, "Plotting 64 filter responses …")
    with matplotlib.rc_context(_RCPARAMS):
        _plot_filters(conv_first_feat["data"], out_dir / "01_filter_responses_64.png")

    _log(3, 6, "Plotting 23 RRDB block progression …")
    with matplotlib.rc_context(_RCPARAMS):
        _plot_blocks(block_feats, out_dir / "02_block_progression_23.png")

    _log(4, 6, "Plotting frequency domain analysis …")
    with matplotlib.rc_context(_RCPARAMS):
        _plot_freq_analysis(input_np, output_np, out_dir / "03_frequency_before_after.png")

    _log(5, 6, "Plotting new frequency generation …")
    with matplotlib.rc_context(_RCPARAMS):
        _plot_new_freqs(input_np, output_np, out_dir / "04_new_frequencies_generated.png")

    _log(6, 6, "Plotting radar summary …")
    with matplotlib.rc_context(_RCPARAMS):
        _plot_radar(input_np, output_np, out_dir / "05_radar_summary.png")

    print(f"\n  ✓ All visualizations saved → {out_dir}\n")


# ---------------------------------------------------------------------------
# 1.  64 Filter Responses
# ---------------------------------------------------------------------------

def _plot_filters(feat, save_path: Path) -> None:
    """feat: Tensor (1, 64, H, W)"""
    fmaps = feat.squeeze(0).numpy()          # (64, H, W)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "Real-ESRGAN — 64 First-Layer Filter Responses\n"
        "Each cell shows what spatial patterns one filter detects in the input image",
        color=TEXT, fontsize=11, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(8, 8, figure=fig, hspace=0.25, wspace=0.15,
                           left=0.01, right=0.99, top=0.91, bottom=0.02)
    for i in range(64):
        ax  = fig.add_subplot(gs[i // 8, i % 8])
        fm  = fmaps[i]
        lo, hi = fm.min(), fm.max()
        if hi > lo:
            fm = (fm - lo) / (hi - lo)
        ax.imshow(fm, cmap="plasma", interpolation="nearest", aspect="auto")
        ax.set_title(f"F{i+1:02d}", color=TEXT2, fontsize=6, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(BG3)

    fig.text(0.5, 0.005,
             "Plasma colourmap · dark=low activation · bright=high activation",
             ha="center", color=TEXT2, fontsize=8, style="italic")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2.  23 RRDB Block Progression
# ---------------------------------------------------------------------------

def _plot_blocks(block_feats: dict, save_path: Path) -> None:
    n = len(block_feats)   # should be 23

    fig = plt.figure(figsize=(26, 10))
    fig.suptitle(
        "Real-ESRGAN — 23 RRDB Block Feature Progression\n"
        "Top: mean activation map per block  |  Bottom: activation energy trend",
        color=TEXT, fontsize=11, fontweight="bold", y=0.98,
    )

    # Top row: mean activation heatmaps
    gs_top = gridspec.GridSpec(1, n, figure=fig,
                               left=0.01, right=0.99, top=0.88, bottom=0.40,
                               wspace=0.04)
    energies = []
    for i in range(n):
        feat    = block_feats[i].squeeze(0).numpy()   # (64, H, W)
        mean_fm = feat.mean(axis=0)
        energies.append(float(np.mean(np.abs(feat))))
        lo, hi  = mean_fm.min(), mean_fm.max()
        if hi > lo:
            mean_fm = (mean_fm - lo) / (hi - lo)

        ax = fig.add_subplot(gs_top[0, i])
        ax.imshow(mean_fm, cmap="inferno", interpolation="nearest", aspect="auto")
        ax.set_title(f"B{i+1:02d}", color=(CYAN if i % 3 == 2 else TEXT2),
                     fontsize=6, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        border = ACCENT if i == n - 1 else BG3
        for sp in ax.spines.values():
            sp.set_edgecolor(border)

    # Bottom: energy bar chart
    gs_bot = gridspec.GridSpec(1, 1, figure=fig,
                               left=0.05, right=0.99, top=0.34, bottom=0.07)
    ax_b = fig.add_subplot(gs_bot[0, 0])
    _style(ax_b,
           title="Mean |Activation| Energy Across 23 RRDB Blocks — higher = richer representation",
           xlabel="Block", ylabel="Mean |Activation|")

    xs = np.arange(1, n + 1)
    pk = int(np.argmax(energies))
    cols = [ACCENT2 if j == pk else ACCENT for j in range(n)]
    ax_b.bar(xs, energies, color=cols, alpha=0.8, width=0.7)

    # smoothed trend
    from scipy.ndimage import uniform_filter1d
    trend = uniform_filter1d(energies, size=3)
    ax_b.plot(xs, trend, color=CYAN, linewidth=1.8, alpha=0.9, label="Smoothed trend")
    ax_b.fill_between(xs, trend, alpha=0.07, color=CYAN)
    ax_b.annotate(
        f"Peak B{pk+1:02d}",
        xy=(xs[pk], energies[pk]),
        xytext=(xs[pk] + 1.5, energies[pk] * 1.03),
        color=GOLD, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=GOLD),
    )
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([str(x) for x in xs], fontsize=7)
    ax_b.legend(facecolor=BG3, labelcolor=TEXT2, fontsize=8)

    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3.  Frequency Domain — Before / After / Difference
# ---------------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    return (0.2126 * img[:, :, 0] +
            0.7152 * img[:, :, 1] +
            0.0722 * img[:, :, 2]).astype(np.float32) / 255.0


def _fft_mag(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img)
    fsh  = np.fft.fftshift(np.fft.fft2(gray))
    return np.log1p(np.abs(fsh))


def _plot_freq_analysis(in_np: np.ndarray, out_np: np.ndarray, save_path: Path) -> None:
    # resize output to same dims as input for fair comparison
    out_r = np.array(
        PILImage.fromarray(out_np).resize(
            (in_np.shape[1], in_np.shape[0]), PILImage.LANCZOS
        )
    )
    mag_in  = _fft_mag(in_np)
    mag_out = _fft_mag(out_r)
    mag_new = np.clip(mag_out - mag_in, 0, None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Real-ESRGAN — Frequency Domain Analysis (2D FFT Magnitude)\n"
        "Centre = DC (low freq)   Edges = high freq   Brighter = more energy",
        color=TEXT, fontsize=10, fontweight="bold",
    )

    data_list = [
        (mag_in,  "Before Enhancement\n(Input FFT)",            "magma"),
        (mag_out, "After Enhancement\n(Output FFT)",             "magma"),
        (mag_new, "New Frequencies Created\n(Output − Input ≥ 0)", "hot"),
    ]
    for ax, (data, title, cmap) in zip(axes, data_list):
        im = ax.imshow(data, cmap=cmap, interpolation="bilinear", aspect="equal")
        _style(ax, title=title)
        ax.set_xticks([]); ax.set_yticks([])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        cb.ax.tick_params(colors=TEXT2, labelsize=7)
        h, w = data.shape
        ax.text(w * 0.85, h * 0.10, "High\nfreq", color=CYAN,
                fontsize=7, ha="center", alpha=0.75)
        ax.text(w * 0.50, h * 0.50, "DC\n(low)", color=GREEN,
                fontsize=7, ha="center", alpha=0.75)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4.  New Frequency Generation — Radial Profile
# ---------------------------------------------------------------------------

def _radial_profile(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img)
    fsh  = np.fft.fftshift(np.fft.fft2(gray))
    mag  = np.abs(fsh)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X   = np.ogrid[:h, :w]
    R      = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r  = min(cx, cy)
    return np.array([
        mag[R == r].mean() if (R == r).any() else 0.0
        for r in range(max_r)
    ])


def _plot_new_freqs(in_np: np.ndarray, out_np: np.ndarray, save_path: Path) -> None:
    out_r = np.array(
        PILImage.fromarray(out_np).resize(
            (in_np.shape[1], in_np.shape[0]), PILImage.LANCZOS
        )
    )
    p_in  = _radial_profile(in_np)
    p_out = _radial_profile(out_r)

    mx = max(p_in.max(), p_out.max())
    if mx > 0:
        p_in /= mx; p_out /= mx
    generated = np.clip(p_out - p_in, 0, None)
    fx = np.linspace(0, 1, len(p_in))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        "Real-ESRGAN — New Frequency Content Generated\n"
        "Radial power profile shows exactly which spatial frequencies are added by the model",
        color=TEXT, fontsize=11, fontweight="bold",
    )

    # --- panel 1: overlay ---
    _style(ax1,
           title="Input vs Output Radial Frequency Profile",
           xlabel="Normalised Spatial Frequency  (0 = DC,  1 = Nyquist)",
           ylabel="Normalised Power")
    ax1.plot(fx, p_in,  color=ACCENT,  lw=1.5, label="Input  (low-res)",  alpha=0.9)
    ax1.plot(fx, p_out, color=GREEN,   lw=1.5, label="Output (enhanced)", alpha=0.9)
    ax1.fill_between(fx, p_in, p_out,
                     where=p_out >= p_in, color=GREEN, alpha=0.20,
                     label="New energy added ▲")
    ax1.fill_between(fx, p_in, p_out,
                     where=p_out <  p_in, color=RED,   alpha=0.10,
                     label="Noise reduced ▼")
    ax1.legend(facecolor=BG3, labelcolor=TEXT, fontsize=9)
    ax1.set_xlim(0, 1)

    bands = [(0.0, 0.1, "DC", ACCENT2),
             (0.1, 0.35, "Low", CYAN),
             (0.35, 0.65, "Mid", GOLD),
             (0.65, 1.0,  "High", RED)]
    for x0, x1, lbl, col in bands:
        ax1.axvspan(x0, x1, alpha=0.04, color=col)
        ax1.text((x0 + x1) / 2, ax1.get_ylim()[1] * 0.92, lbl,
                 ha="center", color=col, fontsize=8, alpha=0.8)

    # --- panel 2: generated only ---
    _style(ax2,
           title="Frequency Energy Generated by Real-ESRGAN  (Output − Input, clipped ≥ 0)",
           xlabel="Normalised Spatial Frequency",
           ylabel="Generated Power")
    ax2.plot(fx, generated, color=GOLD, lw=1.5)
    ax2.fill_between(fx, 0, generated, color=GOLD, alpha=0.28,
                     label="New frequency energy hallucinated by ESRGAN")
    pk = int(np.argmax(generated))
    ax2.annotate(
        f"Peak generation\nf = {fx[pk]:.2f}",
        xy=(fx[pk], generated[pk]),
        xytext=(fx[pk] + 0.12, generated[pk] * 0.85),
        color=RED, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=RED),
    )
    ax2.legend(facecolor=BG3, labelcolor=TEXT, fontsize=9)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5.  Radar Summary Chart
# ---------------------------------------------------------------------------

def _band_energies(img: np.ndarray, n_bands: int = 8) -> np.ndarray:
    gray = _to_gray(img)
    fsh  = np.fft.fftshift(np.fft.fft2(gray))
    mag  = np.abs(fsh)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X   = np.ogrid[:h, :w]
    R      = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r  = float(min(cx, cy))
    edges  = np.linspace(0, max_r, n_bands + 1)
    return np.array([
        float(mag[(R >= edges[i]) & (R < edges[i + 1])].mean())
        for i in range(n_bands)
    ])


def _plot_radar(in_np: np.ndarray, out_np: np.ndarray, save_path: Path) -> None:
    out_r = np.array(
        PILImage.fromarray(out_np).resize(
            (in_np.shape[1], in_np.shape[0]), PILImage.LANCZOS
        )
    )

    N   = 8
    lbs = ["DC", "Very Low", "Low", "Low-Mid", "Mid", "High-Mid", "High", "Ultra-High"]
    e_in  = _band_energies(in_np,  N)
    e_out = _band_energies(out_r,  N)
    mx    = max(e_in.max(), e_out.max())
    if mx > 0:
        e_in /= mx; e_out /= mx

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.append(angles, angles[0])
    e_in   = np.append(e_in,  e_in[0])
    e_out  = np.append(e_out, e_out[0])

    fig = plt.figure(figsize=(10, 10))
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor(BG2)
    fig.patch.set_facecolor(BG)

    ax.plot(angles, e_in,  color=ACCENT,  lw=2.0, label="Input",    alpha=0.9)
    ax.fill(angles, e_in,  color=ACCENT,  alpha=0.12)
    ax.plot(angles, e_out, color=GREEN,   lw=2.0, label="Enhanced",  alpha=0.9)
    ax.fill(angles, e_out, color=GREEN,   alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(lbs, color=TEXT2, fontsize=9)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color(BG3)
    ax.grid(color=BG3, alpha=0.4)

    # improvement delta arrows
    for i, (ang, ei, eo) in enumerate(zip(angles[:-1], e_in[:-1], e_out[:-1])):
        if eo > ei + 0.02:
            ax.annotate("", xy=(ang, eo), xytext=(ang, ei),
                        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))

    ax.legend(facecolor=BG3, labelcolor=TEXT,
              fontsize=10, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.suptitle(
        "Real-ESRGAN — Frequency Band Improvement Radar\n"
        "Shows energy gained in each frequency band after enhancement",
        color=TEXT, fontsize=11, fontweight="bold", y=1.02,
    )

    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

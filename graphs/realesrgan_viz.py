"""
Real-ESRGAN visualization suite.

Generates:
  01_filter_responses_64.png
  01_filter_responses_top16.png
  02_block_progression_23.png
  03_frequency_before_after.png
  04_new_frequencies_generated.png
  05_radar_summary.png
  06_tiling_grid_diagram.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def run_visualization_suite(
    model,
    input_np: np.ndarray,
    output_np: np.ndarray,
    out_dir: str | Path,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    features = _extract_features(model, input_np)
    steps = [
        ("filters", lambda: _plot_filters(features, out_path)),
        ("blocks", lambda: _plot_blocks(features, input_np, out_path)),
        ("frequency", lambda: _plot_freq_analysis(input_np, output_np, out_path)),
        ("new_freq", lambda: _plot_new_freqs(input_np, output_np, out_path)),
        ("radar", lambda: _plot_radar(input_np, output_np, out_path)),
        ("tiling", lambda: _plot_tiling_stub(input_np, output_np, out_path)),
    ]
    total = len(steps)
    for idx, (name, fn) in enumerate(steps, start=1):
        fn()
        if progress_cb:
            progress_cb(idx, total, name)


def _extract_features(model, input_np: np.ndarray) -> dict[str, object]:
    data: dict[str, object] = {"weights": None, "conv": None, "blocks": []}
    if model is None or not hasattr(model, "model"):
        return data

    try:
        import torch
    except Exception:
        return data

    net = model.model
    if not hasattr(net, "conv_first") or not hasattr(net, "body"):
        return data

    hooks = []
    conv_store: dict[str, np.ndarray] = {}
    block_store: dict[int, np.ndarray] = {}

    def _hook_conv(_, __, out):
        conv_store["v"] = out.detach().float().cpu().numpy()[0]

    hooks.append(net.conv_first.register_forward_hook(_hook_conv))

    for i, blk in enumerate(net.body):
        def _hook_blk(_, __, out, k=i):
            block_store[k] = out.detach().float().cpu().numpy()[0]

        hooks.append(blk.register_forward_hook(_hook_blk))

    try:
        img = input_np.astype(np.float32)[:, :, ::-1] / 255.0  # RGB -> BGR
        ten = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0)
        dev = next(net.parameters()).device
        ten = ten.to(dev)
        ten = ten.half() if next(net.parameters()).dtype == torch.float16 else ten.float()
        with torch.no_grad():
            net.eval()
            net(ten)
        data["weights"] = net.conv_first.weight.detach().float().cpu().numpy()
        data["conv"] = conv_store.get("v")
        data["blocks"] = [block_store[k] for k in sorted(block_store)]
    except Exception:
        pass
    finally:
        for h in hooks:
            h.remove()
    return data


def _filter_type(kernel_2d: np.ndarray) -> str:
    gx = np.abs(np.diff(kernel_2d, axis=1)).mean()
    gy = np.abs(np.diff(kernel_2d, axis=0)).mean()
    center = kernel_2d[1, 1]
    corners = (kernel_2d[0, 0] + kernel_2d[0, 2] + kernel_2d[2, 0] + kernel_2d[2, 2]) / 4.0
    spread = kernel_2d.std()
    if spread < 0.08:
        return "Smooth"
    if gx > gy * 1.2:
        return "Edge-V"
    if gy > gx * 1.2:
        return "Edge-H"
    if center > corners:
        return "Blob"
    return "Texture"


def _plot_filters(features: dict[str, object], out_dir: Path) -> None:
    weights = features.get("weights")
    conv = features.get("conv")

    # Image 1: 64 kernel weights as 3x3 patches with diverging map.
    fig1, axes1 = plt.subplots(8, 8, figsize=(18, 18))
    fig1.suptitle("Real-ESRGAN conv_first Kernel Weights (64 filters)", fontsize=14, fontweight="bold")
    if isinstance(weights, np.ndarray) and weights.shape[0] >= 64:
        for i, ax in enumerate(axes1.flat[:64]):
            k = weights[i].mean(axis=0)  # (3,3) as requested
            max_abs = max(float(np.max(np.abs(k))), 1e-6)
            k_norm = np.clip(k / max_abs, -1.0, 1.0)
            label = _filter_type(k_norm)
            im = ax.imshow(k_norm, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_title(f"{i:02d} {label}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            cb = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
    else:
        for ax in axes1.flat:
            ax.axis("off")
        fig1.text(0.5, 0.5, "Feature weights unavailable (model hooks missing)", ha="center", va="center")
    fig1.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig1.savefig(out_dir / "01_filter_responses_64.png", dpi=160, bbox_inches="tight")
    plt.close(fig1)

    # Image 2: top-16 most active responses by variance, per-filter normalization.
    fig2, axes2 = plt.subplots(4, 4, figsize=(14, 14))
    fig2.suptitle("Top-16 conv_first Responses by Variance", fontsize=14, fontweight="bold")
    if isinstance(conv, np.ndarray) and conv.shape[0] >= 16:
        variances = conv.reshape(conv.shape[0], -1).var(axis=1)
        top16 = np.argsort(variances)[-16:][::-1]
        for ax, idx in zip(axes2.flat, top16):
            fmap = conv[idx]
            vmin = float(np.min(fmap))
            vmax = float(np.max(fmap))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6
            label = "Unknown"
            if isinstance(weights, np.ndarray) and idx < weights.shape[0]:
                label = _filter_type(np.clip(weights[idx].mean(axis=0), -1, 1))
            im = ax.imshow(fmap, cmap="plasma", vmin=vmin, vmax=vmax)
            ax.set_title(f"{idx:02d} {label} | mean={fmap.mean():.4f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            cb = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
    else:
        for ax in axes2.flat:
            ax.axis("off")
        fig2.text(0.5, 0.5, "Feature responses unavailable (model hooks missing)", ha="center", va="center")
    fig2.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig2.savefig(out_dir / "01_filter_responses_top16.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)


def _plot_blocks(features: dict[str, object], input_np: np.ndarray, out_dir: Path) -> None:
    blocks = features.get("blocks") or []
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0])

    # Thumbnails with per-block normalization.
    top = gs[0].subgridspec(4, 6, wspace=0.2, hspace=0.35)
    # 4×6 = 24 slots; RRDBNet has 23 blocks so slot 24 stays blank intentionally.
    for i in range(min(24, len(blocks) + 1)):
        ax = fig.add_subplot(top[i // 6, i % 6])
        if i < len(blocks):
            m = blocks[i].mean(axis=0)
            lo, hi = float(m.min()), float(m.max())
            if np.isclose(lo, hi):
                hi = lo + 1e-6
            im = ax.imshow(m, cmap="magma", vmin=lo, vmax=hi)
            ax.set_title(f"Block {i+1}", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    # Bar + cumulative energy + spatial variance — twin y-axis so bars and
    # cumulative line don't compete for the same scale.
    ax_bar = fig.add_subplot(gs[1])
    if blocks:
        energy      = np.array([np.mean(np.abs(b)) for b in blocks], dtype=np.float64)
        spatial_var = np.array([np.var(b.mean(axis=0)) for b in blocks], dtype=np.float64)
        cumulative  = np.cumsum(energy)
        x = np.arange(1, len(blocks) + 1)

        gray_in      = np.dot(input_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float64)
        input_energy = float(np.mean(np.abs(gray_in - gray_in.mean())))

        # Left axis — per-block energy bars + input baseline
        ax_bar.bar(x, energy, color="#4c78a8", alpha=0.75, label="Block energy (mean abs)")
        ax_bar.axhline(input_energy, linestyle="--", color="#e45756", linewidth=1.5,
                       label=f"Input energy baseline ({input_energy:.4f})")
        ax_bar.set_xlabel("RRDB Block Index")
        ax_bar.set_ylabel("Per-block energy (mean |activation|)", color="#4c78a8")
        ax_bar.tick_params(axis="y", labelcolor="#4c78a8")
        ax_bar.set_xticks(x)
        ax_bar.set_title("Block Progression: Per-block Energy  |  Cumulative Build-up  |  Spatial Variance",
                         fontweight="bold")
        ax_bar.grid(alpha=0.2)

        # Right axis — cumulative energy and spatial variance (different scale)
        ax2 = ax_bar.twinx()
        ax2.plot(x, cumulative,  color="#f58518", marker="o", linewidth=2.0,
                 label="Cumulative energy (right axis)")
        ax2.plot(x, spatial_var, color="#54a24b", marker="s", linewidth=1.8,
                 label="Spatial variance (right axis)")
        ax2.set_ylabel("Cumulative / Spatial variance", color="#f58518")
        ax2.tick_params(axis="y", labelcolor="#f58518")

        # Combined legend from both axes
        lines1, labels1 = ax_bar.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_bar.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        ax_bar.text(0.5, 0.5, "Block activations unavailable (model hooks missing)",
                    ha="center", va="center")
        ax_bar.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_dir / "02_block_progression_23.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _fft_mag(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f).astype(np.float64)
    h, w = mag.shape
    mag[h // 2, w // 2] = 0.0  # remove DC spike for visibility
    return mag


def _resize_nearest(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = img.shape[:2]
    row = np.minimum((np.arange(out_h) * in_h // out_h), in_h - 1)
    col = np.minimum((np.arange(out_w) * in_w // out_w), in_w - 1)
    if img.ndim == 2:
        return img[row[:, None], col[None, :]]
    return img[row[:, None], col[None, :], :]


def _plot_freq_analysis(input_np: np.ndarray, output_np: np.ndarray, out_dir: Path) -> None:
    gray_in = np.dot(input_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float64)
    gray_out = np.dot(output_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float64)

    mag_in = _fft_mag(gray_in)
    mag_out = _fft_mag(gray_out)

    # Upscale the INPUT IMAGE to output size, then compute its FFT.
    # This is correct — upscaling the magnitude spectrum directly creates
    # tiled artifacts in frequency space that don't represent anything real.
    gray_in_up = _resize_nearest(gray_in, gray_out.shape[0], gray_out.shape[1])
    mag_in_up  = _fft_mag(gray_in_up)
    diff       = mag_out - mag_in_up
    # Percentile clip: the raw diff range is huge (1e7) but most signal is
    # concentrated in a small range. Clip to ±99th percentile so the colormap
    # actually shows the structure instead of being blank grey.
    p99        = max(float(np.percentile(np.abs(diff), 99)), 1.0)
    max_abs    = p99

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    specs = [
        (mag_in, "Input FFT (native size)"),
        (mag_out, "Output FFT (native size)"),
    ]
    for ax, (mag, ttl) in zip(axes[:2], specs):
        vmax = max(float(np.percentile(mag, 99.5)), 1.0)
        im = ax.imshow(mag, cmap="inferno", norm=LogNorm(vmin=1, vmax=vmax))
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("← Low freq | High freq →")
        ax.set_ylabel("← Low freq | High freq →")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    imd = axes[2].imshow(diff, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    axes[2].set_title("Difference Spectrum (output - input upsampled)", fontsize=11)
    axes[2].set_xlabel("← Low freq | High freq →")
    axes[2].set_ylabel("← Low freq | High freq →")
    fig.colorbar(imd, ax=axes[2], fraction=0.046, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_dir / "03_frequency_before_after.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _radial_profile(mag: np.ndarray) -> np.ndarray:
    h, w = mag.shape
    yy, xx = np.indices((h, w))
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def _plot_new_freqs(input_np: np.ndarray, output_np: np.ndarray, out_dir: Path) -> None:
    in_h, in_w = input_np.shape[:2]
    out_h, out_w = output_np.shape[:2]

    nearest  = _resize_nearest(input_np, out_h, out_w).astype(np.float64)
    out_f    = output_np.astype(np.float64)
    diff_rgb = np.abs(out_f - nearest)
    diff     = diff_rgb.mean(axis=2)

    gray_nearest = np.dot(nearest[..., :3], [0.2989, 0.5870, 0.1140])
    gray_out     = np.dot(out_f[..., :3],   [0.2989, 0.5870, 0.1140])
    prof_nearest = _radial_profile(_fft_mag(gray_nearest))
    prof_out     = _radial_profile(_fft_mag(gray_out))

    max_r        = min(len(prof_nearest), len(prof_out))
    x            = np.arange(max_r)
    prof_nearest = prof_nearest[:max_r]
    prof_out     = prof_out[:max_r]

    # Log-scale: raw FFT magnitude is dominated by low frequencies by orders of
    # magnitude — log1p makes high-frequency differences visible.
    log_nearest = np.log1p(prof_nearest)
    log_out     = np.log1p(prof_out)
    # Linear gain keeps sign so we can see where ESRGAN added vs suppressed.
    gain        = prof_out - prof_nearest

    out_nyquist             = min(out_h, out_w) / 2.0
    input_nyquist_on_output = out_nyquist * (min(in_h, in_w) / max(min(out_h, out_w), 1))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel A — log-scaled radial profiles
    axes[0, 0].plot(x, log_nearest, label="Nearest-upscaled baseline", linewidth=2)
    axes[0, 0].plot(x, log_out,     label="ESRGAN output",             linewidth=2)
    axes[0, 0].axvline(input_nyquist_on_output, color="red", linestyle="--", label="Input Nyquist")
    axes[0, 0].set_title("Radial Frequency Profile — log scale (full output size)")
    axes[0, 0].set_xlabel("Radius (frequency bins)")
    axes[0, 0].set_ylabel("log(1 + Mean FFT magnitude)")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    # Panel B — linear gain (ESRGAN minus nearest baseline)
    axes[0, 1].plot(x, gain, color="#8e44ad", linewidth=2)
    axes[0, 1].axvline(input_nyquist_on_output, color="red", linestyle="--", label="Input Nyquist")
    axes[0, 1].axhline(0, color="grey", linewidth=0.8, linestyle=":")
    axes[0, 1].set_title("New Frequency Gain (ESRGAN − Nearest baseline)\n"
                         "Above 0 = ESRGAN added energy  |  Below 0 = suppressed")
    axes[0, 1].set_xlabel("Radius (frequency bins)")
    axes[0, 1].set_ylabel("Magnitude gain (linear)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.25)

    # Panel C — spatial diff heatmap, clipped to actual data range
    vmax_diff = max(float(np.percentile(diff, 99)), 1.0)
    im = axes[1, 0].imshow(diff, cmap="inferno", vmin=0, vmax=vmax_diff)
    axes[1, 0].set_title(f"Spatial Difference |output − nearest upscale|\n"
                         f"(colormap clipped to 99th percentile = {vmax_diff:.1f})")
    axes[1, 0].set_xlabel(f"{out_w} px")
    axes[1, 0].set_ylabel(f"{out_h} px")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.02)

    # Panel D — histogram
    axes[1, 1].hist(diff.ravel(), bins=60, color="#2ca02c", alpha=0.85)
    axes[1, 1].set_title("Difference Histogram\n"
                         "Most pixels near 0 = ESRGAN preserves structure; tail = new detail")
    axes[1, 1].set_xlabel("Absolute pixel difference (0–255)")
    axes[1, 1].set_ylabel("Pixel count")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "04_new_frequencies_generated.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def _blur3(gray: np.ndarray) -> np.ndarray:
    p = np.pad(gray, 1, mode="reflect")
    out = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
        p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0
    return out


def _lap_var(gray: np.ndarray) -> float:
    p = np.pad(gray, 1, mode="reflect")
    lap = (
        p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * p[1:-1, 1:-1]
    )
    return float(np.var(lap))


def _high_freq_energy(gray: np.ndarray) -> float:
    mag = _fft_mag(gray)
    h, w = mag.shape
    yy, xx = np.indices((h, w))
    cy, cx = h // 2, w // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rr.max()
    mask = rr >= (0.6 * rmax)  # outer 40%
    return float(np.sum(mag[mask]))


def _plot_radar(input_np: np.ndarray, output_np: np.ndarray, out_dir: Path) -> None:
    out_h, out_w = output_np.shape[:2]
    base = _resize_nearest(input_np, out_h, out_w).astype(np.float64)
    out = output_np.astype(np.float64)

    gray_base = np.dot(base[..., :3], [0.2989, 0.5870, 0.1140])
    gray_out = np.dot(out[..., :3], [0.2989, 0.5870, 0.1140])

    metrics = ["Sharpness", "High-freq energy", "Detail gain", "Noise level", "Edge strength"]

    base_vals = np.array([
        _lap_var(gray_base),
        _high_freq_energy(gray_base),
        0.0,   # Detail gain baseline is always 0 — ESRGAN is measured against this
        float(np.std(gray_base - _blur3(gray_base))),
        float(np.mean(_sobel_mag(gray_base))),
    ])
    out_vals = np.array([
        _lap_var(gray_out),
        _high_freq_energy(gray_out),
        float(np.mean(np.abs(out - base)) / 255.0),
        float(np.std(gray_out - _blur3(gray_out))),
        float(np.mean(_sobel_mag(gray_out))),
    ])

    # Normalize each metric independently to [0, 10].
    # Use the max of the two values as the ceiling so both are on the same scale.
    # For "Detail gain", base is always 0 (reference point) so out_norm shows
    # how much ESRGAN added relative to a nearest-neighbor baseline.
    denom = np.maximum(np.maximum(base_vals, out_vals), 1e-9)
    base_norm = 10.0 * (base_vals / denom)
    out_norm  = 10.0 * (out_vals  / denom)

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    base_plot = np.append(base_norm, base_norm[0])
    out_plot = np.append(out_norm, out_norm[0])

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.plot(angles, base_plot, linewidth=2, color="#1f77b4", label="Input baseline (nearest)")
    ax.fill(angles, base_plot, color="#1f77b4", alpha=0.2)
    ax.plot(angles, out_plot, linewidth=2, color="#d62728", label="ESRGAN output")
    ax.fill(angles, out_plot, color="#d62728", alpha=0.2)
    ax.set_title("Real Metrics Radar: Input Baseline vs ESRGAN", y=1.08, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.15), fontsize=9)

    for i, a in enumerate(angles[:-1]):
        ax.text(a, out_norm[i] + 0.45, f"{out_vals[i]:.3g}", color="#d62728", fontsize=8, ha="center")
        ax.text(a, max(base_norm[i] - 0.45, 0.2), f"{base_vals[i]:.3g}", color="#1f77b4", fontsize=8, ha="center")

    fig.tight_layout()
    fig.savefig(out_dir / "05_radar_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_tiling_stub(input_np: np.ndarray, output_np: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ih, iw = input_np.shape[:2]
    oh, ow = output_np.shape[:2]
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"Tiling/shape summary\nInput: {iw}x{ih}\nOutput: {ow}x{oh}",
        ha="center",
        va="center",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "06_tiling_grid_diagram.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

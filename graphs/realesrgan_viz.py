"""
graphs/realesrgan_viz.py — Real-ESRGAN Deep Feature Visualization Suite v2

Each plot is fully labelled and tells a clear story about the processing.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from PIL import Image as PILImage

# ── dark palette ─────────────────────────────────────────────────────────────
BG="#07080f"; BG2="#0d0f1a"; BG3="#111526"
TEXT="#e0e4ff"; TEXT2="#8890b8"
ACCENT="#6c8bff"; ACCENT2="#a259ff"; CYAN="#00e5ff"
GREEN="#39ff8a"; RED="#ff4d6a"; GOLD="#ffd166"

RC = {
    "figure.facecolor":BG,"axes.facecolor":BG2,"axes.edgecolor":BG3,
    "text.color":TEXT,"axes.labelcolor":TEXT2,"xtick.color":TEXT2,
    "ytick.color":TEXT2,"grid.color":BG3,"font.family":"monospace",
    "savefig.facecolor":BG,
}

def _s(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG2)
    for sp in ax.spines.values(): sp.set_edgecolor(BG3)
    if title:  ax.set_title(title, color=TEXT, fontsize=8, pad=5)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT2, fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT2, fontsize=7)
    ax.tick_params(colors=TEXT2, labelsize=6)
    ax.grid(True, alpha=0.1, color=BG3)

def _gray(img):
    return (0.2126*img[:,:,0]+0.7152*img[:,:,1]+0.0722*img[:,:,2]).astype(np.float32)/255.

def _classify_filter(w3x3):
    """Classify a 3x3 kernel: returns (label, color)"""
    lap = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    sharpness = float(np.abs(np.sum(w3x3 * lap)))
    dc = float(abs(w3x3.sum()))
    if sharpness > 0.5:  return "Edge/Detail", RED
    if dc > 0.8:         return "Smooth/Bias", CYAN
    return "Texture/Pattern", GOLD

# ── PUBLIC ENTRY POINT ───────────────────────────────────────────────────────

def run_visualization_suite(model, input_np, output_np, out_dir, progress_cb=None,
                            tile_size=0, tile_pad=10):
    import torch
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def _log(s,t,l):
        if progress_cb: progress_cb(s,t,l)
        print(f"  [ESRGAN-VIZ {s}/{t}] {l}")

    rrdb = model.model
    device = next(rrdb.parameters()).device
    cf_feat, blk = {}, {}
    hooks = []

    def _hf(m,i,o): cf_feat["d"]=o.detach().cpu()
    hooks.append(rrdb.conv_first.register_forward_hook(_hf))
    for idx,b in enumerate(rrdb.body):
        def _hb(m,i,o,k=idx): blk[k]=o.detach().cpu()
        hooks.append(b.register_forward_hook(_hb))

    img_bgr = input_np[:,:,::-1].astype(np.float32)/255.
    img_t   = torch.from_numpy(np.ascontiguousarray(img_bgr.transpose(2,0,1))).float().unsqueeze(0).to(device)

    _log(1,7,"Forward pass capturing hooks …")
    rrdb.eval()
    with torch.no_grad(): rrdb(img_t)
    for h in hooks: h.remove()

    w_first = rrdb.conv_first.weight.detach().cpu().numpy()  # (64,3,3,3)

    _log(2,7,"Plotting 64 filter responses …")
    with matplotlib.rc_context(RC):
        _plot_filters(cf_feat["d"], w_first, input_np, out_dir/"01_filter_responses_64.png")

    _log(3,7,"Plotting 23 RRDB block progression …")
    with matplotlib.rc_context(RC):
        _plot_blocks(blk, out_dir/"02_block_progression_23.png")

    _log(4,7,"Frequency domain analysis …")
    with matplotlib.rc_context(RC):
        _plot_freq_analysis(input_np, output_np, out_dir/"03_frequency_before_after.png")

    _log(5,7,"New frequency generation map …")
    with matplotlib.rc_context(RC):
        _plot_new_freqs(input_np, output_np, out_dir/"04_new_frequencies_generated.png")

    _log(6,7,"Radar summary …")
    with matplotlib.rc_context(RC):
        _plot_radar(input_np, output_np, out_dir/"05_radar_summary.png")

    _log(7,7,"Tiling / grid diagram …")
    with matplotlib.rc_context(RC):
        _plot_tiling(input_np, output_np, tile_size, tile_pad,
                     out_dir/"06_tiling_grid_diagram.png")

    print(f"\n  ✓ All 6 visualizations → {out_dir}\n")


# ── 1. FILTER RESPONSES ──────────────────────────────────────────────────────

def _plot_filters(feat, weights, input_np, save_path):
    """8×8 grid: each cell = feature map + kernel type label."""
    fmaps = feat.squeeze(0).numpy()          # (64, H, W)
    # global min/max for fair comparison
    g_lo, g_hi = fmaps.min(), fmaps.max()
    if g_hi > g_lo: fmaps_n = (fmaps - g_lo)/(g_hi - g_lo)
    else: fmaps_n = fmaps

    counts = {"Edge/Detail":0, "Smooth/Bias":0, "Texture/Pattern":0}
    for i in range(64):
        k = weights[i].mean(axis=0)          # avg over 3 in-channels → (3,3)
        lbl,_ = _classify_filter(k)
        counts[lbl] += 1

    fig = plt.figure(figsize=(22,14))
    fig.suptitle(
        "Real-ESRGAN — First Conv Layer: 64 Feature Responses\n"
        "Each cell shows how one of the 64 learned filters responds to the input image. "
        "Brighter = stronger activation at that spatial location.",
        color=TEXT, fontsize=10, fontweight="bold", y=0.98)

    # legend strip at top
    leg_y = 0.95
    fig.text(0.18, leg_y, f"■ Edge/Detail ({counts['Edge/Detail']})",
             color=RED, fontsize=8, fontfamily="monospace")
    fig.text(0.38, leg_y, f"■ Smooth/Bias ({counts['Smooth/Bias']})",
             color=CYAN, fontsize=8, fontfamily="monospace")
    fig.text(0.58, leg_y, f"■ Texture/Pattern ({counts['Texture/Pattern']})",
             color=GOLD, fontsize=8, fontfamily="monospace")
    fig.text(0.78, leg_y, "Shared intensity scale across all 64",
             color=TEXT2, fontsize=7, fontfamily="monospace")

    gs = gridspec.GridSpec(8,8, figure=fig, hspace=0.35, wspace=0.15,
                           left=0.01, right=0.99, top=0.91, bottom=0.02)

    for i in range(64):
        ax = fig.add_subplot(gs[i//8, i%8])
        ax.imshow(fmaps_n[i], cmap="plasma", interpolation="nearest",
                  aspect="auto", vmin=0, vmax=1)

        k = weights[i].mean(axis=0)
        lbl, col = _classify_filter(k)
        short = {"Edge/Detail":"Edge","Smooth/Bias":"Smooth","Texture/Pattern":"Texture"}[lbl]
        mean_act = fmaps_n[i].mean()

        ax.set_title(f"F{i+1:02d} {short}\nact={mean_act:.2f}",
                     color=col, fontsize=5.5, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(col if mean_act>0.4 else BG3)

    fig.text(0.5, 0.005,
             "Plasma colormap (dark=low, bright=high) | Global scale: all 64 maps normalized together",
             ha="center", color=TEXT2, fontsize=7)
    fig.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ── 2. BLOCK PROGRESSION ─────────────────────────────────────────────────────

_STAGE_LABELS = {
    0:"Feature\nextraction",4:"Low-level\npatterns",
    8:"Mid-level\nstructures",12:"High-level\nfeatures",
    16:"Detail\nrefinement",20:"Pre-upscale\nprep",22:"Final\nrefinement"
}

def _plot_blocks(block_feats, save_path):
    n = len(block_feats)
    energies, deltas = [], [0.0]

    for i in range(n):
        feat = block_feats[i].squeeze(0).numpy()
        e = float(np.mean(np.abs(feat)))
        energies.append(e)
    for i in range(1,n):
        deltas.append(energies[i]-energies[i-1])

    fig = plt.figure(figsize=(28,12))
    fig.suptitle(
        "Real-ESRGAN — 23 RRDB Block Feature Progression\n"
        "Top row: mean feature activation per block (what the model 'sees' as depth increases). "
        "Bottom: energy change between consecutive blocks (positive = new information added).",
        color=TEXT, fontsize=10, fontweight="bold", y=0.98)

    gs_top = gridspec.GridSpec(1,n, figure=fig, left=0.01, right=0.99,
                               top=0.87, bottom=0.42, wspace=0.04)
    gs_bot = gridspec.GridSpec(1,1, figure=fig, left=0.06, right=0.99,
                               top=0.37, bottom=0.07)

    for i in range(n):
        feat = block_feats[i].squeeze(0).numpy()
        mean_fm = feat.mean(axis=0)
        lo,hi = mean_fm.min(),mean_fm.max()
        if hi>lo: mean_fm=(mean_fm-lo)/(hi-lo)

        ax = fig.add_subplot(gs_top[0,i])
        ax.imshow(mean_fm, cmap="inferno", interpolation="nearest", aspect="auto")
        stage = _STAGE_LABELS.get(i,"")
        label_color = GREEN if i in _STAGE_LABELS else TEXT2
        ax.set_title(f"B{i+1:02d}\n{stage}", color=label_color, fontsize=5.5, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(GREEN if i in _STAGE_LABELS else BG3)

        if i > 0 and deltas[i] > 0.01:
            ax.text(0.5,-0.25,f"+{deltas[i]:.3f}", transform=ax.transAxes,
                    ha="center", color=GREEN, fontsize=5)
        elif i > 0 and deltas[i] < -0.01:
            ax.text(0.5,-0.25,f"{deltas[i]:.3f}", transform=ax.transAxes,
                    ha="center", color=RED, fontsize=5)

    ax_b = fig.add_subplot(gs_bot[0,0])
    _s(ax_b,
       title="Block-to-Block Energy Change  (positive = new information injected, negative = compression/noise removal)",
       xlabel="RRDB Block #", ylabel="ΔEnergy vs previous block")

    xs = np.arange(1,n+1)
    cols = [GREEN if d>=0 else RED for d in deltas[1:]]
    ax_b.bar(xs, deltas[1:], color=cols, alpha=0.8, width=0.7)
    ax_b.axhline(0, color=TEXT2, linewidth=0.8, linestyle="--", alpha=0.5)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([str(x) for x in xs], fontsize=6)

    for i, lbl in _STAGE_LABELS.items():
        if i < n:
            ax_b.axvline(i+1, color=GOLD, linewidth=0.8, linestyle=":", alpha=0.5)
            ax_b.text(i+1.1, ax_b.get_ylim()[1]*0.85,
                      lbl.replace("\n"," "), color=GOLD, fontsize=6, rotation=0)

    p1=mpatches.Patch(color=GREEN,label="Energy increase (new detail added)")
    p2=mpatches.Patch(color=RED,  label="Energy decrease (refinement/denoising)")
    ax_b.legend(handles=[p1,p2], facecolor=BG3, labelcolor=TEXT, fontsize=7, loc="upper right")

    fig.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ── 3. FREQUENCY ANALYSIS ─────────────────────────────────────────────────────

def _fft_mag(img):
    gray = _gray(img)
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))

def _add_freq_rings(ax, h, w):
    """Overlay concentric circles labelling frequency bands."""
    cy,cx = h//2, w//2
    for (r_frac, lbl, col) in [
        (0.08, "DC\n(low-freq)", CYAN),
        (0.20, "Low", ACCENT),
        (0.40, "Mid", GOLD),
        (0.65, "High", RED),
        (0.90, "Ultra\nHigh", ACCENT2),
    ]:
        r = int(min(cx,cy)*r_frac*2)
        circ = plt.Circle((cx,cy), r, fill=False, color=col,
                          linewidth=0.8, linestyle="--", alpha=0.45)
        ax.add_patch(circ)
        ax.text(cx+r*0.72, cy-r*0.72, lbl, color=col, fontsize=6, alpha=0.8)

def _plot_freq_analysis(in_np, out_np, save_path):
    out_r = np.array(PILImage.fromarray(out_np).resize(
        (in_np.shape[1],in_np.shape[0]), PILImage.LANCZOS))
    mag_in  = _fft_mag(in_np)
    mag_out = _fft_mag(out_r)
    mag_new = np.clip(mag_out - mag_in, 0, None)

    fig, axes = plt.subplots(1,3, figsize=(20,7))
    fig.suptitle(
        "Real-ESRGAN — 2D FFT Frequency Domain Analysis\n"
        "The Fourier transform shows how much energy exists at each spatial frequency. "
        "Centre = DC (slow variations). Edges = ultra-high freq (fine textures/details).",
        color=TEXT, fontsize=10, fontweight="bold")

    titles = [
        "① Input Image FFT\n(what frequencies existed before)",
        "② Enhanced Image FFT\n(what frequencies exist after ESRGAN)",
        "③ New Frequencies Created\n(output − input, clipped ≥ 0)\n→ bright = AI-hallucinated detail",
    ]
    cmaps = ["magma","magma","hot"]
    for ax,(data,title,cmap) in zip(axes, zip([mag_in,mag_out,mag_new],titles,cmaps)):
        im = ax.imshow(data, cmap=cmap, interpolation="bilinear", aspect="equal")
        _s(ax, title=title)
        ax.set_xticks([]); ax.set_yticks([])
        h,w = data.shape
        _add_freq_rings(ax,h,w)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.75)
        cb.ax.tick_params(colors=TEXT2, labelsize=6)
        cb.set_label("log(1+|FFT|)", color=TEXT2, fontsize=6)

    axes[2].text(0.5,-0.06,
        "Bright regions = frequencies the model INVENTED.\nThese did not exist in the low-res input.",
        transform=axes[2].transAxes, ha="center", color=GOLD, fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ── 4. NEW FREQUENCIES ───────────────────────────────────────────────────────

def _radial_profile(img):
    gray = _gray(img)
    mag  = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h,w  = mag.shape; cy,cx=h//2,w//2
    Y,X  = np.ogrid[:h,:w]
    R    = np.sqrt((X-cx)**2+(Y-cy)**2).astype(int)
    mr   = min(cx,cy)
    return np.array([mag[R==r].mean() if (R==r).any() else 0 for r in range(mr)])

def _plot_new_freqs(in_np, out_np, save_path):
    out_r = np.array(PILImage.fromarray(out_np).resize(
        (in_np.shape[1],in_np.shape[0]), PILImage.LANCZOS))
    p_in  = _radial_profile(in_np)
    p_out = _radial_profile(out_r)
    mx = max(p_in.max(), p_out.max())
    if mx>0: p_in/=mx; p_out/=mx
    gen = np.clip(p_out-p_in,0,None)
    fx  = np.linspace(0,1,len(p_in))

    # spatial difference — where in the image did new content appear?
    diff_rgb = np.abs(out_r.astype(float) - in_np.astype(float))
    diff_gray = diff_rgb.mean(axis=2)

    fig = plt.figure(figsize=(18,12))
    fig.suptitle(
        "Real-ESRGAN — Where and What New Frequency Content Was Generated\n"
        "Left: which spatial frequencies were added (radial FFT profile). "
        "Right: where in the image new detail appeared (pixel-level diff).",
        color=TEXT, fontsize=10, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2,2, figure=fig, hspace=0.4, wspace=0.3,
                           left=0.07,right=0.97,top=0.90,bottom=0.06)

    # panel A: overlaid profiles
    ax1 = fig.add_subplot(gs[0,:])
    _s(ax1, title="A — Input vs Output Radial Frequency Profile  (what ESRGAN changed)",
       xlabel="Normalised Spatial Frequency  (0=DC / 1=Nyquist limit of input)",
       ylabel="Normalised FFT Power")

    ax1.plot(fx, p_in,  color=ACCENT, lw=1.8, label="Input (low-res)",  alpha=0.9)
    ax1.plot(fx, p_out, color=GREEN,  lw=1.8, label="Output (enhanced)", alpha=0.9)
    ax1.fill_between(fx, p_in, p_out, where=p_out>=p_in,
                     color=GREEN, alpha=0.22, label="↑ New energy added")
    ax1.fill_between(fx, p_in, p_out, where=p_out<p_in,
                     color=RED,   alpha=0.12, label="↓ Noise suppressed")

    bands=[(0,.1,"DC\ncolour","#6c8bff"),(0.1,.3,"Low\nedge"   ,CYAN),
           (.3,.6,"Mid\ntexture",GOLD),  (.6,1.0,"High\ndetail",RED)]
    for x0,x1,lbl,col in bands:
        ax1.axvspan(x0,x1,alpha=0.04,color=col)
        ax1.text((x0+x1)/2, ax1.get_ylim()[1]*0.93, lbl,
                 ha="center", color=col, fontsize=7.5, alpha=0.85)
    ax1.legend(facecolor=BG3, labelcolor=TEXT, fontsize=9, loc="upper right")
    ax1.set_xlim(0,1)

    # panel B: generated only
    ax2 = fig.add_subplot(gs[1,0])
    _s(ax2,
       title="B — Frequency Energy Generated by ESRGAN  (output − input, ≥0 only)",
       xlabel="Normalised Spatial Frequency",
       ylabel="Generated Power")
    ax2.plot(fx, gen, color=GOLD, lw=1.5)
    ax2.fill_between(fx,0,gen,color=GOLD,alpha=0.3,label="AI-generated frequency energy")
    pk = int(np.argmax(gen))
    if gen[pk]>0:
        ax2.annotate(f"Peak generation\n@ f={fx[pk]:.2f}\n(model adds most detail here)",
                     xy=(fx[pk],gen[pk]), xytext=(fx[pk]+0.15,gen[pk]*0.8),
                     color=RED, fontsize=8,
                     arrowprops=dict(arrowstyle="->",color=RED,lw=1.2))
    ax2.legend(facecolor=BG3,labelcolor=TEXT,fontsize=8)
    ax2.set_xlim(0,1)

    # panel C: spatial diff heatmap
    ax3 = fig.add_subplot(gs[1,1])
    _s(ax3, title="C — Spatial Map of New Detail  (where in the image ESRGAN added content)\n"
               "Bright = high difference from input (new texture/edge generated there)")
    im = ax3.imshow(diff_gray, cmap="inferno", interpolation="bilinear", aspect="auto")
    ax3.set_xticks([]); ax3.set_yticks([])
    cb = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, shrink=0.85)
    cb.ax.tick_params(colors=TEXT2, labelsize=6)
    cb.set_label("Pixel difference (0–255)", color=TEXT2, fontsize=6)

    fig.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ── 5. RADAR SUMMARY ─────────────────────────────────────────────────────────

def _band_energies(img, n=8):
    gray = _gray(img)
    mag  = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h,w  = mag.shape; cy,cx=h//2,w//2
    Y,X  = np.ogrid[:h,:w]
    R    = np.sqrt((X-cx)**2+(Y-cy)**2)
    mr   = float(min(cx,cy))
    edges= np.linspace(0,mr,n+1)
    return np.array([float(mag[(R>=edges[i])&(R<edges[i+1])].mean()) for i in range(n)])

def _plot_radar(in_np, out_np, save_path):
    out_r = np.array(PILImage.fromarray(out_np).resize(
        (in_np.shape[1],in_np.shape[0]),PILImage.LANCZOS))
    N  = 8
    lbs= ["DC","Very Low","Low","Low-Mid","Mid","High-Mid","High","Ultra-High"]
    ei = _band_energies(in_np,N)
    eo = _band_energies(out_r,N)
    mx = max(ei.max(),eo.max())
    if mx>0: ei/=mx; eo/=mx
    pct_gain = np.where(ei>0,(eo-ei)/ei*100,0.)

    angles = np.linspace(0,2*np.pi,N,endpoint=False)
    angles = np.append(angles,angles[0])
    ei_p   = np.append(ei, ei[0])
    eo_p   = np.append(eo, eo[0])

    fig = plt.figure(figsize=(12,12)); fig.patch.set_facecolor(BG)
    ax  = fig.add_subplot(111, polar=True); ax.set_facecolor(BG2)

    ax.plot(angles,ei_p, color=ACCENT, lw=2.2, label="Input",    alpha=0.9)
    ax.fill(angles,ei_p, color=ACCENT, alpha=0.12)
    ax.plot(angles,eo_p, color=GREEN,  lw=2.2, label="Enhanced",  alpha=0.9)
    ax.fill(angles,eo_p, color=GREEN,  alpha=0.15)
    ax.fill_between(angles, ei_p, eo_p,
                    where=np.array(eo_p)>=np.array(ei_p),
                    color=GREEN, alpha=0.25, label="New energy")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [f"{lbs[i]}\n{pct_gain[i]:+.1f}%" for i in range(N)],
        color=TEXT2, fontsize=9)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color(BG3)
    ax.grid(color=BG3, alpha=0.35)

    for i,(ang,ei_v,eo_v) in enumerate(zip(angles[:-1],ei[:-1] if len(ei)>N else ei,
                                            eo[:-1] if len(eo)>N else eo)):
        if eo_v > ei_v+0.03:
            ax.annotate("",xy=(ang,eo_v),xytext=(ang,ei_v),
                        arrowprops=dict(arrowstyle="->",color=GOLD,lw=1.5))

    ax.legend(facecolor=BG3, labelcolor=TEXT, fontsize=11,
              loc="upper right", bbox_to_anchor=(1.35,1.12))

    total_gain = float(np.mean(pct_gain[4:]))  # high-freq bands
    fig.suptitle(
        "Real-ESRGAN — Frequency Band Energy Radar\n"
        f"Shows energy in 8 frequency bands before vs after enhancement.\n"
        f"Labels show % gain per band. Avg high-freq gain: {total_gain:+.1f}%",
        color=TEXT, fontsize=11, fontweight="bold", y=1.03)

    fig.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ── 6. TILING / GRID DIAGRAM ─────────────────────────────────────────────────

def _plot_tiling(in_np, out_np, tile_size, tile_pad, save_path):
    """
    4-panel diagram explaining Real-ESRGAN tiling:
      A) Input image with tile grid overlay + overlap padding
      B) One example tile zoomed in (shows core + padding)
      C) Output image with stitched tile grid overlay
      D) Table of tiling parameters and stats
    """
    H, W = in_np.shape[:2]
    sH, sW = out_np.shape[:2]

    # compute tile grid
    if tile_size <= 0 or tile_size >= min(H, W):
        effective_tile = min(H, W)
        n_rows, n_cols = 1, 1
        mode_label = f"Full-image mode (tile=0 or tile≥image size)\nEntire {W}×{H} image processed as one tile — no stitching needed."
    else:
        effective_tile = tile_size
        import math
        n_rows = math.ceil(H / tile_size)
        n_cols = math.ceil(W / tile_size)
        mode_label = (f"Tiled mode (tile={tile_size}px, pad={tile_pad}px)\n"
                      f"{n_cols}×{n_rows} = {n_cols*n_rows} tiles  |"
                      f"  Each tile: {tile_size}×{tile_size}px core + {tile_pad}px overlap padding")

    # tile colours (cycle)
    COLOURS = [
        "#6c8bff","#a259ff","#00e5ff","#39ff8a",
        "#ffd166","#ff4d6a","#ff9f1c","#06d6a0",
        "#ef476f","#118ab2","#ffd166","#aeffd8",
    ]

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        "Real-ESRGAN — Tiling & Grid Processing Diagram\n"
        + mode_label,
        color=TEXT, fontsize=10, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25,
                           left=0.05, right=0.97, top=0.89, bottom=0.05)

    # ── Panel A: input image + tile grid ──────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(in_np, interpolation="bilinear", aspect="auto")
    _s(ax_a, title=f"A — Input Image with Tile Grid  ({W}×{H} px)\n"
                   f"Coloured boxes = individual tiles sent to GPU.  "
                   f"Overlap padding = {tile_pad}px (dashed margin)")
    ax_a.set_xticks([]); ax_a.set_yticks([])

    tile_idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = c * effective_tile
            y0 = r * effective_tile
            tw = min(effective_tile, W - x0)
            th = min(effective_tile, H - y0)
            col = COLOURS[tile_idx % len(COLOURS)]
            tile_idx += 1

            # core tile rect
            rect = mpatches.Rectangle((x0, y0), tw, th,
                linewidth=1.5, edgecolor=col, facecolor=col, alpha=0.18)
            ax_a.add_patch(rect)
            rect2 = mpatches.Rectangle((x0, y0), tw, th,
                linewidth=1.5, edgecolor=col, facecolor="none")
            ax_a.add_patch(rect2)

            # padding overlay (dashed)
            px0 = max(0, x0 - tile_pad); py0 = max(0, y0 - tile_pad)
            px1 = min(W, x0 + tw + tile_pad); py1 = min(H, y0 + th + tile_pad)
            pad_rect = mpatches.Rectangle((px0, py0), px1-px0, py1-py0,
                linewidth=0.8, edgecolor=col, facecolor="none",
                linestyle="--", alpha=0.5)
            ax_a.add_patch(pad_rect)

            # tile number label
            ax_a.text(x0 + tw/2, y0 + th/2,
                      f"T{r*n_cols+c+1:02d}",
                      ha="center", va="center",
                      color=col, fontsize=max(5, 9-n_rows),
                      fontweight="bold")

    ax_a.set_xlim(0, W); ax_a.set_ylim(H, 0)

    # ── Panel B: zoomed tile detail (first tile) ───────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _s(ax_b, title="B — Zoomed: One Tile (T01) with Overlap Padding\n"
                   "Solid border = core tile sent for upscale.  "
                   "Dashed border = overlap padding (blends seams)")

    # crop first tile + padding from the input
    p = tile_pad
    x0e = min(effective_tile, W)
    y0e = min(effective_tile, H)
    xp0 = max(0, 0 - p); xp1 = min(W, x0e + p)
    yp0 = max(0, 0 - p); yp1 = min(H, y0e + p)
    tile_crop = in_np[yp0:yp1, xp0:xp1]

    ax_b.imshow(tile_crop, interpolation="nearest", aspect="auto")
    ax_b.set_xticks([]); ax_b.set_yticks([])

    # core boundary
    core_x = 0 - xp0; core_y = 0 - yp0
    core_w = x0e - 0; core_h = y0e - 0
    ax_b.add_patch(mpatches.Rectangle(
        (core_x, core_y), core_w, core_h,
        linewidth=2.5, edgecolor=GREEN, facecolor="none", label="Core tile"))
    # pad boundary
    ax_b.add_patch(mpatches.Rectangle(
        (0, 0), xp1-xp0-1, yp1-yp0-1,
        linewidth=1.5, edgecolor=GOLD, facecolor="none",
        linestyle="--", label=f"+ {p}px padding"))

    ax_b.legend(facecolor=BG3, labelcolor=TEXT, fontsize=8, loc="lower right")
    ax_b.text(core_x + core_w/2, core_y + core_h/2,
              f"Core\n{x0e}×{y0e}px\n→GPU",
              ha="center", va="center", color=GREEN, fontsize=9, fontweight="bold")
    ax_b.text(2, 2, f"Padded region\n({xp1-xp0}×{yp1-yp0}px total)",
              color=GOLD, fontsize=7)

    # ── Panel C: output image + stitching grid ─────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.imshow(out_np, interpolation="bilinear", aspect="auto")
    _s(ax_c, title=f"C — Output Image with Stitched Tile Grid  ({sW}×{sH} px, {sW//max(W,1)}× upscale)\n"
                   "Tiles are upscaled independently then seamlessly stitched together")
    ax_c.set_xticks([]); ax_c.set_yticks([])

    scale_x = sW / max(W, 1); scale_y = sH / max(H, 1)
    tile_idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = int(c * effective_tile * scale_x)
            y0 = int(r * effective_tile * scale_y)
            tw = int(min(effective_tile, W - c*effective_tile) * scale_x)
            th = int(min(effective_tile, H - r*effective_tile) * scale_y)
            col = COLOURS[tile_idx % len(COLOURS)]
            tile_idx += 1
            rect = mpatches.Rectangle((x0,y0), tw, th,
                linewidth=1.5, edgecolor=col, facecolor="none")
            ax_c.add_patch(rect)
            ax_c.text(x0+tw/2, y0+th/2, f"T{r*n_cols+c+1:02d}",
                      ha="center", va="center",
                      color=col, fontsize=max(5, 9-n_rows), fontweight="bold")
    ax_c.set_xlim(0, sW); ax_c.set_ylim(sH, 0)

    # ── Panel D: stats table ───────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(BG2)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    for sp in ax_d.spines.values(): sp.set_edgecolor(BG3)
    ax_d.set_title("D — Tiling Parameters & Processing Stats",
                   color=TEXT, fontsize=9, pad=6)

    rows = [
        ("Input size",         f"{W} × {H} px"),
        ("Output size",        f"{sW} × {sH} px"),
        ("Scale factor",       f"{sW // max(W,1)}×"),
        ("Tile mode",          "Tiled" if tile_size>0 else "Full-image (tile=0)"),
        ("Tile size (core)",   f"{effective_tile} × {effective_tile} px" if tile_size>0 else "N/A"),
        ("Overlap padding",    f"{tile_pad} px" if tile_size>0 else "N/A"),
        ("Grid layout",        f"{n_cols} cols × {n_rows} rows"),
        ("Total tiles",        f"{n_cols * n_rows}"),
        ("Pixels per tile",    f"{effective_tile**2:,}" if tile_size>0 else f"{W*H:,}"),
        ("Padded tile size",   f"{effective_tile+2*tile_pad} × {effective_tile+2*tile_pad} px" if tile_size>0 else "N/A"),
        ("VRAM saving vs full",f"~{100*(1 - (effective_tile+2*tile_pad)**2 / max(W*H,1)):.0f}%" if tile_size>0 else "0% (full image)"),
    ]

    y = 0.95
    for label, val in rows:
        ax_d.text(0.04, y, label, transform=ax_d.transAxes,
                  color=TEXT2, fontsize=9, va="top")
        ax_d.text(0.55, y, val, transform=ax_d.transAxes,
                  color=GOLD if label in ("Total tiles","VRAM saving vs full") else TEXT,
                  fontsize=9, va="top", fontweight="bold")
        y -= 0.085

    ax_d.text(0.5, 0.03,
        "Tiling prevents GPU OOM errors on large images.\n"
        "Overlap padding eliminates seam artefacts at tile edges.",
        transform=ax_d.transAxes, ha="center", color=CYAN, fontsize=8,
        style="italic", va="bottom")

    fig.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close(fig)

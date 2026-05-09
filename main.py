"""
main.py — Image Decoding Pipeline Orchestrator

Entry point and controller for the entire pipeline.

This file owns:
    - CLI argument parsing (argparse)
    - Input validation before any processing starts
    - Calling loader → grid → interpolation → saver in order
    - Progress reporting to the terminal
    - Optional compare-mode visualization

This file owns NO algorithmic logic — no math, no array operations, no
file I/O beyond what argparse/pathlib do natively.

Usage examples:
    python main.py --input photo.jpg
    python main.py --input photo.jpg --method bicubic --scale 4
    python main.py --input photo.jpg --compare --visualize
    python main.py --input photo.jpg --scale 2 --quiet
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows — prevents UnicodeEncodeError when the
# default terminal codec (cp1252) can't encode symbols like checkmarks.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np

import loader
import grid
import interpolation
import saver
import graphs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_STAGES   = 5
VALID_SCALES   = [2, 4, 8]
VALID_METHODS  = ["bicubic", "lanczos", "realesrgan"]
VALID_LANCZOS_A = [2, 3]


# ---------------------------------------------------------------------------
# Progress-reporting helpers
# ---------------------------------------------------------------------------

# Module-level quiet flag — set once in run_pipeline, read by helpers.
_quiet: bool = False


def print_stage(stage_num: int, message: str) -> None:
    """Print a [N/M] stage header. Silent when --quiet is active."""
    if not _quiet:
        print(f"\n[{stage_num}/{TOTAL_STAGES}] {message}...")


def print_result(label: str, value: str) -> None:
    """Print a '>> Label : Value' completion line. Silent when --quiet."""
    if not _quiet:
        print(f"      >> {label:<12} :  {value}")


def print_error(message: str) -> None:
    """Print an 'ERROR: Message' line and exit. Never silenced by --quiet."""
    print(f"\n      ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def print_note(message: str) -> None:
    """Print a neutral informational note. Silent when --quiet."""
    if not _quiet:
        print(f"      NOTE: {message}")


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Build the argparse parser and return the parsed namespace.

    All defaults are chosen to produce good results for a first-time user
    running the script without reading documentation.
    """

    DESCRIPTION = """\
╔══════════════════════════════════════════════════════════════════════╗
║             ANTIGRAVITY — Image Enhancement Pipeline               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Reconstruct high-resolution images from low-res / pixelated       ║
║  sources using classical interpolation or AI super-resolution.     ║
║                                                                    ║
║  Supported methods:                                                ║
║    • lanczos    — Lanczos sinc interpolation (classical, fast)      ║
║    • bicubic    — Bicubic spline interpolation (classical, fast)    ║
║    • realesrgan — Real-ESRGAN deep neural network (AI, slow/GPU)   ║
║                                                                    ║
║  Backends:                                                         ║
║    • local  — runs on this machine (CPU or GPU)                    ║
║    • colab  — offloads AI work to a Google Colab GPU               ║
║    • remote — offloads to a Hugging Face Space                     ║
╚══════════════════════════════════════════════════════════════════════╝"""

    EPILOG = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Basic upscale (Lanczos 2x, default):
    python main.py -i photo.jpg

  Bicubic 4x upscale:
    python main.py -i photo.jpg -m bicubic -s 4

  AI upscale with Real-ESRGAN (local GPU/CPU):
    python main.py -i photo.jpg -m realesrgan -s 4

  AI upscale via Colab GPU (no local GPU needed):
    python main.py -i photo.jpg -s 4 --backend colab --remote-url https://xxxx.gradio.live

  Full AI visualization suite (64 filters, 23 blocks, FFT, radar):
    python main.py -i photo.jpg --analyze-esrgan -s 4

  Full AI viz via Colab GPU (enhancement + features on Colab, plots locally):
    python main.py -i photo.jpg --analyze-esrgan -s 4 --backend colab --remote-url https://xxxx.gradio.live

  DSP analysis (kernel shapes, weighted sums, frequency response):
    python main.py -i photo.jpg --analyze-dsp

  Compare Bicubic vs Lanczos side-by-side:
    python main.py -i photo.jpg --compare

  Compare with visualizations + sharpening:
    python main.py -i photo.jpg --compare --visualize --sharpen

  Quiet mode (suppress progress, show only result path):
    python main.py -i photo.jpg -q

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLAB GPU WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  If you don't have a local GPU, use Google Colab for free:

  1. Open colab.google.com → new notebook → Runtime → T4 GPU
  2. Cell 1: !pip install -q gradio torch torchvision basicsr-fixed realesrgan
  3. Upload colab_worker.py to Colab's Files panel (folder icon on left)
  4. Cell 2: !python colab_worker.py
     → Cell stays spinning (that means GPU server is alive)
     → Copy the https://xxxx.gradio.live URL from the output
  5. On your PC, run:
     python main.py -i photo.jpg --analyze-esrgan -s 4 \\
         --backend colab --remote-url https://PASTE_URL_HERE.gradio.live

  NOTE: Your image is sent to Colab automatically over the internet.
        You do NOT need to upload images to Colab manually.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Standard upscale:
    output/<method>_<scale>x_<filename>.png

  --analyze-esrgan generates (in output/realesrgan/<name>/):
    00_enhanced_output.png        — the 4x upscaled image
    01_filter_responses_64.png    — 64 first-layer filter activations
    02_block_progression_23.png   — 23 RRDB block energy progression
    03_frequency_before_after.png — 2D FFT: input vs output vs new
    04_new_frequencies_generated.png — radial profile + spatial diff
    05_radar_summary.png          — 8-band frequency radar chart
    06_tiling_grid_diagram.png    — tile grid + processing stats

  --analyze-dsp generates (in output/dsp_analysis/):
    kernel shape, weighted sum, frequency response, diff map, radar

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Real-ESRGAN on CPU is VERY slow (10+ min for a 512px image).
    Use --backend colab for free GPU access.
  • --analyze-esrgan forces method=realesrgan regardless of --method.
  • --compare only applies to classical methods (bicubic vs lanczos).
  • Colab gradio.live URLs expire after ~72 hours. Re-run the notebook
    to get a new URL.
  • The web UI (python app.py) provides a browser interface for all
    the same features.
"""

    parser = argparse.ArgumentParser(
        prog="main.py",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )

    # ── Core arguments ────────────────────────────────────────────────────
    core = parser.add_argument_group("Core Options")
    core.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Path to the input image. Supports PNG, JPEG, BMP, TIFF, WEBP.",
    )
    core.add_argument(
        "--output", "-o",
        default="./output",
        metavar="DIR",
        help="Output directory. Created if missing.  [default: ./output]",
    )
    core.add_argument(
        "--scale", "-s",
        type=int,
        default=2,
        choices=VALID_SCALES,
        metavar="N",
        help=f"Upscale factor. Valid: {VALID_SCALES}.  [default: 2]",
    )

    # ── Enhancement method ────────────────────────────────────────────────
    method_grp = parser.add_argument_group("Enhancement Method")
    method_grp.add_argument(
        "--method", "-m",
        default="lanczos",
        metavar="METHOD",
        help=(
            f"Interpolation method. Valid: {VALID_METHODS}.\n"
            "  lanczos    = windowed sinc (sharpest classical method)\n"
            "  bicubic    = cubic spline (smoother, fewer artifacts)\n"
            "  realesrgan = AI super-resolution (best quality, needs GPU)\n"
            "[default: lanczos]"
        ),
    )
    method_grp.add_argument(
        "--lanczos-a",
        type=int,
        default=3,
        dest="lanczos_a",
        metavar="A",
        help="Lanczos window size (2 or 3). Only used with --method lanczos.  [default: 3]",
    )
    method_grp.add_argument(
        "--sharpen",
        action="store_true",
        help="Apply UnsharpMask after upscaling to restore edge crispness.",
    )
    method_grp.add_argument(
        "--tile",
        type=int,
        default=0,
        metavar="PX",
        help=(
            "Tile size for Real-ESRGAN VRAM management (pixels).\n"
            "  0   = process entire image at once (fastest, needs most VRAM)\n"
            "  128 = small tiles (low VRAM, slower)\n"
            "  256 = medium tiles (balanced)\n"
            "  512 = large tiles (fast, needs more VRAM)\n"
            "[default: 0]"
        ),
    )
    method_grp.add_argument(
        "--face-enhance",
        action="store_true",
        dest="face_enhance",
        help=(
            "Enable GFPGAN face restoration after Real-ESRGAN upscaling.\n"
            "Improves facial details (eyes, teeth, skin) in portrait photos.\n"
            "Requires gfpgan package on the backend."
        ),
    )

    # ── Backend / remote ──────────────────────────────────────────────────
    backend_grp = parser.add_argument_group(
        "Backend (where AI processing runs)",
        "Controls whether processing happens locally or on a remote GPU.\n"
        "Only relevant for realesrgan and --analyze-esrgan."
    )
    backend_grp.add_argument(
        "--backend",
        choices=["local", "remote", "colab"],
        default="local",
        help=(
            "Where to run AI processing.\n"
            "  local  = this machine's CPU/GPU  [default]\n"
            "  colab  = Google Colab GPU (needs --remote-url)\n"
            "  remote = Hugging Face Space"
        ),
    )
    backend_grp.add_argument(
        "--remote-url",
        default="",
        metavar="URL",
        help=(
            "Public URL of the Colab worker or HF Space.\n"
            "  For Colab: the https://xxxx.gradio.live URL from colab_worker.py\n"
            "  For HF:    your HF Space URL (e.g. username/space-name)"
        ),
    )

    # ── Visualization & analysis ──────────────────────────────────────────
    viz_grp = parser.add_argument_group(
        "Visualization & Analysis",
        "Generate diagnostic plots and technical analysis."
    )
    viz_grp.add_argument(
        "--analyze-esrgan",
        action="store_true",
        dest="analyze_esrgan",
        help=(
            "Run Real-ESRGAN + generate full visualization suite:\n"
            "  • 64 first-layer filter response maps (classified by type)\n"
            "  • 23 RRDB block activation progression + energy deltas\n"
            "  • 2D FFT frequency analysis (before / after / new)\n"
            "  • Radial frequency profile + spatial diff heatmap\n"
            "  • 8-band frequency radar chart with %% gain labels\n"
            "  • Tiling grid diagram with processing stats\n"
            "Saves all plots to output/realesrgan/<image_name>/"
        ),
    )
    viz_grp.add_argument(
        "--analyze-dsp",
        action="store_true",
        help=(
            "Run DSP analysis suite for classical methods:\n"
            "  • Kernel shape visualization\n"
            "  • Weighted sum computation diagram\n"
            "  • Frequency domain response\n"
            "  • Spatial difference map\n"
            "  • Comparative radar chart"
        ),
    )
    viz_grp.add_argument(
        "--visualize",
        action="store_true",
        help="Show animated grid-mapping and interpolation visualizations inline.",
    )
    viz_grp.add_argument(
        "--compare",
        action="store_true",
        help="Run both Bicubic and Lanczos, save both, show side-by-side comparison.",
    )

    # ── Output control ────────────────────────────────────────────────────
    out_grp = parser.add_argument_group("Output Control")
    out_grp.add_argument(
        "--save-channels",
        action="store_true",
        dest="save_channels",
        help="Save R, G, B channels as separate grayscale PNGs (debug mode).",
    )
    out_grp.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all progress output except the final saved path.",
    )

    return parser.parse_args()



# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_args(args: argparse.Namespace) -> bool:
    """
    Validate all CLI arguments before any pipeline work starts.

    Fails fast on the first problem with a clear human-readable message.
    Either returns True or calls print_error (which calls sys.exit(1)).

    Args:
        args: Parsed argparse namespace.

    Returns:
        True — always, if execution reaches the return statement.
    """
    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"Input file not found: {input_path.resolve()}")
    if not input_path.is_file():
        print_error(f"Input path is not a file: {input_path.resolve()}")

    if args.scale not in VALID_SCALES:
        print_error(
            f"Invalid --scale value '{args.scale}'. "
            f"Valid options: {VALID_SCALES}"
        )

    method = args.method.lower()
    if method not in VALID_METHODS:
        print_error(
            f"Invalid --method '{args.method}'. "
            f"Valid options: {VALID_METHODS}"
        )
    args.method = method   # normalise to lowercase in-place

    if method == "lanczos" and args.lanczos_a not in VALID_LANCZOS_A:
        print_error(
            f"Invalid --lanczos-a value '{args.lanczos_a}'. "
            f"Valid options: {VALID_LANCZOS_A}"
        )

    if args.compare and not args.quiet:
        print_note("--compare is set: --method will be ignored. Both algorithms will run.")

    return True


# ---------------------------------------------------------------------------
# Compare visualization (lives here — simple imshow, no complex animation)
# ---------------------------------------------------------------------------

def show_compare_visualization(
    bicubic_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    lanczos_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    scale_factor: int,
    path_bicubic: str,
    path_lanczos: str,
) -> None:
    """
    Side-by-side comparison: Bicubic | Difference Map | Lanczos.

    The difference map (center panel) shows absolute per-pixel difference
    across all channels, colourised with a heat map — brighter means the
    two algorithms disagree more there.

    Args:
        bicubic_arrays: (r, g, b) float64 filled arrays from bicubic pass.
        lanczos_arrays: (r, g, b) float64 filled arrays from lanczos pass.
        scale_factor:   Used in the figure title.
        path_bicubic:   Saved file path — shown below Bicubic panel.
        path_lanczos:   Saved file path — shown below Lanczos panel.
    """
    bg  = "#0a0a0f"
    txt = "#e0e0ff"
    mono = {"family": "monospace"}

    def _to_rgb(r, g, b):
        """Stack float64 channels into an (H, W, 3) float display array."""
        stacked = np.stack([r, g, b], axis=2)
        return np.clip(stacked, 0.0, 1.0)

    bic_rgb = _to_rgb(*bicubic_arrays)
    lan_rgb = _to_rgb(*lanczos_arrays)
    diff    = np.abs(bic_rgb - lan_rgb).mean(axis=2)   # mean across channels

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=bg)
    fig.suptitle(
        f"BICUBIC  vs  LANCZOS  —  {scale_factor}× upscale",
        color=txt, fontsize=14, fontweight="bold", **mono
    )

    panels = [
        (bic_rgb,  "BICUBIC",     "Blues",  Path(path_bicubic).name),
        (diff,     "DIFF MAP\n(brighter = larger disagreement)",
                                  "inferno", ""),
        (lan_rgb,  "LANCZOS",     "Reds",   Path(path_lanczos).name),
    ]

    for ax, (data, title, cmap, caption) in zip(axes, panels):
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_edgecolor("#00ffff")
            spine.set_alpha(0.25)
        ax.set_title(title, color=txt, fontsize=10, **mono, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

        if data.ndim == 3:
            ax.imshow(data, interpolation="nearest", aspect="equal")
        else:
            ax.imshow(data, cmap=cmap, interpolation="nearest",
                      aspect="equal", vmin=0.0)

        if caption:
            ax.set_xlabel(caption, color="#888888", fontsize=7, **mono)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Execute the full image decoding pipeline.

    Calls loader → grid → interpolation → saver in order, passing data
    between stages.  All progress output is handled here; the modules
    themselves print nothing.
    """
    global _quiet
    _quiet = args.quiet

    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # analyze-esrgan: enhance via any backend, then run full viz suite
    # ------------------------------------------------------------------
    if getattr(args, "analyze_esrgan", False):
        # Force realesrgan method — this flag is specifically for AI analysis
        args.method = "realesrgan"

        import cv2
        import enhancer as _enh
        from graphs.realesrgan_viz import run_visualization_suite
        from PIL import Image as _PILImg
        import numpy as _np

        t0 = time.time()
        out_root = Path(args.output)
        stem     = Path(args.input).stem
        viz_dir  = out_root / "realesrgan" / stem
        viz_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Enhancement ─────────────────────────────────────────────
        if args.backend != "local":
            # Colab GPU does the heavy 4× upscaling
            print_stage(1, f"Colab GPU: Enhancement (4× upscale)")
            import enhancer_remote
            out_path = viz_dir / "00_enhanced_output.png"
            try:
                enhancer_remote.enhance_with_realesrgan(
                    input_path=args.input,
                    output_path=out_path,
                    outscale=args.scale,
                    tile=getattr(args, "tile", 0),
                    face_enhance=getattr(args, "face_enhance", False),
                    remote_url=getattr(args, "remote_url", ""),
                    method=args.method,
                )
                print_result("Colab GPU", f"Enhancement done ({time.time()-t0:.2f}s)")
            except Exception as exc:
                print_error(f"Remote enhancement failed: {exc}")
                return

            # Load model on CPU for diagnostic hooks only (NOT re-doing upscale)
            # This is a lightweight forward pass on the small INPUT image
            print_note("Loading model on CPU for visualization hooks (not re-upscaling) …")
            try:
                import torch, urllib.request
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model_path = Path("models") / "RealESRGAN_x4plus.pth"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                if not model_path.exists():
                    print_note("Downloading RealESRGAN_x4plus.pth …")
                    urllib.request.urlretrieve(
                        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                        str(model_path))
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=4)
                upsampler = RealESRGANer(scale=4, model_path=str(model_path), model=net,
                                         tile=0, tile_pad=10, pre_pad=0, half=False,
                                         device=torch.device("cpu"))
                print_result("Model loaded", "CPU (for viz hooks only)")
            except Exception as exc:
                print_note(f"Model load failed ({exc}) — filter/block plots skipped.")
                upsampler = None


        else:
            # ── Local GPU/CPU ──────────────────────────────────────────────────
            print_stage(1, "Local Real-ESRGAN + Visualization Suite")
            out_path = viz_dir / "00_enhanced_output.png"
            feats    = None
            try:
                import torch, cv2 as _cv2, urllib.request
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model_path = Path("models") / "RealESRGAN_x4plus.pth"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                if not model_path.exists():
                    print_note("Downloading RealESRGAN_x4plus.pth …")
                    urllib.request.urlretrieve(
                        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                        str(model_path))
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=4)
                upsampler = RealESRGANer(scale=4, model_path=str(model_path), model=net,
                                          tile=0, tile_pad=10, pre_pad=0,
                                          half=torch.cuda.is_available())
                img_bgr = _cv2.imread(str(args.input), _cv2.IMREAD_COLOR)
                out_bgr, _ = upsampler.enhance(img_bgr, outscale=args.scale)
                _cv2.imwrite(str(out_path), out_bgr)
                print_result("Local AI", f"Enhancement done ({time.time()-t0:.2f}s)")
            except Exception as exc:
                print_error(f"Local enhancement failed: {exc}"); return

        # ── Step 2: load image arrays ──────────────────────────────────────────
        try:
            img_in_np  = _np.array(_PILImg.open(args.input).convert("RGB"))
            img_out_np = _np.array(_PILImg.open(str(out_path)).convert("RGB"))
        except Exception as exc:
            print_error(f"Could not read images: {exc}"); return

        # ── Step 3: visualization suite ────────────────────────────────────────
        print_stage(2, "Generating all 6 visualization plots …")
        try:
            run_visualization_suite(
                model=upsampler,
                input_np=img_in_np,
                output_np=img_out_np,
                out_dir=viz_dir,
            )
        except Exception as exc:
            import traceback
            print(f"\n  ERROR in visualization suite: {exc}")
            traceback.print_exc()
            print("  Enhanced image was saved. Plots may be partial.")



        print_result("Suite done", f"{time.time()-t0:.2f}s total")
        print_result("Output dir", str(viz_dir))

        # ── Also run DSP analysis if requested alongside --analyze-esrgan ──────
        if getattr(args, "analyze_dsp", False):
            print_stage(3, "DSP analysis …")
            try:
                # DSP analysis needs interpolated channels — run lanczos quickly
                r_n, g_n, b_n, _ = loader.prepare_image(args.input)
                r_sp, g_sp, b_sp, _ = grid.prepare_all_channels(r_n, g_n, b_n, args.scale)
                r_f, g_f, b_f = interpolation.interpolate_all_channels(
                    r_sp, g_sp, b_sp, r_n, g_n, b_n,
                    scale_factor=args.scale, method="lanczos",
                )
                graphs.run_analysis(
                    method="lanczos",
                    r_orig=r_n, g_orig=g_n, b_orig=b_n,
                    r_filled=r_f, g_filled=g_f, b_filled=b_f,
                    scale_factor=args.scale,
                    output_dir=args.output,
                    lanczos_a=3,
                )
                print_result("DSP plots", str(Path(args.output) / "dsp_analysis"))
            except Exception as exc:
                import traceback
                print(f"\n  ERROR in DSP analysis: {exc}")
                traceback.print_exc()

        return


    # ------------------------------------------------------------------
    # Simple remote (no viz)
    # ------------------------------------------------------------------
    if args.backend != "local":
        print_stage(1, f"Remote Execution ({args.backend})")
        t0 = time.time()
        import enhancer_remote
        out_name = Path(args.input).stem + f"_{args.method}_{args.scale}x_remote.png"
        out_path = Path(args.output) / out_name
        try:
            final_path = enhancer_remote.enhance_with_realesrgan(
                input_path=args.input,
                output_path=out_path,
                outscale=args.scale,
                tile=0,
                face_enhance=False,
                remote_url=getattr(args, "remote_url", ""),
                method=args.method,
            )
            print_result("Remote", f"Done ({time.time()-t0:.2f}s)")
            print_result("Saved to", str(final_path))
        except Exception as exc:
            print_error(f"Remote processing failed: {exc}")
        return

    # ------------------------------------------------------------------
    # Simple local AI (no viz)
    # ------------------------------------------------------------------
    if args.backend == "local" and args.method == "realesrgan":
        print_stage(1, "Local AI Execution (Real-ESRGAN)")
        t0 = time.time()
        import enhancer
        out_name = Path(args.input).stem + f"_realesrgan_{args.scale}x.png"
        out_path = Path(args.output) / out_name
        try:
            final_path = enhancer.enhance_with_realesrgan(
                input_path=args.input,
                output_path=out_path,
                outscale=args.scale,
                tile=0,
                face_enhance=False,
            )
            print_result("Local AI", f"Done ({time.time()-t0:.2f}s)")
            print_result("Saved to", str(final_path))
        except Exception as exc:
            print_error(f"Local AI processing failed: {exc}")
        return

    # ------------------------------------------------------------------
    # Stage 1 — Load
    # ------------------------------------------------------------------
    print_stage(1, "Loading image")
    t0 = time.time()
    try:
        r_norm, g_norm, b_norm, image_info = loader.prepare_image(args.input)
    except Exception as exc:
        print_error(f"Failed at loading stage: {exc}")

    print_result(
        "Loaded",
        f"{Path(args.input).name}  "
        f"({image_info['width']} x {image_info['height']}, RGB)  "
        f"[{time.time()-t0:.2f}s]"
    )

    # ------------------------------------------------------------------
    # Stage 2 — Prepare grid
    # ------------------------------------------------------------------
    print_stage(2, "Preparing grid")
    try:
        r_sparse, g_sparse, b_sparse, grid_info = grid.prepare_all_channels(
            r_norm, g_norm, b_norm, args.scale
        )
    except Exception as exc:
        print_error(f"Failed at grid-preparation stage: {exc}")

    print_result(
        "Grid",
        f"{grid_info['new_width']} x {grid_info['new_height']}  "
        f"(scale {args.scale}x,  {grid_info['total_pixels']:,} positions)"
    )
    print_result(
        "Mapped",
        f"{grid_info['filled_pixels']:,} anchors placed  /  "
        f"{grid_info['empty_pixels']:,} gaps to fill"
    )

    if args.visualize:
        for ch_array, sparse, name in [
            (r_norm, r_sparse, "R"),
            (g_norm, g_sparse, "G"),
            (b_norm, b_sparse, "B"),
        ]:
            try:
                grid.visualize_grid_mapping(ch_array, sparse, args.scale, name)
            except Exception as exc:
                print_note(f"Grid visualization skipped ({exc})")

    # ------------------------------------------------------------------
    # Stage 3 — Interpolate
    # ------------------------------------------------------------------
    method_label = (
        f"bicubic + lanczos — compare mode" if args.compare
        else f"{args.method} · {'a='+str(args.lanczos_a) if args.method == 'lanczos' else 'default'}"
    )
    print_stage(3, f"Interpolating  [{method_label}]")

    if args.compare:
        # Run bicubic first, save, then free memory before Lanczos
        t0 = time.time()
        try:
            r_bic, g_bic, b_bic = interpolation.interpolate_all_channels(
                r_sparse, g_sparse, b_sparse,
                r_norm, g_norm, b_norm,
                scale_factor=args.scale,
                method="bicubic",
            )
        except Exception as exc:
            print_error(f"Failed at interpolation stage (bicubic): {exc}")
        print_result("Bicubic", f"done  ({time.time()-t0:.2f}s)")

        t0 = time.time()
        try:
            r_lan, g_lan, b_lan = interpolation.interpolate_all_channels(
                r_sparse, g_sparse, b_sparse,
                r_norm, g_norm, b_norm,
                scale_factor=args.scale,
                method="lanczos",
                lanczos_a=args.lanczos_a,
            )
        except Exception as exc:
            print_error(f"Failed at interpolation stage (lanczos): {exc}")
        print_result("Lanczos", f"done  ({time.time()-t0:.2f}s)")

    else:
        channel_names = ["R", "G", "B"]
        filled_channels: list[np.ndarray] = []

        for ch_sparse, ch_orig, ch_name in zip(
            [r_sparse, g_sparse, b_sparse],
            [r_norm, g_norm, b_norm],
            channel_names,
        ):
            t0 = time.time()
            try:
                if args.method == "bicubic":
                    filled = interpolation.bicubic_interpolate(
                        ch_sparse, ch_orig, args.scale
                    )
                else:
                    filled = interpolation.lanczos_interpolate(
                        ch_sparse, ch_orig, args.scale, a=args.lanczos_a
                    )
            except Exception as exc:
                print_error(
                    f"Failed at interpolation stage (channel {ch_name}): {exc}"
                )

            print_result(f"Channel {ch_name}", f"done  ({time.time()-t0:.2f}s)")
            filled_channels.append(filled)

        r_filled, g_filled, b_filled = filled_channels

        if args.visualize:
            for sparse, filled, name in [
                (r_sparse, r_filled, "R"),
                (g_sparse, g_filled, "G"),
                (b_sparse, b_filled, "B"),
            ]:
                try:
                    interpolation.visualize_interpolation(
                        sparse, filled, name, args.method.capitalize()
                    )
                except Exception as exc:
                    print_note(f"Interpolation visualization skipped ({exc})")

        if args.analyze_dsp:
            try:
                graphs.run_analysis(
                    method=args.method,
                    r_orig=r_norm, g_orig=g_norm, b_orig=b_norm,
                    r_filled=r_filled, g_filled=g_filled, b_filled=b_filled,
                    scale_factor=args.scale,
                    output_dir=args.output,
                    lanczos_a=args.lanczos_a
                )
            except Exception as exc:
                print_error(f"Failed during DSP analysis: {exc}")

    # ------------------------------------------------------------------
    # Stage 4 — Save
    # ------------------------------------------------------------------
    print_stage(4, "Saving")

    if args.compare:
        try:
            path_bicubic = saver.save_all_channels(
                r_bic, g_bic, b_bic,
                input_path=args.input,
                output_dir=args.output,
                method="bicubic",
                scale_factor=args.scale,
                save_channels=args.save_channels,
                sharpen=args.sharpen,
            )
        except Exception as exc:
            print_error(f"Failed at save stage (bicubic): {exc}")

        try:
            path_lanczos = saver.save_all_channels(
                r_lan, g_lan, b_lan,
                input_path=args.input,
                output_dir=args.output,
                method="lanczos",
                scale_factor=args.scale,
                save_channels=args.save_channels,
                sharpen=args.sharpen,
            )
        except Exception as exc:
            print_error(f"Failed at save stage (lanczos): {exc}")

        print_result("Saved [bicubic]", path_bicubic)
        print_result("Saved [lanczos]", path_lanczos)
        saved_path = path_lanczos   # reference for summary

        show_compare_visualization(
            (r_bic, g_bic, b_bic),
            (r_lan, g_lan, b_lan),
            args.scale,
            path_bicubic,
            path_lanczos,
        )

    else:
        try:
            saved_path = saver.save_all_channels(
                r_filled, g_filled, b_filled,
                input_path=args.input,
                output_dir=args.output,
                method=args.method,
                scale_factor=args.scale,
                save_channels=args.save_channels,
                sharpen=args.sharpen,
            )
        except Exception as exc:
            print_error(f"Failed at save stage: {exc}")

        print_result("Saved", saved_path)

    # ------------------------------------------------------------------
    # Stage 5 — Summary (always printed, even in --quiet mode)
    # ------------------------------------------------------------------
    total_time = time.time() - pipeline_start
    was_quiet = _quiet
    _quiet = False                          # force summary to print
    print_stage(5, "Complete")
    print_result("Total time", f"{total_time:.2f}s")
    print_result("Output", saved_path)
    print()
    
    if args.analyze_dsp:
        print("================================================================")
        print("                  DSP ANALYSIS COMMANDS CHEAT SHEET             ")
        print("================================================================")
        print("Basic Upscale:         python main.py -i img.jpg")
        print("Bicubic Method:        python main.py -i img.jpg -m bicubic")
        print("Lanczos Method:        python main.py -i img.jpg -m lanczos")
        print("Specific Scale (4x):   python main.py -i img.jpg -s 4")
        print("Run DSP Analysis:      python main.py -i img.jpg --analyze-dsp")
        print("Compare Both Methods:  python main.py -i img.jpg --compare")
        print("================================================================")
        print()
        
    _quiet = was_quiet


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate_args(args)
    run_pipeline(args)


if __name__ == "__main__":
    main()

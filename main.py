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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_STAGES   = 5
VALID_SCALES   = [2, 4, 8]
VALID_METHODS  = ["bicubic", "lanczos"]
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
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Image Decoding Pipeline — reconstruct a high-resolution image "
            "from a pixelated (undersampled) source using Bicubic or Lanczos "
            "interpolation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --input photo.jpg\n"
            "  python main.py --input photo.jpg --method bicubic --scale 4\n"
            "  python main.py --input photo.jpg --compare --visualize\n"
            "  python main.py --input photo.jpg --scale 2 --quiet"
        ),
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Path to the input (degraded) image file. Supports PNG, JPEG, BMP, TIFF.",
    )
    parser.add_argument(
        "--output", "-o",
        default="./output",
        metavar="DIR",
        help="Output directory. Created automatically if missing.  [default: ./output]",
    )
    parser.add_argument(
        "--scale", "-s",
        type=int,
        default=2,
        choices=VALID_SCALES,
        metavar="SCALE",
        help=f"Integer upscale factor. Valid: {VALID_SCALES}.  [default: 2]",
    )
    parser.add_argument(
        "--method", "-m",
        default="lanczos",
        metavar="METHOD",
        help=f"Interpolation algorithm. Valid: {VALID_METHODS}.  [default: lanczos]",
    )
    parser.add_argument(
        "--lanczos-a",
        type=int,
        default=3,
        dest="lanczos_a",
        metavar="A",
        help="Lanczos window size (2 or 3). Ignored when --method bicubic.  [default: 3]",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show animated grid-mapping and interpolation visualizations inline.",
    )
    parser.add_argument(
        "--save-channels",
        action="store_true",
        dest="save_channels",
        help="Also save R, G, B as separate grayscale PNGs (debug mode).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Run both Bicubic and Lanczos, save both outputs, and show a "
            "side-by-side comparison. Overrides --method."
        ),
    )
    parser.add_argument(
        "--sharpen",
        action="store_true",
        help=(
            "Apply UnsharpMask after interpolation to recover edge crispness. "
            "Recommended — bicubic/Lanczos upscaling inherently softens edges."
        ),
    )
    parser.add_argument(
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

    Error handling: every stage is wrapped in its own try/except so the
    failure message always names the stage that broke.
    """
    global _quiet
    _quiet = args.quiet

    pipeline_start = time.time()

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

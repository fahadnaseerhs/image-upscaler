"""
app.py — Flask Web Server for the Image Decoding Pipeline

Wraps the pipeline as a web API with Server-Sent Events (SSE) for live
progress streaming. Serves the HTML frontend at GET /.

Routes:
    GET  /                          → serve index.html
    POST /api/process               → run pipeline, stream SSE progress
    GET  /api/output/<filename>     → serve a saved output image
    GET  /api/grid-data             → return latest grid snapshot (for 3D viz)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

# Make sure the pipeline modules are importable from this directory
sys.path.insert(0, str(Path(__file__).parent))

import loader
import grid as grid_module
import interpolation
import saver
import enhancer

app = Flask(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Latest grid snapshot for the 3D visualization — updated after each run
_latest_grid_data: dict | None = None
_latest_grid_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _build_sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _run_pipeline(
    input_path: Path,
    scale: int,
    method: str,
    lanczos_a: int,
    sharpen: bool,
    compare: bool,
    realesrgan_tile: int,
    face_enhance: bool,
    emit,            # callable(event, data) — sends SSE to the client
) -> None:
    """
    Run the full pipeline and emit SSE progress events.
    Called from a generator; emit() queues strings into the SSE stream.
    """
    global _latest_grid_data

    try:
        # ── Stage 1: Load ────────────────────────────────────────────────
        emit("stage", {"stage": 1, "label": "Loading image"})
        t0 = time.time()
        r_norm, g_norm, b_norm, image_info = loader.prepare_image(input_path)
        emit("progress", {
            "stage": 1,
            "detail": f"{image_info['width']} × {image_info['height']} px",
            "elapsed": round(time.time() - t0, 2),
        })

        # ── Stage 2: Grid ────────────────────────────────────────────────
        emit("stage", {"stage": 2, "label": "Preparing grid"})
        t0 = time.time()
        r_sparse, g_sparse, b_sparse, grid_info = grid_module.prepare_all_channels(
            r_norm, g_norm, b_norm, scale
        )
        emit("progress", {
            "stage": 2,
            "detail": f"{grid_info['new_width']} × {grid_info['new_height']} · "
                      f"{grid_info['filled_pixels']:,} anchors",
            "elapsed": round(time.time() - t0, 2),
        })

        # ── Build grid_data snapshot for 3D visualization ────────────────
        viz_tile = 12  # sample tile size for 3D visualization snapshot
        import numpy as np
        H, W = image_info["height"], image_info["width"]
        tr = min(H, viz_tile)
        tc = min(W, viz_tile)
        tile_r = r_norm[:tr, :tc]
        tile_g = g_norm[:tr, :tc]
        tile_b = b_norm[:tr, :tc]
        pixels = []
        for row in range(tr):
            for col in range(tc):
                pixels.append({
                    "orig": [col, row],
                    "mapped": [col * scale, row * scale],
                    "r": round(float(tile_r[row, col]), 3),
                    "g": round(float(tile_g[row, col]), 3),
                    "b": round(float(tile_b[row, col]), 3),
                })
        grid_data = {
            "scale": scale,
            "orig_w": tc, "orig_h": tr,
            "new_w": tc * scale, "new_h": tr * scale,
            "pixels": pixels,
        }
        with _latest_grid_lock:
            _latest_grid_data = grid_data
        emit("griddata", grid_data)

        saved_paths = {}
        if method == "realesrgan" and not compare:
            # ── Stage 3: AI enhancement ────────────────────────────────────
            emit("stage", {"stage": 3, "label": "AI enhancement"})
            t0 = time.time()
            emit("channel", {"method": method, "channel": "all"})
            out_name = saver.generate_filename(
                input_path=input_path,
                output_dir=str(OUTPUT_DIR),
                method=method,
                scale_factor=scale,
            )
            out_path = OUTPUT_DIR / out_name
            final_path = enhancer.enhance_with_realesrgan(
                input_path=input_path,
                output_path=out_path,
                outscale=scale,
                tile=realesrgan_tile,
                face_enhance=face_enhance,
            )
            saved_paths[method] = Path(final_path).name
            emit("progress", {
                "stage": 3,
                "detail": "Real-ESRGAN done",
                "elapsed": round(time.time() - t0, 2),
            })

            # ── Stage 4: Save (already saved by enhancer) ────────────────
            emit("stage", {"stage": 4, "label": "Saving"})
            emit("progress", {
                "stage": 4,
                "detail": "Written to disk",
                "elapsed": 0.0,
            })
        else:
            # ── Stage 3: Interpolate ─────────────────────────────────────
            emit("stage", {"stage": 3, "label": "Interpolating"})
            results = {}
            if compare:
                for meth in ("bicubic", "lanczos"):
                    t0 = time.time()
                    emit("channel", {"method": meth, "channel": "all"})
                    r_f, g_f, b_f = interpolation.interpolate_all_channels(
                        r_sparse, g_sparse, b_sparse,
                        r_norm, g_norm, b_norm,
                        scale_factor=scale, method=meth, lanczos_a=lanczos_a,
                    )
                    results[meth] = (r_f, g_f, b_f)
                    emit("progress", {
                        "stage": 3,
                        "detail": f"{meth} done",
                        "elapsed": round(time.time() - t0, 2),
                    })
            else:
                channels = [r_sparse, g_sparse, b_sparse]
                originals = [r_norm, g_norm, b_norm]
                filled = []
                for ch_sparse, ch_orig, ch_name in zip(channels, originals, "RGB"):
                    t0 = time.time()
                    emit("channel", {"method": method, "channel": ch_name})
                    if method == "bicubic":
                        f = interpolation.bicubic_interpolate(ch_sparse, ch_orig, scale)
                    else:
                        f = interpolation.lanczos_interpolate(ch_sparse, ch_orig, scale, a=lanczos_a)
                    filled.append(f)
                    emit("progress", {
                        "stage": 3,
                        "detail": f"Channel {ch_name} done",
                        "elapsed": round(time.time() - t0, 2),
                    })
                results[method] = tuple(filled)

            # ── Stage 4: Save ────────────────────────────────────────────
            emit("stage", {"stage": 4, "label": "Saving"})
            t0 = time.time()
            for meth, (r_f, g_f, b_f) in results.items():
                path = saver.save_all_channels(
                    r_f, g_f, b_f,
                    input_path=input_path,
                    output_dir=str(OUTPUT_DIR),
                    method=meth,
                    scale_factor=scale,
                    sharpen=sharpen,
                )
                saved_paths[meth] = Path(path).name
            emit("progress", {
                "stage": 4,
                "detail": "Written to disk",
                "elapsed": round(time.time() - t0, 2),
            })

        # ── Stage 5: Done ────────────────────────────────────────────────
        emit("stage", {"stage": 5, "label": "Complete"})
        emit("done", {
            "outputs": saved_paths,
            "image_info": image_info,
            "grid_info": {k: v for k, v in grid_info.items()},
        })

    except Exception as exc:
        emit("error", {"message": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/output/<filename>")
def serve_output(filename: str):
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route("/api/input/<filename>")
def serve_input(filename: str):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/api/grid-data")
def get_grid_data():
    with _latest_grid_lock:
        if _latest_grid_data is None:
            return jsonify({"error": "No grid data yet"}), 404
        return jsonify(_latest_grid_data)


@app.route("/api/hardware")
def get_hardware():
    """Return lightweight CPU/GPU info for the UI."""
    info: dict = {"has_torch": False, "device": "unknown", "cuda": False}
    try:
        import torch  # type: ignore

        info["has_torch"] = True
        info["torch_version"] = getattr(torch, "__version__", "unknown")

        if torch.cuda.is_available():
            info["cuda"] = True
            info["device"] = "cuda"
        else:
            try:
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    info["device"] = "mps"
                else:
                    info["device"] = "cpu"
            except Exception:
                info["device"] = "cpu"
    except Exception:
        info["device"] = "cpu"

    return jsonify(info)


@app.route("/api/process", methods=["POST"])
def process():
    """
    Accepts a multipart/form-data POST with:
        file      — the image file
        scale     — int (2, 4, 8)
        method    — "bicubic", "lanczos", or "realesrgan"
        lanczos_a — int (2 or 3)
        sharpen   — "true" / "false"
        compare   — "true" / "false"

    Returns a text/event-stream (SSE) response streaming progress.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename or not _allowed(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    scale     = int(request.form.get("scale", 2))
    method    = request.form.get("method", "lanczos").lower()
    lanczos_a = int(request.form.get("lanczos_a", 3))
    sharpen   = request.form.get("sharpen", "false").lower() == "true"
    compare   = request.form.get("compare", "false").lower() == "true"
    tile      = int(request.form.get("tile", 0))
    face_enhance = request.form.get("face_enhance", "false").lower() == "true"
    if method not in {"bicubic", "lanczos", "realesrgan"}:
        return jsonify({"error": "Invalid method"}), 400
    if method == "realesrgan":
        compare = False

    ext = Path(f.filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    input_path = UPLOAD_DIR / unique_name
    f.save(str(input_path))

    # SSE generator — pipeline runs synchronously inside the request thread
    def generate():
        queue: list[str] = []

        def emit(event: str, data: dict):
            queue.append(_build_sse(event, data))

        _run_pipeline(
            input_path=input_path,
            scale=scale,
            method=method,
            lanczos_a=lanczos_a,
            sharpen=sharpen,
            compare=compare,
            realesrgan_tile=tile,
            face_enhance=face_enhance,
            emit=emit,
        )

        for msg in queue:
            yield msg

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  Image Decoding Pipeline — Web UI")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000, threaded=True)

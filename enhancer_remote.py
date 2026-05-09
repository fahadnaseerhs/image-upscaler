"""
enhancer_remote.py — Remote AI Enhancement via Colab GPU Worker or HF Space

Priority order:
  1. COLAB_WORKER_URL env var  (your Colab gradio.live URL — GPU, fastest)
  2. HF_SPACE_URL env var      (Hugging Face Space — CPU, fallback)
  3. Local enhancer.py         (on-device, last resort)

Set the Colab URL:
    Windows:  set COLAB_WORKER_URL=https://xxxx.gradio.live
    Linux/Mac: export COLAB_WORKER_URL=https://xxxx.gradio.live

Or pass it directly in the UI's Remote URL field (stored in app.py).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Config — read from environment variables
# ---------------------------------------------------------------------------

# Your Colab worker URL (set after launching colab_worker.ipynb)
COLAB_WORKER_URL: str = os.environ.get("COLAB_WORKER_URL", "").strip()

# Fallback HF Space URL
HF_SPACE_URL: str = os.environ.get("HF_SPACE_URL", "hsfahadnaseer/image-enhancer").strip()

# Timeout in seconds (Colab GPU is fast, but cold start can take ~30s)
REMOTE_TIMEOUT: int = int(os.environ.get("HF_REMOTE_TIMEOUT", "180"))


# ---------------------------------------------------------------------------
# Public function — same signature as enhancer.enhance_with_realesrgan
# ---------------------------------------------------------------------------

def enhance_with_realesrgan(
    input_path: str | Path,
    output_path: str | Path,
    outscale: int = 4,
    tile: int = 0,
    face_enhance: bool = False,
    progress_callback=None,
    remote_url: str | None = None,
    method: str = "realesrgan",
) -> str:
    """
    Enhance an image via remote GPU worker (Colab) or HF Space.

    Args:
        input_path:        Source image file path.
        output_path:       Destination image file path.
        outscale:          Upscale factor (2, 4, 8).
        tile:              Tile size for VRAM control (0 = full image).
        face_enhance:      Reserved — passed to worker but not implemented.
        progress_callback: fn(current, total, dt, gpu_util) for live progress.
        remote_url:        Override URL (from UI field). Takes highest priority.
        method:            'realesrgan', 'bicubic', or 'lanczos'.

    Returns:
        Absolute output path string.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Determine which backend to use ────────────────────────────────────
    # Priority: explicit remote_url > COLAB_WORKER_URL > HF_SPACE_URL > local
    target_url = (remote_url or COLAB_WORKER_URL or HF_SPACE_URL or "").strip()

    if not target_url:
        print("[enhancer_remote] No remote URL configured — running locally")
        return _fallback_local(input_path, output_path, outscale, tile,
                               face_enhance, progress_callback)

    try:
        from gradio_client import Client, handle_file
    except ImportError:
        print("[enhancer_remote] gradio_client not installed — running locally")
        print("  Fix: pip install gradio-client")
        return _fallback_local(input_path, output_path, outscale, tile,
                               face_enhance, progress_callback)

    # ── Detect backend type from URL ──────────────────────────────────────
    is_colab = "gradio.live" in target_url or "ngrok" in target_url
    is_hf    = "huggingface" in target_url or ("/" in target_url and "." not in target_url.split("/")[0])

    backend_label = "Colab GPU" if is_colab else "HF Space"
    print(f"[enhancer_remote] Connecting to {backend_label}: {target_url}")

    try:
        t0 = time.time()
        client = Client(target_url, verbose=False)

        # ── Submit job with correct argument signature ─────────────────────
        if is_colab:
            # Colab worker (colab_worker.ipynb) expects these exact args
            job = client.submit(
                image        = handle_file(str(input_path)),
                method       = method,
                scale        = outscale,
                tile         = tile,
                face_enhance = face_enhance,
                api_name     = "/enhance",
            )
        else:
            # HF Space (hf_space/app.py) has a simpler 3-arg signature
            job = client.submit(
                input_image  = handle_file(str(input_path)),
                outscale     = outscale,
                face_enhance = face_enhance,
                api_name     = "/enhance",
            )

        # ── Poll for progress ──────────────────────────────────────────────
        import json
        while not job.done():
            status = job.status()
            if status.progress_data and progress_callback:
                for p in status.progress_data:
                    if p.desc:
                        try:
                            data = json.loads(p.desc)
                            progress_callback(
                                data.get("current", 0),
                                data.get("total",   1),
                                data.get("dt",      0.0),
                                data.get("cpu",     0),   # GPU util % from Colab
                            )
                        except Exception:
                            pass
            time.sleep(0.4)

        elapsed = time.time() - t0
        print(f"[enhancer_remote] ✅ Done in {elapsed:.1f}s via {backend_label}")

        # ── Retrieve result ────────────────────────────────────────────────
        outputs = job.outputs()
        if not outputs:
            raise RuntimeError("Remote job returned no output.")

        result = outputs[0]
        result_path = Path(result)

        if result_path.exists():
            img = Image.open(result_path).convert("RGB")
            img.save(str(output_path), format="PNG")
            try:
                result_path.unlink()
            except OSError:
                pass
        else:
            raise FileNotFoundError(f"Result file not found: {result_path}")

        return str(output_path.resolve())

    except Exception as exc:
        print(f"[enhancer_remote] ❌ Remote failed ({backend_label}): {exc}")
        print("[enhancer_remote] Falling back to local enhancer...")
        return _fallback_local(input_path, output_path, outscale, tile,
                               face_enhance, progress_callback)


# ---------------------------------------------------------------------------
# Status check — called by app.py /api/hardware
# ---------------------------------------------------------------------------

def get_remote_status() -> dict:
    """
    Return info about which remote backend is configured.
    Used by the UI to show the backend label.
    """
    if COLAB_WORKER_URL:
        return {"backend": "colab", "url": COLAB_WORKER_URL, "label": "Colab GPU Worker"}
    elif HF_SPACE_URL:
        return {"backend": "hf_space", "url": HF_SPACE_URL, "label": "HF Space (CPU)"}
    else:
        return {"backend": "local", "url": "", "label": "Local (CPU)"}


def is_remote_available() -> bool:
    """Quick reachability check for the configured remote backend."""
    url = COLAB_WORKER_URL or HF_SPACE_URL
    if not url:
        return False
    try:
        from gradio_client import Client
        Client(url, verbose=False)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Local fallback
# ---------------------------------------------------------------------------

def _fallback_local(
    input_path: Path,
    output_path: Path,
    outscale: int,
    tile: int,
    face_enhance: bool,
    progress_callback=None,
) -> str:
    import enhancer
    return enhancer.enhance_with_realesrgan(
        input_path=input_path,
        output_path=output_path,
        outscale=outscale,
        tile=tile,
        face_enhance=face_enhance,
        progress_callback=progress_callback,
    )


# ---------------------------------------------------------------------------
# New: enhance + extract features in one Colab call
# ---------------------------------------------------------------------------

def enhance_and_extract(
    input_path: str | Path,
    output_path: str | Path,
    npz_path: str | Path,
    outscale: int = 4,
    tile: int = 0,
    remote_url: str | None = None,
) -> dict:
    """
    Call the Colab worker's /enhance_and_extract endpoint.

    Colab GPU runs:
      1. Real-ESRGAN enhancement
      2. Forward pass with hooks → captures conv_first + 23 block features

    Returns dict with keys 'conv_first', 'block_0'..'block_22' (numpy arrays).
    The enhanced image is saved to output_path.
    The raw npz is saved to npz_path (for debugging).
    """
    import numpy as np
    input_path  = Path(input_path)
    output_path = Path(output_path)
    npz_path    = Path(npz_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_url = (remote_url or COLAB_WORKER_URL or "").strip()
    if not target_url:
        raise RuntimeError("No Colab URL — set --remote-url or COLAB_WORKER_URL")

    from gradio_client import Client, handle_file
    print(f"[enhancer_remote] Connecting to Colab GPU: {target_url}")
    t0     = time.time()
    client = Client(target_url, verbose=False)

    job = client.submit(
        image    = handle_file(str(input_path)),
        scale    = outscale,
        tile     = tile,
        api_name = "/enhance_and_extract",
    )

    while not job.done():
        time.sleep(0.5)

    elapsed = time.time() - t0
    outputs = job.outputs()
    if not outputs or len(outputs) < 2:
        raise RuntimeError(f"Expected 2 outputs (image + npz), got {len(outputs or [])}")

    enhanced_file, npz_file = outputs[0], outputs[1]

    # Save enhanced image
    Image.open(enhanced_file).convert("RGB").save(str(output_path), format="PNG")

    # Copy and load NPZ
    import shutil
    shutil.copy(npz_file, str(npz_path))
    feats = dict(np.load(npz_path, allow_pickle=False))

    print(f"[enhancer_remote] ✅ Enhancement + feature extraction done in {elapsed:.1f}s")
    return feats


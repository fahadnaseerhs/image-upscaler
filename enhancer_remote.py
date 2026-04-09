"""
enhancer_remote.py — Remote AI Enhancement via Hugging Face Space

Calls a deployed Hugging Face Space running Real-ESRGAN instead of
running the model locally. Falls back to local enhancer.py if the
remote is unreachable.

Configure via environment variable:
    HF_SPACE_URL  — e.g. "fahadnaseerhs/image-enhancer"
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Set this to your HF Space name, e.g. "fahadnaseerhs/image-enhancer"
HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "hsfahadnaseer/image-enhancer")

# Maximum seconds to wait for the remote API (includes cold start)
REMOTE_TIMEOUT = int(os.environ.get("HF_REMOTE_TIMEOUT", "180"))


# ---------------------------------------------------------------------------
# Public function — same signature as enhancer.enhance_with_realesrgan
# ---------------------------------------------------------------------------

def enhance_with_realesrgan(
    input_path: str | Path,
    output_path: str | Path,
    outscale: int = 4,
    tile: int = 0,           # ignored for remote (Space handles tiling)
    face_enhance: bool = False,
    progress_callback = None,
) -> str:
    """
    Enhance an image via the remote HF Space API.

    Falls back to local enhancer.py if:
      - HF_SPACE_URL is not set
      - gradio_client is not installed
      - The remote API is unreachable or errors out

    Returns:
        Absolute output path as string.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not HF_SPACE_URL:
        print("[enhancer_remote] HF_SPACE_URL not set — falling back to local")
        return _fallback_local(input_path, output_path, outscale, tile, face_enhance, progress_callback)

    try:
        from gradio_client import Client, handle_file
    except ImportError:
        print("[enhancer_remote] gradio_client not installed — falling back to local")
        print("  Install with: pip install gradio-client")
        return _fallback_local(input_path, output_path, outscale, tile, face_enhance, progress_callback)

    try:
        print(f"[enhancer_remote] Connecting to HF Space: {HF_SPACE_URL}")
        t0 = time.time()

        client = Client(HF_SPACE_URL)
        job = client.submit(
            input_image=handle_file(str(input_path)),
            outscale=outscale,
            face_enhance=face_enhance,
            api_name="/enhance",
        )

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
                                data.get("total", 1), 
                                data.get("dt", 0), 
                                data.get("cpu", 0)
                            )
                        except Exception:
                            pass
            time.sleep(0.5)

        elapsed = time.time() - t0
        print(f"[enhancer_remote] Enhancement complete in {elapsed:.1f}s")
        
        err = job.outputs()
        if not err:
            raise RuntimeError("Remote task failed to produce output.")
        result = err[0]

        # result is a path to a temp file on the client side
        result_path = Path(result)
        if result_path.exists():
            # Copy to the desired output path
            img = Image.open(result_path)
            img.save(str(output_path), format="PNG")
            # Clean up the temp file
            try:
                result_path.unlink()
            except OSError:
                pass
        else:
            raise FileNotFoundError(f"Remote result file not found: {result}")

        return str(output_path.resolve())

    except Exception as exc:
        print(f"[enhancer_remote] Remote failed: {exc}")
        print("[enhancer_remote] Falling back to local enhancer")
        return _fallback_local(input_path, output_path, outscale, tile, face_enhance, progress_callback)


def is_remote_available() -> bool:
    """Check if the remote backend is configured and reachable."""
    if not HF_SPACE_URL:
        return False
    try:
        from gradio_client import Client
        client = Client(HF_SPACE_URL)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fallback to local
# ---------------------------------------------------------------------------

def _fallback_local(
    input_path: Path,
    output_path: Path,
    outscale: int,
    tile: int,
    face_enhance: bool,
    progress_callback=None,
) -> str:
    """Run local Real-ESRGAN as a fallback."""
    import enhancer
    return enhancer.enhance_with_realesrgan(
        input_path=input_path,
        output_path=output_path,
        outscale=outscale,
        tile=tile,
        face_enhance=face_enhance,
        progress_callback=progress_callback,
    )

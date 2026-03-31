"""
enhancer.py — Optional AI enhancement using Real-ESRGAN.

This module is intentionally isolated so the rest of the app keeps working
even if AI dependencies are not installed.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def _download_if_missing(model_path: Path, url: str) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return
    urllib.request.urlretrieve(url, str(model_path))


def enhance_with_realesrgan(
    input_path: str | Path,
    output_path: str | Path,
    outscale: int = 4,
    tile: int = 0,
    face_enhance: bool = False,
) -> str:
    """
    Enhance an image with Real-ESRGAN and save it to output_path.

    Args:
        input_path: Source image file path.
        output_path: Destination image file path.
        outscale: Final upscale factor requested by the UI (2/4/8).
        tile: Real-ESRGAN tile size (0 = full image). Smaller tiles reduce
              peak VRAM/RAM usage.
        face_enhance: If True, run GFPGAN on the enhanced output for face
              restoration (requires `pip install gfpgan` + model download).

    Returns:
        Absolute output path as string.

    Raises:
        RuntimeError: If required AI dependencies are unavailable.
    """
    try:
        import cv2
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except Exception as exc:
        raise RuntimeError(
            "Real-ESRGAN dependencies missing. Install with: "
            "pip install torch torchvision realesrgan basicsr opencv-python"
        ) from exc

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_dir = Path(__file__).parent / "models"
    model_path = model_dir / "RealESRGAN_x4plus.pth"
    model_url = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
        "RealESRGAN_x4plus.pth"
    )
    _download_if_missing(model_path, model_url)

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=tile if tile and tile > 0 else 0,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )

    # RealESRGAN works with BGR ndarray via OpenCV.
    pil_input = Image.open(input_path).convert("RGB")
    bgr_input = cv2.cvtColor(np.array(pil_input), cv2.COLOR_RGB2BGR)
    bgr_output, _ = upsampler.enhance(bgr_input, outscale=outscale)
    if face_enhance:
        bgr_output = _enhance_faces_gfpgan(
            bgr_output=bgr_output,
            output_path=output_path,
            outscale=outscale,
        )

    rgb_output = cv2.cvtColor(bgr_output, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb_output).save(output_path, format="PNG")

    return str(output_path.resolve())


def _enhance_faces_gfpgan(
    bgr_output,
    output_path: Path,
    outscale: int,
):
    """
    Optional face restoration with GFPGAN.

    Note: since Real-ESRGAN already upscaled the whole image, we use
    GFPGAN's `upscale=1` so the output keeps the same resolution.
    """
    try:
        import cv2  # noqa: F401
        from gfpgan import GFPGANer
    except Exception as exc:
        raise RuntimeError(
            "Face enhancement requested but GFPGAN is not available. "
            "Install with: pip install gfpgan"
        ) from exc

    model_dir = Path(__file__).parent / "models"
    model_path = model_dir / "GFPGANv1.4.pth"
    model_url = (
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/"
        "GFPGANv1.4.pth"
    )
    _download_if_missing(model_path, model_url)

    # We apply GFPGAN after Real-ESRGAN, so we set upscale=1 to preserve size.
    restorer = GFPGANer(
        model_path=str(model_path),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )

    restored_bgr, _ = restorer.enhance(
        bgr_output,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    return restored_bgr

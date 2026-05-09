"""
enhancer.py — Optional AI enhancement using Real-ESRGAN.

This module is intentionally isolated so the rest of the app keeps working
even if AI dependencies are not installed.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
import sys
import types
import threading

import numpy as np
from PIL import Image

_cache_lock = threading.Lock()
_upsampler_cache: dict[int, object] = {}
_gfpgan_cache: object | None = None


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
    progress_callback = None,
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
    _ensure_compat()
    try:
        import cv2
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

    # Reuse initialized model objects across requests for faster CPU runs.
    upsampler = _get_realesrgan_upsampler(
        model_path=model_path,
        tile=tile if tile and tile > 0 else 0,
    )

    # RealESRGAN works with BGR ndarray via OpenCV.
    pil_input = Image.open(input_path).convert("RGB")
    bgr_input = cv2.cvtColor(np.array(pil_input), cv2.COLOR_RGB2BGR)
    
    # Calculate total tiles for progress tracking
    height, width, _ = bgr_input.shape
    if upsampler.tile_size > 0:
        y_steps = len(range(0, height, upsampler.tile_size))
        x_steps = len(range(0, width, upsampler.tile_size))
        total_tiles = y_steps * x_steps
    else:
        total_tiles = 1

    tile_tracking = {"current": 0, "total": total_tiles}
    original_model = upsampler.model

    class ProgressModelWrapper:
        def __init__(self, model):
            self.model = model
            
        def __call__(self, *args, **kwargs):
            import time
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=None)
            except ImportError:
                cpu = 0.0
                
            t0 = time.time()
            res = self.model(*args, **kwargs)
            dt = time.time() - t0
            
            tile_tracking["current"] += 1
            if progress_callback:
                progress_callback(tile_tracking["current"], tile_tracking["total"], dt, cpu)
            return res
            
        def __getattr__(self, name):
            return getattr(self.model, name)

    try:
        upsampler.model = ProgressModelWrapper(original_model)
        bgr_output, _ = upsampler.enhance(bgr_input, outscale=outscale)
    finally:
        upsampler.model = original_model

    if face_enhance:
        bgr_output = _enhance_faces_gfpgan(
            bgr_output=bgr_output,
            output_path=output_path,
            outscale=outscale,
        )

    rgb_output = cv2.cvtColor(bgr_output, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb_output).save(output_path, format="PNG")

    return str(output_path.resolve())


def _ensure_compat() -> None:
    """
    Provide minimal shims for Python 3.12+ and modern torchvision.
    
    Some dependencies (via basicsr) still rely on modules that have been removed.
    """
    # 1. Provide a distutils shim
    if "distutils.version" not in sys.modules:
        try:
            from setuptools._distutils.version import LooseVersion
        except ImportError:
            try:
                from packaging.version import parse as LooseVersion
            except ImportError:
                pass
        
        if "LooseVersion" in locals():
            distutils_mod = types.ModuleType("distutils")
            version_mod = types.ModuleType("distutils.version")
            version_mod.LooseVersion = LooseVersion
            distutils_mod.version = version_mod
            sys.modules["distutils"] = distutils_mod
            sys.modules["distutils.version"] = version_mod

    # 2. Provide a torchvision.transforms.functional_tensor shim
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        try:
            import torchvision.transforms.functional
            sys.modules["torchvision.transforms.functional_tensor"] = torchvision.transforms.functional
        except ImportError:
            pass


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
    restorer = _get_gfpgan_restorer(model_path=model_path)

    restored_bgr, _ = restorer.enhance(
        bgr_output,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    return restored_bgr


def _get_realesrgan_upsampler(model_path: Path, tile: int):
    """
    Cache RealESRGANer by tile size to avoid repeated heavy initialization.
    This improves latency without changing output quality.
    """
    _ensure_compat()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    key = int(tile)
    with _cache_lock:
        if key in _upsampler_cache:
            return _upsampler_cache[key]

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
            tile=key,
            tile_pad=10,
            pre_pad=0,
            half=False,
        )
        _upsampler_cache[key] = upsampler
        return upsampler


def _get_gfpgan_restorer(model_path: Path):
    """Cache GFPGANer instance to reduce repeated startup overhead."""
    global _gfpgan_cache
    from gfpgan import GFPGANer

    with _cache_lock:
        if _gfpgan_cache is not None:
            return _gfpgan_cache

        _gfpgan_cache = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        return _gfpgan_cache


# ---------------------------------------------------------------------------
# Visualization-enhanced wrapper
# ---------------------------------------------------------------------------

def enhance_with_visualization(
    input_path: str | Path,
    output_dir: Path,
    outscale: int = 4,
    tile: int = 0,
    progress_callback=None,
) -> Path:
    """
    Run Real-ESRGAN on *input_path*, save the enhanced image and the full
    visualization suite under:

        output_dir/realesrgan/{stem}/
            00_enhanced_output.png
            01_filter_responses_64.png
            02_block_progression_23.png
            03_frequency_before_after.png
            04_new_frequencies_generated.png
            05_radar_summary.png

    Returns the path to the enhanced output image.
    """
    import torch
    import cv2
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    input_path  = Path(input_path)
    stem        = input_path.stem
    viz_dir     = output_dir / "realesrgan" / stem
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ---- build model (reuse cache if available) ----
    model_path = Path("models") / "RealESRGAN_x4plus.pth"
    _download_if_missing(
        model_path,
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    )

    with _cache_lock:
        if 4 not in _upsampler_cache:
            net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                          num_block=23, num_grow_ch=32, scale=4)
            _upsampler_cache[4] = RealESRGANer(
                scale=4, model_path=str(model_path), model=net,
                tile=tile, tile_pad=10, pre_pad=0,
                half=torch.cuda.is_available(),
            )
        upsampler = _upsampler_cache[4]

    # ---- read input ----
    img_in_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img_in_bgr is None:
        raise FileNotFoundError(f"Could not read: {input_path}")
    img_in_rgb = cv2.cvtColor(img_in_bgr, cv2.COLOR_BGR2RGB)

    # ---- enhance ----
    out_bgr, _ = upsampler.enhance(img_in_bgr, outscale=outscale)
    img_out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    # ---- save enhanced image ----
    out_img_path = viz_dir / "00_enhanced_output.png"
    cv2.imwrite(str(out_img_path), out_bgr)
    print(f"  Saved enhanced image → {out_img_path}")

    # ---- run visualization suite ----
    try:
        from graphs.realesrgan_viz import run_visualization_suite

        def _viz_cb(step, total, label):
            if progress_callback:
                progress_callback(step, total, 0, 0)
            print(f"  [VIZ {step}/{total}] {label}")

        run_visualization_suite(
            model=upsampler,
            input_np=img_in_rgb,
            output_np=img_out_rgb,
            out_dir=viz_dir,
            progress_cb=_viz_cb,
        )
    except Exception as exc:
        print(f"  WARNING: visualization suite failed ({exc}). Enhanced image still saved.")

    return out_img_path

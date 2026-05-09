"""
colab_worker.py — Antigravity Colab GPU Worker

Upload this file AND the graphs/ folder to Colab, then run:
    !python colab_worker.py

Endpoints:
    /enhance          — standard enhancement (returns image)
    /analyze          — enhancement + all 6 plots on GPU, returns zip file
                        Download the zip from Colab Files panel or via the UI
"""

import os, sys, time, zipfile, tempfile, traceback
import numpy as np
import torch
import cv2
from PIL import Image
from urllib.request import urlretrieve
from pathlib import Path

# ── Compatibility shims — MUST run before any basicsr/realesrgan import ──────
# torchvision removed functional_tensor in newer versions; basicsr still needs it
import types as _types
if "torchvision.transforms.functional_tensor" not in sys.modules:
    try:
        import torchvision.transforms.functional as _tf
        sys.modules["torchvision.transforms.functional_tensor"] = _tf
    except ImportError:
        pass

# distutils.version was removed in Python 3.12; basicsr still imports it
if "distutils.version" not in sys.modules:
    try:
        from setuptools._distutils.version import LooseVersion as _LV
    except ImportError:
        try:
            from packaging.version import parse as _LV
        except ImportError:
            _LV = None
    if _LV is not None:
        _dm = _types.ModuleType("distutils")
        _vm = _types.ModuleType("distutils.version")
        _vm.LooseVersion = _LV
        _dm.version = _vm
        sys.modules["distutils"] = _dm
        sys.modules["distutils.version"] = _vm

print("=" * 60)
print("  Antigravity Colab GPU Worker")
print("=" * 60)

# ── 1. Install dependencies ───────────────────────────────────────
print("\n[1/4] Checking dependencies ...")
try:
    import gradio as gr
    import basicsr
    import realesrgan
    import matplotlib
    print("  All dependencies already installed")
except ImportError:
    print("  Installing dependencies (~2-3 min) ...")
    # Install torch first — basicsr/realesrgan setup.py imports torch
    os.system("pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    # Apply shims before installing basicsr so its setup doesn't fail
    os.system("pip install -q numpy<2 opencv-python-headless matplotlib pillow")
    os.system("pip install -q basicsr")
    os.system("pip install -q realesrgan")
    os.system("pip install -q gradio")
    print("  Installation complete")

import gradio as gr
import matplotlib
matplotlib.use("Agg")

# ── 2. GPU check ──────────────────────────────────────────────────
print("\n[2/4] Checking GPU ...")
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    print("  WARNING: No GPU — using CPU (slow!)")
    print("  Go to Runtime -> Change runtime type -> T4 GPU")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda:0")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name}  ({mem:.1f} GB VRAM)")

# ── 3. Mount Google Drive and copy graphs/ folder ────────────────
print("\n[3/4] Setting up graphs/ from Google Drive ...")

GRAPHS_DST = Path("/content/graphs")

def _setup_graphs_from_drive():
    """
    Drive mounting must be done in a notebook cell before running this script.
    This function just checks if /content/graphs was already copied there.
    If not found, tries to copy from /content/drive/MyDrive/graphs directly.
    """
    import shutil

    # Already copied — nothing to do
    if GRAPHS_DST.exists() and any(GRAPHS_DST.iterdir()):
        py_files = list(GRAPHS_DST.rglob("*.py"))
        print(f"  graphs/ ready at {GRAPHS_DST} — {len(py_files)} .py files")
        return True

    # Try to copy from Drive if it's already mounted
    src = Path("/content/drive/MyDrive/graphs")
    if src.exists():
        print(f"  Drive already mounted. Copying {src} -> {GRAPHS_DST} ...")
        if GRAPHS_DST.exists():
            shutil.rmtree(str(GRAPHS_DST))
        shutil.copytree(str(src), str(GRAPHS_DST))
        py_files = list(GRAPHS_DST.rglob("*.py"))
        print(f"  Copied — {len(py_files)} .py files: {[f.name for f in py_files]}")
        return True

    # Drive not mounted and graphs not present
    print("  WARNING: /content/graphs not found and Drive not mounted.")
    print("  Run this in a Colab cell BEFORE starting the worker:")
    print()
    print("    from google.colab import drive")
    print("    drive.mount('/content/drive')")
    print("    import shutil")
    print("    shutil.copytree('/content/drive/MyDrive/graphs', '/content/graphs')")
    print()
    print("  Then re-run: !python colab_worker.py")
    return False

_graphs_ready = _setup_graphs_from_drive()

# ── 4. Model cache ────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
_cache = {}

def _get_upsampler(scale: int, tile: int):
    key = (scale, tile)
    if key in _cache:
        return _cache[key]

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model_path = "models/RealESRGAN_x4plus.pth"
    if not os.path.exists(model_path):
        print("  Downloading RealESRGAN_x4plus.pth ...")
        urlretrieve(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth",
            model_path,
        )
        print("  Downloaded")

    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                  num_block=23, num_grow_ch=32, scale=4)
    sampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=net,
        tile=tile if tile > 0 else 0,
        tile_pad=10,
        pre_pad=0,
        half=(DEVICE.type == "cuda"),
        device=DEVICE,
    )
    _cache[key] = sampler
    print(f"  Model loaded on {DEVICE}  (half={sampler.half})")
    return sampler


# ── 4. Standard enhancement ───────────────────────────────────────
def enhance(image, method: str, scale: int, tile: int, face_enhance: bool):
    scale = int(scale); tile = int(tile)
    method = str(method).strip().lower()
    t0 = time.time()
    print(f"\n  /enhance  method={method}  scale={scale}  tile={tile}  size={image.shape}")

    if method == "realesrgan":
        up  = _get_upsampler(scale, tile)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out_bgr, _ = up.enhance(bgr, outscale=scale)
        result = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    elif method in ("bicubic", "lanczos"):
        pil = Image.fromarray(image)
        rsm = Image.BICUBIC if method == "bicubic" else Image.LANCZOS
        w, h = pil.size
        result = np.array(pil.resize((w * scale, h * scale), rsm))
    else:
        raise ValueError(f"Unknown method: {method!r}")

    print(f"  Done in {time.time()-t0:.1f}s  output={result.shape}")
    return result


# ── 5. Full analyze: enhance + all plots on GPU, return zip ───────
def analyze(image, scale: int, tile: int, face_enhance: bool):
    """
    Full pipeline on Colab GPU:
      1. Real-ESRGAN enhancement
      2. All 6 visualization plots (graphs/ folder must be uploaded)
      3. Zip everything into /content/results/<stem>/
      4. Return the zip file path — download from Colab Files panel

    Returns: zip file path (gr.File)
    """
    scale = int(scale); tile = int(tile)
    t0 = time.time()

    # Guard against Gradio warmup pings with no image
    if image is None:
        return None

    print(f"\n  /analyze  scale={scale}  tile={tile}  size={image.shape}")

    # Output folder in /content so it persists and is visible in Files panel
    stem    = f"analysis_{scale}x_{int(time.time())}"
    out_dir = Path("/content/results") / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {out_dir}")

    # ── Step 1: Enhancement on GPU ────────────────────────────────
    print("  [1/3] Running Real-ESRGAN on GPU ...")
    up  = _get_upsampler(scale, tile)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out_bgr, _ = up.enhance(bgr, outscale=scale)
    enhanced   = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    # Save enhanced image
    enhanced_path = out_dir / "00_enhanced_output.png"
    Image.fromarray(enhanced).save(str(enhanced_path))
    print(f"  Enhanced: {enhanced.shape}  saved to {enhanced_path}")

    # ── Step 2: All plots on GPU (model still in VRAM) ────────────
    print("  [2/3] Generating all visualization plots ...")
    if not _graphs_ready:
        print("  SKIPPING plots — graphs/ folder not available from Drive")
        print("  Make sure My Drive/graphs/ exists and re-run the worker")
    else:
        try:
            # Import realesrgan_viz directly by file path — avoids triggering
            # graphs/__init__.py which imports kernel_plot -> interpolation
            # (interpolation.py is a local module not present on Colab)
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "realesrgan_viz",
                "/content/graphs/realesrgan_viz.py"
            )
            viz_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(viz_mod)
            run_visualization_suite = viz_mod.run_visualization_suite

            def _progress(step, total, name):
                print(f"    Plot {step}/{total}: {name}")

            run_visualization_suite(
                model=up,
                input_np=image,
                output_np=enhanced,
                out_dir=out_dir,
                progress_cb=_progress,
            )
            print("  All 6 plots generated successfully")
        except Exception as e:
            print(f"  ERROR in plot generation: {e}")
            traceback.print_exc()

    # ── Step 3: Zip everything ────────────────────────────────────
    print("  [3/3] Zipping results ...")
    zip_path = Path("/content/results") / f"{stem}.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.iterdir()):
            zf.write(str(f), f.name)
            print(f"    Added: {f.name}  ({f.stat().st_size // 1024} KB)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  zip={zip_path}")
    print(f"\n  DOWNLOAD: Go to Colab Files panel (folder icon) -> results -> {stem}.zip")
    return str(zip_path)


# ── 6. Gradio interface ───────────────────────────────────────────
print("\n[5/5] Starting Gradio server ...")

with gr.Blocks(title="Antigravity Colab Worker") as demo:
    gr.Markdown(
        "## Antigravity Colab GPU Worker\n\n"
        "**Two modes:**\n"
        "- `/enhance` — standard enhancement, returns image\n"
        "- `/analyze` — enhancement + all 6 plots on GPU, returns zip\n\n"
        f"> GPU: {torch.cuda.get_device_name(0) if num_gpus > 0 else 'CPU only'}"
    )

    with gr.Tab("Enhance"):
        with gr.Row():
            img_in  = gr.Image(type="numpy", label="Input Image")
            img_out = gr.Image(type="numpy", label="Enhanced Output")
        method_in = gr.Textbox(value="realesrgan", label="Method")
        scale_in  = gr.Number(value=4, label="Scale", precision=0)
        tile_in   = gr.Number(value=0, label="Tile Size (0=full GPU)", precision=0)
        face_in   = gr.Checkbox(value=False, label="Face Enhance")
        gr.Button("Run Enhancement", variant="primary").click(
            fn=enhance,
            inputs=[img_in, method_in, scale_in, tile_in, face_in],
            outputs=img_out,
            api_name="enhance",
        )

    with gr.Tab("Analyze (GPU plots + zip)"):
        gr.Markdown(
            "Runs enhancement + all 6 visualization plots on the GPU.\n"
            "Returns a zip file — download from the **Files panel** (folder icon on left) "
            "under `/content/results/`\n\n"
            "**Make sure you uploaded the `graphs/` folder to Colab before running.**"
        )
        with gr.Row():
            az_img_in  = gr.Image(type="numpy", label="Input Image")
            az_zip_out = gr.File(label="Download ZIP (plots + enhanced image)")
        az_scale_in = gr.Number(value=4, label="Scale (2/4/8)", precision=0)
        az_tile_in  = gr.Number(value=256, label="Tile Size (256 recommended for 8x)", precision=0)
        az_face_in  = gr.Checkbox(value=False, label="Face Enhance")
        gr.Button("Run Full Analysis on GPU", variant="primary").click(
            fn=analyze,
            inputs=[az_img_in, az_scale_in, az_tile_in, az_face_in],
            outputs=az_zip_out,
            api_name="analyze",
        )

demo.queue(max_size=4)

print("\n" + "=" * 60)
print("  SERVER RUNNING — cell stays spinning")
print("  Copy the gradio.live URL for --remote-url")
print("  For full analysis: use the Analyze tab in the browser")
print("  Results saved to /content/results/ — download from Files panel")
print("=" * 60 + "\n")

demo.launch(share=True, show_error=True, quiet=False, inline=False)
print("\n  Server stopped.")

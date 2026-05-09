"""
colab_worker.py — Antigravity Colab GPU Worker (script version)

Upload this file to Colab and run in a cell:
    !python colab_worker.py

Because this is a plain Python script (not a notebook cell),
demo.launch() BLOCKS execution — the cell stays ⌛ running,
the GPU stays allocated, and every request from your local
machine is processed by the Colab GPU.

Press the STOP button (■) in Colab when you are done.
"""

import os
import sys
import time
import numpy as np
import torch
import cv2
from PIL import Image
from urllib.request import urlretrieve

print("=" * 60)
print("  Antigravity Colab GPU Worker")
print("=" * 60)

# ── 1. Install dependencies if not already installed ──────────────
print("\n[1/3] Checking dependencies ...")
try:
    import gradio as gr
    import basicsr
    import realesrgan
    print("  ✓ All dependencies already installed")
except ImportError:
    print("  Installing dependencies (this takes ~2-3 min) ...")
    os.system("pip install -q gradio basicsr realesrgan opencv-python-headless")
    os.system("pip install -q basicsr-fixed")  # torchvision compat fix
    import gradio as gr

# ── 2. GPU check ──────────────────────────────────────────────────
print("\n[2/3] Checking GPU ...")
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    print("  ⚠ No GPU detected — using CPU (slow!)")
    print("  Tip: Runtime → Change runtime type → T4 GPU")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda:0")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  ✓ GPU {i}: {name}  ({mem:.1f} GB VRAM)")

# ── 3. Model cache ────────────────────────────────────────────────
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
        print("  ✓ Downloaded")

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
    print(f"  ✓ Model loaded on {DEVICE}")
    return sampler


# ── 4. Enhancement function ───────────────────────────────────────
def enhance(image, method: str, scale: int, tile: int, face_enhance: bool):
    """
    Called by gradio_client from the local machine.
    image    : numpy uint8 RGB (Gradio converts the uploaded file automatically)
    method   : 'realesrgan' | 'bicubic' | 'lanczos'
    scale    : 2, 4, or 8
    tile     : tile size (0 = whole image)
    """
    scale = int(scale); tile = int(tile)
    method = str(method).strip().lower()
    t0 = time.time()
    print(f"\n  → Request: method={method}  scale={scale}  "
          f"tile={tile}  size={image.shape}  device={DEVICE}")

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

    elapsed = time.time() - t0
    print(f"  ✓ Done in {elapsed:.1f}s  output={result.shape}")
    return result


# ── 5. Enhancement + feature extraction (for visualization suite) ──
def enhance_and_extract(image, scale: int, tile: int):
    """
    Called by /enhance_and_extract API.
    Runs Real-ESRGAN on GPU, then captures conv_first + 23 block features
    via forward hooks on the INPUT image.
    Returns: (enhanced_img_np, npz_file_path)
    """
    import tempfile
    scale = int(scale); tile = int(tile)
    t0 = time.time()
    print(f"\n  → enhance_and_extract: scale={scale} tile={tile} size={image.shape}")

    # Step 1: enhance
    up  = _get_upsampler(scale, tile)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out_bgr, _ = up.enhance(bgr, outscale=scale)
    enhanced   = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    # Step 2: forward pass with hooks on the INPUT image
    net    = up.model
    cf_out = {}
    blk_out = {}
    hooks  = []

    def _hf(m, i, o): cf_out["d"] = o.detach().cpu().numpy()
    hooks.append(net.conv_first.register_forward_hook(_hf))
    for idx, b in enumerate(net.body):
        def _hb(m, i, o, k=idx): blk_out[k] = o.detach().cpu().numpy()
        hooks.append(b.register_forward_hook(_hb))

    img_f  = image.astype(np.float32)[:, :, ::-1] / 255.0   # RGB→BGR, float
    img_t  = torch.from_numpy(
        np.ascontiguousarray(img_f.transpose(2, 0, 1))
    ).float().unsqueeze(0).to(DEVICE)

    net.eval()
    with torch.no_grad(): net(img_t)
    for h in hooks: h.remove()

    # Step 3: save NPZ
    npz_data = {"conv_first": cf_out["d"]}
    for i, arr in blk_out.items():
        npz_data[f"block_{i}"] = arr

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(tmp.name, **npz_data)
    tmp.close()

    print(f"  ✓ enhance_and_extract done in {time.time()-t0:.1f}s")
    return enhanced, tmp.name


# ── 6. Build Gradio interface ─────────────────────────────────────
print("\n[3/3] Starting Gradio server ...")

import gradio as gr

with gr.Blocks(title="Antigravity Colab Worker") as demo:
    gr.Markdown(
        "## Antigravity Colab GPU Worker\n"
        "This server runs on a Colab GPU. "
        "Paste the public URL into your local CLI `--remote-url` flag.\n\n"
        f"> **GPU:** {torch.cuda.get_device_name(0) if num_gpus > 0 else 'CPU only'}"
    )
    with gr.Row():
        img_in  = gr.Image(type="numpy", label="Input Image")
        img_out = gr.Image(type="numpy", label="Enhanced Output")

    method_in = gr.Textbox(value="realesrgan", label="Method")
    scale_in  = gr.Number(value=4, label="Scale", precision=0)
    tile_in   = gr.Number(value=0, label="Tile Size (0=full)", precision=0)
    face_in   = gr.Checkbox(value=False, label="Face Enhance")

    # Standard enhancement endpoint (/enhance)
    run_btn = gr.Button("▶ Run (manual test)", variant="primary")
    run_btn.click(
        fn=enhance,
        inputs=[img_in, method_in, scale_in, tile_in, face_in],
        outputs=img_out,
        api_name="enhance",
    )

    # Visualization endpoint (/enhance_and_extract)
    viz_img_out  = gr.Image(type="numpy",  label="Enhanced (viz)", visible=False)
    viz_file_out = gr.File(label="Feature NPZ", visible=False)
    img_in.change(
        fn=enhance_and_extract,
        inputs=[img_in, scale_in, tile_in],
        outputs=[viz_img_out, viz_file_out],
        api_name="enhance_and_extract",
    )

demo.queue(max_size=8)

print("\n" + "=" * 60)
print("  SERVER STARTING — this cell stays ⌛ RUNNING")
print("  Copy the public gradio.live URL below")
print("  Use: python main.py -i img.jpg --analyze-esrgan -s 4")
print("       --backend colab --remote-url https://xxxx.gradio.live")
print("  Press ■ Stop in Colab when done")
print("=" * 60 + "\n")

demo.launch(
    share=True,
    show_error=True,
    quiet=False,
    inline=False,
)

print("\n  Server stopped.")


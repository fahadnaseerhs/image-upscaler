<![CDATA[<div align="center">

# 🚀 Image Enhancement Pipeline — DSP Project

### A Full-Stack Image Enhancement System | Classical DSP + AI Super-Resolution

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_UI-000000?logo=flask)
![Three.js](https://img.shields.io/badge/Three.js-3D_Viz-black?logo=three.js)
![Real-ESRGAN](https://img.shields.io/badge/Real--ESRGAN-AI_SR-ff6b35)
![Colab](https://img.shields.io/badge/Google_Colab-GPU_Worker-F9AB00?logo=googlecolab)

</div>

---

<div align="center">

## ⚡ QUICK START — CLONE & RUN IN ONE COMMAND ⚡

</div>

> [!IMPORTANT]
> **Prerequisites:** You only need **Python 3.10+** installed and added to your PATH. Everything else is automated.

---

### 📥 Step 1 — Clone the Repository

Open a terminal (Command Prompt / PowerShell / Git Bash) and run:

```bash
git clone https://github.com/fahadnaseerhs/image-upscaler.git "%USERPROFILE%\Desktop\DSP_Project"
```

> [!TIP]
> This clones the project into a folder named **`DSP_Project`** on your **Desktop**.
>
> **Linux / macOS users:**
> ```bash
> git clone https://github.com/fahadnaseerhs/image-upscaler.git ~/Desktop/DSP_Project
> ```

---

### 🖥️ Step 2 — Run the Project

Navigate into the project and launch:

<table>
<tr>
<th>🪟 Windows</th>
<th>🐧 Linux / 🍎 macOS</th>
</tr>
<tr>
<td>

```bat
cd %USERPROFILE%\Desktop\DSP_Project
run.bat
```

</td>
<td>

```bash
cd ~/Desktop/DSP_Project
chmod +x run.sh
./run.sh
```

</td>
</tr>
</table>

> [!NOTE]
> **What happens automatically when you run `run.bat` / `run.sh`:**
> 1. ✅ Creates a Python virtual environment (`venv/`)
> 2. ✅ Installs all dependencies from `requirements.txt`
> 3. ✅ Activates the environment
> 4. ✅ Launches the Flask web server on `http://localhost:5000`
>
> **You don't need to install anything manually. Just clone → run.**

---

> [!CAUTION]
> **Windows users** → Use **`run.bat`** (double-click or run from CMD/PowerShell)
>
> **Linux / macOS users** → Use **`run.sh`** (run from terminal with `./run.sh`)
>
> ❌ Do **NOT** use `run.sh` on Windows or `run.bat` on Linux/macOS.

---

## 🔧 Manual Setup (Optional — Only If You Prefer)

If you don't want to use the one-click scripts, you can set up manually:

```bash
# 1. Clone
git clone https://github.com/fahadnaseerhs/image-upscaler.git "%USERPROFILE%\Desktop\DSP_Project"
cd "%USERPROFILE%\Desktop\DSP_Project"

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run
python app.py
```

---

## What It Does

Takes a low-resolution or degraded image and reconstructs a high-resolution version using one of three methods:

| Method | Type | Quality | Speed |
|---|---|---|---|
| **Bicubic** | Classical DSP — Keys' cubic kernel | Good | Fast (CPU) |
| **Lanczos** | Classical DSP — windowed sinc kernel | Better | Fast (CPU) |
| **Real-ESRGAN** | AI super-resolution — RRDBNet | Best | Slow (needs GPU) |

Scale factors: **2×, 4×, 8×**

---

## Project Structure

```
DSP_Project/
├── app.py                  # Flask web server with SSE live streaming
├── main.py                 # CLI entry point and pipeline orchestrator
├── run.bat                 # 🪟 One-click launcher (Windows)
├── run.sh                  # 🐧 One-click launcher (Linux/macOS)
├── setup.bat               # 🪟 Environment setup (Windows)
├── setup.sh                # 🐧 Environment setup (Linux/macOS)
├── requirements.txt        # Python dependencies
├── loader.py               # Image loading, validation, channel splitting, normalization
├── grid.py                 # Sparse grid creation — upsampling step (DSP)
├── interpolation.py        # Bicubic & Lanczos interpolation engines
├── saver.py                # Denormalization, channel merging, sharpening, PNG export
├── enhancer.py             # Local Real-ESRGAN inference (CPU/GPU)
├── enhancer_remote.py      # Remote Real-ESRGAN via Colab or HF Space
├── colab_worker.py         # Colab GPU worker script — /enhance + /analyze endpoints
├── colab_worker.ipynb      # Colab notebook launcher
├── generate_dsp_pdf.py     # DSP Concepts PDF report generator
├── graphs/
│   ├── realesrgan_viz.py   # Real-ESRGAN visualization suite (6 plots)
│   ├── spatial_domain.py   # Spatial domain before/after + diff + zoom
│   ├── frequency_domain.py # FFT frequency analysis
│   ├── kernel_plot.py      # Interpolation kernel shape visualization
│   ├── weighted_sum.py     # Weighted sum computation diagram
│   ├── radar_chart.py      # Algorithm comparison radar chart
│   ├── antialiasing.py     # Anti-aliasing property visualization
│   └── uint8_to_float.py   # Normalization step visualization
├── templates/
│   └── index.html          # Web UI — Three.js 3D visualization
├── hf_space/
│   ├── app.py              # Hugging Face Space Gradio app
│   └── requirements.txt
├── models/                 # Model weights (auto-downloaded)
├── uploads/                # Temporary uploaded images
└── output/                 # Generated output images and plots
```

---

## DSP Pipeline

The pipeline processes images through 5 stages:

```
Input Image
    │
    ▼
[1] LOAD — Read file, validate, split into R/G/B channels, normalize to [0.0, 1.0]
    │
    ▼
[2] GRID — Create sparse grid (H×scale, W×scale), place known pixels at scale-spaced
           positions, fill rest with NaN  ← DSP upsampling by factor L
    │
    ▼
[3] INTERPOLATE — Fill every NaN using weighted kernel sums of neighboring pixels
                  Bicubic: Keys' piecewise cubic, 4×4 neighborhood
                  Lanczos: windowed sinc, 6×6 neighborhood (a=3)
    │
    ▼
[4] SAVE — Denormalize [0,1] → [0,255], merge channels, optional UnsharpMask, write PNG
    │
    ▼
[5] COMPLETE
```

---

## Web UI

```bash
python app.py
```

Open `http://localhost:5000`

**Features:**
- Drag-and-drop image upload
- Scale factor: 2×, 4×, 8×
- Method: Bicubic, Lanczos, AI (Real-ESRGAN)
- Execution backend: Local CPU/GPU, Hugging Face Space, Google Colab GPU
- Real-time 3D visualization (Three.js) showing the pipeline stages
- Live progress streaming via Server-Sent Events (SSE)
- Before/after comparison slider

---

## CLI

```bash
# Basic 2× Lanczos upscale
python main.py --input photo.jpg

# Bicubic 4× upscale
python main.py --input photo.jpg --method bicubic --scale 4

# Lanczos 8× with sharpening
python main.py --input photo.jpg --scale 8 --sharpen

# Compare Bicubic vs Lanczos side-by-side
python main.py --input photo.jpg --compare

# Real-ESRGAN locally
python main.py --input photo.jpg --method realesrgan --scale 4

# Real-ESRGAN on Colab GPU
python main.py --input photo.jpg --method realesrgan --scale 4 \
    --backend colab --remote-url https://xxxx.gradio.live

# Full ESRGAN visualization suite (6 plots)
python main.py --input photo.jpg --analyze-esrgan --scale 4

# Full ESRGAN viz on Colab GPU
python main.py --input photo.jpg --analyze-esrgan --scale 4 \
    --backend colab --remote-url https://xxxx.gradio.live

# DSP analysis plots (classical methods)
python main.py --input photo.jpg --analyze-dsp

# Both analyses together
python main.py --input photo.jpg --analyze-esrgan --analyze-dsp --scale 4 \
    --backend colab --remote-url https://xxxx.gradio.live
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--input`, `-i` | required | Input image path |
| `--output`, `-o` | `./output` | Output directory |
| `--scale`, `-s` | `2` | Upscale factor: 2, 4, or 8 |
| `--method`, `-m` | `lanczos` | `bicubic`, `lanczos`, or `realesrgan` |
| `--lanczos-a` | `3` | Lanczos window size: 2 (fast) or 3 (sharp) |
| `--sharpen` | off | Apply UnsharpMask post-processing |
| `--compare` | off | Run both Bicubic and Lanczos, show diff map |
| `--visualize` | off | Show animated grid/interpolation visualizations |
| `--analyze-esrgan` | off | Generate full Real-ESRGAN visualization suite |
| `--analyze-dsp` | off | Generate DSP analysis plots |
| `--tile` | `0` | Real-ESRGAN tile size in pixels (0 = full image) |
| `--face-enhance` | off | GFPGAN face restoration after Real-ESRGAN |
| `--backend` | `local` | `local`, `remote` (HF Space), or `colab` |
| `--remote-url` | — | Colab `gradio.live` URL or HF Space name |
| `--quiet`, `-q` | off | Suppress progress output |

---

## Google Colab GPU Worker

Use Colab's free T4 GPU for Real-ESRGAN when you don't have a local GPU.

### Setup

**1. Open Colab and set GPU runtime**

Go to [colab.google.com](https://colab.google.com) → open `colab_worker.ipynb` → `Runtime → Change runtime type → T4 GPU`

**2. Upload `graphs/` folder to Google Drive**

Upload the `graphs/` folder to the root of your Google Drive (`My Drive/graphs/`).

**3. Mount Drive and copy graphs (run in a Colab cell)**

```python
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.copytree('/content/drive/MyDrive/graphs', '/content/graphs')
print('Done')
```

**4. Upload `colab_worker.py` to Colab Files panel and run**

```python
!python colab_worker.py
```

**5. Copy the `gradio.live` URL from the output**

### Colab Endpoints

| Endpoint | Description |
|---|---|
| `/enhance` | Standard enhancement — returns upscaled image |
| `/analyze` | Enhancement + all 6 visualization plots on GPU → returns zip file |

### Using `/analyze` (browser)

Open the `gradio.live` URL → **Analyze tab** → upload image → set scale/tile → **Run Full Analysis on GPU** → download zip from Colab Files panel (`/content/results/`)

### Using `/enhance` (CLI)

```bash
python main.py --input photo.jpg --scale 4 \
    --backend colab --remote-url https://xxxx.gradio.live
```

---

## Real-ESRGAN Visualization Suite

Running `--analyze-esrgan` generates 6 diagnostic plots saved to `output/realesrgan/<image_name>/`:

| File | Description |
|---|---|
| `00_enhanced_output.png` | The upscaled image |
| `01_filter_responses_64.png` | All 64 first-layer kernel weights (3×3 patches, RdBu colormap) |
| `01_filter_responses_top16.png` | Top 16 most active feature maps by variance |
| `02_block_progression_23.png` | 23 RRDB block activations + energy/variance chart (twin y-axis) |
| `03_frequency_before_after.png` | 2D FFT: input vs output vs difference spectrum |
| `04_new_frequencies_generated.png` | Radial frequency profile (log scale) + spatial diff heatmap + histogram |
| `05_radar_summary.png` | 5-metric radar: sharpness, high-freq energy, detail gain, noise, edge strength |
| `06_tiling_grid_diagram.png` | Tiling parameters and processing stats |

---

## DSP Concepts Used

| Concept | Where |
|---|---|
| Discrete 2D signal | Image as pixel array |
| Signal normalization | `loader.py` — uint8 → float64 |
| Upsampling by factor L | `grid.py` — sparse grid with NaN placeholders |
| Reconstruction filter | `interpolation.py` — both algorithms |
| Piecewise cubic filter (Keys') | Bicubic interpolation via SciPy `map_coordinates` |
| Ideal sinc / windowed sinc | Lanczos interpolation via PIL `Image.LANCZOS` |
| Filter separability | 2D weight = w_row × w_col |
| Ringing artifact | Lanczos negative side-lobes, clipped in `saver.py` |
| High-frequency emphasis | UnsharpMask post-processing |
| Learned non-linear filter | Real-ESRGAN (23-stage RRDBNet) |
| Block/overlap-add processing | Real-ESRGAN tiling with `tile_pad=10` |

---

## Output Files

```
output/
├── <name>_lanczos_4x.png           # Classical upscale result
├── <name>_bicubic_4x.png           # Classical upscale result
└── realesrgan/
    └── <name>/
        ├── 00_enhanced_output.png
        ├── 01_filter_responses_64.png
        ├── 01_filter_responses_top16.png
        ├── 02_block_progression_23.png
        ├── 03_frequency_before_after.png
        ├── 04_new_frequencies_generated.png
        ├── 05_radar_summary.png
        └── 06_tiling_grid_diagram.png
```

---

## Requirements

- Python 3.10+
- See installation section above for package requirements
- GPU recommended for Real-ESRGAN (or use Colab worker)
- Google Colab free tier (T4 GPU) supported

---

## License

Open source — available for educational and personal use.
]]>

# 🔬 Image Upscaler

A Python-based image upscaling pipeline that reconstructs high-resolution images from low-resolution sources using **Bicubic** and **Lanczos** interpolation algorithms. Features both a command-line interface and a stunning web UI with real-time 3D visualization powered by Three.js.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_UI-000000?logo=flask)
![Three.js](https://img.shields.io/badge/Three.js-3D_Viz-black?logo=three.js)

---

## ✨ Features

- **Bicubic Interpolation** — 4×4 neighborhood, Keys' cubic kernel via SciPy
- **Lanczos Interpolation** — Windowed sinc kernel (a=2 or a=3) via Pillow
- **Multiple Scale Factors** — 2×, 4×, or 8× upscaling
- **Post-Processing Sharpening** — Optional UnsharpMask to recover edge crispness
- **AI Enhancement (Optional)** — Real-ESRGAN model for true detail enhancement
- **Compare Mode** — Run both algorithms side-by-side with a difference map
- **Web UI** — Dark-themed glassmorphism interface with drag-and-drop upload
- **3D Visualization** — Real-time Three.js animation showing the decoding pipeline:
  - Stage 1: Uploaded image rendered as colored 3D cubes
  - Stage 2: Cubes dissolve into a sparse grid with NaN wireframes
  - Stage 3: Interpolation fills empty positions with animated vectors
- **Live Progress Streaming** — Server-Sent Events (SSE) for real-time pipeline feedback
- **CLI Mode** — Full command-line interface with progress reporting and visualization

---

## 📦 Requirements

### Python Packages

| Package      | Purpose                                |
|--------------|----------------------------------------|
| `flask`      | Web server and API routes              |
| `numpy`      | Array operations, grid math            |
| `pillow`     | Image I/O, Lanczos resize, sharpening  |
| `scipy`      | Bicubic interpolation (map_coordinates)|
| `matplotlib` | CLI visualizations and compare plots   |
| `torch` + `torchvision` | Real-ESRGAN inference backend (optional) |
| `realesrgan` + `basicsr` | AI model loader/runtime (optional) |
| `opencv-python` | Image conversion for Real-ESRGAN I/O (optional) |

### Install

```bash
pip install flask numpy pillow scipy matplotlib
```

### Optional AI Enhancer Dependencies (Real-ESRGAN)

```bash
pip install torch torchvision realesrgan basicsr opencv-python gfpgan
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/fahadnaseerhs/image-upscaler.git
cd image-upscaler
```

### 2. Install Dependencies

```bash
pip install flask numpy pillow scipy matplotlib
```

### 3. Run the Web UI

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 4. Or Use the CLI

```bash
python main.py --input photo.jpg
```

---

## 🖥️ Web UI Usage

1. **Upload** an image (PNG, JPEG, BMP, TIFF, WebP)
2. **Configure** parameters:
   - Scale factor: `2×`, `4×`, or `8×`
   - Method: `Bicubic`, `Lanczos`, or `AI (Real-ESRGAN)`
   - Lanczos window: `a=2` (fast) or `a=3` (sharp)
   - Toggle **Sharpen** or **Compare** mode
   - In AI mode you can set a Real-ESRGAN `tile` size and optionally enable face restoration (GFPGAN)
3. Click **Run Pipeline** and watch the 3D visualization
4. **Download** the upscaled result from the results panel

---

## ⌨️ CLI Commands

```bash
# Basic upscale (2× Lanczos, default)
python main.py --input photo.jpg

# Bicubic 4× upscale
python main.py --input photo.jpg --method bicubic --scale 4

# Lanczos with window a=2
python main.py --input photo.jpg --method lanczos --lanczos-a 2

# Compare both methods side-by-side
python main.py --input photo.jpg --compare

# With sharpening
python main.py --input photo.jpg --sharpen

# Show animated visualizations
python main.py --input photo.jpg --visualize

# Save individual R, G, B channel images
python main.py --input photo.jpg --save-channels

# Custom output directory
python main.py --input photo.jpg --output ./results

# Quiet mode (minimal output)
python main.py --input photo.jpg --quiet

# Full example
python main.py --input photo.jpg --scale 4 --method lanczos --lanczos-a 3 --sharpen --visualize
```

### CLI Options Reference

| Flag              | Default    | Description                                      |
|-------------------|------------|--------------------------------------------------|
| `--input`, `-i`   | *required* | Path to input image                              |
| `--output`, `-o`  | `./output` | Output directory                                 |
| `--scale`, `-s`   | `2`        | Scale factor (2, 4, or 8)                        |
| `--method`, `-m`  | `lanczos`  | Interpolation method (`bicubic` or `lanczos`)    |
| `--lanczos-a`     | `3`        | Lanczos window size (2 or 3)                     |
| `--sharpen`       | off        | Apply UnsharpMask post-processing                |
| `--compare`       | off        | Run both methods and show comparison             |
| `--visualize`     | off        | Show animated grid/interpolation visualizations  |
| `--save-channels` | off        | Save R, G, B as separate grayscale PNGs          |
| `--quiet`, `-q`   | off        | Suppress progress output                         |

---

## 📁 Project Structure

```
image-upscaler/
├── app.py              # Flask web server with SSE streaming
├── main.py             # CLI entry point and pipeline orchestrator
├── loader.py           # Image loading, validation, normalization
├── grid.py             # Sparse grid creation and pixel mapping
├── interpolation.py    # Bicubic & Lanczos interpolation engines
├── saver.py            # Denormalization, merging, sharpening, saving
├── templates/
│   └── index.html      # Web UI (Three.js 3D visualization)
├── uploads/            # Temporary uploaded images
├── output/             # Generated upscaled images
└── README.md
```

---

## 🔧 How It Works

The pipeline processes images through 5 stages:

1. **Load** — Read image, validate, split into R/G/B channels, normalize to `[0.0, 1.0]`
2. **Grid** — Create a larger sparse grid (`H×scale`, `W×scale`), place known pixels at `scale`-spaced positions (rest = NaN)
3. **Interpolate** — Fill every NaN position using weighted sums of neighboring known pixels
4. **Save** — Denormalize back to `[0, 255]`, merge channels, optionally sharpen, write PNG
5. **Complete** — Display summary with output path and timing

---

## 📄 License

This project is open source and available for educational and personal use.

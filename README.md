<div align="center">

# Antigravity Image Enhancement Pipeline
### Next-Generation Image Upscaling: Classical DSP and Deep Learning Super-Resolution

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#)
[![Flask](https://img.shields.io/badge/Flask-Web_UI-000000?logo=flask)](#)
[![Three.js](https://img.shields.io/badge/Three.js-3D_Viz-black?logo=three.js)](#)
[![Real-ESRGAN](https://img.shields.io/badge/Real--ESRGAN-AI_SR-ff6b35)](#)
[![Google Colab](https://img.shields.io/badge/Google_Colab-GPU_Worker-F9AB00?logo=googlecolab)](#)

*An industry-grade pipeline designed to reconstruct high-resolution images from degraded sources utilizing both Digital Signal Processing (DSP) algorithms and Neural Network-based inference, featuring advanced diagnostic tools and distributed GPU orchestration.*

</div>

---

## Executive Summary

The Antigravity Image Enhancement Pipeline provides a transparent, modular architecture for image super-resolution. It bridges the gap between conventional interpolation methodologies and modern Deep Learning approaches. Designed for researchers, developers, and imaging professionals, the software supports local CPU/GPU execution as well as remote inference offloading via Google Colab and Hugging Face.

### Key Capabilities
- **Classical DSP Algorithms**: High-performance implementations of Bicubic and Lanczos interpolation.
- **Neural Super-Resolution**: Integration of Real-ESRGAN (RRDBNet) for superior detail hallucination.
- **Distributed Inference**: Built-in orchestration to seamlessly stream inference workloads to remote GPU endpoints (Google Colab, Hugging Face).
- **DSP Diagnostic Suite**: In-depth analytical tools to generate frequency domain plots, filter response maps, and comparative metrics.
- **Interactive Visualization**: A Flask-based web interface featuring real-time Server-Sent Events (SSE) and 3D data-flow visualization using Three.js.

---

## Installation and Environment Setup

The system requires **Python 3.10 or higher**. All dependencies are isolated within a Python virtual environment to prevent system-wide package conflicts.

### Step 1: Repository Cloning

First, clone the repository to your local machine:

```bash
git clone https://github.com/fahadnaseerhs/image-upscaler.git
cd image-upscaler
```

### Step 2: Automated Initialization

We provide automated bootstrap scripts that handle virtual environment creation, dependency installation, and server execution.

**For Windows Environments:**
Execute the batch script from the Command Prompt or PowerShell:
```cmd
run.bat
```
*(Note: If you only wish to install dependencies without running the server, use `setup.bat` instead).*

**For Linux / macOS Environments:**
Ensure the shell scripts have execution permissions, then run:
```bash
chmod +x run.sh setup.sh
./run.sh
```

### Step 3: Manual Initialization (Alternative)

If you prefer to manually configure your environment, follow these steps:

1. **Create the Virtual Environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment:**
   - Windows: `venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Start the Application:**
   ```bash
   python app.py
   ```

Upon successful startup, the Web UI will be accessible at `http://localhost:5000`.

---

## Architecture and Supported Algorithms

The pipeline normalizes input signals and reconstructs them using one of the following methods. Scale factors supported are **2x, 4x, and 8x**.

| Algorithm | Processing Type | Hardware Utilization | Characteristics |
|---|---|---|---|
| **Bicubic** | Classical DSP (Keys' Cubic Filter) | Local CPU | Computationally inexpensive. Produces smooth reconstructions. |
| **Lanczos** | Classical DSP (Windowed Sinc) | Local CPU | Enhanced high-frequency preservation and anti-aliasing properties. |
| **Real-ESRGAN** | Neural Network (RRDBNet) | Local GPU / Cloud GPU | State-of-the-art detail synthesis. Requires significant compute power. |

---

## Command Line Interface (CLI) Reference

The `main.py` entry point exposes the core engine for batch processing, automation, and advanced diagnostic generation.

### Core Processing Commands

**Standard Local Inference:**
```bash
# Upscale using Lanczos algorithm by a factor of 4
python main.py --input source_image.jpg --method lanczos --scale 4

# Upscale using AI (Real-ESRGAN) with post-process unsharp masking
python main.py --input source_image.jpg --method realesrgan --scale 4 --sharpen
```

**Remote Execution (Google Colab / Hugging Face):**
To execute inference on remote hardware, supply the target backend and the API endpoint URL.

```bash
# Offload to Google Colab GPU
python main.py --input source_image.jpg --method realesrgan --scale 4 \
    --backend colab --remote-url https://<dynamic-id>.gradio.live

# Offload to Hugging Face Space
python main.py --input source_image.jpg --method realesrgan --scale 4 \
    --backend remote --remote-url <hf-space-id>
```

### Diagnostic and Analysis Commands

The pipeline includes a comprehensive suite to visualize algorithmic behavior in both the spatial and frequency domains.

```bash
# Generate analytical plots for classical algorithms (Bicubic vs Lanczos)
python main.py --input source_image.jpg --analyze-dsp

# Generate AI interpretation diagnostics (Neural activations, FFT responses)
python main.py --input source_image.jpg --analyze-esrgan --scale 4

# Perform a direct visual comparison between classical algorithms
python main.py --input source_image.jpg --compare
```

### Complete CLI Argument Matrix

| Argument | Requirement | Default | Description |
|---|---|---|---|
| `--input`, `-i` | **Required** | None | Filepath of the target source image. |
| `--output`, `-o` | Optional | `./output` | Output directory destination. |
| `--scale`, `-s` | Optional | `2` | Magnification factor (`2`, `4`, or `8`). |
| `--method`, `-m` | Optional | `lanczos` | Selected interpolation algorithm: `bicubic`, `lanczos`, or `realesrgan`. |
| `--backend` | Optional | `local` | Computing target: `local`, `remote` (HF), or `colab`. |
| `--remote-url` | Optional | None | The Gradio API URL or Hugging Face Space ID for remote endpoints. |
| `--sharpen` | Optional | False | Applies Unsharp Masking post-processing. |
| `--compare` | Optional | False | Executes Bicubic and Lanczos simultaneously, producing a visual difference map. |
| `--analyze-dsp` | Optional | False | Executes classical signal analysis plotting suite. |
| `--analyze-esrgan`| Optional | False | Executes AI inference diagnostic plotting suite. |
| `--tile` | Optional | `0` | Specifies tile size (e.g., 400) to mitigate GPU Out-Of-Memory (OOM) errors. |

---

## Remote Worker Configuration

To utilize free tier cloud GPUs, the system can seamlessly delegate processing to external endpoints.

### Configuring the Google Colab GPU Worker

Google Colab provides access to NVIDIA T4 GPUs. This is highly recommended for executing Real-ESRGAN efficiently if your local workstation lacks a dedicated graphics card.

1. Navigate to [Google Colab](https://colab.google.com).
2. Upload the `colab_worker.ipynb` notebook located in the repository root.
3. In the Colab menu, select `Runtime -> Change runtime type` and set the hardware accelerator to **T4 GPU**.
4. Upload the `graphs/` directory from the repository to your Google Drive to enable remote diagnostic plotting.
5. Execute the cells in the notebook. It will initialize the pipeline and output a public URL (e.g., `https://xxxx.gradio.live`).
6. Pass this URL to your local CLI using the `--remote-url` flag or input it directly into the Web UI.

### Configuring a Hugging Face Space

For a persistent, 24/7 inference endpoint, deploy the provided Hugging Face configuration.

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) and select the **Gradio** SDK.
2. Upload the contents of the local `hf_space/` directory to the newly created Space.
3. Allow the Docker container to build.
4. Utilize the `--backend remote --remote-url <your-space-name>` command from your local CLI.

---

## Signal Processing Diagnostics

When operating the CLI with the `--analyze-esrgan` flag, the pipeline bypasses traditional black-box AI implementation. It documents the transformation process by producing standard metrics in the `output/realesrgan/<filename>/` directory:

- **Filter Responses (`01_filter_responses.png`)**: Visualizes first-layer convolutional kernel weights.
- **Block Progression (`02_block_progression.png`)**: Graphs spatial variance across all 23 Residual-in-Residual Dense Blocks (RRDB).
- **Frequency Domain Mapping (`03_frequency_before_after.png`)**: Produces 2D Fast Fourier Transform (FFT) spectrograms demonstrating hallucinated high-frequency details.
- **Radar Summary (`05_radar_summary.png`)**: Evaluates the processed image across five key axes (Sharpness, Artifacts, Noise, Edge Consistency, Frequency Energy).

---

## Extensibility and Contributions

The Antigravity pipeline is built upon standard Python imaging and scientific computation libraries (NumPy, SciPy, Pillow, Torch). It is structured for modularity.

We welcome external contributions. Areas of active development include:
- Integration of newer AI models (e.g., SwinIR, DAT).
- Optimization of Lanczos implementations using PyTorch CUDA bindings.
- Further expansion of the spatial and frequency analysis modules.

---

## License

This software is released under an open-source license. It is available for academic, personal, and professional use.

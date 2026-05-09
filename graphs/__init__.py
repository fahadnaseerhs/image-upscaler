"""
graphs/__init__.py — DSP Analysis Orchestrator
"""

import sys
from pathlib import Path
import numpy as np

from . import uint8_to_float
from . import kernel_plot
from . import weighted_sum
from . import frequency_domain
from . import spatial_domain
from . import antialiasing
from . import radar_chart

def run_analysis(method: str, 
                 r_orig: np.ndarray, g_orig: np.ndarray, b_orig: np.ndarray,
                 r_filled: np.ndarray, g_filled: np.ndarray, b_filled: np.ndarray,
                 scale_factor: int, output_dir: str | Path, lanczos_a: int = 3):
    """
    Runs all DSP analysis graphs sequentially.
    """
    print("\n--- STARTING DSP ANALYSIS ---")
    out_dir = Path(output_dir) / "dsp_analysis"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. uint8 to float (need to simulate uint8 original since we get normalized floats here)
    print(">> Generating: uint8 to float conversion...")
    r_uint8 = (np.clip(r_orig, 0, 1) * 255).astype(np.uint8)
    uint8_to_float.plot(r_uint8, r_orig, out_dir)
    
    # 2. Kernel Plot
    print(f">> Generating: 1D Kernel Shape ({method})...")
    kernel_plot.plot(method, out_dir, lanczos_a)
    
    # 3. Weighted Sum Heatmap
    print(f">> Generating: 2D Weighted Sum Heatmap ({method})...")
    weighted_sum.plot(method, out_dir, lanczos_a)
    
    # 4. Frequency Domain (FFT)
    print(">> Generating: Frequency Domain (FFT) Before/After...")
    frequency_domain.plot(r_orig, g_orig, b_orig, r_filled, g_filled, b_filled, out_dir)
    
    # 5. Spatial Domain (Enhanced pixels overlay)
    print(">> Generating: Spatial Domain Enhanced Pixels...")
    spatial_domain.plot(r_orig, g_orig, b_orig, r_filled, g_filled, b_filled, scale_factor, out_dir)
    
    # 6. Anti-aliasing (specific to Lanczos)
    if method == "lanczos":
        print(">> Generating: Anti-aliasing properties (Lanczos vs Bicubic)...")
        antialiasing.plot(out_dir, lanczos_a)
        
    # 7. Radar Chart
    print(">> Generating: Performance Radar Chart...")
    radar_chart.plot(method, out_dir)
    
    print(f"\nAll DSP graphs saved to: {out_dir.resolve()}")

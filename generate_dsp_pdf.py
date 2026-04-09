"""
generate_dsp_pdf.py — Generate a PDF document covering the DSP concepts
used in the Image Enhancer / Upscaler project.

Run:
    python generate_dsp_pdf.py

Output:
    DSP_Concepts_Image_Upscaler.pdf  (in the same directory)
"""

from fpdf import FPDF


class DSPConceptsPDF(FPDF):
    """Custom PDF with styled headers, footers, and section helpers."""

    # ── colours (R, G, B) ─────────────────────────────────────────────
    DARK_BG    = (15, 15, 25)
    ACCENT     = (0, 200, 255)
    HEADING    = (30, 130, 230)
    SUBHEADING = (120, 200, 255)
    BODY       = (50, 50, 60)
    LIGHT_GRAY = (120, 120, 140)
    FORMULA_BG = (240, 245, 255)
    CODE_BG    = (235, 240, 250)
    WHITE      = (255, 255, 255)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*self.LIGHT_GRAY)
        self.cell(0, 6, "DSP Concepts  |  Image Enhancer Project", align="L")
        self.ln(4)
        # thin accent line
        self.set_draw_color(*self.ACCENT)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.LIGHT_GRAY)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── helper methods ────────────────────────────────────────────────

    def section_title(self, num: int, title: str):
        """Major numbered section heading."""
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*self.HEADING)
        self.cell(0, 10, f"{num}.  {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.ACCENT)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 120, self.get_y())
        self.ln(3)

    def sub_heading(self, title: str):
        """Sub-section heading."""
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*self.SUBHEADING)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text: str):
        """Normal paragraph."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.BODY)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def formula_box(self, formula: str, caption: str = ""):
        """Highlighted formula / equation box."""
        self.set_fill_color(*self.FORMULA_BG)
        self.set_font("Courier", "B", 11)
        self.set_text_color(30, 60, 120)
        x = self.get_x()
        self.cell(0, 8, f"    {formula}", fill=True, new_x="LMARGIN", new_y="NEXT")
        if caption:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*self.LIGHT_GRAY)
            self.cell(0, 5, f"    {caption}", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def code_box(self, label: str, code: str):
        """Small code reference box."""
        self.set_fill_color(*self.CODE_BG)
        self.set_font("Helvetica", "BI", 9)
        self.set_text_color(60, 60, 80)
        self.cell(0, 6, f"  File: {label}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Courier", "", 9)
        self.set_text_color(40, 40, 60)
        for line in code.strip().split("\n"):
            self.cell(0, 5, f"    {line}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.BODY)
        self.cell(6, 5.5, "-  ")
        self.multi_cell(0, 5.5, text)
        self.ln(1)


def build_pdf() -> str:
    pdf = DSPConceptsPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ══════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*DSPConceptsPDF.HEADING)
    pdf.cell(0, 14, "Digital Signal Processing", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 14, "Concepts", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(*DSPConceptsPDF.LIGHT_GRAY)
    pdf.cell(0, 8, "As Applied in the Image Enhancer / Upscaler Project", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)
    pdf.set_draw_color(*DSPConceptsPDF.ACCENT)
    pdf.set_line_width(0.6)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(12)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*DSPConceptsPDF.LIGHT_GRAY)
    pdf.cell(0, 6, "Project modules:  loader.py  |  grid.py  |  interpolation.py  |  saver.py  |  enhancer.py", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.cell(0, 6, "Pipeline:  Load  ->  Decode  ->  Upsample  ->  Interpolate  ->  Sharpen  ->  Save", align="C", new_x="LMARGIN", new_y="NEXT")

    # ══════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*DSPConceptsPDF.HEADING)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    toc = [
        ("1", "Sampling & Quantization"),
        ("2", "Normalization (ADC Analogy)"),
        ("3", "Channel Decomposition"),
        ("4", "Sparse Grid & Upsampling"),
        ("5", "Interpolation Kernels"),
        ("5.1", "   Bicubic (Keys') Kernel"),
        ("5.2", "   Lanczos (Windowed Sinc) Kernel"),
        ("5.3", "   Separable 2-D Kernels"),
        ("6", "Boundary Handling"),
        ("7", "Unsharp Mask (Post-Processing)"),
        ("8", "Denormalization (DAC Analogy)"),
        ("9", "AI Super-Resolution (Real-ESRGAN)"),
        ("10", "Full Pipeline Summary"),
    ]
    pdf.set_font("Helvetica", "", 11)
    for num, title in toc:
        pdf.set_text_color(*DSPConceptsPDF.BODY)
        pdf.cell(0, 7, f"  {num:<6}{title}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ══════════════════════════════════════════════════════════════════
    # 1. SAMPLING & QUANTIZATION
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title(1, "Sampling & Quantization")

    pdf.body_text(
        "An image is a 2-D discrete signal. When a camera captures a scene, two "
        "fundamental DSP operations happen simultaneously:"
    )
    pdf.sub_heading("Spatial Sampling")
    pdf.body_text(
        "The continuous light field is sampled at a finite grid of H x W locations "
        "(pixels). Each pixel represents one sample of the scene's intensity at "
        "that spatial coordinate. In the project, this discrete grid arrives as a "
        "NumPy array of shape (H, W, 3) when loaded via Pillow."
    )
    pdf.sub_heading("Amplitude Quantization")
    pdf.body_text(
        "Each sample's continuous brightness is quantized to one of 256 discrete "
        "levels (0-255) and stored as an unsigned 8-bit integer (uint8). This "
        "is uniform scalar quantization with 8-bit depth, giving 2^8 = 256 "
        "possible values per channel."
    )
    pdf.formula_box("Q(x) = round(x * 255)  ->  uint8   [0, 255]",
                     "Quantization: continuous intensity mapped to 8-bit integer")

    pdf.code_box("loader.py  ->  load_image()", 
                 'image = image.convert("RGB")\n'
                 'image_array = np.array(image)    # shape (H, W, 3), dtype uint8')

    # ══════════════════════════════════════════════════════════════════
    # 2. NORMALIZATION
    # ══════════════════════════════════════════════════════════════════
    pdf.section_title(2, "Normalization (ADC Analogy)")
    pdf.body_text(
        "Before any mathematical processing, the integer pixel values are "
        "converted to floating-point in the range [0.0, 1.0]. This is analogous "
        "to an Analog-to-Digital Converter (ADC) producing a normalized "
        "continuous representation from discrete samples."
    )
    pdf.body_text(
        "Why is this necessary?  Interpolation kernels compute weighted sums "
        "of neighbours. Operating in integer space would cause truncation errors "
        "(e.g., 127 / 255 = 0 in integer division). Float64 preserves full "
        "precision throughout the pipeline."
    )
    pdf.formula_box("x_norm = x_uint8 / 255.0        float64 in [0.0, 1.0]",
                     "Normalization: discrete integers to continuous float space")

    pdf.code_box("loader.py  ->  normalize()",
                 'normalized = channel_array.astype(np.float64) / 255.0\n'
                 'normalized = np.clip(normalized, 0.0, 1.0)  # defensive clip')

    # ══════════════════════════════════════════════════════════════════
    # 3. CHANNEL DECOMPOSITION
    # ══════════════════════════════════════════════════════════════════
    pdf.section_title(3, "Channel Decomposition")
    pdf.body_text(
        "An RGB image is a vector-valued 2-D signal with three components "
        "(Red, Green, Blue). The project decomposes this into three independent "
        "scalar signals, each processed through the same pipeline separately."
    )
    pdf.body_text(
        "This is valid because the interpolation kernels are linear and "
        "channel-independent: each colour channel can be upsampled in isolation "
        "and recombined at the end without cross-channel interference."
    )
    pdf.formula_box("I(r, c) = [R(r,c), G(r,c), B(r,c)]   ->   3 independent 2-D signals")

    pdf.code_box("loader.py  ->  split_channels()",
                 'r = image_array[:, :, 0].copy()\n'
                 'g = image_array[:, :, 1].copy()\n'
                 'b = image_array[:, :, 2].copy()')

    # ══════════════════════════════════════════════════════════════════
    # 4. SPARSE GRID & UPSAMPLING
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title(4, "Sparse Grid & Upsampling")
    pdf.body_text(
        "The core of the upscaling process is creating a larger output grid "
        "and placing the known pixel samples at regularly spaced positions. "
        "This is a direct implementation of the DSP concept of upsampling "
        "(inserting zeros / NaN between samples)."
    )

    pdf.sub_heading("Upsampling by Factor L")
    pdf.body_text(
        "Given scale_factor L, the output grid has dimensions (H*L, W*L). "
        "Each original pixel at position (r, c) is placed at (r*L, c*L) in "
        "the new grid. All other positions are filled with NaN (not zero, "
        "because 0.0 is a valid pixel value representing black)."
    )
    pdf.formula_box("sparse[r*L, c*L] = original[r, c]     all other positions = NaN",
                     "Upsampling rule: place known samples at multiples of L")

    pdf.body_text(
        "The fill ratio of the sparse grid is exactly 1/L^2. For a 4x upscale, "
        "only 1/16 of the output pixels are known; the remaining 15/16 must be "
        "computed by interpolation. This sparse-to-dense reconstruction is the "
        "central DSP challenge of the project."
    )

    pdf.code_box("grid.py  ->  map_pixels()",
                 'empty_grid = np.full((H*L, W*L), np.nan, dtype=np.float64)\n'
                 'for r in range(H):\n'
                 '    for c in range(W):\n'
                 '        empty_grid[r*L, c*L] = channel[r, c]')

    # ══════════════════════════════════════════════════════════════════
    # 5. INTERPOLATION KERNELS
    # ══════════════════════════════════════════════════════════════════
    pdf.section_title(5, "Interpolation Kernels")
    pdf.body_text(
        "Interpolation reconstructs the continuous signal from discrete samples, "
        "then re-samples it at the new (higher) resolution. The quality of this "
        "reconstruction depends entirely on the kernel (also called the "
        "interpolation filter) used to weight the contributions of known neighbours."
    )
    pdf.formula_box(
        "f(x) = SUM[ f(n) * h(x - n) ]     for all known neighbours n",
        "General interpolation formula: weighted sum of neighbours"
    )

    # 5.1 Bicubic
    pdf.sub_heading("5.1  Bicubic (Keys') Kernel")
    pdf.body_text(
        "The bicubic kernel is a piecewise cubic polynomial that uses a 4x4 "
        "neighbourhood (16 known pixels) around each unknown position. The "
        "parameter a = -0.5 (Keys' value) gives the best approximation to the "
        "ideal sinc reconstructor among all piecewise cubics."
    )
    pdf.formula_box(
        "h(t) = (a+2)|t|^3 - (a+3)|t|^2 + 1          for |t| <= 1\n"
        "      h(t) = a|t|^3 - 5a|t|^2 + 8a|t| - 4a        for 1 < |t| <= 2\n"
        "      h(t) = 0                                       for |t| > 2",
        "Keys' bicubic kernel (a = -0.5)"
    )
    pdf.body_text(
        "Properties:  Continuous first derivative (smooth curve). "
        "Compact support (radius = 2). Weights sum to 1 for interior pixels. "
        "Produces smooth results with mild blurring."
    )

    pdf.code_box("interpolation.py  ->  bicubic_kernel(t, a=-0.5)",
                 'abs_t = abs(t)\n'
                 'if abs_t <= 1.0:\n'
                 '    return (a + 2.0)*abs_t**3 - (a + 3.0)*abs_t**2 + 1.0\n'
                 'elif abs_t <= 2.0:\n'
                 '    return a*abs_t**3 - 5.0*a*abs_t**2 + 8.0*a*abs_t - 4.0*a')

    # 5.2 Lanczos
    pdf.add_page()
    pdf.sub_heading("5.2  Lanczos (Windowed Sinc) Kernel")
    pdf.body_text(
        "The ideal reconstruction filter is the sinc function, which has "
        "infinite support (it never decays to zero). The Lanczos kernel "
        "approximates the ideal sinc by windowing it with a scaled copy of "
        "itself, truncated to a finite support of radius 'a' (typically 3)."
    )
    pdf.formula_box(
        "L(t) = sinc(t) * sinc(t/a)      for |t| < a\n"
        "      L(t) = 0                          for |t| >= a",
        "Lanczos-a kernel (a=3: 6x6 neighbourhood)"
    )
    pdf.body_text(
        "The normalized sinc is defined as sinc(x) = sin(pi*x) / (pi*x). "
        "NumPy's np.sinc() already uses this normalized form."
    )
    pdf.body_text(
        "Properties:  Closest practical approximation to ideal reconstruction. "
        "Produces sharper output than bicubic. Negative side-lobes cause mild "
        "'ringing' artifacts (values slightly outside [0, 1]) near high-contrast "
        "edges. These are clipped during denormalization."
    )

    pdf.code_box("interpolation.py  ->  lanczos_kernel(t, a=3)",
                 'if t == 0.0:  return 1.0\n'
                 'if abs(t) >= a:  return 0.0\n'
                 'return float(np.sinc(t) * np.sinc(t / a))')

    # 5.3 Separable kernels
    pdf.sub_heading("5.3  Separable 2-D Kernels")
    pdf.body_text(
        "Both bicubic and Lanczos kernels are separable: the 2-D weight for "
        "a neighbour at displacement (dr, dc) is the product of two 1-D kernel "
        "evaluations. This property is fundamental to efficient implementation."
    )
    pdf.formula_box(
        "w_2D(dr, dc) = h(dr) * h(dc)",
        "Separability: 2-D weight = product of two 1-D weights"
    )
    pdf.body_text(
        "An N x N neighbourhood with a non-separable kernel requires N^2 "
        "multiplications, while a separable kernel needs only 2N. The project "
        "exploits this in interpolate_pixel() where w_row and w_col are "
        "computed independently."
    )

    pdf.code_box("interpolation.py  ->  interpolate_pixel()",
                 'for value, row_dist, col_dist in neighbors:\n'
                 '    w_row = kernel_fn(row_dist)    # 1-D evaluation\n'
                 '    w_col = kernel_fn(col_dist)    # 1-D evaluation\n'
                 '    weight = w_row * w_col          # separability\n'
                 '    weighted_sum += weight * value')

    # ══════════════════════════════════════════════════════════════════
    # 6. BOUNDARY HANDLING
    # ══════════════════════════════════════════════════════════════════
    pdf.section_title(6, "Boundary Handling (Edge Extension)")
    pdf.body_text(
        "When interpolating near image edges, the kernel window extends beyond "
        "the image boundary. The project uses 'clamp-to-edge' (replicate) "
        "padding: out-of-bounds indices are clamped to the nearest valid pixel."
    )
    pdf.formula_box(
        "n_clamped = max(0, min(N-1, n))",
        "Clamp-to-edge boundary extension"
    )
    pdf.body_text(
        "This is equivalent to extending the border pixels outward infinitely. "
        "It avoids introducing artificial dark edges (zero-padding) or "
        "discontinuities (wrap-around). SciPy's map_coordinates uses "
        "mode='nearest' which implements identical clamping behaviour."
    )

    # ══════════════════════════════════════════════════════════════════
    # 7. UNSHARP MASK
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title(7, "Unsharp Mask (Post-Processing)")
    pdf.body_text(
        "Interpolation inherently acts as a low-pass filter: it smooths the "
        "image by computing weighted averages. While this eliminates pixelation, "
        "it also softens edges. The UnsharpMask is a classic DSP technique to "
        "restore perceived sharpness."
    )

    pdf.sub_heading("Algorithm")
    pdf.bullet("Blur the image with a Gaussian of radius r pixels")
    pdf.bullet("Subtract the blurred version from the original to extract high-frequency edges")
    pdf.bullet("Add a percentage of the extracted edges back to the original")
    pdf.bullet("Only sharpen where the difference exceeds a threshold (suppresses noise)")

    pdf.formula_box(
        "sharpened = original + percent/100 * (original - blur(original))",
        "UnsharpMask formula (applied only where |diff| > threshold)"
    )

    pdf.body_text(
        "In DSP terms, this is high-frequency emphasis (high-boost filtering). "
        "The Gaussian blur acts as a low-pass filter; subtracting it yields a "
        "band-pass (edge) component; adding it back amplifies high frequencies "
        "relative to low frequencies."
    )

    pdf.code_box("saver.py  ->  sharpen_image()",
                 'pil_image = Image.fromarray(image_array, mode="RGB")\n'
                 'sharpened = pil_image.filter(\n'
                 '    ImageFilter.UnsharpMask(\n'
                 '        radius=1.5, percent=120, threshold=3\n'
                 '    )\n'
                 ')')

    # ══════════════════════════════════════════════════════════════════
    # 8. DENORMALIZATION
    # ══════════════════════════════════════════════════════════════════
    pdf.section_title(8, "Denormalization (DAC Analogy)")
    pdf.body_text(
        "After interpolation, the float64 values must be converted back to "
        "uint8 integers for storage and display. This is analogous to a "
        "Digital-to-Analog Converter (DAC)."
    )
    pdf.body_text(
        "The process has a careful ordering to avoid artifacts:"
    )
    pdf.bullet("Multiply by 255.0 to scale back to integer range")
    pdf.bullet("Clip to [0, 255] to handle Lanczos ringing overshoot")
    pdf.bullet("Round (not truncate) to avoid systematic darkening bias")
    pdf.bullet("Cast to uint8")

    pdf.formula_box(
        "x_uint8 = uint8( round( clip(x_float * 255, 0, 255) ) )",
        "Denormalization: float64 back to 8-bit integer"
    )

    pdf.code_box("saver.py  ->  denormalize()",
                 'scaled = channel_array * 255.0\n'
                 'clipped = np.clip(scaled, 0.0, 255.0)\n'
                 'rounded = np.round(clipped)\n'
                 'denormalized = rounded.astype(np.uint8)')

    # ══════════════════════════════════════════════════════════════════
    # 9. AI SUPER-RESOLUTION
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title(9, "AI Super-Resolution (Real-ESRGAN)")
    pdf.body_text(
        "Beyond classical DSP interpolation, the project includes an optional "
        "AI-driven super-resolution path using Real-ESRGAN, a deep neural "
        "network trained on millions of image pairs."
    )

    pdf.sub_heading("How It Differs from Classical DSP")
    pdf.bullet(
        "Classical kernels (bicubic, Lanczos) are fixed mathematical functions "
        "that rely only on local pixel neighbourhoods."
    )
    pdf.bullet(
        "Real-ESRGAN uses a learned RRDB (Residual-in-Residual Dense Block) "
        "network that captures semantic understanding: it can synthesize "
        "plausible texture details that no fixed kernel could produce."
    )
    pdf.bullet(
        "The GFPGAN face enhancement module adds even more domain-specific "
        "knowledge, restoring facial features with dedicated priors."
    )

    pdf.sub_heading("DSP Connection")
    pdf.body_text(
        "Neural super-resolution can be understood as a learned, non-linear, "
        "adaptive interpolation filter. Where classical filters apply "
        "the same weights everywhere, the network adapts its 'filter' "
        "to the local content of each image patch."
    )

    # ══════════════════════════════════════════════════════════════════
    # 10. FULL PIPELINE SUMMARY
    # ══════════════════════════════════════════════════════════════════
    pdf.section_title(10, "Full Pipeline Summary")

    steps = [
        ("LOAD",          "loader.py",         "Decode image file -> (H,W,3) uint8 array"),
        ("VALIDATE",      "loader.py",         "Check shape, dtype, range, minimum size"),
        ("SPLIT",         "loader.py",         "Decompose RGB -> 3 independent channels"),
        ("NORMALIZE",     "loader.py",         "uint8 [0,255] -> float64 [0.0,1.0]  (ADC)"),
        ("UPSAMPLE",      "grid.py",           "Create (H*L, W*L) sparse grid with NaN gaps"),
        ("MAP PIXELS",    "grid.py",           "Place known samples at (r*L, c*L) positions"),
        ("INTERPOLATE",   "interpolation.py",  "Fill NaN positions via bicubic or Lanczos"),
        ("DENORMALIZE",   "saver.py",          "float64 [0.0,1.0] -> uint8 [0,255]  (DAC)"),
        ("MERGE",         "saver.py",          "Recombine R, G, B channels into RGB array"),
        ("SHARPEN",       "saver.py",          "Optional UnsharpMask high-frequency boost"),
        ("SAVE",          "saver.py",          "Write final PNG to disk"),
        ("AI ENHANCE",    "enhancer.py",       "Optional Real-ESRGAN neural upscaling"),
    ]

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 235, 250)
    pdf.set_text_color(40, 40, 60)
    pdf.cell(30, 7, "  Stage", border=1, fill=True)
    pdf.cell(35, 7, "  Module", border=1, fill=True)
    pdf.cell(0, 7, "  DSP Operation", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    for i, (stage, module, desc) in enumerate(steps):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(245, 248, 255)
        pdf.cell(30, 6, f"  {stage}", border=1, fill=fill)
        pdf.cell(35, 6, f"  {module}", border=1, fill=fill)
        pdf.cell(0, 6, f"  {desc}", border=1, fill=fill, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*DSPConceptsPDF.LIGHT_GRAY)
    pdf.cell(0, 6, "End of document.", align="C")

    # ── Save ──────────────────────────────────────────────────────────
    output_path = "DSP_Concepts_Image_Upscaler.pdf"
    pdf.output(output_path)
    return output_path


if __name__ == "__main__":
    path = build_pdf()
    print(f"PDF generated: {path}")

#!/usr/bin/env python3
"""
Convert RESEARCH_PAPER_DRAFT.md to PDF and DOCX
Includes embedded figures from the project directory.
"""

import pypandoc
import os
import shutil

PROJECT_DIR = "/data/gait/2d_project"
OUTPUT_DIR = "/data/gait/2d_project/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Key figures to include (copy to output dir for relative paths)
FIGURES = [
    "segmentation_comparison.png",
    "gt_free_scalars_S1.png",
    "left_vs_right_S1.png",
    "ablation_dtw_S1.png",
    "fp_vs_atdtw_demo.png",
    "frontal_N26_grid.png",
]

# Copy figures to output
for fig in FIGURES:
    src = os.path.join(PROJECT_DIR, fig)
    dst = os.path.join(OUTPUT_DIR, fig)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied: {fig}")
    else:
        print(f"Missing: {fig}")

# Read markdown
md_path = os.path.join(PROJECT_DIR, "RESEARCH_PAPER_DRAFT.md")
with open(md_path, 'r') as f:
    md_content = f.read()

# Add figure references if not already present (append at end)
figure_section = """

---

## Appendix: Key Figures

### Figure 1: AT-RTM Cycle Segmentation Example
![Segmentation Comparison](segmentation_comparison.png)

### Figure 2: Force Plate vs AT-RTM Detection
![FP vs AT-RTM](fp_vs_atdtw_demo.png)

### Figure 3: GT-Free Scalar Parameter Extraction
![GT-Free Scalars](gt_free_scalars_S1.png)

### Figure 4: Left vs Right Knee Comparison
![Left vs Right](left_vs_right_S1.png)

### Figure 5: DTW Ablation Study
![DTW Ablation](ablation_dtw_S1.png)

"""

# Check if figures already referenced
if "![" not in md_content:
    md_content += figure_section

# Write temp file
temp_md = os.path.join(OUTPUT_DIR, "paper_with_figures.md")
with open(temp_md, 'w') as f:
    f.write(md_content)
print(f"Created: {temp_md}")

# Convert to DOCX
docx_path = os.path.join(OUTPUT_DIR, "RESEARCH_PAPER.docx")
try:
    pypandoc.convert_file(temp_md, 'docx', outputfile=docx_path)
    print(f"✅ Created: {docx_path}")
except Exception as e:
    print(f"DOCX Error: {e}")

# Convert to PDF (requires LaTeX or wkhtmltopdf)
pdf_path = os.path.join(OUTPUT_DIR, "RESEARCH_PAPER.pdf")
try:
    pypandoc.convert_file(temp_md, 'pdf', outputfile=pdf_path,
                          extra_args=['--pdf-engine=xelatex', '-V', 'geometry:margin=1in'])
    print(f"✅ Created: {pdf_path}")
except Exception as e:
    print(f"PDF Error (trying html): {e}")
    # Fallback: HTML to PDF via wkhtmltopdf or weasyprint
    try:
        html_path = os.path.join(OUTPUT_DIR, "RESEARCH_PAPER.html")
        pypandoc.convert_file(temp_md, 'html', outputfile=html_path,
                              extra_args=['--standalone', '--self-contained'])
        print(f"✅ Created HTML: {html_path}")
    except Exception as e2:
        print(f"HTML Error: {e2}")

print("\n✅ Conversion complete!")
print(f"Output directory: {OUTPUT_DIR}")

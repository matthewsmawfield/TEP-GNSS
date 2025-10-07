# HTML to PDF Converter

## Overview

This package provides advanced HTML to PDF conversion capabilities with fine-grained control over output quality, page breaks, scaling, and formatting. Built on Playwright (Chromium) for maximum compatibility and quality.

## Features

- **High-Quality Output**: Crisp images, proper font rendering, exact color reproduction
- **Smart Page Breaks**: Automatic avoidance of breaking figures, tables, and headings
- **Flexible Scaling**: Custom zoom levels and device pixel ratios for optimal quality
- **Professional Formatting**: Headers, footers, custom margins, multiple page formats
- **Wait Controls**: Wait for dynamic content, selectors, or specific timeouts
- **Preset Configurations**: Pre-optimized settings for common use cases

## Installation

```bash
# Install dependencies
pip install -r requirements/requirements.txt

# Setup Playwright browser (one-time)
./scripts/utils/setup_pdf_converter.sh
```

## Usage

### Basic Conversion

```bash
# Simple HTML to PDF
python scripts/utils/html_to_pdf.py input.html output.pdf

# With quality preset
python scripts/utils/html_to_pdf.py input.html output.pdf --preset high_quality
```

### Advanced Options

```bash
# Custom scaling and format
python scripts/utils/html_to_pdf.py input.html output.pdf \
  --scale 1.5 \
  --format A3 \
  --dpi 300 \
  --margin-top 2cm

# With headers and footers
python scripts/utils/html_to_pdf.py input.html output.pdf \
  --header-footer \
  --wait-time 5

# Custom page dimensions
python scripts/utils/html_to_pdf.py input.html output.pdf \
  --width 21cm \
  --height 29.7cm \
  --scale 1.2
```

### Site-Specific Converter

```bash
# Generate PDF from the main TEP-GNSS site
python scripts/generate_site_pdf.py

# With custom output path
python scripts/generate_site_pdf.py -o custom_report.pdf

# Different preset
python scripts/generate_site_pdf.py --preset large_format
```

## Available Presets

- **`high_quality`**: 2x device scale, A4, optimized margins, headers/footers
- **`print_ready`**: Standard print format with page numbers
- **`web_optimized`**: Compact layout for web content
- **`large_format`**: A3 format with enhanced scaling

## Advanced Features

### Custom CSS Injection

```python
# Example: Custom CSS for better PDF rendering
custom_css = """
@media print {
    .no-break { page-break-inside: avoid !important; }
    img { image-rendering: crisp-edges !important; }
}
"""
```

### Wait Strategies

```bash
# Wait for specific content to load
python scripts/utils/html_to_pdf.py input.html output.pdf \
  --wait-for-selector ".content-loaded"

# Wait fixed time for dynamic content
python scripts/utils/html_to_pdf.py input.html output.pdf \
  --wait-time 10
```

### Quality Control

- **Device Scale Factor**: Controls image sharpness (1.0-3.0)
- **Scale**: Page zoom level (0.1-2.0) 
- **Format**: A4, A3, Letter, Legal, Tabloid, or custom dimensions
- **Background Printing**: Always enabled for full visual fidelity

## Command Line Reference

```
positional arguments:
  input                 Input HTML file path
  output                Output PDF file path

optional arguments:
  --preset {high_quality,print_ready,web_optimized,large_format}
                        Use preset configuration
  --format FORMAT       Page format (A4, A3, Letter, Legal, Tabloid)
  --width WIDTH         Custom page width (e.g., 21cm, 8.5in)
  --height HEIGHT       Custom page height (e.g., 29.7cm, 11in)
  --scale SCALE         Scale factor for the page (0.1 to 2.0)
  --dpi DPI             Device scale factor equivalent (96, 150, 192, 300)
  --margin-top MARGIN_TOP     Top margin
  --margin-right MARGIN_RIGHT Right margin  
  --margin-bottom MARGIN_BOTTOM Bottom margin
  --margin-left MARGIN_LEFT   Left margin
  --header-footer       Enable headers and footers
  --header-template HEADER_TEMPLATE Custom header HTML template
  --footer-template FOOTER_TEMPLATE Custom footer HTML template
  --wait-time WAIT_TIME Wait time in seconds before PDF generation
  --wait-for-selector WAIT_FOR_SELECTOR CSS selector to wait for
  --page-ranges PAGE_RANGES Page ranges to print (e.g., "1-3, 5, 8-")
  --css-page-size       Prefer CSS page size over format
  --custom-css-file CUSTOM_CSS_FILE Path to custom CSS file
  --viewport-width VIEWPORT_WIDTH Browser viewport width
  --viewport-height VIEWPORT_HEIGHT Browser viewport height
```

## Technical Details

- **Engine**: Playwright Chromium (headless)
- **CSS Support**: Full modern CSS including Flexbox, Grid, custom fonts
- **Image Handling**: Automatic optimization, crisp rendering at high DPI
- **Page Breaks**: CSS `page-break-*` properties fully supported
- **Print Media**: Proper `@media print` CSS evaluation

## Troubleshooting

### Common Issues

1. **Missing Browser**: Run `./scripts/utils/setup_pdf_converter.sh`
2. **Images Not Loading**: Check file paths are absolute or use `--wait-time`
3. **Font Issues**: Ensure system fonts are available or embed web fonts
4. **Large Files**: Use `--wait-time` for complex content or large images

### Performance Tips

- Use `--preset web_optimized` for faster generation
- Set appropriate `--wait-time` (2-5 seconds usually sufficient)
- Consider `--scale` < 1.0 for very large documents
- Use `--page-ranges` to generate partial documents during testing

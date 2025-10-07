#!/usr/bin/env python3
"""
PDF Generator for TEP-GNSS Site
===============================
Simple script to generate PDF from localhost:8347
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'utils'))
from html_to_pdf import HTMLToPDFConverter

logger = logging.getLogger(__name__)


async def main():
    """Generate PDF from localhost site."""
    
    project_root = Path(__file__).parent.parent
    
    # Read version info
    import json
    version_file = project_root / 'VERSION.json'
    version_info = json.loads(version_file.read_text())
    version = version_info['version'].split('.')[0] + '.' + version_info['version'].split('.')[1]  # Just major.minor
    codename = version_info['codename'].split('-')[0]  # Remove any suffix after dash
    
    # Build filename from version info
    filename = f"Smawfield_2025_GlobalTimeEchoes_v{version}_{codename}.pdf"
    output_path = project_root / 'results' / filename
    public_path = project_root / 'site' / 'public' / 'docs' / filename
    
    options = {
        'scale': 0.65,  # Back to previous better setting
        'device_scale_factor': 1.5,  # Back to previous better setting
        'format': 'A4',
        'margin_top': '1.5cm',
        'margin_bottom': '1.5cm', 
        'margin_left': '1.2cm',
        'margin_right': '1.2cm',
        'display_header_footer': True,
        'header_template': '<div style="font-size:8px; text-align:center; width:100%; color:#666;">Global Time Echoes: Distance-Structured Correlations in GNSS Clocks | Matthew Lukin Smawfield</div>',
        'footer_template': '<div style="font-size:8px; text-align:center; width:100%; color:#666;">Page <span class="pageNumber"></span> of <span class="totalPages"></span> | TEP-GNSS Analysis</div>',
        'wait_time': 5,  # Reduced from 8 seconds
        'viewport': {'width': 1600, 'height': 900},  # Slightly smaller viewport
        'custom_css': '''
            @media print {
                body, p, div, span, td, th { font-weight: normal !important; }
                h1, h2, h3, h4, h5, h6 { font-weight: bold !important; }
                img { 
                    max-width: 100% !important; 
                    height: auto !important; 
                    page-break-inside: avoid !important;
                    image-rendering: auto !important;
                }
                figure { page-break-inside: avoid !important; text-align: center !important; }
                * { print-color-adjust: exact !important; }
            }
        '''
    }
    
    logger.info("🌐 Generating PDF from http://localhost:8347/")
    
    async with HTMLToPDFConverter() as converter:
        converter.context = await converter.browser.new_context(
            viewport=options['viewport'],
            device_scale_factor=options['device_scale_factor']
        )
        page = await converter.context.new_page()
        
        try:
            # Load site
            await page.goto('http://localhost:8347/', wait_until='networkidle', timeout=60000)
            await page.add_style_tag(content=converter._get_css_for_pdf(options))
            await page.wait_for_timeout(options['wait_time'] * 1000)
            
            # Scroll to load lazy images - optimized for speed
            logger.info("📜 Loading lazy images...")
            await page.evaluate("""
                async () => {
                    const totalHeight = document.body.scrollHeight;
                    const viewportHeight = window.innerHeight;
                    // Faster scrolling with fewer stops
                    for (let y = 0; y < totalHeight; y += viewportHeight) {
                        window.scrollTo(0, y);
                        await new Promise(resolve => setTimeout(resolve, 200)); // Reduced from 500ms
                    }
                    window.scrollTo(0, 0);
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
            """)
            
            # Quick image check - don't wait too long
            try:
                await page.wait_for_function("""() => {
                    const images = Array.from(document.querySelectorAll('img'));
                    return images.length === 0 || images.every(img => img.complete && img.naturalHeight !== 0);
                }""", timeout=10000)  # Reduced from 30 seconds
                logger.info("✅ Images loaded")
            except:
                logger.info("⏭️ Proceeding with partial image loading")
            
            # Generate PDF
            logger.info("🎯 Generating PDF...")
            pdf_options = converter._build_pdf_options(options)
            await page.pdf(path=str(output_path), **pdf_options)
            
            file_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ PDF created: {output_path.name} ({file_size:.1f} MB)")
            
            # Copy to site public folder
            public_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(output_path, public_path)
            logger.info(f"📋 Copied to: {public_path}")
            
        finally:
            await converter.context.close()
    
    # Clean up temporary files
    logger.info("🧹 Cleaning up temporary files...")
    temp_files = [
        project_root / 'results' / 'compiled_site.html',
        project_root / 'results' / 'compiled_site_fixed.html', 
        project_root / 'results' / 'compiled_site_with_images.html'
    ]
    for temp_file in temp_files:
        if temp_file.exists():
            temp_file.unlink()
            logger.info(f"🗑️  Removed: {temp_file.name}")
    
    # Clean up temporary image folders
    temp_dirs = [
        project_root / 'results' / 'figures',
        project_root / 'results' / 'og-image.jpg',
        project_root / 'results' / 'twitter-image.jpg'
    ]
    for temp_item in temp_dirs:
        if temp_item.exists():
            if temp_item.is_dir():
                import shutil
                shutil.rmtree(temp_item)
                logger.info(f"🗑️  Removed directory: {temp_item.name}")
            else:
                temp_item.unlink()
                logger.info(f"🗑️  Removed: {temp_item.name}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    asyncio.run(main())

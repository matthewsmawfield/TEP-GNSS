#!/usr/bin/env python3
"""
Component Management Utility for TEP-GNSS Manuscript
Provides tools for managing the modular HTML components
"""

import json
import os
from pathlib import Path

def load_manifest():
    """Load the manifest.json file"""
    with open('manifest.json', 'r') as f:
        return json.load(f)

def list_components():
    """List all components with their sizes"""
    manifest = load_manifest()
    print(f"\n📄 {manifest['title']}")
    print(f"👤 {manifest['author']} - {manifest['version']}")
    print("=" * 80)
    
    total_lines = 0
    for section in sorted(manifest['sections'], key=lambda x: x['order']):
        file_path = Path('components') / section['file']
        if file_path.exists():
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
            total_lines += lines
            print(f"{section['order']:2d}. {section['title']:<40} ({lines:4d} lines) - {section['file']}")
        else:
            print(f"{section['order']:2d}. {section['title']:<40} (MISSING) - {section['file']}")
    
    print("=" * 80)
    print(f"Total component lines: {total_lines}")
    
    # Check main index.html
    if Path('index.html').exists():
        with open('index.html', 'r') as f:
            index_lines = len(f.readlines())
        print(f"Main index.html: {index_lines} lines")
        print(f"Reduction: {total_lines + index_lines} → {index_lines} main file + {len(manifest['sections'])} components")

def reorder_sections(new_order):
    """Reorder sections in the manifest
    
    Args:
        new_order: List of section IDs in new order
    """
    manifest = load_manifest()
    
    # Update order numbers
    for i, section_id in enumerate(new_order, 1):
        for section in manifest['sections']:
            if section['id'] == section_id:
                section['order'] = i
                break
    
    # Save updated manifest
    with open('manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Reordered sections: {' → '.join(new_order)}")

def add_section(section_id, title, filename, position=None):
    """Add a new section to the manifest"""
    manifest = load_manifest()
    
    if position is None:
        position = len(manifest['sections']) + 1
    
    new_section = {
        "id": section_id,
        "file": filename,
        "title": title,
        "order": position
    }
    
    # Adjust order numbers for existing sections
    for section in manifest['sections']:
        if section['order'] >= position:
            section['order'] += 1
    
    manifest['sections'].append(new_section)
    
    # Save updated manifest
    with open('manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Added section: {title} ({filename}) at position {position}")

def validate_components():
    """Validate that all component files exist and are readable"""
    manifest = load_manifest()
    missing = []
    total_size = 0
    
    print("🔍 Validating components...")
    
    for section in manifest['sections']:
        file_path = Path('components') / section['file']
        if file_path.exists():
            size = file_path.stat().st_size
            total_size += size
            print(f"✅ {section['file']:<35} ({size:6d} bytes)")
        else:
            missing.append(section['file'])
            print(f"❌ {section['file']:<35} (MISSING)")
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} component(s): {', '.join(missing)}")
        return False
    else:
        print(f"\n✅ All {len(manifest['sections'])} components validated ({total_size:,} bytes total)")
        return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_components.py list          - List all components")
        print("  python manage_components.py validate      - Validate all components exist")
        print("  python manage_components.py reorder id1,id2,id3  - Reorder sections")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        list_components()
    elif command == "validate":
        validate_components()
    elif command == "reorder" and len(sys.argv) == 3:
        new_order = sys.argv[2].split(',')
        reorder_sections(new_order)
    else:
        print("Unknown command or missing arguments")
        sys.exit(1)

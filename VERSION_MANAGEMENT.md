# TEP-GNSS Version Management System

## Overview
Centralized version management system that automatically updates version numbers across all project files from a single source of truth.

## Quick Start

### View Current Version
```bash
python scripts/utils/version_manager.py --info
```

### Update Core Files
```bash
python scripts/utils/version_manager.py --core
```

### Update Python Scripts (Dynamic Loading)
```bash
python scripts/utils/version_manager.py --python
```

### Check Consistency
```bash
python scripts/utils/version_manager.py --check
```

### Full System Info
```bash
python scripts/utils/version_manager.py --system
```

## How It Works

1. **Single Source of Truth**: All version information is stored in `VERSION.json`
2. **Dynamic Loading**: Scripts read version data dynamically from `VERSION.json`
3. **Python Script Integration**: Python scripts use `from scripts.utils.version_utils import VERSION_STRING`
4. **Multi-Format Support**: Handles Python, HTML, Markdown, JSON, CFF, TOML files
5. **Smart Exclusions**: Automatically excludes virtual environments and output data

## File Structure

```
VERSION.json                    # Central version file
scripts/utils/version_manager.py # Single consolidated script
scripts/utils/version_utils.py   # Python version utilities
```

## Updating Version

1. Edit `VERSION.json` with new version details
2. Run: `python scripts/utils/version_manager.py --core`
3. Verify: `python scripts/utils/version_manager.py --check`
4. Commit changes to git

## Benefits

- ✅ Single source of truth
- ✅ Automated updates across all file types
- ✅ Consistent versioning
- ✅ Easy change tracking
- ✅ Reduced human error
- ✅ Professional presentation

## Current Status
- **Version**: v0.19 (Jaipur)
- **Date**: 2025-10-13
- **System**: ✅ Active and ready

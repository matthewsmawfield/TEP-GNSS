# Multi-Paper Site Structure

## Overview

The TEP-GNSS project now hosts **two independent papers** with separate websites, deployed to a single GitHub Pages site with different routes.

## Structure

```
TEP-GNSS/
├── site/                      # Paper 1: Multi-Center Validation (2.5 years)
│   ├── index.html
│   ├── components/
│   ├── manifest.json
│   ├── build.js
│   └── dist/                  # Built output
│
├── site-code-longspan/        # Paper 2: 25-Year Confirmatory Analysis
│   ├── index.html
│   ├── components/
│   ├── manifest.json
│   ├── build.js
│   └── dist/                  # Built output
│
├── deploy.sh                  # Original deploy script (Paper 1 only)
├── deploy-all.sh             # NEW: Deploy both papers
└── final-dist/               # Temporary: Combined deployment (auto-generated)
```

## URLs

**Production (GitHub Pages):**
- Paper 1: `https://matthewsmawfield.github.io/TEP-GNSS/`
- Paper 2: `https://matthewsmawfield.github.io/TEP-GNSS/code-longspan/`

**Local Development:**
- Paper 1: `http://localhost:8347/` (via `npm run dev`)
- Paper 2: `http://localhost:8347/` (via `npm run dev` in `site-code-longspan/`)

## Development Workflow

### Working on Paper 1 (Multi-Center)
```bash
cd site
npm run dev          # Develop with live reload
npm run build        # Build for production
```

### Working on Paper 2 (CODE Longspan)
```bash
cd site-code-longspan
npm run dev          # Develop with live reload
npm run build        # Build for production
```

### Deploying Both Papers
```bash
./deploy-all.sh      # Builds and deploys both papers
```

## Key Design Decisions

1. **Complete Isolation**: Each paper has its own complete site folder with independent:
   - Build system
   - Dependencies (`package.json`)
   - Components
   - Figures
   - Manifest

2. **Zero Risk to Paper 1**: The original site (`site/`) is completely untouched. All Paper 2 work happens in the separate `site-code-longspan/` folder.

3. **URL Preservation**: All existing Paper 1 links continue to work at the root URL.

4. **Unified Deployment**: The `deploy-all.sh` script builds both sites and combines them into a single deployment.

## Building and Testing

### Build Both Papers Locally
```bash
# Build Paper 1
cd site && npm run build && cd ..

# Build Paper 2
cd site-code-longspan && npm run build && cd ..

# View built Paper 1
cd site/dist && python3 -m http.server 8349

# View built Paper 2
cd site-code-longspan/dist && python3 -m http.server 8350
```

### Test Combined Deployment Locally
```bash
./deploy-all.sh
# Manually inspect final-dist/ folder
cd final-dist && python3 -m http.server 8351
# Visit:
#   http://localhost:8351/ → Paper 1
#   http://localhost:8351/code-longspan/ → Paper 2
```

## Maintenance Notes

- **Updating Paper 1**: Only edit files in `site/`
- **Updating Paper 2**: Only edit files in `site-code-longspan/`
- **Shared Assets**: If you need to share assets (e.g., utility scripts), keep them in the repo root and symlink or reference them from both sites
- **Dependencies**: Each site has its own `node_modules/`. Run `npm install` in each site folder after cloning.

## Netlify Configuration

If deploying to Netlify instead of GitHub Pages, update `netlify.toml` in each site folder:

**site/netlify.toml** (Paper 1):
```toml
[build]
  publish = "dist"
  command = "node build.js"
```

**site-code-longspan/netlify.toml** (Paper 2):
```toml
[build]
  publish = "dist"
  command = "node build.js"
  
[[redirects]]
  from = "/*"
  to = "/code-longspan/:splat"
  status = 200
```

## Future Papers

To add Paper 3, 4, etc.:
1. Copy `site-code-longspan/` to `site-paper3/`
2. Update metadata (title, description, URLs)
3. Update `deploy-all.sh` to include the new paper
4. Deploy to `/paper3/` route

---

**Last Updated**: 2025-11-02  
**Structure Version**: 2.0 (Multi-Paper)

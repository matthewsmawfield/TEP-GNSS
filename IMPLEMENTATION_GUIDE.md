# Inline Styles Migration - Implementation Guide

## Objective
Migrate all inline styles to CSS classes across TEP manuscript repos while preserving exact published appearance.

## Approach

### 1. Preserve Individual Paper Styles
Each paper has slight differences in color palettes and styling:
- **TEP-GNSS (I)**: Purple/Slate (#220126, #495773)
- **TEP-GNSS-RINEX (III)**: Similar purple/slate palette
- **TEP-RBH**: Yellow/Grey palette (#D4AC0D, #2C3E50)
- **TEP-UCD**: Similar to GNSS
- **TEP-SLR**: Similar to GNSS
- **TEP-GL**: Similar to GNSS
- **TEP-GTE**: Similar to GNSS
- **TEP-EXP**: Similar to GNSS

### 2. CSS Architecture

Each repo will have:
```
site/
  styles/
    manuscript.css          # Main stylesheet with paper-specific colors
    manuscript-tables.css   # Table styles (optional, can be in main)
    manuscript-layout.css   # Layout utilities (optional, can be in main)
```

### 3. Migration Process

#### Step 1: Create CSS Stylesheet
- Extract all inline style patterns
- Create semantic CSS classes
- Preserve exact colors, spacing, borders, etc.

#### Step 2: Update Components
- Replace inline `style=""` attributes with `class=""` attributes
- Test each component individually

#### Step 3: Link Stylesheet
- Add `<link rel="stylesheet" href="styles/manuscript.css">` to index.html
- Ensure build process copies CSS to dist/

#### Step 4: Visual Regression Testing
- Take screenshots before migration
- Take screenshots after migration
- Compare pixel-by-pixel to ensure identical rendering

## Common Inline Style Patterns

### Executive Summary
```html
<!-- Before -->
<div style="background: linear-gradient(...); border: 1px solid ...; padding: 25px;">

<!-- After -->
<div class="executive-summary">
```

### Tables
```html
<!-- Before -->
<table style="width: 100%; border-collapse: collapse;">
  <tr style="background-color: rgba(...);">
    <th style="padding: 10px; border: 1px solid ...;">

<!-- After -->
<table class="data-table">
  <tr>
    <th>
```

### Callout Boxes
```html
<!-- Before -->
<div style="background-color: rgba(245, 245, 245, 0.6); border-left: 3px solid #666; padding: 12px;">

<!-- After -->
<div class="callout-box">
```

## Build Process Updates

Each repo's `build.js` needs to:
1. Copy CSS files from `site/styles/` to `dist/styles/`
2. Ensure HTML references correct CSS path
3. Minify CSS for production (optional)

## Testing Checklist

For each manuscript:
- [ ] All inline styles removed from components
- [ ] CSS stylesheet created with paper-specific colors
- [ ] Stylesheet linked in index.html
- [ ] Build process copies CSS to dist/
- [ ] Visual regression test passes
- [ ] Responsive behavior preserved
- [ ] Print styles work correctly
- [ ] PDF generation unaffected

## Deployment

After migration:
1. Build each manuscript: `npm run build`
2. Test locally: `npm run dev`
3. Deploy to GitHub Pages (auto via GitHub Actions)
4. Deploy to mlsmawfield.com: `./scripts/deploy-to-linode.sh`

## Rollback Plan

If issues arise:
1. Git revert to previous commit
2. Rebuild and redeploy
3. Investigate issue
4. Fix and redeploy

## Timeline

- TEP-GNSS: 1-2 hours (pilot)
- TEP-GNSS-RINEX: 1 hour (similar to GNSS)
- TEP-RBH: 1 hour (different palette)
- TEP-UCD: 1 hour
- TEP-SLR: 1 hour
- TEP-GL: 1 hour
- TEP-GTE: 1 hour
- TEP-EXP: 1 hour

Total: ~8-10 hours

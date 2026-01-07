# Inline Styles Migration Guide - TEP-GNSS

## Overview
This document tracks the migration of inline styles to CSS classes for the TEP-GNSS manuscript.

## Color Palette
- Primary: `#220126` (Deep Purple)
- Secondary: `#495773` (Slate Blue)
- Background Gradient: `rgba(34, 1, 38, 0.05)` to `rgba(73, 87, 115, 0.05)`

## CSS Classes Created

### Executive Summary
- `.executive-summary` - Main container with gradient background
- `.summary-section` - White content boxes
- `.summary-conclusion` - Conclusion box with slate background
- `.summary-footnotes` - Footnotes section

### Tables
- `.data-table` - Standard data tables
- `.results-table` - Results tables
- `.comparison-table` - Comparison tables

### Callout Boxes
- `.callout-box` - Standard callout
- `.callout-box-info` - Info callout (slate border)
- `.callout-box-warning` - Warning callout (yellow border)

### Content Boxes
- `.experimental-section` - Experimental sections
- `.highlight-box` - Highlight boxes
- `.stats-box` - Statistics boxes
- `.method-box` - Methodology boxes
- `.validation-box` - Validation boxes
- `.discussion-box` - Discussion boxes

### Utilities
- Spacing: `.mt-0` to `.mt-3`, `.mb-0` to `.mb-3`, `.p-0` to `.p-3`
- Text alignment: `.text-left`, `.text-center`, `.text-right`
- Font sizes: `.text-xs` to `.text-2xl`
- Line heights: `.leading-tight`, `.leading-normal`, `.leading-relaxed`
- Border radius: `.rounded-sm`, `.rounded`, `.rounded-lg`

## Migration Status

### Completed
- [x] `components/executive_summary.html` - All inline styles migrated
- [x] Created `styles/manuscript.css` with comprehensive class system

### In Progress
- [ ] Link stylesheet in `index.html`
- [ ] Migrate remaining components

### Pending
- [ ] `components/section_1_introduction.html`
- [ ] `components/section_2_methods.html`
- [ ] `components/section_3_results.html`
- [ ] `components/section_4_validation.html`
- [ ] `components/section_5_discussion.html`
- [ ] `components/section_6_conclusions.html`
- [ ] `components/references.html`

## Testing Checklist
- [ ] Visual regression test - compare before/after screenshots
- [ ] Verify all colors match original
- [ ] Check responsive behavior
- [ ] Validate print stylesheet
- [ ] Test PDF generation

## Notes
- All inline styles preserve exact colors and spacing from published version
- CSS classes are semantic and reusable
- Stylesheet is modular and can be adapted for other TEP manuscripts

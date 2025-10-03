# TEP-GNSS Manuscript Component System

The manuscript has been restructured into a modular component-based system for better maintainability and AI-assisted editing.

## 📁 Structure

```
site/
├── index.html                 # Main file with component loading logic (774 lines)
├── manifest.json             # Section configuration and order (55 lines)
├── components/               # Individual manuscript sections
│   ├── abstract.html         # Abstract section (10 lines)
│   ├── section_1_introduction.html  # Introduction (43 lines)
│   ├── section_2_methods.html       # Methods (388 lines)
│   ├── section_3_results.html       # Results (1692 lines)
│   ├── section_4_discussion.html    # Discussion (680 lines)
│   ├── section_5_conclusions.html   # Conclusions (17 lines)
│   ├── section_6_analysis_package.html  # Analysis Package (349 lines)
│   └── references.html       # References & Contact (70 lines)
├── manage_components.py      # Component management utility
└── index_original.html       # Backup of original monolithic file
```

**Total reduction:** 3915 lines → 774 main file + 8 focused components

## 🔧 Component Management

### List all components
```bash
cd site && python manage_components.py list
```

### Validate components
```bash
cd site && python manage_components.py validate
```

### Reorder sections
```bash
cd site && python manage_components.py reorder abstract,section_1,section_3,section_2
```

## ✨ Benefits for AI Editing

1. **Focused Context**: AI can work on individual sections (10-1692 lines) instead of searching through 3915 lines
2. **Easy Reordering**: Change section order by editing `manifest.json` 
3. **Independent Editing**: Modify sections without affecting others
4. **Better Understanding**: Each component has clear scope and purpose
5. **Parallel Development**: Multiple sections can be edited simultaneously

## 🔄 How Component Loading Works

1. **Page Load**: `index.html` loads with header, styles, and loading indicator
2. **Manifest Fetch**: JavaScript loads `manifest.json` to get section configuration
3. **Component Loading**: Each component is fetched in order and inserted into the DOM
4. **Feature Initialization**: MathJax and other features are re-initialized after loading
5. **Error Handling**: Missing components show error messages instead of breaking the page

## 📝 Editing Workflow

### To edit a specific section:
1. Open the relevant component file (e.g., `components/section_2_methods.html`)
2. Make your changes
3. Refresh the page to see changes (no build step needed)

### To add a new section:
1. Create the component file in `components/`
2. Add entry to `manifest.json` with appropriate order number
3. The section will automatically appear on next page load

### To reorder sections:
1. Edit the `order` values in `manifest.json`, OR
2. Use the management utility: `python manage_components.py reorder section_1,section_3,section_2`

## 🚀 Performance Notes

- **First Load**: Slightly slower due to multiple HTTP requests
- **Subsequent Loads**: Fast due to browser caching
- **SEO**: All content loads client-side, maintaining full SEO compatibility
- **Print**: Works normally as all content is in the DOM after loading

## 🔧 Development Tips

- Use browser developer tools to see component loading progress
- Components are cached by the browser for faster subsequent loads
- Each component is independent - changes don't affect others
- The system gracefully handles missing components with error messages
- **Images**: All figure references use `public/figures/` path - images are served from `site/public/figures/`

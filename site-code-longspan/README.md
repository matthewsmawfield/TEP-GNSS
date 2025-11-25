# TEP-GNSS Paper 2: CODE Longspan Analysis

**Publication website for:**  
"Long-Term Stability and Geophysical Coupling of Temporal-Gravitational Signatures in GNSS: A 25-Year Confirmatory Analysis"

**Author:** Matthew Lukin Smawfield  
**Version:** v0.15 (Cairo)  
**Date:** 25 November 2025  
**Status:** Preprint

## Overview

This is the website for Paper 2 of the TEP-GNSS project, which presents a 25-year confirmatory analysis using CODE analysis center data (165M+ station pairs).

## Key Findings

- Confirms orbital velocity correlation (r = -0.888, p < 2×10⁻⁷, 5.1σ; 5 M surrogates)
- 56 planetary event responses (25 surviving Bonferroni correction, 33 BY-FDR)
- 18.6-year nutation cycle detection (R² = 0.641)
- 21+ cycles of Chandler wobble (R² = 0.106)
- Network coherence over 25.3 years

## Development

```bash
# Install dependencies
npm install

# Development server with live reload
npm run dev

# Build for production
npm run build

# View built site
npm run serve:dist
```

## Deployment

This site is deployed as part of the unified TEP-GNSS deployment. Use the root-level `deploy-all.sh` script to deploy both Paper 1 and Paper 2 together.

```bash
cd ..
./deploy-all.sh
```

## URLs

- **Production**: https://matthewsmawfield.github.io/TEP-GNSS/code-longspan/
- **Paper 1**: https://matthewsmawfield.github.io/TEP-GNSS/ (root)

## Structure

See `../MULTI_PAPER_STRUCTURE.md` for the complete multi-paper site architecture.

---

**Part of the TEP-GNSS research project**  
**Repository**: https://github.com/matthewsmawfield/TEP-GNSS

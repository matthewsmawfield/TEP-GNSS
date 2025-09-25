# Historical CODE Data Investigation Summary

**Date**: 2025-09-25  
**Objective**: Investigate acquiring 2010-2022 CODE data for extended TEP analysis

## Key Findings

### 1. Current Data Status
- **Available processed data**: 2023-2025 (2.5 years, 912 files)
- **Current analysis**: Shows strong correlations (r = -0.229, p = 4.09×10⁻¹²)
- **Data quality**: Excellent (38.4M station pair measurements)

### 2. CODE Historical Data Architecture

#### File Format Evolution
- **1991-~2021**: Legacy format `COD#####.CLK.Z` (GPS week/day based)
- **~2022-present**: Modern format `COD0OPSFIN_YYYYDDD0000_01D_30S_CLK.CLK.gz`
- **Transition period**: ~2021-2022 (mixed formats)

#### URL Patterns Identified
```
Legacy:  http://ftp.aiub.unibe.ch/CODE/{year}/COD{gps_week:04d}{gps_day}.CLK.Z
Modern:  http://ftp.aiub.unibe.ch/CODE/{year}/COD0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz
```

### 3. Technical Challenges

#### Format Conversion Requirements
- **Legacy files**: Unix compress (.Z) → need `uncompress` utility
- **Naming conversion**: GPS week/day → YYYY-DDD format
- **Processing pipeline**: Requires adaptation for legacy formats

#### Data Processing Compatibility
- **RINEX CLK format**: Consistent across all years (AR records)
- **Parsing logic**: Current Step 3 parser should work with minor modifications
- **Coordinate matching**: Same station codes across years

## Implementation Strategy

### Phase 1: Extended Analysis with Current Data ✅ COMPLETE
- **Status**: Successfully implemented and tested
- **Results**: Strong correlations over 2.5-year period
- **Files created**: 
  - `scripts/experimental/step_14_extended_code_analysis.py`
  - `scripts/experimental/test_extended_feasibility.py`

### Phase 2: Historical Data Acquisition (READY TO IMPLEMENT)

#### Option A: Conservative Approach (RECOMMENDED)
**Scope**: 2020-2022 (3 additional years)
- **Rationale**: Modern format, easier integration
- **Benefit**: Extends analysis to 5.5 years total
- **Effort**: Low - use existing pipeline with minor modifications
- **Files needed**: ~1,095 additional files

#### Option B: Aggressive Approach
**Scope**: 2010-2022 (13 additional years) 
- **Rationale**: Maximum historical depth
- **Benefit**: 15.5 years total for groundbreaking analysis
- **Effort**: High - requires legacy format handling
- **Files needed**: ~4,745 additional files

#### Option C: Hybrid Approach
**Phase 2a**: 2020-2022 (modern format)
**Phase 2b**: 2015-2019 (legacy format) 
**Phase 2c**: 2010-2014 (if Phase 2b successful)

## Created Infrastructure

### Scripts Ready for Deployment
1. **`scripts/historical_code/acquire_historical_code_data.py`**
   - Handles both legacy and modern formats
   - GPS week/day conversion
   - Parallel downloads with retry logic
   - Format detection and conversion

2. **`scripts/historical_code/test_historical_download.py`**
   - Tests accessibility of different years/formats
   - Validates download and parsing
   - Generates feasibility reports

3. **`scripts/experimental/step_14_extended_code_analysis.py`**
   - Extended analysis using CODE-only data
   - Works with current 2.5-year dataset
   - Ready for historical data integration

### Directory Structure Created
```
data/historical_code/
├── raw/           # Downloaded CLK files by year
├── processed/     # Processed correlation data  
└── test_downloads/# Test downloads

results/historical_code/
├── format_analysis.json
├── acquisition_results.json
└── investigation_summary.md

scripts/historical_code/
├── acquire_historical_code_data.py
├── test_historical_download.py
└── investigate_historical_formats.py
```

## Recommendations

### 🎯 Immediate Action: Option A (Conservative)
1. **Download 2020-2022 data** using modern format
2. **Integrate with existing pipeline** (minimal modifications needed)
3. **Extend Step 14 analysis** to 5.5-year timeframe
4. **Publish results** with enhanced statistical power

**Command to execute:**
```bash
cd /Users/matthewsmawfield/www/TEP-GNSS
python scripts/historical_code/acquire_historical_code_data.py \
  --start-year 2020 --end-year 2022 \
  --output-dir data/historical_code/raw
```

### 🔬 Future Enhancement: Option C (Hybrid)
After successful Option A implementation:
1. **Evaluate results** from 5.5-year analysis
2. **Implement legacy format handling** if needed
3. **Extend back to 2010** for maximum scientific impact

### 📊 Expected Outcomes

#### With Option A (2020-2025: 5.5 years)
- **Statistical power**: ~2x improvement
- **Seasonal patterns**: Multiple complete cycles
- **Publication impact**: Strong multi-year validation

#### With Option B/C (2010-2025: 15.5 years)
- **Statistical power**: ~6x improvement  
- **Solar cycle coverage**: Complete 11-year cycle
- **Breakthrough potential**: Unprecedented TEP validation

## Risk Assessment

### Low Risk (Option A)
- ✅ Modern format compatibility
- ✅ Existing pipeline integration
- ✅ Proven download methods
- ⚠️ Limited to recent years

### Medium Risk (Option B/C)
- ⚠️ Legacy format complexity
- ⚠️ GPS week conversion requirements
- ⚠️ Potential data gaps
- ✅ Massive scientific impact potential

## Next Steps

1. **Execute Option A** - download 2020-2022 data
2. **Test integration** with existing Step 14 analysis
3. **Validate results** against current 2.5-year analysis
4. **Assess Option B/C feasibility** based on Option A success

## Files Ready for Use

All scripts are complete and tested. The infrastructure is ready for immediate deployment of historical CODE data acquisition and analysis.


# SW California Groundwater Analysis (2023-2025) Groundwater Analysis Report

Generated: 2026-01-16 17:13:23

---

## 1. Data Overview

- **Total Monitoring Sites**: 70
- **Latitude Range**: 32.5536° ~ 34.1576°
- **Longitude Range**: -117.4270° ~ -116.6623°
- **Analysis Period**: 2023-01-01 ~ 2025-12-31
- **Total Data Points**: 1096 dates

## 2. Summary Statistics

### Overall Statistics (Original Data)

- **Depth Range**: 5.0 ft ~ 592.7 ft
- **Mean Depth Range**: 135.2 ft ~ 224.4 ft
- **Average Mean Depth**: 190.0 ft

### Temporal Analysis

| Date | Sites | Min (ft) | Max (ft) | Mean (ft) |
|------|-------|----------|----------|-----------|
| 2023-01-01 | 64 | 7.0 | 578.6 | 205.6 |
| 2023-07-02 | 66 | 8.6 | 587.2 | 190.7 |
| 2023-12-31 | 68 | 8.7 | 586.2 | 197.2 |
| 2024-01-01 | 68 | 8.7 | 586.2 | 197.2 |
| 2024-07-02 | 68 | 7.6 | 588.7 | 189.4 |
| 2024-12-31 | 69 | 8.2 | 588.3 | 188.8 |
| 2025-01-01 | 69 | 8.2 | 588.4 | 188.8 |
| 2025-07-02 | 56 | 7.9 | 585.7 | 175.9 |
| 2025-12-31 | 43 | 6.5 | 415.1 | 143.2 |

## 3. Kriging Interpolation Results

### Interpolated Surface Statistics

- **Interpolated Depth Range**: 10.3 ft ~ 555.9 ft
- **Mean Interpolated Depth**: 143.0 ft
- **Total Frames**: 1096

### Yearly Summary (Kriging Results)

| Year | Frames | Avg Min (ft) | Avg Max (ft) | Avg Mean (ft) |
|------|--------|--------------|--------------|---------------|
| 2023 | 365 | 17.7 | 549.0 | 154.0 |
| 2024 | 366 | 16.0 | 534.9 | 145.4 |
| 2025 | 365 | 16.8 | 508.7 | 129.4 |

## 4. Spatial Patterns

Based on the kriging interpolation results:

- **Deep Aquifer Zone**: Areas with depth > 400 ft (typically in mountainous/inland regions)
- **Shallow Aquifer Zone**: Areas with depth < 100 ft (typically near coast/valleys)
- **Transition Zone**: Areas with depth 100-400 ft

## 5. Data Quality Warnings

> **CAUTION**: The following data quality issues were detected:

- Site count dropped 39% from peak (70 sites on 2023-02-12) to 43 sites on 2025-12-31

These issues may significantly affect the reliability of trend analysis and kriging results.
Results for periods with reduced site counts should be interpreted with caution.

## 6. Trend Analysis

- **Early Period Mean** (2023-01-01 ~ 2023-01-10): 158.3 ft
- **Late Period Mean** (2025-12-22 ~ 2025-12-31): 110.2 ft
- **Change**: -48.1 ft (rising)

> **Note**: This trend may be affected by the significant reduction in monitoring sites 
> during the late period. The apparent change could be due to:
> - Actual groundwater level changes
> - Sampling bias from site dropout (remaining sites may not be representative)
> - Seasonal data collection patterns

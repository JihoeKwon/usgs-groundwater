# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent project for collecting, analyzing, and visualizing groundwater data from the USGS NWIS (National Water Information System) API. The project uses MCP (Model Context Protocol) servers to enable AI-driven groundwater analysis workflows.

## Development Commands

### Setup
```bash
# Core dependencies
pip install requests pandas numpy mcp

# Full functionality (kriging + visualization)
pip install pykrige matplotlib pillow

# Basemap support (satellite/map backgrounds)
pip install contextily

# GeoTIFF output (optional)
pip install rasterio
```

### Running MCP Servers
MCP servers are configured in `.mcp.json` and run automatically when invoked by Claude Code.

```bash
# Manual testing
python usgs_gwinfo_mcp.py       # USGS data collection
python kriging_mcp.py           # Kriging interpolation
python visualize_mcp.py         # Visualization
python analysis_report_mcp.py   # Analysis reports
```

## Architecture

### MCP Server Pipeline

The project implements a four-stage groundwater analysis pipeline via MCP servers:

```
[usgs-gwinfo] → [kriging] → [visualize-kriging] → [analysis-report]
     │              │              │                    │
  Data Collection   Interpolation  Visualization       Reporting
```

**1. usgs-gwinfo** (`usgs_gwinfo_mcp.py`)
- Tools: `get_groundwater_sites`, `get_groundwater_data`, `get_groundwater_data_single_date`, `get_site_history`
- Input: Bounding box (west,south,east,north), date range, aquifer type
- Output: CSV with site coordinates and water depth values (wide format with dates as columns)
- Supports `output_dir` parameter for organized file management

**2. kriging** (`kriging_mcp.py`)
- Tools: `kriging_interpolate`, `kriging_interpolate_multiple`, `get_variogram_models`
- Input: CSV from usgs-gwinfo with Lat/Lon columns
- Single date output: Grid CSV, GeoTIFF (if rasterio available), metadata
- Multiple dates output: Sectioned format file (.dat) with all dates in single file
- Variogram models: spherical (default), gaussian, exponential, linear, power
- Supports `output_dir` parameter for organized file management

**3. visualize-kriging** (`visualize_mcp.py`)
- Tools: `visualize_kriging_result`, `get_available_colormaps`, `create_comparison_plot`
- Input: Grid CSV or sectioned format file (.dat) from kriging
- Single frame output: PNG with contour map and variance plot (side-by-side)
- Multiple frames output: GIF animation with time series visualization
- **Basemap support**: Satellite/map backgrounds via contextily
  - Providers: `OpenStreetMap`, `CartoDB.Positron`, `CartoDB.Voyager`, `Esri.WorldImagery`
  - Parameters: `show_basemap`, `basemap_provider`, `basemap_alpha`, `basemap_zoom_buffer`
- Supports `output_dir` parameter for organized file management

**4. analysis-report** (`analysis_report_mcp.py`)
- Tools: `generate_analysis_report`, `get_quick_summary`
- Input: Original CSV + kriging .dat file
- Output: Markdown or JSON report with statistics, trends, and data quality warnings
- Includes yearly summaries, temporal analysis, and spatial patterns
- Supports `output_dir` parameter for organized file management

### USGS API Integration

Three distinct USGS endpoints:
- `site/` - Site metadata (RDB format, requires parsing)
- `dv/` - Daily values (JSON, aggregated statistics)
- `gwlevels/` - Individual measurements (JSON)

Key parameter code: `72019` (Depth to water level, feet below land surface)

### Aquifer Type Codes
- `U` - Unconfined (default)
- `C` - Confined
- `X` - Mixed
- `M` - Multiple

### Data Flow

1. **Query** → Bounding box + date range + aquifer filter
2. **Site Lookup** → Parse RDB response for matching sites
3. **Data Fetch** → JSON daily values for filtered sites
4. **CSV Export** → Wide format with dates as columns (UTF-8-sig for Excel)
5. **Kriging** → Ordinary Kriging with configurable variogram
   - Single date → Grid CSV + GeoTIFF + metadata
   - Multiple dates → Sectioned format (.dat) with [HEADER], [COORDINATES], [DEPTH], [VARIANCE] sections
6. **Visualization** → Contour map with observation point overlay and basemap
   - Single frame → PNG (depth map + variance map side by side)
   - Multiple frames → GIF animation with time series
   - Basemap → Satellite (`Esri.WorldImagery`) or map backgrounds (`OpenStreetMap`)
7. **Reporting** → Analysis report with statistics and trends
   - Markdown report with data overview, statistics, and quality warnings
   - JSON format for programmatic access

## API Constraints

- **Timeouts**: State-wide queries: 120s, site-specific: 60s
- **Data Lag**: Measurements delayed by days (not real-time)
- **Coordinate System**: WGS84 (EPSG:4326)
- **Period Format**: ISO 8601 duration (`P7D`, `P30D`, `P365D`) or date range

## MCP Integration

External MCP servers (`.mcp.json`):
- Notion, Slack (Team: T09F3PG1H1T), Playwright, Linear
- Slack requires `SLACK_BOT_TOKEN` environment variable

## Key Files

- `usgs_gwinfo_mcp.py` - USGS data collection MCP server
- `kriging_mcp.py` - Kriging interpolation MCP server
- `visualize_mcp.py` - Visualization MCP server
- `analysis_report_mcp.py` - Analysis report generation MCP server
- `USGS_NWIS_GUIDE.md` - API documentation with Python examples
- `USGS_GWINFO_MCP_GUIDE.md` - MCP server usage guide
- `USGS_MCP_SERVER.md` - Integrated MCP server guide
- `backup/` - Legacy standalone scripts (pre-MCP architecture)

## Typical Workflow Example

```
User: "남서부 캘리포니아 지역의 2023년-2025년 지하수위 분포 변화 추이를 10일간격으로 분석해줘"

Claude Code automatically:
1. get_groundwater_data → CSV with multi-year data
2. kriging_interpolate_multiple → .dat file with 100+ frames
3. visualize_kriging_result → GIF animation
4. generate_analysis_report → Markdown report with trends
```

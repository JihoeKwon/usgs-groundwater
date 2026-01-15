# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GeoAI agent project for collecting, analyzing, and visualizing groundwater data from the USGS NWIS (National Water Information System) API. The project uses MCP (Model Context Protocol) servers to enable AI-driven groundwater analysis workflows.

## Development Commands

### Setup
```bash
# Core dependencies
pip install -r requirements_usgs.txt

# Full functionality (kriging + visualization)
pip install pykrige matplotlib rasterio
```

### Running MCP Servers
MCP servers are configured in `.mcp.json` and run automatically when invoked by Claude Code.

```bash
# Manual testing
python usgs_gwinfo_mcp.py    # USGS data collection
python kriging_mcp.py        # Kriging interpolation
python visualize_mcp.py      # Visualization
```

## Architecture

### MCP Server Pipeline

The project implements a three-stage groundwater analysis pipeline via MCP servers:

```
[usgs-gwinfo] → [kriging] → [visualize-kriging]
     │              │              │
  Data Collection   Interpolation  Visualization
```

**1. usgs-gwinfo** (`usgs_gwinfo_mcp.py`)
- Tools: `get_groundwater_sites`, `get_groundwater_data`, `get_site_history`
- Input: Bounding box (west,south,east,north), date range, aquifer type
- Output: CSV with site coordinates and water depth values

**2. kriging** (`kriging_mcp.py`)
- Tools: `kriging_interpolate`, `get_variogram_models`
- Input: CSV from usgs-gwinfo with Lat/Lon columns
- Output: Grid CSV, GeoTIFF (if rasterio available), metadata
- Variogram models: spherical (default), gaussian, exponential, linear

**3. visualize-kriging** (`visualize_mcp.py`)
- Tools: `visualize_kriging_result`, `get_available_colormaps`, `create_comparison_plot`
- Input: Grid CSV from kriging
- Output: PNG visualization with contour map and variance plot

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
6. **Visualization** → Contour map with observation point overlay

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

- `USGS_NWIS_GUIDE.md` - API documentation with Python examples
- `backup/` - Legacy standalone scripts (pre-MCP architecture)

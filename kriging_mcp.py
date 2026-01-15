"""
Kriging Interpolation MCP Server
- Perform Ordinary Kriging on groundwater data
- Generate regular grid
- Export to CSV/GeoTIFF
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from mcp.server.fastmcp import FastMCP

try:
    from pykrige.ok import OrdinaryKriging
    PYKRIGE_AVAILABLE = True
except ImportError:
    PYKRIGE_AVAILABLE = False

try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Create MCP server
mcp = FastMCP("kriging")


@mcp.tool()
def kriging_interpolate(
    input_csv: str,
    date_column: str = None,
    grid_resolution: int = 50,
    variogram_model: str = "spherical",
    output_prefix: str = None
) -> str:
    """
    Perform Ordinary Kriging interpolation on groundwater data.

    Args:
        input_csv: Input CSV file with groundwater data (must have Lat, Lon columns)
        date_column: Specific date column to interpolate (default: last date column)
        grid_resolution: Number of grid cells in each direction (default: 50)
        variogram_model: Variogram model - 'spherical', 'gaussian', 'exponential', 'linear'
        output_prefix: Output file prefix (default: auto-generated)

    Returns:
        JSON string with kriging results and output file paths
    """
    if not PYKRIGE_AVAILABLE:
        return json.dumps({
            "error": "pykrige not installed. Install with: pip install pykrige"
        })

    # Load data
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        return json.dumps({"error": f"Failed to load CSV: {str(e)}"})

    # Find date columns
    date_cols = [c for c in df.columns if c.startswith('20') and len(c) == 10]

    if not date_cols:
        return json.dumps({"error": "No date columns found in CSV"})

    if date_column and date_column in date_cols:
        use_col = date_column
    else:
        use_col = date_cols[-1]

    # Extract valid data
    valid_mask = df[use_col].notna() & (df[use_col] != '')
    df_valid = df[valid_mask].copy()

    if len(df_valid) < 3:
        return json.dumps({"error": "Need at least 3 data points for kriging"})

    lons = df_valid['Lon'].astype(float).values
    lats = df_valid['Lat'].astype(float).values
    values = df_valid[use_col].astype(float).values

    # Define grid bounds
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()

    buffer_lon = (lon_max - lon_min) * 0.05
    buffer_lat = (lat_max - lat_min) * 0.05

    grid_lon = np.linspace(lon_min - buffer_lon, lon_max + buffer_lon, grid_resolution)
    grid_lat = np.linspace(lat_min - buffer_lat, lat_max + buffer_lat, grid_resolution)

    # Perform kriging
    try:
        OK = OrdinaryKriging(
            lons, lats, values,
            variogram_model=variogram_model,
            variogram_parameters={
                'sill': np.var(values),
                'range': max(lon_max - lon_min, lat_max - lat_min) / 2,
                'nugget': np.var(values) * 0.1
            },
            verbose=False,
            enable_plotting=False,
            nlags=10
        )

        z_grid, ss_grid = OK.execute('grid', grid_lon, grid_lat)
    except Exception as e:
        return json.dumps({"error": f"Kriging failed: {str(e)}"})

    # Generate output filenames
    if output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"kriging_{use_col}_{timestamp}"

    output_files = {}

    # Save to CSV
    csv_file = f"{output_prefix}_grid.csv"
    rows = []
    for i, lat in enumerate(grid_lat):
        for j, lon in enumerate(grid_lon):
            rows.append({
                'Lon': lon,
                'Lat': lat,
                'Depth_ft': z_grid[i, j],
                'Variance': ss_grid[i, j]
            })

    pd.DataFrame(rows).to_csv(csv_file, index=False, encoding='utf-8-sig')
    output_files['csv'] = csv_file

    # Save to GeoTIFF if available
    if RASTERIO_AVAILABLE:
        tiff_file = f"{output_prefix}.tif"
        transform = from_bounds(grid_lon.min(), grid_lat.min(),
                                grid_lon.max(), grid_lat.max(),
                                z_grid.shape[1], z_grid.shape[0])

        with rasterio.open(
            tiff_file, 'w',
            driver='GTiff',
            height=z_grid.shape[0],
            width=z_grid.shape[1],
            count=1,
            dtype=z_grid.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(np.flipud(z_grid), 1)

        output_files['geotiff'] = tiff_file

    # Save metadata
    meta_file = f"{output_prefix}_metadata.txt"
    with open(meta_file, 'w') as f:
        f.write("Kriging Interpolation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {input_csv}\n")
        f.write(f"Date: {use_col}\n")
        f.write(f"Input points: {len(values)}\n")
        f.write(f"Variogram model: {variogram_model}\n")
        f.write(f"Grid resolution: {grid_resolution} x {grid_resolution}\n")
        f.write(f"Grid bounds:\n")
        f.write(f"  Lon: [{grid_lon.min():.6f}, {grid_lon.max():.6f}]\n")
        f.write(f"  Lat: [{grid_lat.min():.6f}, {grid_lat.max():.6f}]\n")
        f.write(f"\nInput Statistics:\n")
        f.write(f"  Min: {values.min():.2f} ft\n")
        f.write(f"  Max: {values.max():.2f} ft\n")
        f.write(f"  Mean: {values.mean():.2f} ft\n")
        f.write(f"\nOutput Statistics:\n")
        f.write(f"  Min: {z_grid.min():.2f} ft\n")
        f.write(f"  Max: {z_grid.max():.2f} ft\n")
        f.write(f"  Mean: {z_grid.mean():.2f} ft\n")

    output_files['metadata'] = meta_file

    return json.dumps({
        "status": "success",
        "input_csv": input_csv,
        "date_column": use_col,
        "input_points": len(values),
        "grid_resolution": grid_resolution,
        "variogram_model": variogram_model,
        "grid_bounds": {
            "lon_min": float(grid_lon.min()),
            "lon_max": float(grid_lon.max()),
            "lat_min": float(grid_lat.min()),
            "lat_max": float(grid_lat.max())
        },
        "input_stats": {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std())
        },
        "output_stats": {
            "min": float(z_grid.min()),
            "max": float(z_grid.max()),
            "mean": float(z_grid.mean())
        },
        "output_files": output_files
    }, indent=2)


@mcp.tool()
def get_variogram_models() -> str:
    """
    Get list of available variogram models.

    Returns:
        JSON string with available models and descriptions
    """
    models = {
        "spherical": "Most commonly used. Reaches sill at finite range.",
        "gaussian": "Smooth at origin. Good for continuous phenomena.",
        "exponential": "Linear at origin. Reaches sill asymptotically.",
        "linear": "Simple linear model. No sill.",
        "power": "Power model with exponent parameter."
    }

    return json.dumps({
        "available_models": models,
        "default": "spherical"
    }, indent=2)


if __name__ == "__main__":
    mcp.run()

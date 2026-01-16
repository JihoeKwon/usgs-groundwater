"""
Kriging Interpolation MCP Server
- Perform Ordinary Kriging on groundwater data
- Support single date or multiple dates
- Generate regular grid
- Export to CSV/GeoTIFF or sectioned format
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from mcp.server.fastmcp import FastMCP


def _ensure_output_dir(output_dir: str) -> str:
    """Create output directory if it doesn't exist."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _get_output_path(filename: str, output_dir: str = None) -> str:
    """Get full output path, creating directory if needed."""
    if output_dir:
        _ensure_output_dir(output_dir)
        return os.path.join(output_dir, filename)
    return filename

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


def _get_date_columns(df):
    """Find all date columns in DataFrame (YYYY-MM-DD format)"""
    return [c for c in df.columns if c.startswith('20') and len(c) == 10]


def _perform_kriging(lons, lats, values, grid_resolution=50, variogram_model='spherical'):
    """
    Internal: Perform Ordinary Kriging interpolation.

    Returns:
        tuple: (grid_lon, grid_lat, z_grid, ss_grid)
    """
    # Define grid bounds with small buffer
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()

    buffer_lon = (lon_max - lon_min) * 0.05
    buffer_lat = (lat_max - lat_min) * 0.05

    grid_lon = np.linspace(lon_min - buffer_lon, lon_max + buffer_lon, grid_resolution)
    grid_lat = np.linspace(lat_min - buffer_lat, lat_max + buffer_lat, grid_resolution)

    # Create Ordinary Kriging object
    OK = OrdinaryKriging(
        lons,
        lats,
        values,
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

    # Execute kriging on grid
    z_grid, ss_grid = OK.execute('grid', grid_lon, grid_lat)

    return grid_lon, grid_lat, z_grid, ss_grid


@mcp.tool()
def kriging_interpolate(
    input_csv: str,
    date_column: str = None,
    grid_resolution: int = 50,
    variogram_model: str = "spherical",
    output_prefix: str = None,
    output_dir: str = None
) -> str:
    """
    Perform Ordinary Kriging interpolation on groundwater data for a single date.

    Args:
        input_csv: Input CSV file with groundwater data (must have Lat, Lon columns)
        date_column: Specific date column to interpolate (default: last date column)
        grid_resolution: Number of grid cells in each direction (default: 50)
        variogram_model: Variogram model - 'spherical', 'gaussian', 'exponential', 'linear'
        output_prefix: Output file prefix (default: auto-generated)
        output_dir: Optional output directory (created if not exists)

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
    date_cols = _get_date_columns(df)

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

    # Perform kriging
    try:
        grid_lon, grid_lat, z_grid, ss_grid = _perform_kriging(
            lons, lats, values,
            grid_resolution=grid_resolution,
            variogram_model=variogram_model
        )
    except Exception as e:
        return json.dumps({"error": f"Kriging failed: {str(e)}"})

    # Generate output filenames
    if output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"kriging_{use_col}_{timestamp}"

    output_files = {}

    # Save to CSV
    csv_file = _get_output_path(f"{output_prefix}_grid.csv", output_dir)
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
        tiff_file = _get_output_path(f"{output_prefix}.tif", output_dir)
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
    meta_file = _get_output_path(f"{output_prefix}_metadata.txt", output_dir)
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
        "output_dir": output_dir,
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
def kriging_interpolate_multiple(
    input_csv: str,
    grid_resolution: int = 50,
    variogram_model: str = "spherical",
    output_file: str = None,
    output_dir: str = None
) -> str:
    """
    Perform Ordinary Kriging interpolation on groundwater data for multiple dates.
    Outputs sectioned format file with all dates in a single file.

    Args:
        input_csv: Input CSV file with groundwater data (must have Lat, Lon, date columns)
        grid_resolution: Number of grid cells in each direction (default: 50)
        variogram_model: Variogram model - 'spherical', 'gaussian', 'exponential', 'linear'
        output_file: Output file path (default: auto-generated .dat file)
        output_dir: Optional output directory (created if not exists)

    Returns:
        JSON string with kriging results for all dates
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
    date_cols = _get_date_columns(df)

    if not date_cols:
        return json.dumps({"error": "No date columns found in CSV"})

    # Generate output filename
    if output_file is None:
        base_name = os.path.basename(input_csv).replace('.csv', '')
        output_file = f"{base_name}_kriging_multiframe.dat"

    output_file = _get_output_path(output_file, output_dir)

    results = {
        'input_csv': input_csv,
        'grid_resolution': grid_resolution,
        'variogram_model': variogram_model,
        'dates_processed': [],
        'dates_skipped': [],
        'stats': {}
    }

    # Store all kriging results
    all_grids = {}      # {date: z_grid}
    all_variances = {}  # {date: ss_grid}
    grid_lon = None
    grid_lat = None

    for date_col in date_cols:
        # Extract valid data for this date
        valid_mask = df[date_col].notna() & (df[date_col] != '')
        df_valid = df[valid_mask].copy()

        if len(df_valid) < 3:
            results['dates_skipped'].append(date_col)
            continue

        lons = df_valid['Lon'].astype(float).values
        lats = df_valid['Lat'].astype(float).values
        values = df_valid[date_col].astype(float).values

        # Perform kriging
        try:
            g_lon, g_lat, z_grid, ss_grid = _perform_kriging(
                lons, lats, values,
                grid_resolution=grid_resolution,
                variogram_model=variogram_model
            )

            # Store grid coordinates (same for all dates)
            if grid_lon is None:
                grid_lon = g_lon
                grid_lat = g_lat

            all_grids[date_col] = z_grid
            all_variances[date_col] = ss_grid
            results['dates_processed'].append(date_col)
            results['stats'][date_col] = {
                'input_points': len(values),
                'input_min': float(values.min()),
                'input_max': float(values.max()),
                'input_mean': float(values.mean()),
                'output_min': float(z_grid.min()),
                'output_max': float(z_grid.max()),
                'output_mean': float(z_grid.mean())
            }

        except Exception as e:
            results['dates_skipped'].append(date_col)
            continue

    if not all_grids:
        return json.dumps({
            "error": "No dates were successfully processed",
            "dates_skipped": results['dates_skipped']
        })

    # Build sectioned output file
    n_points = grid_resolution * grid_resolution
    dates_str = ','.join(results['dates_processed'])

    with open(output_file, 'w', encoding='utf-8') as f:
        # [HEADER] section
        f.write("[HEADER]\n")
        f.write(f"NFRAME={len(results['dates_processed'])}\n")
        f.write(f"GRID={grid_resolution}x{grid_resolution}\n")
        f.write(f"POINTS={n_points}\n")
        f.write(f"DATES={dates_str}\n")
        f.write("\n")

        # [COORDINATES] section
        f.write("[COORDINATES]\n")
        f.write("Lon,Lat\n")
        for i, lat in enumerate(grid_lat):
            for j, lon in enumerate(grid_lon):
                f.write(f"{lon:.6f},{lat:.6f}\n")
        f.write("\n")

        # [DEPTH] section - each row is one point, columns are dates
        f.write("[DEPTH]\n")
        for i in range(len(grid_lat)):
            for j in range(len(grid_lon)):
                row_values = [f"{all_grids[d][i,j]:.2f}" for d in results['dates_processed']]
                f.write(','.join(row_values) + '\n')
        f.write("\n")

        # [VARIANCE] section - same structure as DEPTH
        f.write("[VARIANCE]\n")
        for i in range(len(grid_lat)):
            for j in range(len(grid_lon)):
                row_values = [f"{all_variances[d][i,j]:.2f}" for d in results['dates_processed']]
                f.write(','.join(row_values) + '\n')

    results['output_file'] = output_file
    results['grid_points'] = grid_resolution * grid_resolution
    results['grid_bounds'] = {
        'lon_min': float(grid_lon.min()),
        'lon_max': float(grid_lon.max()),
        'lat_min': float(grid_lat.min()),
        'lat_max': float(grid_lat.max())
    }

    # Save metadata
    if '.' in os.path.basename(output_file):
        base = output_file.rsplit('.', 1)[0]
        meta_file = f"{base}_metadata.txt"
    else:
        meta_file = f"{output_file}_metadata.txt"

    with open(meta_file, 'w') as f:
        f.write("Kriging Interpolation Results - Multiple Dates\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input file: {input_csv}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Grid resolution: {grid_resolution} x {grid_resolution}\n")
        f.write(f"Grid points per frame: {grid_resolution * grid_resolution}\n")
        f.write(f"Variogram model: {variogram_model}\n")
        f.write(f"\nGrid bounds:\n")
        f.write(f"  Lon: [{grid_lon.min():.6f}, {grid_lon.max():.6f}]\n")
        f.write(f"  Lat: [{grid_lat.min():.6f}, {grid_lat.max():.6f}]\n")
        f.write(f"\nDates processed: {len(results['dates_processed'])}\n")
        f.write(f"Dates skipped: {len(results['dates_skipped'])}\n")

        if results['dates_skipped']:
            f.write(f"\nSkipped dates (insufficient data):\n")
            for d in results['dates_skipped']:
                f.write(f"  - {d}\n")

        f.write(f"\n{'='*60}\n")
        f.write("Statistics by Date\n")
        f.write(f"{'='*60}\n\n")

        for date_col in results['dates_processed']:
            s = results['stats'][date_col]
            f.write(f"{date_col}:\n")
            f.write(f"  Input: {s['input_points']} pts, {s['input_min']:.2f} ~ {s['input_max']:.2f} ft\n")
            f.write(f"  Output: {s['output_min']:.2f} ~ {s['output_max']:.2f} ft (mean: {s['output_mean']:.2f})\n")
            f.write("\n")

    results['metadata_file'] = meta_file

    return json.dumps({
        "status": "success",
        "input_csv": input_csv,
        "dates_processed": len(results['dates_processed']),
        "dates_skipped": len(results['dates_skipped']),
        "grid_resolution": grid_resolution,
        "variogram_model": variogram_model,
        "output_dir": output_dir,
        "grid_bounds": results['grid_bounds'],
        "output_file": output_file,
        "metadata_file": meta_file,
        "stats": results['stats']
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

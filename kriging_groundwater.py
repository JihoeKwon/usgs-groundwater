"""
Kriging Interpolation for Groundwater Level Data
- Load groundwater data from CSV
- Perform Ordinary Kriging interpolation
- Generate regular grid model
- Save results as CSV and GeoTIFF
"""

import numpy as np
import pandas as pd
from datetime import datetime

try:
    from pykrige.ok import OrdinaryKriging
    PYKRIGE_AVAILABLE = True
except ImportError:
    PYKRIGE_AVAILABLE = False
    print("Warning: pykrige not installed. Install with: pip install pykrige")

try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not installed. GeoTIFF export disabled.")


def load_groundwater_data(csv_file, date_column=None):
    """
    Load groundwater data from CSV file

    Parameters:
    -----------
    csv_file : str
        Path to CSV file
    date_column : str
        Specific date column to use (if None, uses last date column)

    Returns:
    --------
    tuple: (lons, lats, values, date_used)
    """
    print(f"Loading data from: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df)}")

    # Find date columns (YYYY-MM-DD format)
    date_cols = [c for c in df.columns if c.startswith('20') and len(c) == 10]

    if not date_cols:
        raise ValueError("No date columns found in CSV")

    if date_column and date_column in date_cols:
        use_col = date_column
    else:
        use_col = date_cols[-1]  # Use last date

    print(f"  Using date column: {use_col}")

    # Extract data
    valid_mask = df[use_col].notna() & (df[use_col] != '')
    df_valid = df[valid_mask].copy()

    lons = df_valid['Lon'].astype(float).values
    lats = df_valid['Lat'].astype(float).values
    values = df_valid[use_col].astype(float).values

    print(f"  Valid data points: {len(values)}")

    return lons, lats, values, use_col


def perform_kriging(lons, lats, values, grid_resolution=100, variogram_model='spherical'):
    """
    Perform Ordinary Kriging interpolation

    Parameters:
    -----------
    lons : array
        Longitude values
    lats : array
        Latitude values
    values : array
        Groundwater depth values
    grid_resolution : int
        Number of grid cells in each direction
    variogram_model : str
        Variogram model ('linear', 'power', 'gaussian', 'spherical', 'exponential')

    Returns:
    --------
    tuple: (grid_lon, grid_lat, z_grid, ss_grid)
        - grid_lon: 1D array of longitude values
        - grid_lat: 1D array of latitude values
        - z_grid: 2D array of interpolated values
        - ss_grid: 2D array of kriging variance (uncertainty)
    """
    if not PYKRIGE_AVAILABLE:
        raise ImportError("pykrige is required for kriging. Install with: pip install pykrige")

    print("\nPerforming Ordinary Kriging...")
    print(f"  Variogram model: {variogram_model}")
    print(f"  Grid resolution: {grid_resolution} x {grid_resolution}")

    # Define grid bounds with small buffer
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()

    buffer_lon = (lon_max - lon_min) * 0.05
    buffer_lat = (lat_max - lat_min) * 0.05

    grid_lon = np.linspace(lon_min - buffer_lon, lon_max + buffer_lon, grid_resolution)
    grid_lat = np.linspace(lat_min - buffer_lat, lat_max + buffer_lat, grid_resolution)

    print(f"  Grid bounds: Lon [{grid_lon.min():.4f}, {grid_lon.max():.4f}]")
    print(f"               Lat [{grid_lat.min():.4f}, {grid_lat.max():.4f}]")

    # Create Ordinary Kriging object
    # Use custom variogram parameters for better interpolation
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

    print(f"  Kriging complete!")
    print(f"  Z range: [{z_grid.min():.2f}, {z_grid.max():.2f}]")

    return grid_lon, grid_lat, z_grid, ss_grid


def save_grid_to_csv(grid_lon, grid_lat, z_grid, ss_grid, output_file):
    """
    Save kriging results to CSV
    """
    print(f"\nSaving grid to CSV: {output_file}")

    rows = []
    for i, lat in enumerate(grid_lat):
        for j, lon in enumerate(grid_lon):
            rows.append({
                'Lon': lon,
                'Lat': lat,
                'Depth_ft': z_grid[i, j],
                'Variance': ss_grid[i, j]
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  Saved {len(rows)} grid points")

    return output_file


def save_grid_to_geotiff(grid_lon, grid_lat, z_grid, output_file):
    """
    Save kriging results to GeoTIFF
    """
    if not RASTERIO_AVAILABLE:
        print("Warning: rasterio not available, skipping GeoTIFF export")
        return None

    print(f"\nSaving grid to GeoTIFF: {output_file}")

    # Get bounds
    lon_min, lon_max = grid_lon.min(), grid_lon.max()
    lat_min, lat_max = grid_lat.min(), grid_lat.max()

    # Create transform
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max,
                            z_grid.shape[1], z_grid.shape[0])

    # Write GeoTIFF
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=z_grid.shape[0],
        width=z_grid.shape[1],
        count=1,
        dtype=z_grid.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        # Flip array vertically (rasterio expects top-to-bottom)
        dst.write(np.flipud(z_grid), 1)

    print(f"  GeoTIFF saved successfully")
    return output_file


def kriging_groundwater(input_csv, date_column=None, grid_resolution=100,
                        variogram_model='spherical', output_prefix=None):
    """
    Main function: Load data, perform kriging, save results

    Parameters:
    -----------
    input_csv : str
        Input CSV file with groundwater data
    date_column : str
        Specific date column to interpolate (default: last date)
    grid_resolution : int
        Grid resolution (default: 100)
    variogram_model : str
        Variogram model (default: 'spherical')
    output_prefix : str
        Output file prefix (default: auto-generated)

    Returns:
    --------
    dict: Output file paths
    """
    print("=" * 70)
    print("Kriging Interpolation for Groundwater Data")
    print("=" * 70)

    # Load data
    lons, lats, values, date_used = load_groundwater_data(input_csv, date_column)

    if len(values) < 3:
        raise ValueError("Need at least 3 data points for kriging")

    # Print input statistics
    print(f"\nInput Statistics:")
    print(f"  Points: {len(values)}")
    print(f"  Depth range: {values.min():.2f} ~ {values.max():.2f} ft")
    print(f"  Depth mean: {values.mean():.2f} ft")
    print(f"  Depth std: {values.std():.2f} ft")

    # Perform kriging
    grid_lon, grid_lat, z_grid, ss_grid = perform_kriging(
        lons, lats, values,
        grid_resolution=grid_resolution,
        variogram_model=variogram_model
    )

    # Generate output filenames
    if output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"kriging_{date_used}_{timestamp}"

    output_files = {}

    # Save to CSV
    csv_file = f"{output_prefix}_grid.csv"
    output_files['csv'] = save_grid_to_csv(grid_lon, grid_lat, z_grid, ss_grid, csv_file)

    # Save to GeoTIFF
    if RASTERIO_AVAILABLE:
        tiff_file = f"{output_prefix}.tif"
        output_files['geotiff'] = save_grid_to_geotiff(grid_lon, grid_lat, z_grid, tiff_file)

    # Save metadata
    meta_file = f"{output_prefix}_metadata.txt"
    with open(meta_file, 'w') as f:
        f.write("Kriging Interpolation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {input_csv}\n")
        f.write(f"Date: {date_used}\n")
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
        f.write(f"  Std: {values.std():.2f} ft\n")
        f.write(f"\nOutput Statistics:\n")
        f.write(f"  Min: {z_grid.min():.2f} ft\n")
        f.write(f"  Max: {z_grid.max():.2f} ft\n")
        f.write(f"  Mean: {z_grid.mean():.2f} ft\n")

    output_files['metadata'] = meta_file
    print(f"\nMetadata saved to: {meta_file}")

    print("\n" + "=" * 70)
    print("KRIGING COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    for key, path in output_files.items():
        if path:
            print(f"  {key}: {path}")

    return output_files


if __name__ == "__main__":
    import glob

    # Find most recent groundwater CSV file
    csv_files = glob.glob("groundwater_bbox_*.csv")

    if csv_files:
        latest_csv = max(csv_files)
        print(f"Using: {latest_csv}\n")

        kriging_groundwater(
            input_csv=latest_csv,
            grid_resolution=50,  # 50x50 grid
            variogram_model='spherical'
        )
    else:
        print("No groundwater CSV file found.")
        print("Run usgs_api_test.py first to generate data.")

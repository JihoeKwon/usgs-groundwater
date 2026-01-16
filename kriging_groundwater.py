"""
Kriging Interpolation for Groundwater Level Data
- Load groundwater data from CSV
- Perform Ordinary Kriging interpolation
- Support single date or multiple dates (batch processing)
- Generate regular grid model
- Save results as CSV and GeoTIFF
"""

import argparse
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


def get_date_columns(df):
    """Find all date columns in DataFrame (YYYY-MM-DD format)"""
    return [c for c in df.columns if c.startswith('20') and len(c) == 10]


def load_groundwater_data(csv_file, date_column=None):
    """
    Load groundwater data from CSV file for a specific date.

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
    df = pd.read_csv(csv_file)
    date_cols = get_date_columns(df)

    if not date_cols:
        raise ValueError("No date columns found in CSV")

    if date_column and date_column in date_cols:
        use_col = date_column
    else:
        use_col = date_cols[-1]

    # Extract valid data
    valid_mask = df[use_col].notna() & (df[use_col] != '')
    df_valid = df[valid_mask].copy()

    lons = df_valid['Lon'].astype(float).values
    lats = df_valid['Lat'].astype(float).values
    values = df_valid[use_col].astype(float).values

    return lons, lats, values, use_col


def perform_kriging(lons, lats, values, grid_resolution=100, variogram_model='spherical', verbose=True):
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
    verbose : bool
        Print progress messages

    Returns:
    --------
    tuple: (grid_lon, grid_lat, z_grid, ss_grid)
    """
    if not PYKRIGE_AVAILABLE:
        raise ImportError("pykrige is required for kriging. Install with: pip install pykrige")

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


def save_grid_to_csv(grid_lon, grid_lat, z_grid, ss_grid, output_file):
    """Save kriging results to CSV"""
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
    return output_file


def save_grid_to_geotiff(grid_lon, grid_lat, z_grid, output_file):
    """Save kriging results to GeoTIFF"""
    if not RASTERIO_AVAILABLE:
        return None

    lon_min, lon_max = grid_lon.min(), grid_lon.max()
    lat_min, lat_max = grid_lat.min(), grid_lat.max()

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max,
                            z_grid.shape[1], z_grid.shape[0])

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
        dst.write(np.flipud(z_grid), 1)

    return output_file


def kriging_single_date(input_csv, date_column=None, grid_resolution=100,
                        variogram_model='spherical', output_prefix=None, verbose=True):
    """
    Perform kriging for a single date.

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
    verbose : bool
        Print progress messages

    Returns:
    --------
    dict: Result with output file paths and statistics
    """
    if verbose:
        print(f"\n[Kriging] Date: {date_column or 'auto'}")

    # Load data
    lons, lats, values, date_used = load_groundwater_data(input_csv, date_column)

    if len(values) < 3:
        if verbose:
            print(f"  Skipping - only {len(values)} points (need >= 3)")
        return None

    if verbose:
        print(f"  Points: {len(values)}, Range: {values.min():.2f} ~ {values.max():.2f} ft")

    # Perform kriging
    grid_lon, grid_lat, z_grid, ss_grid = perform_kriging(
        lons, lats, values,
        grid_resolution=grid_resolution,
        variogram_model=variogram_model,
        verbose=verbose
    )

    # Generate output filenames
    if output_prefix is None:
        output_prefix = f"kriging_{date_used}"

    output_files = {}

    # Save to CSV
    csv_file = f"{output_prefix}_grid.csv"
    output_files['csv'] = save_grid_to_csv(grid_lon, grid_lat, z_grid, ss_grid, csv_file)

    # Save to GeoTIFF
    if RASTERIO_AVAILABLE:
        tiff_file = f"{output_prefix}.tif"
        output_files['geotiff'] = save_grid_to_geotiff(grid_lon, grid_lat, z_grid, tiff_file)

    if verbose:
        print(f"  Output: {csv_file}")

    return {
        'date': date_used,
        'input_points': len(values),
        'input_stats': {
            'min': float(values.min()),
            'max': float(values.max()),
            'mean': float(values.mean()),
            'std': float(values.std())
        },
        'output_stats': {
            'min': float(z_grid.min()),
            'max': float(z_grid.max()),
            'mean': float(z_grid.mean())
        },
        'grid_bounds': {
            'lon_min': float(grid_lon.min()),
            'lon_max': float(grid_lon.max()),
            'lat_min': float(grid_lat.min()),
            'lat_max': float(grid_lat.max())
        },
        'output_files': output_files
    }


def kriging_multiple_dates(input_csv, dates=None, grid_resolution=100,
                           variogram_model='spherical', output_file=None):
    """
    Perform kriging for multiple dates and save all results to a single file.

    Parameters:
    -----------
    input_csv : str
        Input CSV file with groundwater data
    dates : list
        List of date columns to process (default: all date columns)
    grid_resolution : int
        Grid resolution (default: 100)
    variogram_model : str
        Variogram model (default: 'spherical')
    output_file : str
        Output CSV filename (default: auto-generated)

    Returns:
    --------
    dict: Results for all dates
    """
    print("=" * 70)
    print("Kriging Interpolation - Multiple Dates")
    print("=" * 70)

    # Load CSV to get date columns
    df = pd.read_csv(input_csv)
    all_date_cols = get_date_columns(df)

    if not all_date_cols:
        raise ValueError("No date columns found in CSV")

    # Determine which dates to process
    if dates is None:
        dates_to_process = all_date_cols
    else:
        dates_to_process = [d for d in dates if d in all_date_cols]

    print(f"\nInput file: {input_csv}")
    print(f"Total rows: {len(df)}")
    print(f"Date columns found: {len(all_date_cols)}")
    print(f"Dates to process: {len(dates_to_process)}")
    print(f"Grid resolution: {grid_resolution} x {grid_resolution}")
    print(f"Variogram model: {variogram_model}")

    # Generate output filename
    if output_file is None:
        base_name = input_csv.replace('.csv', '')
        output_file = f"{base_name}_kriging_grid.csv"

    results = {
        'input_csv': input_csv,
        'grid_resolution': grid_resolution,
        'variogram_model': variogram_model,
        'dates_processed': [],
        'dates_skipped': [],
        'stats': {}
    }

    print(f"\nProcessing {len(dates_to_process)} dates...")
    print("-" * 70)

    # Store all kriging results
    all_grids = {}      # {date: z_grid}
    all_variances = {}  # {date: ss_grid}
    grid_lon = None
    grid_lat = None

    for date_col in dates_to_process:
        print(f"\n[Kriging] Date: {date_col}")

        # Load data for this date
        try:
            lons, lats, values, date_used = load_groundwater_data(input_csv, date_col)
        except Exception as e:
            print(f"  Skipping - error loading data: {e}")
            results['dates_skipped'].append(date_col)
            continue

        if len(values) < 3:
            print(f"  Skipping - only {len(values)} points (need >= 3)")
            results['dates_skipped'].append(date_col)
            continue

        print(f"  Points: {len(values)}, Range: {values.min():.2f} ~ {values.max():.2f} ft")

        # Perform kriging
        try:
            g_lon, g_lat, z_grid, ss_grid = perform_kriging(
                lons, lats, values,
                grid_resolution=grid_resolution,
                variogram_model=variogram_model,
                verbose=False
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
            print(f"  Output range: {z_grid.min():.2f} ~ {z_grid.max():.2f} ft")

        except Exception as e:
            print(f"  Skipping - kriging error: {e}")
            results['dates_skipped'].append(date_col)
            continue

    if not all_grids:
        print("\nNo dates were successfully processed!")
        return results

    # Build sectioned output file
    print(f"\nSaving combined results to: {output_file}")

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
    # Handle various extensions
    if '.' in output_file:
        base, ext = output_file.rsplit('.', 1)
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

    print("\n" + "=" * 70)
    print("KRIGING COMPLETE!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Frames: {len(results['dates_processed'])}")
    print(f"  Dates skipped: {len(results['dates_skipped'])}")
    print(f"  Grid points per frame: {grid_resolution * grid_resolution} ({grid_resolution}x{grid_resolution})")
    print(f"\nOutput files:")
    print(f"  Data: {output_file}")
    print(f"  Metadata: {meta_file}")

    return results


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Kriging Interpolation for Groundwater Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single date (last date in file)
  python kriging_groundwater.py input.csv

  # Single specific date
  python kriging_groundwater.py input.csv --date 2026-01-13

  # All dates in file
  python kriging_groundwater.py input.csv --all-dates

  # Specific dates
  python kriging_groundwater.py input.csv --dates 2026-01-01 2026-01-07 2026-01-13

  # Custom grid resolution and variogram
  python kriging_groundwater.py input.csv --all-dates --resolution 100 --variogram gaussian
        """
    )

    parser.add_argument('input_csv', help="Input CSV file with groundwater data")
    parser.add_argument('--date', '-d', help="Single date column to process (YYYY-MM-DD)")
    parser.add_argument('--all-dates', '-a', action='store_true',
                        help="Process all date columns in the file")
    parser.add_argument('--dates', nargs='+',
                        help="List of specific dates to process")
    parser.add_argument('--resolution', '-r', type=int, default=50,
                        help="Grid resolution (default: 50)")
    parser.add_argument('--variogram', '-v', default='spherical',
                        choices=['linear', 'power', 'gaussian', 'spherical', 'exponential'],
                        help="Variogram model (default: spherical)")
    parser.add_argument('--output', '-o', help="Output prefix")

    args = parser.parse_args()

    # Determine mode
    if args.all_dates:
        # Process all dates
        results = kriging_multiple_dates(
            input_csv=args.input_csv,
            dates=None,  # All dates
            grid_resolution=args.resolution,
            variogram_model=args.variogram,
            output_file=args.output
        )
    elif args.dates:
        # Process specific dates
        results = kriging_multiple_dates(
            input_csv=args.input_csv,
            dates=args.dates,
            grid_resolution=args.resolution,
            variogram_model=args.variogram,
            output_file=args.output
        )
    else:
        # Single date mode
        print("=" * 70)
        print("Kriging Interpolation for Groundwater Data")
        print("=" * 70)

        results = kriging_single_date(
            input_csv=args.input_csv,
            date_column=args.date,
            grid_resolution=args.resolution,
            variogram_model=args.variogram,
            output_prefix=args.output
        )

        if results:
            print("\n" + "=" * 70)
            print("KRIGING COMPLETE!")
            print("=" * 70)
            print(f"\nDate: {results['date']}")
            print(f"Input points: {results['input_points']}")
            print(f"Output files: {list(results['output_files'].keys())}")

    return results


if __name__ == "__main__":
    main()

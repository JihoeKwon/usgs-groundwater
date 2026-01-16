"""
Visualize Kriging Interpolation Results
- Load kriging grid from sectioned format (.dat) or CSV
- Single frame: PNG output
- Multiple frames: GIF animation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import argparse
import glob
import re


def parse_sectioned_file(filepath):
    """
    Parse sectioned kriging output file.

    Returns:
    --------
    dict: {
        'header': {nframe, grid, points, dates},
        'coordinates': DataFrame with Lon, Lat,
        'depth': 2D array [points, dates],
        'variance': 2D array [points, dates]
    }
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    result = {'header': {}}

    # Parse [HEADER]
    header_match = re.search(r'\[HEADER\]\n(.*?)\n\n', content, re.DOTALL)
    if header_match:
        for line in header_match.group(1).strip().split('\n'):
            if '=' in line:
                key, val = line.split('=', 1)
                result['header'][key.lower()] = val

    # Parse dates
    dates_str = result['header'].get('dates', '')
    result['dates'] = dates_str.split(',') if dates_str else []

    # Parse grid dimensions
    grid_str = result['header'].get('grid', '50x50')
    grid_parts = grid_str.lower().split('x')
    result['grid_size'] = (int(grid_parts[0]), int(grid_parts[1]))

    n_points = int(result['header'].get('points', 0))
    n_frames = int(result['header'].get('nframe', 1))

    # Parse [COORDINATES]
    coord_match = re.search(r'\[COORDINATES\]\nLon,Lat\n(.*?)\n\n', content, re.DOTALL)
    if coord_match:
        coord_lines = coord_match.group(1).strip().split('\n')
        coords = [line.split(',') for line in coord_lines]
        result['coordinates'] = pd.DataFrame(coords, columns=['Lon', 'Lat']).astype(float)

    # Parse [DEPTH]
    depth_match = re.search(r'\[DEPTH\]\n(.*?)\n\n', content, re.DOTALL)
    if depth_match:
        depth_lines = depth_match.group(1).strip().split('\n')
        result['depth'] = np.array([[float(v) for v in line.split(',')] for line in depth_lines])

    # Parse [VARIANCE]
    var_match = re.search(r'\[VARIANCE\]\n(.*?)$', content, re.DOTALL)
    if var_match:
        var_lines = var_match.group(1).strip().split('\n')
        result['variance'] = np.array([[float(v) for v in line.split(',')] for line in var_lines])

    return result


def load_kriging_data(filepath):
    """
    Load kriging data from sectioned file (.dat) or CSV.

    Returns:
    --------
    dict with grid data
    """
    if filepath.endswith('.dat') or not filepath.endswith('.csv'):
        # Try sectioned format first
        try:
            return parse_sectioned_file(filepath)
        except:
            pass

    # Fall back to CSV format
    df = pd.read_csv(filepath)

    # Check if it's old single-frame format
    if 'Depth_ft' in df.columns:
        lons = sorted(df['Lon'].unique())
        lats = sorted(df['Lat'].unique())

        grid_size = (len(lats), len(lons))
        n_points = len(lats) * len(lons)

        # Reshape to 2D
        depth = np.zeros((n_points, 1))
        variance = np.zeros((n_points, 1))
        coords = []

        idx = 0
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                row = df[(df['Lat'] == lat) & (df['Lon'] == lon)]
                if not row.empty:
                    depth[idx, 0] = row['Depth_ft'].values[0]
                    if 'Variance' in df.columns:
                        variance[idx, 0] = row['Variance'].values[0]
                coords.append({'Lon': lon, 'Lat': lat})
                idx += 1

        return {
            'header': {'nframe': '1', 'grid': f'{len(lons)}x{len(lats)}', 'points': str(n_points)},
            'dates': ['unknown'],
            'grid_size': (len(lons), len(lats)),
            'coordinates': pd.DataFrame(coords),
            'depth': depth,
            'variance': variance
        }

    raise ValueError(f"Unknown file format: {filepath}")


def get_bbox_string(lon_grid, lat_grid):
    """Generate bbox string for title."""
    lon_min, lon_max = lon_grid.min(), lon_grid.max()
    lat_min, lat_max = lat_grid.min(), lat_grid.max()
    return f"[{lon_min:.1f}, {lat_min:.1f}, {lon_max:.1f}, {lat_max:.1f}]"


def create_frame_plot(ax, lon_grid, lat_grid, z_grid, var_grid, date_str,
                      orig_data=None, cmap='viridis_r', vmin=None, vmax=None,
                      region_name=None):
    """Create a single frame plot."""
    ax.clear()

    if vmin is None:
        vmin = z_grid.min()
    if vmax is None:
        vmax = z_grid.max()

    # Filled contour
    levels = np.linspace(vmin, vmax, 20)
    cf = ax.contourf(lon_grid, lat_grid, z_grid, levels=levels, cmap=cmap, extend='both')

    # Contour lines
    cs = ax.contour(lon_grid, lat_grid, z_grid, levels=10, colors='black',
                    linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

    # Original points overlay
    if orig_data is not None:
        ax.scatter(orig_data['lon'], orig_data['lat'], c='red',
                   edgecolors='white', linewidths=1, s=50, zorder=5)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Title with region/bbox info
    bbox_str = get_bbox_string(lon_grid, lat_grid)
    if region_name:
        ax.set_title(f'{region_name}\nGroundwater Depth - {date_str}')
    else:
        ax.set_title(f'Groundwater Depth - {date_str}\nBBox: {bbox_str}')

    ax.grid(True, alpha=0.3, linestyle='--')

    # Stats box
    stats_text = f'Min: {z_grid.min():.1f} ft\nMax: {z_grid.max():.1f} ft\nMean: {z_grid.mean():.1f} ft'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return cf


def visualize_single_frame(data, frame_idx=0, output_file=None, cmap='viridis_r',
                           original_csv=None, show_variance=True, region_name=None):
    """
    Visualize a single frame.
    """
    coords = data['coordinates']
    depth = data['depth']
    variance = data['variance']
    dates = data['dates']
    grid_w, grid_h = data['grid_size']

    # Get unique coordinates
    lons = coords['Lon'].values.reshape(grid_h, grid_w)[0, :]
    lats = coords['Lat'].values.reshape(grid_h, grid_w)[:, 0]

    # Get depth grid for this frame
    z_grid = depth[:, frame_idx].reshape(grid_h, grid_w)
    var_grid = variance[:, frame_idx].reshape(grid_h, grid_w) if variance is not None else None

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    date_str = dates[frame_idx] if frame_idx < len(dates) else 'Unknown'
    bbox_str = get_bbox_string(lon_grid, lat_grid)

    # Load original data
    orig_data = None
    if original_csv:
        try:
            orig_df = pd.read_csv(original_csv)
            date_cols = [c for c in orig_df.columns if c.startswith('20') and len(c) == 10]
            if date_str in date_cols:
                valid_mask = orig_df[date_str].notna()
                orig_data = {
                    'lon': orig_df.loc[valid_mask, 'Lon'].astype(float).values,
                    'lat': orig_df.loc[valid_mask, 'Lat'].astype(float).values,
                    'val': orig_df.loc[valid_mask, date_str].astype(float).values
                }
        except:
            pass

    # Create figure
    if show_variance and var_grid is not None and var_grid.max() > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Depth plot
        cf1 = create_frame_plot(axes[0], lon_grid, lat_grid, z_grid, var_grid,
                                date_str, orig_data, cmap, region_name=region_name)
        plt.colorbar(cf1, ax=axes[0], label='Depth to Water (ft)', shrink=0.8)

        # Variance plot
        cf2 = axes[1].contourf(lon_grid, lat_grid, var_grid, levels=20, cmap='Reds')
        plt.colorbar(cf2, ax=axes[1], label='Kriging Variance')
        if orig_data:
            axes[1].scatter(orig_data['lon'], orig_data['lat'], c='blue',
                           edgecolors='white', s=50, zorder=5)
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title(f'Uncertainty - {date_str}')
        axes[1].grid(True, alpha=0.3)

        # Main title with region/bbox
        if region_name:
            fig.suptitle(f'{region_name} - Groundwater Level Kriging Analysis\nBBox: {bbox_str}',
                        fontsize=14, fontweight='bold')
        else:
            fig.suptitle(f'Groundwater Level Kriging Analysis\nBBox: {bbox_str}',
                        fontsize=14, fontweight='bold')
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        cf = create_frame_plot(ax, lon_grid, lat_grid, z_grid, var_grid,
                               date_str, orig_data, cmap, region_name=region_name)
        plt.colorbar(cf, ax=ax, label='Depth to Water (ft)', shrink=0.8)

    plt.tight_layout()

    # Save
    if output_file is None:
        output_file = f"kriging_{date_str}.png"

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_file}")
    return output_file


def visualize_animation(data, output_file=None, cmap='viridis_r',
                        original_csv=None, fps=2, dpi=100, region_name=None):
    """
    Create animated GIF from multiple frames.
    """
    coords = data['coordinates']
    depth = data['depth']
    dates = data['dates']
    grid_w, grid_h = data['grid_size']
    n_frames = len(dates)

    print(f"Creating animation with {n_frames} frames...")

    # Get unique coordinates
    lons = coords['Lon'].values.reshape(grid_h, grid_w)[0, :]
    lats = coords['Lat'].values.reshape(grid_h, grid_w)[:, 0]
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # BBox string for title
    bbox_str = get_bbox_string(lon_grid, lat_grid)

    # Global min/max for consistent color scale
    vmin = depth.min()
    vmax = depth.max()

    # Load original data
    orig_df = None
    if original_csv:
        try:
            orig_df = pd.read_csv(original_csv)
        except:
            pass

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Initial contour
    z_grid = depth[:, 0].reshape(grid_h, grid_w)
    levels = np.linspace(vmin, vmax, 20)
    cf = ax.contourf(lon_grid, lat_grid, z_grid, levels=levels, cmap=cmap, extend='both')
    cbar = plt.colorbar(cf, ax=ax, label='Depth to Water (ft)', shrink=0.8)

    # Title with region/bbox
    if region_name:
        title_text = f'{region_name} - Groundwater Depth\nBBox: {bbox_str}\n{dates[0]}'
    else:
        title_text = f'Groundwater Depth - {dates[0]}\nBBox: {bbox_str}'
    title = ax.set_title(title_text, fontsize=12, fontweight='bold')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Stats text
    stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Original points scatter (if available)
    scatter = None

    def update(frame_idx):
        nonlocal cf, scatter

        # Clear previous contours
        for c in ax.collections:
            c.remove()
        for t in ax.texts:
            if t != stats_text:
                t.remove()

        # Get data for this frame
        z_grid = depth[:, frame_idx].reshape(grid_h, grid_w)
        date_str = dates[frame_idx]

        # Redraw contours
        cf = ax.contourf(lon_grid, lat_grid, z_grid, levels=levels, cmap=cmap, extend='both')
        cs = ax.contour(lon_grid, lat_grid, z_grid, levels=10, colors='black',
                        linewidths=0.5, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

        # Original points for this date
        if orig_df is not None and date_str in orig_df.columns:
            valid_mask = orig_df[date_str].notna()
            if valid_mask.any():
                ax.scatter(orig_df.loc[valid_mask, 'Lon'].astype(float),
                          orig_df.loc[valid_mask, 'Lat'].astype(float),
                          c='red', edgecolors='white', linewidths=1, s=50, zorder=5)

        # Update title and stats
        if region_name:
            title.set_text(f'{region_name} - Groundwater Depth\nBBox: {bbox_str}\n{date_str}')
        else:
            title.set_text(f'Groundwater Depth - {date_str}\nBBox: {bbox_str}')
        stats_text.set_text(f'Min: {z_grid.min():.1f} ft\nMax: {z_grid.max():.1f} ft\nMean: {z_grid.mean():.1f} ft')

        return [cf]

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)

    # Save as GIF
    if output_file is None:
        output_file = "kriging_animation.gif"

    print(f"Saving animation to: {output_file}")
    anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
    plt.close()

    print(f"Animation saved: {output_file} ({n_frames} frames, {fps} fps)")
    return output_file


def visualize_kriging(input_file, output_file=None, cmap='viridis_r',
                      original_csv=None, fps=2, frame=None, region_name=None):
    """
    Main visualization function.

    - Single frame or specified frame index -> PNG
    - Multiple frames without frame specified -> GIF animation
    """
    print("=" * 60)
    print("Kriging Visualization")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {input_file}")
    data = load_kriging_data(input_file)

    n_frames = int(data['header'].get('nframe', 1))
    print(f"  Frames: {n_frames}")
    print(f"  Grid: {data['grid_size']}")
    print(f"  Dates: {data['dates'][:3]}{'...' if n_frames > 3 else ''}")
    if region_name:
        print(f"  Region: {region_name}")

    if n_frames == 1 or frame is not None:
        # Single frame -> PNG
        frame_idx = frame if frame is not None else 0
        if output_file is None:
            date_str = data['dates'][frame_idx] if frame_idx < len(data['dates']) else 'frame'
            output_file = f"kriging_{date_str}.png"

        result = visualize_single_frame(data, frame_idx, output_file, cmap, original_csv,
                                        region_name=region_name)
    else:
        # Multiple frames -> GIF
        if output_file is None:
            output_file = "kriging_animation.gif"

        result = visualize_animation(data, output_file, cmap, original_csv, fps,
                                     region_name=region_name)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Kriging Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect: single frame -> PNG, multiple -> GIF
  python visualize_kriging.py kriging_output.dat

  # With region name
  python visualize_kriging.py kriging_output.dat --region "Southern California"

  # Specific frame as PNG
  python visualize_kriging.py kriging_output.dat --frame 5

  # With original data overlay
  python visualize_kriging.py kriging_output.dat --original data.csv

  # Custom GIF settings
  python visualize_kriging.py kriging_output.dat --fps 4 -o animation.gif
        """
    )

    parser.add_argument('input_file', help="Kriging output file (.dat or .csv)")
    parser.add_argument('--output', '-o', help="Output file (PNG for single, GIF for animation)")
    parser.add_argument('--original', help="Original data CSV for point overlay")
    parser.add_argument('--region', '-r', help="Region name for title (e.g., 'Southern California')")
    parser.add_argument('--frame', '-f', type=int, help="Specific frame index to render as PNG")
    parser.add_argument('--fps', type=int, default=2, help="Frames per second for GIF (default: 2)")
    parser.add_argument('--cmap', '-c', default='viridis_r',
                        help="Colormap (default: viridis_r)")

    args = parser.parse_args()

    visualize_kriging(
        input_file=args.input_file,
        output_file=args.output,
        cmap=args.cmap,
        original_csv=args.original,
        fps=args.fps,
        frame=args.frame,
        region_name=args.region
    )


if __name__ == "__main__":
    main()

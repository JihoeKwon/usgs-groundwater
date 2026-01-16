"""
Kriging Visualization MCP Server
- Visualize kriging results from CSV or sectioned format
- Single frame: PNG output
- Multiple frames: GIF animation
- Support region name and bbox in titles
"""

import json
import os
import numpy as np
import pandas as pd
import re
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
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import contextily as cx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False

# Create MCP server
mcp = FastMCP("visualize-kriging")


def _parse_sectioned_file(filepath):
    """
    Parse sectioned kriging output file (.dat format).

    Returns:
        dict: {header, dates, grid_size, coordinates, depth, variance}
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


def _load_kriging_data(filepath):
    """
    Load kriging data from sectioned file (.dat) or CSV.

    Returns:
        dict with grid data
    """
    if filepath.endswith('.dat') or not filepath.endswith('.csv'):
        # Try sectioned format first
        try:
            return _parse_sectioned_file(filepath)
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


def _get_bbox_string(lon_grid, lat_grid):
    """Generate bbox string for title."""
    lon_min, lon_max = lon_grid.min(), lon_grid.max()
    lat_min, lat_max = lat_grid.min(), lat_grid.max()
    return f"[{lon_min:.1f}, {lat_min:.1f}, {lon_max:.1f}, {lat_max:.1f}]"


def _get_basemap_provider(provider_name: str):
    """Get contextily basemap provider by name."""
    if not CONTEXTILY_AVAILABLE:
        return None

    providers = {
        "OpenStreetMap": cx.providers.OpenStreetMap.Mapnik,
        "CartoDB.Positron": cx.providers.CartoDB.Positron,
        "CartoDB.Voyager": cx.providers.CartoDB.Voyager,
        "Esri.WorldImagery": cx.providers.Esri.WorldImagery,
        "Esri.WorldStreetMap": cx.providers.Esri.WorldStreetMap,
        "Stamen.Terrain": cx.providers.Stadia.StamenTerrain,
    }
    return providers.get(provider_name, cx.providers.OpenStreetMap.Mapnik)


def _add_basemap(ax, lon_grid, lat_grid, provider_name: str = "OpenStreetMap",
                 alpha: float = 0.5, zoom_buffer: float = 0.1):
    """
    Add basemap to matplotlib axis with buffered extent.

    Args:
        ax: Matplotlib axis
        lon_grid: Longitude grid
        lat_grid: Latitude grid
        provider_name: Basemap provider name
        alpha: Basemap transparency
        zoom_buffer: Buffer around bbox as fraction
    """
    if not CONTEXTILY_AVAILABLE:
        return False

    # Calculate buffered extent
    lon_min, lon_max = lon_grid.min(), lon_grid.max()
    lat_min, lat_max = lat_grid.min(), lat_grid.max()

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    # Add buffer
    lon_min_buf = lon_min - lon_range * zoom_buffer
    lon_max_buf = lon_max + lon_range * zoom_buffer
    lat_min_buf = lat_min - lat_range * zoom_buffer
    lat_max_buf = lat_max + lat_range * zoom_buffer

    # Set axis limits with buffer
    ax.set_xlim(lon_min_buf, lon_max_buf)
    ax.set_ylim(lat_min_buf, lat_max_buf)

    try:
        provider = _get_basemap_provider(provider_name)
        cx.add_basemap(ax, crs="EPSG:4326", source=provider, alpha=alpha, zorder=0)
        return True
    except Exception as e:
        print(f"Warning: Failed to add basemap: {e}")
        return False


@mcp.tool()
def visualize_kriging_result(
    input_file: str,
    original_csv: str = None,
    output_file: str = None,
    region_name: str = None,
    colormap: str = "viridis_r",
    show_points: bool = True,
    frame: int = None,
    fps: int = 2,
    output_dir: str = None,
    show_basemap: bool = True,
    basemap_provider: str = "OpenStreetMap",
    basemap_alpha: float = 0.5,
    basemap_zoom_buffer: float = 0.1
) -> str:
    """
    Visualize kriging interpolation results.
    Single frame or specified frame -> PNG, Multiple frames -> GIF animation.

    Args:
        input_file: Kriging output file (.dat sectioned format or .csv)
        original_csv: Original data CSV file for overlay (optional)
        output_file: Output image file (PNG for single, GIF for animation)
        region_name: Region name for title (e.g., 'Southern California')
        colormap: Matplotlib colormap name (default: 'viridis_r')
        show_points: Whether to show original data points (default: True)
        frame: Specific frame index to render as PNG (optional)
        fps: Frames per second for GIF animation (default: 2)
        output_dir: Optional output directory (created if not exists)
        show_basemap: Whether to show OpenStreetMap basemap (default: True)
        basemap_provider: Basemap provider - 'OpenStreetMap', 'CartoDB.Positron', 'CartoDB.Voyager', 'Esri.WorldImagery'
        basemap_alpha: Basemap transparency 0-1 (default: 0.5)
        basemap_zoom_buffer: Buffer around bbox as fraction (default: 0.1 = 10%)

    Returns:
        JSON string with output file paths
    """
    if not MATPLOTLIB_AVAILABLE:
        return json.dumps({
            "error": "matplotlib not installed. Install with: pip install matplotlib"
        })

    # Load data
    try:
        data = _load_kriging_data(input_file)
    except Exception as e:
        return json.dumps({"error": f"Failed to load data: {str(e)}"})

    coords = data['coordinates']
    depth = data['depth']
    variance = data.get('variance')
    dates = data['dates']
    grid_w, grid_h = data['grid_size']
    n_frames = int(data['header'].get('nframe', 1))

    # Get unique coordinates
    lons = coords['Lon'].values.reshape(grid_h, grid_w)[0, :]
    lats = coords['Lat'].values.reshape(grid_h, grid_w)[:, 0]
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    bbox_str = _get_bbox_string(lon_grid, lat_grid)

    output_files = []

    if n_frames == 1 or frame is not None:
        # Single frame -> PNG
        frame_idx = frame if frame is not None else 0
        z_grid = depth[:, frame_idx].reshape(grid_h, grid_w)
        var_grid = variance[:, frame_idx].reshape(grid_h, grid_w) if variance is not None else None
        date_str = dates[frame_idx] if frame_idx < len(dates) else 'Unknown'

        # Load original data for overlay
        orig_data = None
        if original_csv and show_points:
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

        # Create figure with depth and variance
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Depth plot
        levels = np.linspace(z_grid.min(), z_grid.max(), 20)
        contour_alpha = 0.7 if show_basemap else 1.0
        cf1 = axes[0].contourf(lon_grid, lat_grid, z_grid, levels=levels, cmap=colormap,
                               alpha=contour_alpha, zorder=1)
        plt.colorbar(cf1, ax=axes[0], label='Depth to Water (ft)', shrink=0.8)
        cs1 = axes[0].contour(lon_grid, lat_grid, z_grid, levels=10, colors='black',
                              linewidths=0.5, alpha=0.5, zorder=2)
        axes[0].clabel(cs1, inline=True, fontsize=8, fmt='%.0f')

        # Add basemap if requested
        basemap_added = False
        if show_basemap:
            basemap_added = _add_basemap(axes[0], lon_grid, lat_grid,
                                         basemap_provider, basemap_alpha, basemap_zoom_buffer)

        if orig_data:
            axes[0].scatter(orig_data['lon'], orig_data['lat'], c='red',
                           edgecolors='white', linewidths=1, s=50, zorder=5)

        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        if region_name:
            axes[0].set_title(f'{region_name}\nGroundwater Depth - {date_str}')
        else:
            axes[0].set_title(f'Groundwater Depth - {date_str}')
        if not show_basemap:
            axes[0].grid(True, alpha=0.3, linestyle='--')

        # Stats box
        stats_text = f'Min: {z_grid.min():.1f} ft\nMax: {z_grid.max():.1f} ft\nMean: {z_grid.mean():.1f} ft'
        axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Variance plot
        if var_grid is not None and var_grid.max() > 0:
            cf2 = axes[1].contourf(lon_grid, lat_grid, var_grid, levels=20, cmap='Reds',
                                   alpha=contour_alpha, zorder=1)
            plt.colorbar(cf2, ax=axes[1], label='Kriging Variance')
            if orig_data:
                axes[1].scatter(orig_data['lon'], orig_data['lat'], c='blue',
                               edgecolors='white', s=50, zorder=5)
        # Add basemap to variance plot too
        if show_basemap:
            _add_basemap(axes[1], lon_grid, lat_grid,
                        basemap_provider, basemap_alpha, basemap_zoom_buffer)
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title(f'Uncertainty - {date_str}')
        if not show_basemap:
            axes[1].grid(True, alpha=0.3)

        # Main title
        if region_name:
            fig.suptitle(f'{region_name} - Groundwater Level Kriging Analysis\nBBox: {bbox_str}',
                        fontsize=14, fontweight='bold')
        else:
            fig.suptitle(f'Groundwater Level Kriging Analysis\nBBox: {bbox_str}',
                        fontsize=14, fontweight='bold')

        plt.tight_layout()

        if output_file is None:
            output_file = f"kriging_{date_str}.png"

        output_path = _get_output_path(output_file, output_dir)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        output_files.append(output_path)
        plt.close()

    else:
        # Multiple frames -> GIF animation
        vmin = depth.min()
        vmax = depth.max()

        # Load original data
        orig_df = None
        if original_csv and show_points:
            try:
                orig_df = pd.read_csv(original_csv)
            except:
                pass

        fig, ax = plt.subplots(figsize=(12, 10))

        # Add basemap FIRST (before contours) so it's in the background
        basemap_added = False
        contour_alpha = 1.0
        if show_basemap:
            basemap_added = _add_basemap(ax, lon_grid, lat_grid,
                                         basemap_provider, basemap_alpha, basemap_zoom_buffer)
            if basemap_added:
                contour_alpha = 0.7  # Make contours semi-transparent over basemap

        # Initial contour
        z_grid = depth[:, 0].reshape(grid_h, grid_w)
        levels = np.linspace(vmin, vmax, 20)
        cf = ax.contourf(lon_grid, lat_grid, z_grid, levels=levels, cmap=colormap,
                         extend='both', alpha=contour_alpha, zorder=1)
        cbar = plt.colorbar(cf, ax=ax, label='Depth to Water (ft)', shrink=0.8)

        # Title
        if region_name:
            title_text = f'{region_name} - Groundwater Depth\nBBox: {bbox_str}\n{dates[0]}'
        else:
            title_text = f'Groundwater Depth - {dates[0]}\nBBox: {bbox_str}'
        title = ax.set_title(title_text, fontsize=12, fontweight='bold')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        if not basemap_added:
            ax.grid(True, alpha=0.3, linestyle='--')

        stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Store initial number of images (basemap) to preserve them during animation
        n_basemap_images = len(ax.images)

        def update(frame_idx):
            nonlocal cf

            # Remove only contour collections, not basemap
            while len(ax.collections) > 0:
                ax.collections[0].remove()
            for t in ax.texts:
                if t != stats_text:
                    t.remove()

            z_grid = depth[:, frame_idx].reshape(grid_h, grid_w)
            date_str = dates[frame_idx]

            cf = ax.contourf(lon_grid, lat_grid, z_grid, levels=levels, cmap=colormap,
                             extend='both', alpha=contour_alpha, zorder=1)
            cs = ax.contour(lon_grid, lat_grid, z_grid, levels=10, colors='black',
                            linewidths=0.5, alpha=0.5, zorder=2)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

            if orig_df is not None and date_str in orig_df.columns:
                valid_mask = orig_df[date_str].notna()
                if valid_mask.any():
                    ax.scatter(orig_df.loc[valid_mask, 'Lon'].astype(float),
                              orig_df.loc[valid_mask, 'Lat'].astype(float),
                              c='red', edgecolors='white', linewidths=1, s=50, zorder=5)

            if region_name:
                title.set_text(f'{region_name} - Groundwater Depth\nBBox: {bbox_str}\n{date_str}')
            else:
                title.set_text(f'Groundwater Depth - {date_str}\nBBox: {bbox_str}')
            stats_text.set_text(f'Min: {z_grid.min():.1f} ft\nMax: {z_grid.max():.1f} ft\nMean: {z_grid.mean():.1f} ft')

            return [cf]

        anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)

        if output_file is None:
            output_file = "kriging_animation.gif"

        output_path = _get_output_path(output_file, output_dir)
        anim.save(output_path, writer='pillow', fps=fps, dpi=100)
        output_files.append(output_path)
        plt.close()

    return json.dumps({
        "status": "success",
        "input_file": input_file,
        "original_csv": original_csv,
        "region_name": region_name,
        "n_frames": n_frames,
        "grid_size": f"{grid_w} x {grid_h}",
        "bbox": bbox_str,
        "output_type": "gif" if n_frames > 1 and frame is None else "png",
        "output_dir": output_dir,
        "output_files": output_files,
        "basemap": {
            "enabled": show_basemap,
            "provider": basemap_provider if show_basemap else None,
            "contextily_available": CONTEXTILY_AVAILABLE
        }
    }, indent=2)


@mcp.tool()
def get_available_colormaps() -> str:
    """
    Get list of recommended colormaps for groundwater visualization.

    Returns:
        JSON string with available colormaps
    """
    colormaps = {
        "viridis_r": "Default. Yellow (shallow) to purple (deep).",
        "plasma_r": "Similar to viridis, warmer colors.",
        "RdYlBu": "Red-Yellow-Blue diverging colormap.",
        "terrain": "Earth-tone colors.",
        "Blues": "Light to dark blue gradient.",
        "coolwarm": "Cool to warm diverging colormap.",
        "RdBu_r": "Blue to red (reversed)."
    }

    return json.dumps({
        "recommended_colormaps": colormaps,
        "default": "viridis_r",
        "note": "Add '_r' suffix to reverse any colormap"
    }, indent=2)


@mcp.tool()
def create_comparison_plot(
    grid_csv_1: str,
    grid_csv_2: str,
    label_1: str = "Dataset 1",
    label_2: str = "Dataset 2",
    output_file: str = None,
    output_dir: str = None
) -> str:
    """
    Create side-by-side comparison of two kriging results.

    Args:
        grid_csv_1: First kriging grid CSV file
        grid_csv_2: Second kriging grid CSV file
        label_1: Label for first dataset
        label_2: Label for second dataset
        output_file: Output image file
        output_dir: Optional output directory (created if not exists)

    Returns:
        JSON string with output file path
    """
    if not MATPLOTLIB_AVAILABLE:
        return json.dumps({
            "error": "matplotlib not installed"
        })

    try:
        df1 = pd.read_csv(grid_csv_1)
        df2 = pd.read_csv(grid_csv_2)
    except Exception as e:
        return json.dumps({"error": f"Failed to load CSV: {str(e)}"})

    # Process both datasets
    def process_grid(df):
        lons = sorted(df['Lon'].unique())
        lats = sorted(df['Lat'].unique())
        z = np.zeros((len(lats), len(lons)))
        for _, row in df.iterrows():
            i = lats.index(row['Lat'])
            j = lons.index(row['Lon'])
            z[i, j] = row['Depth_ft']
        return np.meshgrid(lons, lats), z

    (lon1, lat1), z1 = process_grid(df1)
    (lon2, lat2), z2 = process_grid(df2)

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    vmin = min(z1.min(), z2.min())
    vmax = max(z1.max(), z2.max())

    cf1 = axes[0].contourf(lon1, lat1, z1, levels=20, cmap='viridis_r', vmin=vmin, vmax=vmax)
    axes[0].set_title(label_1, fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(cf1, ax=axes[0], label='Depth (ft)')

    cf2 = axes[1].contourf(lon2, lat2, z2, levels=20, cmap='viridis_r', vmin=vmin, vmax=vmax)
    axes[1].set_title(label_2, fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(cf2, ax=axes[1], label='Depth (ft)')

    fig.suptitle('Groundwater Depth Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"kriging_comparison_{timestamp}.png"

    output_path = _get_output_path(output_file, output_dir)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return json.dumps({
        "status": "success",
        "output_file": output_path,
        "output_dir": output_dir,
        "dataset_1": {
            "file": grid_csv_1,
            "min": float(z1.min()),
            "max": float(z1.max())
        },
        "dataset_2": {
            "file": grid_csv_2,
            "min": float(z2.min()),
            "max": float(z2.max())
        }
    }, indent=2)


if __name__ == "__main__":
    mcp.run()

"""
Kriging Visualization MCP Server
- Visualize kriging results
- Create contour/heatmap plots
- Save as PNG/JPG
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from mcp.server.fastmcp import FastMCP

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Create MCP server
mcp = FastMCP("visualize-kriging")


@mcp.tool()
def visualize_kriging_result(
    grid_csv: str,
    original_csv: str = None,
    output_file: str = None,
    title: str = None,
    colormap: str = "viridis_r",
    show_points: bool = True
) -> str:
    """
    Visualize kriging interpolation results.

    Args:
        grid_csv: Kriging grid CSV file (must have Lon, Lat, Depth_ft columns)
        original_csv: Original data CSV file for overlay (optional)
        output_file: Output image file (PNG/JPG). Auto-generated if not provided.
        title: Plot title (optional)
        colormap: Matplotlib colormap name (default: 'viridis_r')
        show_points: Whether to show original data points (default: True)

    Returns:
        JSON string with output file paths
    """
    if not MATPLOTLIB_AVAILABLE:
        return json.dumps({
            "error": "matplotlib not installed. Install with: pip install matplotlib"
        })

    # Load grid data
    try:
        grid_df = pd.read_csv(grid_csv)
    except Exception as e:
        return json.dumps({"error": f"Failed to load grid CSV: {str(e)}"})

    # Get unique lon/lat values
    lons = sorted(grid_df['Lon'].unique())
    lats = sorted(grid_df['Lat'].unique())

    # Reshape to 2D array
    z_grid = np.zeros((len(lats), len(lons)))
    var_grid = np.zeros((len(lats), len(lons)))

    for _, row in grid_df.iterrows():
        i = lats.index(row['Lat'])
        j = lons.index(row['Lon'])
        z_grid[i, j] = row['Depth_ft']
        if 'Variance' in row:
            var_grid[i, j] = row['Variance']

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Load original data if provided
    orig_lons, orig_lats, orig_vals = None, None, None
    if original_csv and show_points:
        try:
            orig_df = pd.read_csv(original_csv)
            date_cols = [c for c in orig_df.columns if c.startswith('20') and len(c) == 10]
            if date_cols:
                val_col = date_cols[-1]
                valid_mask = orig_df[val_col].notna()
                orig_lons = orig_df.loc[valid_mask, 'Lon'].astype(float).values
                orig_lats = orig_df.loc[valid_mask, 'Lat'].astype(float).values
                orig_vals = orig_df.loc[valid_mask, val_col].astype(float).values
        except Exception as e:
            pass  # Continue without original points

    output_files = []

    # Create main figure (2 panels)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Depth contour
    ax1 = axes[0]
    levels = np.linspace(z_grid.min(), z_grid.max(), 20)
    cf = ax1.contourf(lon_grid, lat_grid, z_grid, levels=levels, cmap=colormap)
    plt.colorbar(cf, ax=ax1, label='Depth to Water (ft)')

    cs = ax1.contour(lon_grid, lat_grid, z_grid, levels=10, colors='white',
                     linewidths=0.5, alpha=0.5)
    ax1.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

    if orig_lons is not None:
        ax1.scatter(orig_lons, orig_lats, c=orig_vals, cmap=colormap,
                   edgecolors='white', linewidths=1.5, s=100,
                   vmin=z_grid.min(), vmax=z_grid.max(), zorder=5)

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Groundwater Depth (Kriging Interpolation)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Variance
    ax2 = axes[1]
    if var_grid.max() > 0:
        cf2 = ax2.contourf(lon_grid, lat_grid, var_grid, levels=20, cmap='Reds')
        plt.colorbar(cf2, ax=ax2, label='Kriging Variance')

        if orig_lons is not None:
            ax2.scatter(orig_lons, orig_lats, c='blue', edgecolors='white',
                       linewidths=1, s=80, zorder=5, label='Observation Points')
            ax2.legend(loc='upper right')

    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Kriging Variance (Uncertainty)')
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save main figure
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"kriging_visualization_{timestamp}.png"

    plt.savefig(output_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    output_files.append(output_file)
    plt.close()

    # Create simple single-plot version
    fig2, ax = plt.subplots(figsize=(12, 10))

    cf = ax.contourf(lon_grid, lat_grid, z_grid, levels=20, cmap=colormap)
    plt.colorbar(cf, ax=ax, label='Depth to Water (ft)', shrink=0.8)

    cs = ax.contour(lon_grid, lat_grid, z_grid, levels=10, colors='black',
                    linewidths=0.5, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=9, fmt='%.0f ft')

    if orig_lons is not None:
        ax.scatter(orig_lons, orig_lats, c=orig_vals, cmap=colormap,
                  edgecolors='black', linewidths=2, s=150,
                  vmin=z_grid.min(), vmax=z_grid.max(), zorder=5)
        for lon, lat, val in zip(orig_lons, orig_lats, orig_vals):
            ax.annotate(f'{val:.1f}', (lon, lat), fontsize=8,
                       ha='center', va='bottom', fontweight='bold',
                       xytext=(0, 8), textcoords='offset points')

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title or 'Groundwater Depth - Kriging Interpolation',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    stats_text = f'Min: {z_grid.min():.1f} ft\nMax: {z_grid.max():.1f} ft\nMean: {z_grid.mean():.1f} ft'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    simple_output = output_file.replace('.png', '_simple.png').replace('.jpg', '_simple.jpg')
    plt.savefig(simple_output, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    output_files.append(simple_output)
    plt.close()

    return json.dumps({
        "status": "success",
        "grid_csv": grid_csv,
        "original_csv": original_csv,
        "grid_size": f"{len(lons)} x {len(lats)}",
        "depth_range": {
            "min": float(z_grid.min()),
            "max": float(z_grid.max()),
            "mean": float(z_grid.mean())
        },
        "output_files": output_files
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
    output_file: str = None
) -> str:
    """
    Create side-by-side comparison of two kriging results.

    Args:
        grid_csv_1: First kriging grid CSV file
        grid_csv_2: Second kriging grid CSV file
        label_1: Label for first dataset
        label_2: Label for second dataset
        output_file: Output image file

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

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return json.dumps({
        "status": "success",
        "output_file": output_file,
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

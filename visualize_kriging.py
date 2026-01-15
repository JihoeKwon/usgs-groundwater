"""
Visualize Kriging Interpolation Results
- Load kriging grid from CSV
- Create contour/heatmap visualization
- Overlay original data points
- Save as PNG/JPG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import glob


def visualize_kriging(grid_csv, original_csv=None, output_file=None,
                      title=None, cmap='viridis_r', show_points=True):
    """
    Visualize kriging results

    Parameters:
    -----------
    grid_csv : str
        Kriging grid CSV file
    original_csv : str
        Original data CSV file (for overlay)
    output_file : str
        Output image file (PNG/JPG)
    title : str
        Plot title
    cmap : str
        Colormap name
    show_points : bool
        Whether to show original data points
    """
    print("=" * 60)
    print("Kriging Visualization")
    print("=" * 60)

    # Load grid data
    print(f"\nLoading grid: {grid_csv}")
    grid_df = pd.read_csv(grid_csv)

    # Get unique lon/lat values
    lons = sorted(grid_df['Lon'].unique())
    lats = sorted(grid_df['Lat'].unique())

    print(f"  Grid size: {len(lons)} x {len(lats)}")

    # Reshape to 2D array
    z_grid = np.zeros((len(lats), len(lons)))
    var_grid = np.zeros((len(lats), len(lons)))

    for _, row in grid_df.iterrows():
        i = lats.index(row['Lat'])
        j = lons.index(row['Lon'])
        z_grid[i, j] = row['Depth_ft']
        if 'Variance' in row:
            var_grid[i, j] = row['Variance']

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Load original data if provided
    orig_lons, orig_lats, orig_vals = None, None, None
    if original_csv and show_points:
        print(f"Loading original data: {original_csv}")
        orig_df = pd.read_csv(original_csv)

        # Find the last date column
        date_cols = [c for c in orig_df.columns if c.startswith('20') and len(c) == 10]
        if date_cols:
            val_col = date_cols[-1]
            valid_mask = orig_df[val_col].notna()
            orig_lons = orig_df.loc[valid_mask, 'Lon'].astype(float).values
            orig_lats = orig_df.loc[valid_mask, 'Lat'].astype(float).values
            orig_vals = orig_df.loc[valid_mask, val_col].astype(float).values
            print(f"  Original points: {len(orig_vals)}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Depth contour/heatmap
    ax1 = axes[0]

    # Filled contour
    levels = np.linspace(z_grid.min(), z_grid.max(), 20)
    cf = ax1.contourf(lon_grid, lat_grid, z_grid, levels=levels, cmap=cmap)
    plt.colorbar(cf, ax=ax1, label='Depth to Water (ft)')

    # Contour lines
    cs = ax1.contour(lon_grid, lat_grid, z_grid, levels=10, colors='white',
                     linewidths=0.5, alpha=0.5)
    ax1.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

    # Overlay original points
    if orig_lons is not None:
        scatter = ax1.scatter(orig_lons, orig_lats, c=orig_vals, cmap=cmap,
                             edgecolors='white', linewidths=1.5, s=100,
                             vmin=z_grid.min(), vmax=z_grid.max(),
                             zorder=5)
        # Add labels
        for lon, lat, val in zip(orig_lons, orig_lats, orig_vals):
            ax1.annotate(f'{val:.0f}', (lon, lat), fontsize=7,
                        ha='center', va='bottom', color='white',
                        xytext=(0, 5), textcoords='offset points')

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Groundwater Depth (Kriging Interpolation)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Variance/Uncertainty
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
    else:
        ax2.text(0.5, 0.5, 'Variance data not available',
                ha='center', va='center', transform=ax2.transAxes)

    # Main title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle('Groundwater Level Kriging Analysis', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"kriging_visualization_{timestamp}.png"

    plt.savefig(output_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved to: {output_file}")

    plt.close()

    # Also create a simple single-plot version
    fig2, ax = plt.subplots(figsize=(12, 10))

    cf = ax.contourf(lon_grid, lat_grid, z_grid, levels=20, cmap=cmap)
    cbar = plt.colorbar(cf, ax=ax, label='Depth to Water (ft)', shrink=0.8)

    cs = ax.contour(lon_grid, lat_grid, z_grid, levels=10, colors='black',
                    linewidths=0.5, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=9, fmt='%.0f ft')

    if orig_lons is not None:
        ax.scatter(orig_lons, orig_lats, c=orig_vals, cmap=cmap,
                  edgecolors='black', linewidths=2, s=150,
                  vmin=z_grid.min(), vmax=z_grid.max(), zorder=5)
        for lon, lat, val in zip(orig_lons, orig_lats, orig_vals):
            ax.annotate(f'{val:.1f}', (lon, lat), fontsize=8,
                       ha='center', va='bottom', fontweight='bold',
                       xytext=(0, 8), textcoords='offset points')

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Groundwater Depth - Kriging Interpolation\n(San Diego Area, 2026-01-13)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add stats box
    stats_text = f'Min: {z_grid.min():.1f} ft\nMax: {z_grid.max():.1f} ft\nMean: {z_grid.mean():.1f} ft'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    simple_output = output_file.replace('.png', '_simple.png').replace('.jpg', '_simple.jpg')
    plt.savefig(simple_output, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved to: {simple_output}")

    plt.close()

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)

    return output_file, simple_output


if __name__ == "__main__":
    # Find most recent kriging grid file
    grid_files = glob.glob("kriging_*_grid.csv")
    orig_files = glob.glob("groundwater_bbox_*.csv")

    if grid_files:
        latest_grid = max(grid_files)
        latest_orig = max(orig_files) if orig_files else None

        print(f"Grid file: {latest_grid}")
        print(f"Original file: {latest_orig}")
        print()

        visualize_kriging(
            grid_csv=latest_grid,
            original_csv=latest_orig,
            cmap='viridis_r'  # reversed viridis (deeper = darker)
        )
    else:
        print("No kriging grid file found.")
        print("Run kriging_groundwater.py first.")

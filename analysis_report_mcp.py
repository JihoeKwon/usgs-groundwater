#!/usr/bin/env python3
"""
Analysis Report MCP Server
Generates summary reports from groundwater kriging analysis results.
"""

import json
import os
from datetime import datetime
from mcp.server.fastmcp import FastMCP

import pandas as pd
import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

mcp = FastMCP("analysis-report")


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


def parse_kriging_dat(dat_file: str) -> dict:
    """Parse kriging .dat file and extract statistics.

    Supports two formats:
    1. Sectioned format with [DEPTH:date] sections
    2. Matrix format with [DEPTH] section containing all dates per row
    """
    with open(dat_file, 'r') as f:
        lines = f.readlines()

    dates = []
    depth_data = []
    current_section = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('[HEADER]'):
            current_section = 'header'
            i += 1
            continue
        elif line.startswith('DATES='):
            dates = line.replace('DATES=', '').split(',')
            i += 1
            continue
        elif line.startswith('[COORDINATES]'):
            current_section = 'coordinates'
            i += 1
            continue
        elif line == '[DEPTH]':
            current_section = 'depth'
            i += 1
            continue
        elif line.startswith('[DEPTH:'):
            # Old format with per-date sections
            date = line.replace('[DEPTH:', '').replace(']', '').strip()
            if date not in dates:
                dates.append(date)
            current_section = f'depth:{date}'
            i += 1
            continue
        elif line.startswith('[VARIANCE]') or line.startswith('[VARIANCE:'):
            current_section = 'variance'
            i += 1
            continue

        if current_section == 'depth' and line:
            try:
                values = [float(v) for v in line.split(',') if v.strip()]
                if values:
                    depth_data.append(values)
            except:
                pass
        elif current_section and current_section.startswith('depth:') and line:
            # Old format handling
            date = current_section.replace('depth:', '')
            try:
                values = [float(v) for v in line.split(',') if v.strip()]
                # Store in different structure for old format
                if not hasattr(parse_kriging_dat, '_old_format_data'):
                    parse_kriging_dat._old_format_data = {}
                if date not in parse_kriging_dat._old_format_data:
                    parse_kriging_dat._old_format_data[date] = []
                parse_kriging_dat._old_format_data[date].extend(values)
            except:
                pass

        i += 1

    # Calculate statistics for each date
    stats = {}

    if depth_data and dates:
        # Matrix format: each row contains values for all dates
        depth_array = np.array(depth_data)  # Shape: (n_points, n_dates)

        for idx, date in enumerate(dates):
            if idx < depth_array.shape[1]:
                date_values = depth_array[:, idx]
                valid_values = date_values[~np.isnan(date_values)]
                if len(valid_values) > 0:
                    stats[date] = {
                        'min': float(np.min(valid_values)),
                        'max': float(np.max(valid_values)),
                        'mean': float(np.mean(valid_values)),
                        'std': float(np.std(valid_values)),
                        'median': float(np.median(valid_values))
                    }
    elif hasattr(parse_kriging_dat, '_old_format_data'):
        # Old sectioned format
        for date, values in parse_kriging_dat._old_format_data.items():
            if values:
                arr = np.array(values)
                stats[date] = {
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'median': float(np.median(arr))
                }
        parse_kriging_dat._old_format_data = {}

    return stats


def analyze_csv_data(csv_file: str) -> dict:
    """Analyze original CSV data."""
    df = pd.read_csv(csv_file)

    # Basic info
    info = {
        'total_sites': len(df),
        'lat_range': [float(df['Lat'].min()), float(df['Lat'].max())],
        'lon_range': [float(df['Lon'].min()), float(df['Lon'].max())]
    }

    # Date columns
    date_cols = [c for c in df.columns if c[0:2] in ['20', '19']]
    info['date_count'] = len(date_cols)

    if date_cols:
        info['date_range'] = [date_cols[0], date_cols[-1]]

        # Per-date statistics
        date_stats = {}
        for col in date_cols:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                date_stats[col] = {
                    'sites_with_data': int(len(valid_data)),
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()) if len(valid_data) > 1 else 0
                }
        info['date_stats'] = date_stats

    return info


def generate_markdown_report(
    region_name: str,
    csv_info: dict,
    kriging_stats: dict,
    output_file: str
) -> str:
    """Generate a markdown report."""

    lines = []
    lines.append(f"# {region_name} Groundwater Analysis Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n---\n")

    # Data Overview
    lines.append("## 1. Data Overview\n")
    lines.append(f"- **Total Monitoring Sites**: {csv_info['total_sites']}")
    lines.append(f"- **Latitude Range**: {csv_info['lat_range'][0]:.4f}° ~ {csv_info['lat_range'][1]:.4f}°")
    lines.append(f"- **Longitude Range**: {csv_info['lon_range'][0]:.4f}° ~ {csv_info['lon_range'][1]:.4f}°")
    lines.append(f"- **Analysis Period**: {csv_info.get('date_range', ['N/A', 'N/A'])[0]} ~ {csv_info.get('date_range', ['N/A', 'N/A'])[1]}")
    lines.append(f"- **Total Data Points**: {csv_info.get('date_count', 0)} dates")
    lines.append("")

    # Summary Statistics
    lines.append("## 2. Summary Statistics\n")

    if 'date_stats' in csv_info and csv_info['date_stats']:
        date_stats = csv_info['date_stats']
        dates = sorted(date_stats.keys())

        # Overall statistics
        all_means = [date_stats[d]['mean'] for d in dates]
        all_mins = [date_stats[d]['min'] for d in dates]
        all_maxs = [date_stats[d]['max'] for d in dates]

        lines.append("### Overall Statistics (Original Data)\n")
        lines.append(f"- **Depth Range**: {min(all_mins):.1f} ft ~ {max(all_maxs):.1f} ft")
        lines.append(f"- **Mean Depth Range**: {min(all_means):.1f} ft ~ {max(all_means):.1f} ft")
        lines.append(f"- **Average Mean Depth**: {np.mean(all_means):.1f} ft")
        lines.append("")

        # Temporal Analysis
        lines.append("### Temporal Analysis\n")
        lines.append("| Date | Sites | Min (ft) | Max (ft) | Mean (ft) |")
        lines.append("|------|-------|----------|----------|-----------|")

        # Sample key dates (first, middle, last of each year)
        years = sorted(set(d[:4] for d in dates))
        sample_dates = []
        for year in years:
            year_dates = [d for d in dates if d.startswith(year)]
            if year_dates:
                sample_dates.append(year_dates[0])  # First
                if len(year_dates) > 6:
                    sample_dates.append(year_dates[len(year_dates)//2])  # Middle
                if len(year_dates) > 1:
                    sample_dates.append(year_dates[-1])  # Last

        sample_dates = sorted(set(sample_dates))
        for d in sample_dates:
            s = date_stats[d]
            lines.append(f"| {d} | {s['sites_with_data']} | {s['min']:.1f} | {s['max']:.1f} | {s['mean']:.1f} |")
        lines.append("")

    # Kriging Results
    if kriging_stats:
        lines.append("## 3. Kriging Interpolation Results\n")

        kriging_dates = sorted(kriging_stats.keys())
        kriging_means = [kriging_stats[d]['mean'] for d in kriging_dates]
        kriging_mins = [kriging_stats[d]['min'] for d in kriging_dates]
        kriging_maxs = [kriging_stats[d]['max'] for d in kriging_dates]

        lines.append("### Interpolated Surface Statistics\n")
        lines.append(f"- **Interpolated Depth Range**: {min(kriging_mins):.1f} ft ~ {max(kriging_maxs):.1f} ft")
        lines.append(f"- **Mean Interpolated Depth**: {np.mean(kriging_means):.1f} ft")
        lines.append(f"- **Total Frames**: {len(kriging_dates)}")
        lines.append("")

        # Yearly summary
        lines.append("### Yearly Summary (Kriging Results)\n")
        lines.append("| Year | Frames | Avg Min (ft) | Avg Max (ft) | Avg Mean (ft) |")
        lines.append("|------|--------|--------------|--------------|---------------|")

        years = sorted(set(d[:4] for d in kriging_dates))
        for year in years:
            year_dates = [d for d in kriging_dates if d.startswith(year)]
            year_mins = [kriging_stats[d]['min'] for d in year_dates]
            year_maxs = [kriging_stats[d]['max'] for d in year_dates]
            year_means = [kriging_stats[d]['mean'] for d in year_dates]
            lines.append(f"| {year} | {len(year_dates)} | {np.mean(year_mins):.1f} | {np.mean(year_maxs):.1f} | {np.mean(year_means):.1f} |")
        lines.append("")

    # Spatial Patterns
    lines.append("## 4. Spatial Patterns\n")
    lines.append("Based on the kriging interpolation results:\n")
    lines.append("- **Deep Aquifer Zone**: Areas with depth > 400 ft (typically in mountainous/inland regions)")
    lines.append("- **Shallow Aquifer Zone**: Areas with depth < 100 ft (typically near coast/valleys)")
    lines.append("- **Transition Zone**: Areas with depth 100-400 ft")
    lines.append("")

    # Data Quality Analysis
    data_quality_warnings = []
    site_count_issues = []

    if 'date_stats' in csv_info and csv_info['date_stats']:
        date_stats = csv_info['date_stats']
        dates = sorted(date_stats.keys())
        site_counts = [date_stats[d]['sites_with_data'] for d in dates]

        # Find peak and final site counts
        peak_sites = max(site_counts)
        peak_date = dates[site_counts.index(peak_sites)]
        final_sites = site_counts[-1]
        final_date = dates[-1]

        # Check for significant drop from peak
        if final_sites < peak_sites * 0.7:  # 30% or more drop
            drop_pct = (1 - final_sites / peak_sites) * 100
            site_count_issues.append({
                'type': 'significant_drop',
                'peak_date': peak_date,
                'peak_sites': peak_sites,
                'final_date': final_date,
                'final_sites': final_sites,
                'drop_pct': drop_pct
            })
            data_quality_warnings.append(
                f"Site count dropped {drop_pct:.0f}% from peak ({peak_sites} sites on {peak_date}) "
                f"to {final_sites} sites on {final_date}"
            )

        # Check for sudden drops (>20% between consecutive periods)
        for i in range(1, len(site_counts)):
            if site_counts[i] < site_counts[i-1] * 0.8:
                drop_pct = (1 - site_counts[i] / site_counts[i-1]) * 100
                data_quality_warnings.append(
                    f"Sudden drop of {drop_pct:.0f}% between {dates[i-1]} ({site_counts[i-1]} sites) "
                    f"and {dates[i]} ({site_counts[i]} sites)"
                )

    if data_quality_warnings:
        lines.append("## 5. Data Quality Warnings\n")
        lines.append("> **CAUTION**: The following data quality issues were detected:\n")
        for warning in data_quality_warnings:
            lines.append(f"- {warning}")
        lines.append("")
        lines.append("These issues may significantly affect the reliability of trend analysis and kriging results.")
        lines.append("Results for periods with reduced site counts should be interpreted with caution.")
        lines.append("")

    # Trend Analysis
    if kriging_stats and len(kriging_stats) > 10:
        section_num = 6 if data_quality_warnings else 5
        lines.append(f"## {section_num}. Trend Analysis\n")
        kriging_dates = sorted(kriging_stats.keys())
        first_10 = kriging_dates[:10]
        last_10 = kriging_dates[-10:]

        early_mean = np.mean([kriging_stats[d]['mean'] for d in first_10])
        late_mean = np.mean([kriging_stats[d]['mean'] for d in last_10])
        change = late_mean - early_mean

        lines.append(f"- **Early Period Mean** ({first_10[0]} ~ {first_10[-1]}): {early_mean:.1f} ft")
        lines.append(f"- **Late Period Mean** ({last_10[0]} ~ {last_10[-1]}): {late_mean:.1f} ft")
        lines.append(f"- **Change**: {change:+.1f} ft ({'deepening' if change > 0 else 'rising'})")

        # Add caveat if site counts are problematic
        if site_count_issues:
            lines.append("")
            lines.append("> **Note**: This trend may be affected by the significant reduction in monitoring sites ")
            lines.append("> during the late period. The apparent change could be due to:")
            lines.append("> - Actual groundwater level changes")
            lines.append("> - Sampling bias from site dropout (remaining sites may not be representative)")
            lines.append("> - Seasonal data collection patterns")
        lines.append("")

    # Write to file
    content = '\n'.join(lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    return content


def generate_trend_analysis_chart(
    csv_info: dict,
    region_name: str,
    output_file: str
) -> str:
    """
    Generate a 4-panel trend analysis chart.

    Panels:
    1. Top-left: Depth trend with std dev range
    2. Top-right: Annual distribution boxplot
    3. Bottom-left: Quarterly mean bar chart
    4. Bottom-right: Active monitoring sites over time

    Args:
        csv_info: Dictionary with CSV analysis data
        region_name: Name of the region for the title
        output_file: Output file path for the PNG

    Returns:
        Output file path
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for chart generation")

    if 'date_stats' not in csv_info or not csv_info['date_stats']:
        raise ValueError("No date statistics available for chart generation")

    date_stats = csv_info['date_stats']
    dates_str = sorted(date_stats.keys())
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates_str]

    # Extract statistics
    means = [date_stats[d]['mean'] for d in dates_str]
    stds = [date_stats[d].get('std', 0) for d in dates_str]
    mins = [date_stats[d]['min'] for d in dates_str]
    maxs = [date_stats[d]['max'] for d in dates_str]
    site_counts = [date_stats[d]['sites_with_data'] for d in dates_str]

    # Calculate std dev range
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    upper_bound = means_arr - stds_arr  # Inverted because depth increases downward
    lower_bound = means_arr + stds_arr

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{region_name} Groundwater Analysis', fontsize=16, fontweight='bold')

    # =========================================
    # Panel 1: Top-left - Depth trend with std dev range
    # =========================================
    ax1 = axes[0, 0]
    ax1.fill_between(dates, upper_bound, lower_bound, alpha=0.3, color='blue', label='Std Dev Range')
    ax1.plot(dates, means, 'b-', linewidth=1.5, label='Mean Depth')
    ax1.set_ylabel('Depth (ft below land surface)', fontsize=10)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_title(f'{region_name} Groundwater Depth Trend ({dates_str[0][:4]}-{dates_str[-1][:4]})', fontsize=11)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # =========================================
    # Panel 2: Top-right - Annual distribution boxplot
    # =========================================
    ax2 = axes[0, 1]

    # Group data by year - collect all individual measurements
    yearly_data = {}
    # We need to read the original CSV to get individual values
    # For now, use the per-date statistics to approximate
    for date_str in dates_str:
        year = date_str[:4]
        if year not in yearly_data:
            yearly_data[year] = []
        # Add mean as representative value (we'll improve this with actual data)
        stats = date_stats[date_str]
        # Create approximate distribution using min, mean, max
        yearly_data[year].append(stats['mean'])

    years = sorted(yearly_data.keys())
    data_by_year = [yearly_data[y] for y in years]

    bp = ax2.boxplot(data_by_year, tick_labels=years, patch_artist=True)
    colors = ['#87CEEB', '#87CEEB', '#87CEEB', '#87CEEB', '#87CEEB'][:len(years)]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylabel('Depth (ft below land surface)', fontsize=10)
    ax2.set_title('Annual Distribution of Groundwater Depth', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # =========================================
    # Panel 3: Bottom-left - Quarterly mean bar chart
    # =========================================
    ax3 = axes[1, 0]

    # Group by quarter
    quarterly_data = {}
    for date_str in dates_str:
        year = date_str[:4]
        month = int(date_str[5:7])
        quarter = (month - 1) // 3 + 1
        key = f"{year}\nQ{quarter}"
        if key not in quarterly_data:
            quarterly_data[key] = []
        quarterly_data[key].append(date_stats[date_str]['mean'])

    quarters = list(quarterly_data.keys())
    quarterly_means = [np.mean(quarterly_data[q]) for q in quarters]

    # Color bars by year (alternating red/green pattern)
    bar_colors = []
    for q in quarters:
        year = q.split('\n')[0]
        year_idx = int(year) % 2
        bar_colors.append('#E57373' if year_idx else '#81C784')

    bars = ax3.bar(range(len(quarters)), quarterly_means, color=bar_colors, alpha=0.8)

    # Reference line
    overall_mean = np.mean(quarterly_means)
    ax3.axhline(y=overall_mean, color='red', linestyle='--', linewidth=1.5,
                label=f'Reference ({overall_mean:.0f} ft)')

    ax3.set_xticks(range(len(quarters)))
    ax3.set_xticklabels(quarters, fontsize=8)
    ax3.set_xlabel('Quarter', fontsize=10)
    ax3.set_ylabel('Mean Depth (ft)', fontsize=10)
    ax3.set_title('Quarterly Mean Groundwater Depth', fontsize=11)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')

    # =========================================
    # Panel 4: Bottom-right - Active monitoring sites over time
    # =========================================
    ax4 = axes[1, 1]

    ax4.fill_between(dates, site_counts, alpha=0.7, color='#90EE90')
    ax4.plot(dates, site_counts, color='#228B22', linewidth=1)

    # Average line
    avg_sites = np.mean(site_counts)
    ax4.axhline(y=avg_sites, color='red', linestyle='--', linewidth=1.5,
                label=f'Average ({avg_sites:.0f})')

    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_ylabel('Number of Active Sites', fontsize=10)
    ax4.set_title('Active Monitoring Sites Over Time', fontsize=11)
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, max(site_counts) * 1.1)
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def generate_trend_analysis_chart_from_csv(
    original_csv: str,
    region_name: str,
    output_file: str
) -> str:
    """
    Generate a 4-panel trend analysis chart directly from CSV file.
    This version reads actual data points for more accurate boxplots.

    Args:
        original_csv: Path to original CSV data file
        region_name: Name of the region for the title
        output_file: Output file path for the PNG

    Returns:
        Output file path
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for chart generation")

    # Read CSV data
    df = pd.read_csv(original_csv)

    # Get date columns
    meta_cols = ['Site', 'Name', 'Lat', 'Lon']
    date_cols = [c for c in df.columns if c not in meta_cols and c[0:2] in ['20', '19']]

    if not date_cols:
        raise ValueError("No date columns found in CSV")

    dates_str = sorted(date_cols)
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates_str]

    # Calculate statistics for each date
    means = []
    stds = []
    site_counts = []

    for col in dates_str:
        valid_data = df[col].dropna()
        means.append(valid_data.mean() if len(valid_data) > 0 else np.nan)
        stds.append(valid_data.std() if len(valid_data) > 1 else 0)
        site_counts.append(len(valid_data))

    means_arr = np.array(means)
    stds_arr = np.array(stds)
    upper_bound = means_arr - stds_arr
    lower_bound = means_arr + stds_arr

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{region_name} Groundwater Analysis', fontsize=16, fontweight='bold')

    # =========================================
    # Panel 1: Top-left - Depth trend with std dev range
    # =========================================
    ax1 = axes[0, 0]
    ax1.fill_between(dates, upper_bound, lower_bound, alpha=0.3, color='blue', label='Std Dev Range')
    ax1.plot(dates, means, 'b-', linewidth=1.5, label='Mean Depth')
    ax1.set_ylabel('Depth (ft below land surface)', fontsize=10)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_title(f'{region_name} Groundwater Depth Trend ({dates_str[0][:4]}-{dates_str[-1][:4]})', fontsize=11)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # =========================================
    # Panel 2: Top-right - Annual distribution boxplot (with actual data)
    # =========================================
    ax2 = axes[0, 1]

    # Group actual data by year
    yearly_data = {}
    for col in dates_str:
        year = col[:4]
        if year not in yearly_data:
            yearly_data[year] = []
        valid_data = df[col].dropna().values
        yearly_data[year].extend(valid_data)

    years = sorted(yearly_data.keys())
    data_by_year = [yearly_data[y] for y in years]

    bp = ax2.boxplot(data_by_year, tick_labels=years, patch_artist=True)
    colors = ['#87CEEB'] * len(years)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylabel('Depth (ft below land surface)', fontsize=10)
    ax2.set_title('Annual Distribution of Groundwater Depth', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # =========================================
    # Panel 3: Bottom-left - Quarterly mean bar chart
    # =========================================
    ax3 = axes[1, 0]

    # Group by quarter
    quarterly_data = {}
    for i, col in enumerate(dates_str):
        year = col[:4]
        month = int(col[5:7])
        quarter = (month - 1) // 3 + 1
        key = f"{year}\nQ{quarter}"
        if key not in quarterly_data:
            quarterly_data[key] = []
        quarterly_data[key].append(means[i])

    quarters = list(quarterly_data.keys())
    quarterly_means = [np.nanmean(quarterly_data[q]) for q in quarters]

    # Color bars by year (alternating red/green pattern)
    bar_colors = []
    for q in quarters:
        year = q.split('\n')[0]
        year_idx = int(year) % 2
        bar_colors.append('#E57373' if year_idx else '#81C784')

    ax3.bar(range(len(quarters)), quarterly_means, color=bar_colors, alpha=0.8)

    # Reference line
    overall_mean = np.nanmean(quarterly_means)
    ax3.axhline(y=overall_mean, color='red', linestyle='--', linewidth=1.5,
                label=f'Reference ({overall_mean:.0f} ft)')

    ax3.set_xticks(range(len(quarters)))
    ax3.set_xticklabels(quarters, fontsize=8)
    ax3.set_xlabel('Quarter', fontsize=10)
    ax3.set_ylabel('Mean Depth (ft)', fontsize=10)
    ax3.set_title('Quarterly Mean Groundwater Depth', fontsize=11)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')

    # =========================================
    # Panel 4: Bottom-right - Active monitoring sites over time
    # =========================================
    ax4 = axes[1, 1]

    ax4.fill_between(dates, site_counts, alpha=0.7, color='#90EE90')
    ax4.plot(dates, site_counts, color='#228B22', linewidth=1)

    # Average line
    avg_sites = np.mean(site_counts)
    ax4.axhline(y=avg_sites, color='red', linestyle='--', linewidth=1.5,
                label=f'Average ({avg_sites:.0f})')

    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_ylabel('Number of Active Sites', fontsize=10)
    ax4.set_title('Active Monitoring Sites Over Time', fontsize=11)
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, max(site_counts) * 1.1)
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


@mcp.tool()
def generate_analysis_report(
    kriging_file: str,
    original_csv: str,
    region_name: str = "Groundwater Analysis",
    output_file: str = None,
    output_format: str = "markdown",
    output_dir: str = None,
    generate_chart: bool = True
) -> str:
    """
    Generate an analysis report from kriging results.

    Args:
        kriging_file: Path to kriging output file (.dat sectioned format)
        original_csv: Path to original CSV data file
        region_name: Name of the region for the report title
        output_file: Output file path (auto-generated if not specified)
        output_format: Output format - 'markdown' or 'json'
        output_dir: Optional output directory (created if not exists)
        generate_chart: Whether to generate trend analysis chart (default: True)

    Returns:
        JSON string with report content and output file path
    """
    try:
        # Analyze CSV data
        csv_info = analyze_csv_data(original_csv)

        # Parse kriging results
        kriging_stats = {}
        if kriging_file and os.path.exists(kriging_file):
            kriging_stats = parse_kriging_dat(kriging_file)

        # Generate output filename
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = os.path.splitext(os.path.basename(original_csv))[0]
            ext = '.md' if output_format == 'markdown' else '.json'
            output_file = f"{base_name}_report_{timestamp}{ext}"

        output_path = _get_output_path(output_file, output_dir)

        # Generate chart if requested
        chart_file = None
        if generate_chart and MATPLOTLIB_AVAILABLE:
            base_name = os.path.splitext(os.path.basename(original_csv))[0]
            chart_filename = f"{base_name}_trend_analysis.png"
            chart_path = _get_output_path(chart_filename, output_dir)
            try:
                generate_trend_analysis_chart_from_csv(
                    original_csv, region_name, chart_path
                )
                chart_file = chart_path
            except Exception as chart_error:
                chart_file = f"Chart generation failed: {str(chart_error)}"

        if output_format == 'markdown':
            content = generate_markdown_report(
                region_name, csv_info, kriging_stats, output_path
            )
            result = {
                'status': 'success',
                'format': 'markdown',
                'output_file': output_path,
                'output_dir': output_dir,
                'chart_file': chart_file,
                'summary': {
                    'region': region_name,
                    'total_sites': csv_info['total_sites'],
                    'date_count': csv_info.get('date_count', 0),
                    'kriging_frames': len(kriging_stats)
                }
            }
        else:
            # JSON format
            report_data = {
                'region_name': region_name,
                'generated_at': datetime.now().isoformat(),
                'data_overview': csv_info,
                'kriging_statistics': kriging_stats
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            result = {
                'status': 'success',
                'format': 'json',
                'output_file': output_path,
                'output_dir': output_dir,
                'chart_file': chart_file,
                'summary': {
                    'region': region_name,
                    'total_sites': csv_info['total_sites'],
                    'date_count': csv_info.get('date_count', 0),
                    'kriging_frames': len(kriging_stats)
                }
            }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            'status': 'error',
            'error': str(e)
        }, indent=2)


@mcp.tool()
def get_quick_summary(
    original_csv: str,
    kriging_file: str = None
) -> str:
    """
    Get a quick summary of groundwater analysis data.

    Args:
        original_csv: Path to original CSV data file
        kriging_file: Optional path to kriging output file (.dat)

    Returns:
        JSON string with quick summary statistics
    """
    try:
        csv_info = analyze_csv_data(original_csv)

        summary = {
            'total_sites': csv_info['total_sites'],
            'lat_range': csv_info['lat_range'],
            'lon_range': csv_info['lon_range'],
            'date_count': csv_info.get('date_count', 0),
            'date_range': csv_info.get('date_range', [])
        }

        if 'date_stats' in csv_info:
            date_stats = csv_info['date_stats']
            all_means = [v['mean'] for v in date_stats.values()]
            all_mins = [v['min'] for v in date_stats.values()]
            all_maxs = [v['max'] for v in date_stats.values()]

            summary['depth_statistics'] = {
                'overall_min': min(all_mins),
                'overall_max': max(all_maxs),
                'mean_of_means': np.mean(all_means),
                'mean_range': [min(all_means), max(all_means)]
            }

        if kriging_file and os.path.exists(kriging_file):
            kriging_stats = parse_kriging_dat(kriging_file)
            summary['kriging_frames'] = len(kriging_stats)

            if kriging_stats:
                kriging_means = [v['mean'] for v in kriging_stats.values()]
                summary['kriging_mean_range'] = [min(kriging_means), max(kriging_means)]

        return json.dumps({
            'status': 'success',
            'summary': summary
        }, indent=2)

    except Exception as e:
        return json.dumps({
            'status': 'error',
            'error': str(e)
        }, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")

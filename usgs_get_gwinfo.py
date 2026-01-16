"""
USGS NWIS API - Groundwater Data Fetcher
- Fetch groundwater data for a specific date or date range
- Save to CSV in pivot format (sites × dates)
"""

import requests
import json
import argparse
from datetime import datetime, timedelta


def get_sites_by_bbox(bbox, aquifer_type='U', timeout=120):
    """
    Fetch groundwater monitoring sites within a bounding box.

    Parameters:
    -----------
    bbox : str
        Bounding box in format 'west,south,east,north'
    aquifer_type : str
        Aquifer type code: 'U' (Unconfined), 'C' (Confined), 'X' (Mixed), 'M' (Multiple)
    timeout : int
        Request timeout in seconds

    Returns:
    --------
    tuple: (list of site codes, dict of site info)
    """
    site_url = 'https://waterservices.usgs.gov/nwis/site/'
    site_params = {
        'format': 'rdb',
        'bBox': bbox,
        'siteType': 'GW',
        'siteOutput': 'expanded',
        'siteStatus': 'active'
    }

    response = requests.get(site_url, params=site_params, timeout=timeout)

    if response.status_code != 200:
        print(f"Error fetching sites: {response.status_code}")
        return [], {}

    target_sites = []
    site_info = {}

    lines = response.text.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('agency_cd'):
            headers = line.split('\t')
            site_idx = headers.index('site_no')
            type_idx = headers.index('aqfr_type_cd')
            name_idx = headers.index('station_nm')
            lat_idx = headers.index('dec_lat_va')
            lon_idx = headers.index('dec_long_va')

            for data_line in lines[i+2:]:
                if data_line.strip() and not data_line.startswith('#'):
                    values = data_line.split('\t')
                    if len(values) > type_idx:
                        aq_type = values[type_idx]
                        if aq_type == aquifer_type:
                            site = values[site_idx]
                            target_sites.append(site)
                            site_info[site] = {
                                'name': values[name_idx] if name_idx < len(values) else '',
                                'lat': values[lat_idx] if lat_idx < len(values) else '',
                                'lon': values[lon_idx] if lon_idx < len(values) else ''
                            }
            break

    return target_sites, site_info


def fetch_groundwater_data(sites, start_date, end_date=None, timeout=120):
    """
    Fetch groundwater data for given sites.

    Parameters:
    -----------
    sites : list
        List of site codes
    start_date : str
        Start date (YYYY-MM-DD) or single date
    end_date : str, optional
        End date (YYYY-MM-DD). If None, uses start_date (single day query)
    timeout : int
        Request timeout in seconds

    Returns:
    --------
    tuple: (dict of site data, set of dates)
    """
    if end_date is None:
        end_date = start_date

    data_url = 'https://waterservices.usgs.gov/nwis/dv/'
    data_params = {
        'format': 'json',
        'sites': ','.join(sites),
        'parameterCd': '72019',  # Depth to water level
        'startDT': start_date,
        'endDT': end_date
    }

    resp = requests.get(data_url, params=data_params, timeout=timeout)

    if resp.status_code != 200:
        print(f"Error fetching data: {resp.status_code}")
        return {}, set()

    data = resp.json()
    ts_list = data.get('value', {}).get('timeSeries', [])

    all_data = {}  # {site: {date: value}}
    all_dates = set()

    for ts in ts_list:
        vlist = ts.get('values', [{}])[0].get('value', [])
        valid = [v for v in vlist if float(v.get('value', -999999)) > -9999]

        if valid:
            si = ts['sourceInfo']
            code = si['siteCode'][0]['value']

            if code not in all_data:
                all_data[code] = {}

            for v in valid:
                date = v['dateTime'][:10]
                all_dates.add(date)
                all_data[code][date] = float(v['value'])

    return all_data, all_dates


def save_to_csv(all_data, all_dates, site_info, output_file, include_change=True):
    """
    Save groundwater data to CSV file.

    Parameters:
    -----------
    all_data : dict
        Data dictionary {site: {date: value}}
    all_dates : set
        Set of date strings
    site_info : dict
        Site metadata {site: {name, lat, lon}}
    output_file : str
        Output CSV filename
    include_change : bool
        Include change column (last - first value)
    """
    sorted_dates = sorted(all_dates)

    with open(output_file, 'w', encoding='utf-8-sig') as f:
        # Header
        header = ['Site', 'Name', 'Lat', 'Lon'] + sorted_dates
        if include_change and len(sorted_dates) > 1:
            header.append('Change')
        f.write(','.join(header) + '\n')

        # Data rows
        for site in sorted(all_data.keys()):
            info = site_info.get(site, {})
            name = info.get('name', '').replace('"', "'")
            row = [
                site,
                f'"{name}"',
                str(info.get('lat', '')),
                str(info.get('lon', ''))
            ]

            # Date values
            values_list = []
            for date in sorted_dates:
                val = all_data[site].get(date, '')
                if val != '':
                    row.append(f'{val:.2f}')
                    values_list.append(val)
                else:
                    row.append('')

            # Calculate change (last - first)
            if include_change and len(sorted_dates) > 1:
                if len(values_list) >= 2:
                    change = values_list[-1] - values_list[0]
                    row.append(f'{change:.2f}')
                else:
                    row.append('')

            f.write(','.join(row) + '\n')

    return output_file


def fetch_single_date(bbox, date, aquifer_type='U', output_file=None):
    """
    Fetch groundwater data for a specific date.

    Parameters:
    -----------
    bbox : str
        Bounding box in format 'west,south,east,north'
    date : str
        Target date (YYYY-MM-DD)
    aquifer_type : str
        Aquifer type code
    output_file : str, optional
        Output CSV filename

    Returns:
    --------
    dict: Result summary
    """
    print("=" * 80)
    print("USGS NWIS API - Single Date Query")
    print("=" * 80)
    print(f"  bBox: {bbox}")
    print(f"  Date: {date}")
    print(f"  Aquifer Type: {aquifer_type}")
    print()

    # Step 1: Get sites
    print("[Step 1] Fetching site list...")
    sites, site_info = get_sites_by_bbox(bbox, aquifer_type)
    print(f"  Found {len(sites)} sites with aquifer type '{aquifer_type}'")

    if not sites:
        print("No sites found.")
        return None

    # Step 2: Fetch data
    print("\n[Step 2] Fetching groundwater data...")
    all_data, all_dates = fetch_groundwater_data(sites, date)
    print(f"  Retrieved data for {len(all_data)} sites")

    if not all_data:
        print("No data available for the specified date.")
        return None

    # Step 3: Save to CSV
    print("\n[Step 3] Saving to CSV...")
    if not output_file:
        output_file = f"groundwater_{date.replace('-', '')}.csv"

    save_to_csv(all_data, all_dates, site_info, output_file, include_change=False)
    print(f"  Saved to: {output_file}")

    result = {
        'type': 'single_date',
        'bbox': bbox,
        'date': date,
        'aquifer_type': aquifer_type,
        'sites_found': len(sites),
        'sites_with_data': len(all_data),
        'output_file': output_file
    }

    print(f"\n{'=' * 80}")
    print("COMPLETE!")
    print("=" * 80)

    return result


def fetch_date_range(bbox, start_date, end_date, aquifer_type='U', output_file=None):
    """
    Fetch groundwater data for a date range.

    Parameters:
    -----------
    bbox : str
        Bounding box in format 'west,south,east,north'
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    aquifer_type : str
        Aquifer type code
    output_file : str, optional
        Output CSV filename

    Returns:
    --------
    dict: Result summary
    """
    print("=" * 80)
    print("USGS NWIS API - Date Range Query")
    print("=" * 80)
    print(f"  bBox: {bbox}")
    print(f"  Date Range: {start_date} ~ {end_date}")
    print(f"  Aquifer Type: {aquifer_type}")
    print()

    # Step 1: Get sites
    print("[Step 1] Fetching site list...")
    sites, site_info = get_sites_by_bbox(bbox, aquifer_type)
    print(f"  Found {len(sites)} sites with aquifer type '{aquifer_type}'")

    if not sites:
        print("No sites found.")
        return None

    # Step 2: Fetch data
    print("\n[Step 2] Fetching groundwater data...")
    all_data, all_dates = fetch_groundwater_data(sites, start_date, end_date)
    print(f"  Retrieved data for {len(all_data)} sites")
    print(f"  Date columns: {len(all_dates)} dates ({min(all_dates)} ~ {max(all_dates)})")

    if not all_data:
        print("No data available for the specified date range.")
        return None

    # Step 3: Save to CSV
    print("\n[Step 3] Saving to CSV...")
    if not output_file:
        start_str = start_date.replace('-', '')
        end_str = end_date.replace('-', '')
        output_file = f"groundwater_{start_str}_{end_str}.csv"

    save_to_csv(all_data, all_dates, site_info, output_file, include_change=True)
    print(f"  Saved to: {output_file}")

    # Calculate statistics
    all_values = []
    for site_data in all_data.values():
        all_values.extend(site_data.values())

    result = {
        'type': 'date_range',
        'bbox': bbox,
        'start_date': start_date,
        'end_date': end_date,
        'aquifer_type': aquifer_type,
        'sites_found': len(sites),
        'sites_with_data': len(all_data),
        'date_count': len(all_dates),
        'output_file': output_file,
        'stats': {
            'min': min(all_values),
            'max': max(all_values),
            'mean': sum(all_values) / len(all_values)
        }
    }

    print(f"\n[Statistics]")
    print(f"  Min Depth: {result['stats']['min']:.2f} ft")
    print(f"  Max Depth: {result['stats']['max']:.2f} ft")
    print(f"  Mean Depth: {result['stats']['mean']:.2f} ft")

    print(f"\n{'=' * 80}")
    print("COMPLETE!")
    print("=" * 80)

    return result


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='USGS NWIS Groundwater Data Fetcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single date query
  python usgs_get_gwinfo.py --bbox "-117.5,32.5,-116.5,33.5" --date 2026-01-13

  # Date range query
  python usgs_get_gwinfo.py --bbox "-117.5,32.5,-116.5,33.5" --start 2026-01-01 --end 2026-01-13

  # With custom output file and aquifer type
  python usgs_get_gwinfo.py --bbox "-121,32,-114,36" --start 2026-01-01 --end 2026-01-13 -o socal_gw.csv -a U

Aquifer Types:
  U - Unconfined (default)
  C - Confined
  X - Mixed
  M - Multiple
        """
    )

    parser.add_argument('--bbox', '-b', required=True,
                        help="Bounding box: 'west,south,east,north' (e.g., '-117.5,32.5,-116.5,33.5')")
    parser.add_argument('--date', '-d',
                        help="Single date (YYYY-MM-DD)")
    parser.add_argument('--start', '-s',
                        help="Start date for range query (YYYY-MM-DD)")
    parser.add_argument('--end', '-e',
                        help="End date for range query (YYYY-MM-DD)")
    parser.add_argument('--aquifer', '-a', default='U',
                        choices=['U', 'C', 'X', 'M'],
                        help="Aquifer type (default: U)")
    parser.add_argument('--output', '-o',
                        help="Output CSV filename (auto-generated if not specified)")

    args = parser.parse_args()

    # Validate arguments
    if args.date and (args.start or args.end):
        parser.error("Cannot use --date with --start/--end. Choose one mode.")

    if not args.date and not (args.start and args.end):
        parser.error("Must specify either --date OR both --start and --end")

    if (args.start and not args.end) or (args.end and not args.start):
        parser.error("Both --start and --end are required for date range query")

    # Execute query
    if args.date:
        result = fetch_single_date(
            bbox=args.bbox,
            date=args.date,
            aquifer_type=args.aquifer,
            output_file=args.output
        )
    else:
        result = fetch_date_range(
            bbox=args.bbox,
            start_date=args.start,
            end_date=args.end,
            aquifer_type=args.aquifer,
            output_file=args.output
        )

    if result:
        print(f"\nResult: {json.dumps(result, indent=2)}")

    return result


if __name__ == "__main__":
    main()

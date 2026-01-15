"""
USGS NWIS API Connection Test
- Fetch and parse groundwater data
"""

import requests
import json
from datetime import datetime


def test_api_connection():
    """Test API connection and parse data"""

    base_url = "https://waterservices.usgs.gov/nwis/dv/"

    params = {
        'format': 'json',
        'sites': '323527117050001',
        'parameterCd': '72019',
        'period': 'P30D',
        'siteStatus': 'all'
    }

    print("=" * 70)
    print("USGS NWIS API - Groundwater Data Parser")
    print("=" * 70)
    print(f"\nURL: {base_url}")
    print(f"Parameters: {params}")
    print("\nSending request...")

    try:
        response = requests.get(base_url, params=params, timeout=30)

        print(f"\n[Response]")
        print(f"  Status Code: {response.status_code}")

        if response.status_code != 200:
            print(f"  Error: {response.text[:500]}")
            return False

        data = response.json()

        # Save raw JSON with request info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"usgs_response_{timestamp}.json"

        save_data = {
            "request": {
                "url": base_url,
                "parameters": params,
                "timestamp": datetime.now().isoformat(),
                "full_url": response.url
            },
            "response": {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "elapsed_seconds": response.elapsed.total_seconds()
            },
            "data": data
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved to: {filename}")

        time_series = data.get('value', {}).get('timeSeries', [])

        print(f"  Time Series Count: {len(time_series)}")

        # Process each time series
        for idx, ts in enumerate(time_series):
            # --- Site Information ---
            site_info = ts.get('sourceInfo', {})
            site_code = site_info.get('siteCode', [{}])[0].get('value', 'N/A')
            site_name = site_info.get('siteName', 'N/A')

            geo = site_info.get('geoLocation', {}).get('geogLocation', {})
            latitude = geo.get('latitude', 'N/A')
            longitude = geo.get('longitude', 'N/A')

            # --- Variable Information ---
            variable = ts.get('variable', {})
            var_code = variable.get('variableCode', [{}])[0].get('value', 'N/A')
            var_name = variable.get('variableName', 'N/A')
            var_desc = variable.get('variableDescription', 'N/A')
            unit = variable.get('unit', {}).get('unitCode', 'N/A')

            # --- Values ---
            values = ts.get('values', [{}])[0].get('value', [])
            valid_values = [v for v in values if float(v.get('value', -999999)) > -9999]

            if not valid_values:
                continue

            print(f"\n{'=' * 70}")
            print(f"[Time Series {idx + 1}]")
            print(f"{'=' * 70}")

            print(f"\n[Site Information]")
            print(f"  Site Code  : {site_code}")
            print(f"  Site Name  : {site_name}")
            print(f"  Latitude   : {latitude}")
            print(f"  Longitude  : {longitude}")

            print(f"\n[Variable Information]")
            print(f"  Code       : {var_code}")
            print(f"  Name       : {var_name}")
            print(f"  Description: {var_desc}")
            print(f"  Unit       : {unit}")

            # --- Statistics ---
            float_values = [float(v['value']) for v in valid_values]
            min_val = min(float_values)
            max_val = max(float_values)
            avg_val = sum(float_values) / len(float_values)

            print(f"\n[Statistics]")
            print(f"  Count      : {len(valid_values)}")
            print(f"  Min        : {min_val:.2f} {unit}")
            print(f"  Max        : {max_val:.2f} {unit}")
            print(f"  Average    : {avg_val:.2f} {unit}")

            # --- All Measurements ---
            print(f"\n[Measurements]")
            print(f"  {'Date':<12} | {'Value':>10} | {'Qualifier'}")
            print(f"  {'-' * 12}-+-{'-' * 10}-+-{'-' * 15}")

            # Sort by date
            sorted_values = sorted(valid_values, key=lambda x: x['dateTime'])
            for v in sorted_values:
                date = v['dateTime'][:10]
                val = float(v['value'])
                qualifiers = v.get('qualifiers', [])
                qual_str = ', '.join(qualifiers) if qualifiers else '-'
                print(f"  {date:<12} | {val:>10.2f} | {qual_str}")

        print(f"\n{'=' * 70}")
        print("PARSING COMPLETE!")
        print("=" * 70)
        return True

    except requests.exceptions.Timeout:
        print("\nTimeout error (exceeded 30 seconds)")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"\nConnection error: {e}")
        return False
    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        return False


def fetch_bbox_data_to_csv(bbox, start_date, end_date, aquifer_type='U', output_file=None):
    """
    Fetch groundwater data by bounding box and save to CSV (pivot format)

    Parameters:
    -----------
    bbox : str
        Bounding box in format 'west,south,east,north'
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    aquifer_type : str
        Aquifer type code ('U' for Unconfined, 'C' for Confined)
    output_file : str
        Output CSV filename (auto-generated if None)
    """

    print("=" * 80)
    print("USGS NWIS API - BBox Query to CSV")
    print("=" * 80)
    print(f"bBox: {bbox}")
    print(f"Date Range: {start_date} ~ {end_date}")
    print(f"Aquifer Type: {aquifer_type}")
    print()

    # Step 1: Get sites with specified aquifer type
    print("[Step 1] Fetching site list...")
    site_url = 'https://waterservices.usgs.gov/nwis/site/'
    site_params = {
        'format': 'rdb',
        'bBox': bbox,
        'siteType': 'GW',
        'siteOutput': 'expanded',
        'siteStatus': 'active'
    }

    response = requests.get(site_url, params=site_params, timeout=120)

    if response.status_code != 200:
        print(f"Error fetching sites: {response.status_code}")
        return None

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

    print(f"  Found {len(target_sites)} sites with aquifer type '{aquifer_type}'")

    if not target_sites:
        print("No sites found.")
        return None

    # Step 2: Fetch data for these sites
    print("\n[Step 2] Fetching groundwater data...")
    data_url = 'https://waterservices.usgs.gov/nwis/dv/'
    data_params = {
        'format': 'json',
        'sites': ','.join(target_sites),
        'parameterCd': '72019',
        'startDT': start_date,
        'endDT': end_date
    }

    resp = requests.get(data_url, params=data_params, timeout=120)

    if resp.status_code != 200:
        print(f"Error fetching data: {resp.status_code}")
        return None

    data = resp.json()
    ts_list = data.get('value', {}).get('timeSeries', [])

    # Collect all data points
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

    print(f"  Retrieved data for {len(all_data)} sites")
    print(f"  Date columns: {sorted(all_dates)}")

    # Step 3: Create pivot table and save to CSV
    print("\n[Step 3] Saving to CSV...")

    sorted_dates = sorted(all_dates)

    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"groundwater_bbox_{timestamp}.csv"

    with open(output_file, 'w', encoding='utf-8-sig') as f:
        # Header
        header = ['Site', 'Name', 'Lat', 'Lon'] + sorted_dates + ['Change']
        f.write(','.join(header) + '\n')

        # Data rows
        for site in sorted(all_data.keys()):
            info = site_info.get(site, {})
            row = [
                site,
                f'"{info.get("name", "")}"',
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
            if len(values_list) >= 2:
                change = values_list[-1] - values_list[0]
                row.append(f'{change:.2f}')
            else:
                row.append('')

            f.write(','.join(row) + '\n')

    print(f"  Saved to: {output_file}")
    print(f"\n{'=' * 80}")
    print("COMPLETE!")
    print("=" * 80)

    return output_file


if __name__ == "__main__":
    # Test 1: Single site query
    # test_api_connection()

    # Test 2: BBox query to CSV
    fetch_bbox_data_to_csv(
        bbox='-117.5,32.5,-116.5,33.5',  # San Diego area
        start_date='2026-01-10',
        end_date='2026-01-13',
        aquifer_type='U'  # Unconfined
    )

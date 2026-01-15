"""
USGS Groundwater Info MCP Server
- Fetch groundwater data by bounding box
- Filter by aquifer type
- Export to CSV
"""

import json
import requests
from datetime import datetime
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("usgs-gwinfo")


@mcp.tool()
def get_groundwater_sites(
    bbox: str,
    aquifer_type: str = "U"
) -> str:
    """
    Get groundwater monitoring sites within a bounding box.

    Args:
        bbox: Bounding box as 'west,south,east,north' (e.g., '-117.5,32.5,-116.5,33.5')
        aquifer_type: Aquifer type - 'U' (Unconfined), 'C' (Confined), 'X' (Mixed), 'M' (Multiple)

    Returns:
        JSON string with site information
    """
    site_url = 'https://waterservices.usgs.gov/nwis/site/'
    params = {
        'format': 'rdb',
        'bBox': bbox,
        'siteType': 'GW',
        'siteOutput': 'expanded',
        'siteStatus': 'active'
    }

    response = requests.get(site_url, params=params, timeout=120)

    if response.status_code != 200:
        return json.dumps({"error": f"API error: {response.status_code}"})

    sites = []
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
                            sites.append({
                                'site_no': values[site_idx],
                                'name': values[name_idx] if name_idx < len(values) else '',
                                'lat': float(values[lat_idx]) if lat_idx < len(values) else None,
                                'lon': float(values[lon_idx]) if lon_idx < len(values) else None,
                                'aquifer_type': aq_type
                            })
            break

    return json.dumps({
        "bbox": bbox,
        "aquifer_type": aquifer_type,
        "count": len(sites),
        "sites": sites
    }, indent=2)


@mcp.tool()
def get_groundwater_data(
    bbox: str,
    start_date: str,
    end_date: str,
    aquifer_type: str = "U",
    output_csv: str = None
) -> str:
    """
    Fetch groundwater level data for a bounding box and date range.

    Args:
        bbox: Bounding box as 'west,south,east,north' (e.g., '-117.5,32.5,-116.5,33.5')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        aquifer_type: Aquifer type - 'U' (Unconfined), 'C' (Confined)
        output_csv: Optional CSV filename to save results

    Returns:
        JSON string with groundwater data
    """
    # Step 1: Get sites
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
        return json.dumps({"error": f"Site API error: {response.status_code}"})

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

    if not target_sites:
        return json.dumps({"error": "No sites found", "bbox": bbox, "aquifer_type": aquifer_type})

    # Step 2: Fetch data
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
        return json.dumps({"error": f"Data API error: {resp.status_code}"})

    data = resp.json()
    ts_list = data.get('value', {}).get('timeSeries', [])

    all_data = {}
    all_dates = set()

    for ts in ts_list:
        vlist = ts.get('values', [{}])[0].get('value', [])
        valid = [v for v in vlist if float(v.get('value', -999999)) > -9999]

        if valid:
            si = ts['sourceInfo']
            code = si['siteCode'][0]['value']
            geo = si.get('geoLocation', {}).get('geogLocation', {})

            if code not in all_data:
                all_data[code] = {
                    'name': site_info.get(code, {}).get('name', ''),
                    'lat': geo.get('latitude', ''),
                    'lon': geo.get('longitude', ''),
                    'values': {}
                }

            for v in valid:
                date = v['dateTime'][:10]
                all_dates.add(date)
                all_data[code]['values'][date] = float(v['value'])

    # Save to CSV if requested
    if output_csv and all_data:
        sorted_dates = sorted(all_dates)
        with open(output_csv, 'w', encoding='utf-8-sig') as f:
            header = ['Site', 'Name', 'Lat', 'Lon'] + sorted_dates + ['Change']
            f.write(','.join(header) + '\n')

            for site in sorted(all_data.keys()):
                info = all_data[site]
                row = [site, f'"{info["name"]}"', str(info['lat']), str(info['lon'])]

                values_list = []
                for date in sorted_dates:
                    val = info['values'].get(date, '')
                    if val != '':
                        row.append(f'{val:.2f}')
                        values_list.append(val)
                    else:
                        row.append('')

                if len(values_list) >= 2:
                    change = values_list[-1] - values_list[0]
                    row.append(f'{change:.2f}')
                else:
                    row.append('')

                f.write(','.join(row) + '\n')

    return json.dumps({
        "bbox": bbox,
        "start_date": start_date,
        "end_date": end_date,
        "aquifer_type": aquifer_type,
        "sites_found": len(target_sites),
        "sites_with_data": len(all_data),
        "dates": sorted(list(all_dates)),
        "output_csv": output_csv,
        "data": all_data
    }, indent=2)


@mcp.tool()
def get_site_history(
    site_no: str,
    days: int = 365
) -> str:
    """
    Get historical groundwater data for a specific site.

    Args:
        site_no: USGS site number (e.g., '323527117050001')
        days: Number of days of history (default: 365)

    Returns:
        JSON string with historical data
    """
    url = 'https://waterservices.usgs.gov/nwis/dv/'
    params = {
        'format': 'json',
        'sites': site_no,
        'parameterCd': '72019',
        'period': f'P{days}D',
        'siteStatus': 'all'
    }

    response = requests.get(url, params=params, timeout=60)

    if response.status_code != 200:
        return json.dumps({"error": f"API error: {response.status_code}"})

    data = response.json()
    ts_list = data.get('value', {}).get('timeSeries', [])

    records = []
    site_name = ''

    for ts in ts_list:
        si = ts.get('sourceInfo', {})
        site_name = si.get('siteName', '')
        geo = si.get('geoLocation', {}).get('geogLocation', {})

        vlist = ts.get('values', [{}])[0].get('value', [])
        for v in vlist:
            if float(v.get('value', -999999)) > -9999:
                records.append({
                    'date': v['dateTime'][:10],
                    'depth_ft': float(v['value'])
                })

    return json.dumps({
        "site_no": site_no,
        "site_name": site_name,
        "days": days,
        "record_count": len(records),
        "records": records
    }, indent=2)


if __name__ == "__main__":
    mcp.run()

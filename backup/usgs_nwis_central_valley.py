"""
USGS NWIS API를 사용하여 캘리포니아 센트럴 밸리 지하수 데이터 수집
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json

class USGSNWISClient:
    """USGS National Water Information System API 클라이언트"""

    def __init__(self):
        self.base_url = "https://waterservices.usgs.gov/nwis"

    def get_groundwater_sites(self, state_code="CA", county_codes=None, bbox=None):
        """
        지하수 측정소 정보 조회

        Parameters:
        -----------
        state_code : str
            주 코드 (CA = California)
        county_codes : list
            카운티 코드 리스트 (예: ['019', '077', '099'])
        bbox : tuple
            경계 상자 (west_lon, south_lat, east_lon, north_lat)
            센트럴 밸리: (-122.5, 35.0, -118.5, 40.5)

        Returns:
        --------
        pandas.DataFrame : 측정소 정보
        """
        url = f"{self.base_url}/site/"

        params = {
            'format': 'rdb',  # Tab-delimited format works better for site service
            'stateCd': state_code,
            'siteType': 'GW',  # Groundwater
        }

        # 센트럴 밸리 경계 상자 설정
        if bbox:
            params['bBox'] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        if county_codes:
            params['countyCd'] = ','.join([f"{state_code}:{code}" for code in county_codes])

        print(f"Requesting: {url}")
        print(f"Parameters: {params}")

        response = requests.get(url, params=params)
        response.raise_for_status()

        # RDB 형식 파싱
        lines = response.text.strip().split('\n')

        # 헤더 찾기 (# 으로 시작하지 않는 첫 번째 줄)
        header_idx = 0
        for i, line in enumerate(lines):
            if not line.startswith('#'):
                header_idx = i
                break

        if header_idx >= len(lines) - 2:
            print("No sites found")
            return pd.DataFrame()

        # 헤더와 데이터 분리
        headers = lines[header_idx].split('\t')
        # 다음 줄은 데이터 타입 정의이므로 스킵
        data_lines = lines[header_idx + 2:]

        # 데이터 파싱
        sites = []
        for line in data_lines:
            if line.strip():
                values = line.split('\t')
                site_dict = dict(zip(headers, values))
                sites.append({
                    'site_no': site_dict.get('site_no', ''),
                    'site_name': site_dict.get('station_nm', 'N/A'),
                    'latitude': float(site_dict.get('dec_lat_va', 0)),
                    'longitude': float(site_dict.get('dec_long_va', 0)),
                    'county': site_dict.get('county_cd', 'N/A'),
                    'state': site_dict.get('state_cd', 'N/A'),
                    'site_type': site_dict.get('site_tp_cd', 'N/A')
                })

        return pd.DataFrame(sites)

    def get_groundwater_levels(self, site_no, start_date=None, end_date=None, period=None):
        """
        지하수 수위 데이터 조회

        Parameters:
        -----------
        site_no : str or list
            측정소 번호(들)
        start_date : str
            시작일 (YYYY-MM-DD)
        end_date : str
            종료일 (YYYY-MM-DD)
        period : str
            기간 (예: 'P7D' = 최근 7일, 'P30D' = 최근 30일, 'P1Y' = 최근 1년)

        Returns:
        --------
        pandas.DataFrame : 지하수 수위 데이터
        """
        url = f"{self.base_url}/gwlevels/"

        # 사이트 번호를 리스트로 변환
        if isinstance(site_no, str):
            site_no = [site_no]

        params = {
            'format': 'json',
            'sites': ','.join(site_no),
            'siteStatus': 'all'
        }

        # 날짜 범위 또는 기간 설정
        if period:
            params['period'] = period
        elif start_date and end_date:
            params['startDT'] = start_date
            params['endDT'] = end_date
        else:
            # 기본값: 최근 1년
            params['period'] = 'P365D'

        print(f"Requesting groundwater levels for {len(site_no)} site(s)")
        print(f"Parameters: {params}")

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # 데이터 파싱
        records = []
        for site_data in data.get('value', []):
            site_no = site_data['sourceInfo']['siteCode'][0]['value']
            site_name = site_data['sourceInfo'].get('siteName', 'N/A')

            for value in site_data.get('values', [{}])[0].get('value', []):
                records.append({
                    'site_no': site_no,
                    'site_name': site_name,
                    'datetime': value['dateTime'],
                    'water_level_ft': float(value['value']) if value['value'] != '' else None,
                    'qualifiers': value.get('qualifiers', [])
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values(['site_no', 'datetime'])

        return df

    def get_daily_values(self, site_no, parameter_code='72019', start_date=None, end_date=None):
        """
        일별 통계 데이터 조회

        Parameters:
        -----------
        site_no : str or list
            측정소 번호(들)
        parameter_code : str
            파라미터 코드
            - '72019': Depth to water level, feet below land surface
            - '62610': Groundwater level above NAVD 1988
        start_date : str
            시작일 (YYYY-MM-DD)
        end_date : str
            종료일 (YYYY-MM-DD)

        Returns:
        --------
        pandas.DataFrame : 일별 데이터
        """
        url = f"{self.base_url}/dv/"

        if isinstance(site_no, str):
            site_no = [site_no]

        params = {
            'format': 'json',
            'sites': ','.join(site_no),
            'parameterCd': parameter_code,
            'siteStatus': 'all'
        }

        if start_date and end_date:
            params['startDT'] = start_date
            params['endDT'] = end_date
        else:
            # 기본값: 최근 1년
            end = datetime.now()
            start = end - timedelta(days=365)
            params['startDT'] = start.strftime('%Y-%m-%d')
            params['endDT'] = end.strftime('%Y-%m-%d')

        print(f"Requesting daily values for {len(site_no)} site(s)")
        print(f"Parameters: {params}")

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        records = []
        for ts in data['value'].get('timeSeries', []):
            site_no = ts['sourceInfo']['siteCode'][0]['value']
            site_name = ts['sourceInfo'].get('siteName', 'N/A')
            variable_name = ts['variable']['variableName']
            unit = ts['variable']['unit']['unitCode']

            for value in ts.get('values', [{}])[0].get('value', []):
                records.append({
                    'site_no': site_no,
                    'site_name': site_name,
                    'date': value['dateTime'],
                    'value': float(value['value']) if value['value'] != '' else None,
                    'variable': variable_name,
                    'unit': unit,
                    'qualifiers': value.get('qualifiers', [])
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['site_no', 'date'])

        return df


def main():
    """캘리포니아 센트럴 밸리 지하수 데이터 수집 예제"""

    client = USGSNWISClient()

    # 센트럴 밸리 주요 카운티 코드
    # https://www.usgs.gov/mission-areas/water-resources/science/usgs-water-data-nation
    central_valley_counties = [
        '019',  # Fresno
        '031',  # Kings
        '039',  # Madera
        '047',  # Merced
        '077',  # San Joaquin
        '099',  # Stanislaus
        '107',  # Tulare
    ]

    # 센트럴 밸리 경계 상자 (대략적)
    central_valley_bbox = (-122.5, 35.0, -118.5, 40.5)

    print("=" * 80)
    print("캘리포니아 센트럴 밸리 지하수 측정소 조회")
    print("=" * 80)

    # 1. 측정소 정보 조회 (카운티 코드 사용)
    sites_df = client.get_groundwater_sites(
        state_code="CA",
        county_codes=central_valley_counties[:3]  # 처음 3개 카운티만 테스트
    )

    if not sites_df.empty:
        print(f"\n총 {len(sites_df)}개 측정소 발견")
        print("\n처음 10개 측정소:")
        print(sites_df.head(10))

        # CSV로 저장
        sites_df.to_csv('central_valley_gw_sites.csv', index=False)
        print(f"\n측정소 정보 저장: central_valley_gw_sites.csv")

        # 2. 샘플 측정소의 지하수 수위 데이터 조회
        if len(sites_df) > 0:
            sample_sites = sites_df['site_no'].head(5).tolist()

            print("\n" + "=" * 80)
            print(f"샘플 측정소 {len(sample_sites)}개의 최근 1년 지하수 수위 데이터 조회")
            print("=" * 80)

            levels_df = client.get_groundwater_levels(
                site_no=sample_sites,
                period='P365D'
            )

            if not levels_df.empty:
                print(f"\n총 {len(levels_df)}개 레코드")
                print("\n샘플 데이터:")
                print(levels_df.head(10))

                # CSV로 저장
                levels_df.to_csv('central_valley_gw_levels.csv', index=False)
                print(f"\n지하수 수위 데이터 저장: central_valley_gw_levels.csv")

                # 측정소별 통계
                print("\n측정소별 통계:")
                stats = levels_df.groupby('site_no')['water_level_ft'].agg([
                    'count', 'mean', 'min', 'max', 'std'
                ]).round(2)
                print(stats)
            else:
                print("\n지하수 수위 데이터를 찾을 수 없습니다.")
    else:
        print("\n측정소를 찾을 수 없습니다.")

    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

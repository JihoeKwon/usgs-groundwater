"""
USGS NWIS API 간단한 테스트 및 사용 예제
"""

import requests
import pandas as pd
from datetime import datetime, timedelta


def test_usgs_api():
    """USGS NWIS API 기본 테스트"""

    print("=" * 80)
    print("USGS NWIS API 테스트")
    print("=" * 80)

    # 1. 특정 측정소의 지하수 수위 데이터 조회 (예제)
    # 캘리포니아의 알려진 지하수 측정소 사용
    site_no = "365543119204701"  # Fresno 지역의 지하수 측정소 예제

    print(f"\n1. 측정소 {site_no}의 최근 지하수 수위 데이터 조회")
    print("-" * 80)

    url = "https://waterservices.usgs.gov/nwis/gwlevels/"

    params = {
        'format': 'json',
        'sites': site_no,
        'period': 'P30D',  # 최근 30일
        'siteStatus': 'all'
    }

    print(f"URL: {url}")
    print(f"Parameters: {params}\n")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        print(f"API 응답 성공!")

        if data.get('value'):
            print(f"\n발견된 데이터:")
            for site_data in data['value']:
                site_info = site_data['sourceInfo']
                print(f"  측정소: {site_info['siteCode'][0]['value']}")
                print(f"  이름: {site_info.get('siteName', 'N/A')}")
                print(f"  위치: {site_info['geoLocation']['geogLocation']}")

                values = site_data.get('values', [{}])[0].get('value', [])
                print(f"  데이터 포인트 수: {len(values)}")

                if values:
                    print(f"\n  최근 5개 측정값:")
                    for v in values[-5:]:
                        print(f"    {v['dateTime']}: {v['value']} feet")
        else:
            print("데이터 없음")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP 에러: {e}")
        print(f"Response text: {response.text[:500]}")
    except Exception as e:
        print(f"에러 발생: {e}")

    # 2. 캘리포니아 주의 지하수 측정소 목록 조회
    print("\n\n2. 캘리포니아 지하수 측정소 조회 (제한된 수)")
    print("-" * 80)

    url = "https://waterservices.usgs.gov/nwis/gwlevels/"

    params = {
        'format': 'json',
        'stateCd': 'CA',
        'period': 'P7D',  # 최근 7일간 데이터가 있는 곳만
        'siteStatus': 'all'
    }

    print(f"URL: {url}")
    print(f"Parameters: {params}\n")

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        print(f"API 응답 성공!")

        sites = data.get('value', [])
        print(f"\n발견된 측정소 수: {len(sites)}")

        if sites:
            print(f"\n처음 10개 측정소:")
            for i, site_data in enumerate(sites[:10], 1):
                site_info = site_data['sourceInfo']
                site_code = site_info['siteCode'][0]['value']
                site_name = site_info.get('siteName', 'N/A')
                lat = site_info['geoLocation']['geogLocation']['latitude']
                lon = site_info['geoLocation']['geogLocation']['longitude']

                values = site_data.get('values', [{}])[0].get('value', [])

                print(f"\n  {i}. {site_code}")
                print(f"     이름: {site_name}")
                print(f"     좌표: ({lat}, {lon})")
                print(f"     데이터 포인트: {len(values)}개")

            # DataFrame으로 변환
            records = []
            for site_data in sites:
                site_info = site_data['sourceInfo']
                records.append({
                    'site_no': site_info['siteCode'][0]['value'],
                    'site_name': site_info.get('siteName', 'N/A'),
                    'latitude': site_info['geoLocation']['geogLocation']['latitude'],
                    'longitude': site_info['geoLocation']['geogLocation']['longitude'],
                    'county': site_info.get('siteProperty', [{}])[0].get('value', 'N/A')
                        if site_info.get('siteProperty') else 'N/A',
                    'data_points': len(site_data.get('values', [{}])[0].get('value', []))
                })

            df = pd.DataFrame(records)
            print(f"\n\n측정소 DataFrame 생성:")
            print(df.head(10))

            # CSV로 저장
            df.to_csv('california_gw_sites.csv', index=False)
            print(f"\n파일 저장: california_gw_sites.csv")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP 에러: {e}")
        if hasattr(response, 'text'):
            print(f"Response text: {response.text[:500]}")
    except Exception as e:
        print(f"에러 발생: {e}")

    # 3. 일별 데이터 조회 예제
    print("\n\n3. 특정 측정소의 일별 지하수 수위 데이터")
    print("-" * 80)

    url = "https://waterservices.usgs.gov/nwis/dv/"

    # 최근 1년
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    params = {
        'format': 'json',
        'sites': site_no,
        'parameterCd': '72019',  # Depth to water level, feet below land surface
        'startDT': start_date.strftime('%Y-%m-%d'),
        'endDT': end_date.strftime('%Y-%m-%d'),
        'siteStatus': 'all'
    }

    print(f"URL: {url}")
    print(f"Parameters: {params}\n")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        print(f"API 응답 성공!")

        if data['value'].get('timeSeries'):
            for ts in data['value']['timeSeries']:
                site = ts['sourceInfo']
                variable = ts['variable']
                values = ts.get('values', [{}])[0].get('value', [])

                print(f"\n  측정소: {site['siteCode'][0]['value']}")
                print(f"  변수: {variable['variableName']}")
                print(f"  단위: {variable['unit']['unitCode']}")
                print(f"  데이터 포인트: {len(values)}개")

                if values:
                    print(f"\n  최근 5개 측정값:")
                    for v in values[-5:]:
                        print(f"    {v['dateTime']}: {v['value']}")
        else:
            print("일별 데이터 없음 (이 측정소는 일별 데이터를 제공하지 않을 수 있습니다)")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP 에러: {e}")
    except Exception as e:
        print(f"에러 발생: {e}")

    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    test_usgs_api()

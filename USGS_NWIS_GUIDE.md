# USGS NWIS API 사용 가이드

## 개요

**USGS NWIS (National Water Information System)**는 미국 전역의 수자원 데이터를 제공하는 공공 API입니다.

## API 엔드포인트

### 1. **Groundwater Levels Service**
```
https://waterservices.usgs.gov/nwis/gwlevels/
```
지하수 수위 측정 데이터 제공

### 2. **Daily Values Service**
```
https://waterservices.usgs.gov/nwis/dv/
```
일별 통계 데이터 제공

### 3. **Site Service**
```
https://waterservices.usgs.gov/nwis/site/
```
측정소 정보 제공

## 주요 파라미터

| 파라미터 | 설명 | 예제 |
|---------|------|------|
| `format` | 응답 형식 | `json`, `rdb`, `waterml` |
| `stateCd` | 주 코드 | `CA` (캘리포니아) |
| `sites` | 측정소 번호 | `365543119204701` |
| `period` | 기간 | `P7D` (최근 7일), `P30D` (최근 30일), `P365D` (최근 1년) |
| `startDT` | 시작일 | `2024-01-01` |
| `endDT` | 종료일 | `2024-12-31` |
| `parameterCd` | 파라미터 코드 | `72019` (지하수 깊이) |

## 센트럴 밸리 지역 정보

### 주요 카운티 코드 (FIPS)
- **019**: Fresno County
- **031**: Kings County
- **039**: Madera County
- **047**: Merced County
- **077**: San Joaquin County
- **099**: Stanislaus County
- **107**: Tulare County

### 경계 상자 (Bounding Box)
```python
# 센트럴 밸리 대략적 좌표
west_lon = -122.5
south_lat = 35.0
east_lon = -118.5
north_lat = 40.5
```

## Python 사용 예제

### 기본 사용법

```python
import requests
import pandas as pd

# 1. 캘리포니아 지하수 측정소 조회 (최근 7일 데이터)
url = "https://waterservices.usgs.gov/nwis/gwlevels/"
params = {
    'format': 'json',
    'stateCd': 'CA',
    'period': 'P7D',
    'siteStatus': 'all'
}

response = requests.get(url, params=params, timeout=120)
data = response.json()

# 데이터 추출
for site_data in data.get('value', []):
    site_info = site_data['sourceInfo']
    site_code = site_info['siteCode'][0]['value']
    site_name = site_info.get('siteName', 'N/A')

    lat = site_info['geoLocation']['geogLocation']['latitude']
    lon = site_info['geoLocation']['geogLocation']['longitude']

    print(f"측정소: {site_code}")
    print(f"이름: {site_name}")
    print(f"위치: ({lat}, {lon})")
    print("-" * 40)
```

### 특정 측정소 데이터 조회

```python
# 2. 특정 측정소의 지하수 수위 데이터
site_no = "365543119204701"  # Fresno 지역 예제

url = "https://waterservices.usgs.gov/nwis/gwlevels/"
params = {
    'format': 'json',
    'sites': site_no,
    'period': 'P365D',  # 최근 1년
    'siteStatus': 'all'
}

response = requests.get(url, params=params)
data = response.json()

# 수위 데이터 추출
for site_data in data.get('value', []):
    values = site_data.get('values', [{}])[0].get('value', [])

    for v in values:
        date = v['dateTime']
        water_level = v['value']
        print(f"{date}: {water_level} feet")
```

### 일별 데이터 조회

```python
# 3. 일별 통계 데이터
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

url = "https://waterservices.usgs.gov/nwis/dv/"
params = {
    'format': 'json',
    'sites': site_no,
    'parameterCd': '72019',  # Depth to water level
    'startDT': start_date.strftime('%Y-%m-%d'),
    'endDT': end_date.strftime('%Y-%m-%d'),
    'siteStatus': 'all'
}

response = requests.get(url, params=params)
data = response.json()

# 데이터 처리
for ts in data['value'].get('timeSeries', []):
    variable = ts['variable']['variableName']
    unit = ts['variable']['unit']['unitCode']
    values = ts.get('values', [{}])[0].get('value', [])

    print(f"변수: {variable} ({unit})")
    print(f"데이터 포인트: {len(values)}개")
```

## 주요 파라미터 코드

| 코드 | 설명 |
|------|------|
| `72019` | Depth to water level, feet below land surface |
| `62610` | Groundwater level above NAVD 1988 |
| `72020` | Elevation above NGVD 1929, feet |

## 데이터 분석 팁

### 1. 지하수 수위 트렌드 분석
```python
import matplotlib.pyplot as plt

# DataFrame 생성
df = pd.DataFrame(records)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['water_level_ft'])
plt.xlabel('Date')
plt.ylabel('Water Level (feet below surface)')
plt.title('Groundwater Level Trend')
plt.grid(True)
plt.show()
```

### 2. 여러 측정소 비교
```python
# 여러 측정소 데이터를 하나의 DataFrame으로
sites = ['site1', 'site2', 'site3']
all_data = []

for site in sites:
    # API 호출
    # ...
    all_data.extend(records)

df = pd.DataFrame(all_data)

# 측정소별 평균 수위
avg_by_site = df.groupby('site_no')['water_level_ft'].mean()
print(avg_by_site)
```

## 참고 자료

- **공식 문서**: https://waterservices.usgs.gov/rest/
- **데이터 카탈로그**: https://waterdata.usgs.gov/nwis
- **센트럴 밸리 지하수 정보**: https://ca.water.usgs.gov/
- **GRACE 위성 데이터**: https://podaac.jpl.nasa.gov/GRACE

## 주의사항

1. **요청 제한**: API는 공개되어 있지만, 대량 요청 시 타임아웃 발생 가능
2. **데이터 지연**: 실시간 데이터가 아니며, 며칠의 지연이 있을 수 있음
3. **측정소 상태**: 일부 측정소는 비활성 또는 데이터가 없을 수 있음
4. **좌표계**: WGS84 사용

## MCP 서버 연동

이 프로젝트는 USGS API를 MCP 서버로 래핑하여 Claude Code에서 자연어로 사용할 수 있습니다.

**현재 MCP 서버:**
- `usgs_gwinfo_mcp.py` - 데이터 수집 MCP 서버
- `kriging_mcp.py` - 공간 보간 MCP 서버
- `visualize_mcp.py` - 시각화 MCP 서버

**필요한 패키지:**
```bash
# 기본 의존성
pip install requests pandas mcp

# Kriging + 시각화
pip install pykrige matplotlib

# GeoTIFF (선택)
pip install rasterio
```

**사용 예:**
```
San Diego 지역의 2026년 1월 지하수 데이터를 시각화해줘
```

자세한 MCP 사용법은 `USGS_GWINFO_MCP_GUIDE.md`를 참조하세요.

# USGS Groundwater Info MCP Server 가이드

USGS NWIS API를 통해 지하수 데이터를 수집하는 MCP 서버입니다.

## 설치 및 실행

### 의존성
```bash
pip install requests pandas mcp
```

### 실행
```bash
python usgs_gwinfo_mcp.py
```

Claude Code에서는 `.mcp.json`에 설정되어 자동으로 실행됩니다.

## 제공 도구 (Tools)

### 1. get_groundwater_sites

지정된 영역 내 지하수 관측소 목록을 조회합니다.

**파라미터:**
| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `bbox` | string | O | 경계 상자 `'서,남,동,북'` (예: `'-117.5,32.5,-116.5,33.5'`) |
| `aquifer_type` | string | X | 대수층 유형 (기본값: `'U'`) |

**대수층 유형 코드:**
- `U` - Unconfined (비피압 대수층)
- `C` - Confined (피압 대수층)
- `X` - Mixed (혼합)
- `M` - Multiple (다중)

**반환값:**
```json
{
  "bbox": "-117.5,32.5,-116.5,33.5",
  "aquifer_type": "U",
  "count": 15,
  "sites": [
    {
      "site_no": "323527117050003",
      "name": "018S002W22E005S",
      "lat": 32.591,
      "lon": -117.083,
      "aquifer_type": "U"
    }
  ]
}
```

---

### 2. get_groundwater_data

지정된 영역과 기간의 지하수위 데이터를 조회합니다.

**파라미터:**
| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `bbox` | string | O | 경계 상자 `'서,남,동,북'` |
| `start_date` | string | O | 시작일 `'YYYY-MM-DD'` |
| `end_date` | string | O | 종료일 `'YYYY-MM-DD'` |
| `aquifer_type` | string | X | 대수층 유형 (기본값: `'U'`) |
| `output_csv` | string | X | CSV 출력 파일명 |

**반환값:**
```json
{
  "bbox": "-117.5,32.5,-116.5,33.5",
  "start_date": "2026-01-13",
  "end_date": "2026-01-13",
  "aquifer_type": "U",
  "sites_found": 22,
  "sites_with_data": 20,
  "dates": ["2026-01-13"],
  "output_csv": "san_diego_gw.csv",
  "data": {
    "323527117050003": {
      "name": "018S002W22E005S",
      "lat": 32.591,
      "lon": -117.083,
      "values": {
        "2026-01-13": 35.01
      }
    }
  }
}
```

**CSV 출력 형식:**
```
Site,Name,Lat,Lon,2026-01-13,Change
323527117050003,"018S002W22E005S",32.591,-117.083,35.01,
```

---

### 3. get_site_history

특정 관측소의 과거 데이터를 조회합니다.

**파라미터:**
| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `site_no` | string | O | USGS 관측소 번호 (예: `'323527117050001'`) |
| `days` | integer | X | 조회 기간 일수 (기본값: 365) |

**반환값:**
```json
{
  "site_no": "323527117050001",
  "site_name": "018S002W22E005S",
  "days": 365,
  "record_count": 250,
  "records": [
    {"date": "2025-01-15", "depth_ft": 34.5},
    {"date": "2025-01-16", "depth_ft": 34.8}
  ]
}
```

---

## 사용 예시

### Claude Code에서 사용

```
San Diego 지역의 1월 13일 지하수 데이터를 가져와줘
```

Claude Code가 자동으로 다음을 수행:
1. `get_groundwater_data` 호출 (bbox, 날짜 지정)
2. CSV 파일 생성
3. 결과 요약 제공

### 파이프라인 워크플로우

```
[get_groundwater_data] → CSV 생성
        ↓
[kriging_interpolate] → 보간 그리드 생성
        ↓
[visualize_kriging_result] → PNG 시각화
```

---

## 주요 지역 경계 상자 (Bounding Box)

| 지역 | bbox |
|-----|------|
| San Diego | `-117.5,32.5,-116.5,33.5` |
| Los Angeles | `-118.7,33.7,-117.5,34.4` |
| Central Valley | `-122.5,35.0,-118.5,40.5` |
| San Francisco Bay | `-123.0,37.0,-121.5,38.5` |

---

## 데이터 설명

- **Depth to Water (ft)**: 지표면에서 지하수면까지의 깊이 (피트)
- 값이 클수록 지하수면이 깊음 (물이 부족)
- 값이 작을수록 지하수면이 얕음 (물이 풍부)

---

## API 제약사항

- **타임아웃**: 대규모 쿼리 시 120초
- **데이터 지연**: 실시간이 아님 (며칠 지연 가능)
- **좌표계**: WGS84 (EPSG:4326)

---

## 문제 해결

### "No sites found" 오류
- 경계 상자 범위가 너무 좁을 수 있음
- 해당 지역에 해당 대수층 유형의 관측소가 없을 수 있음
- `aquifer_type`을 다른 값으로 변경해보기

### 데이터가 비어있는 경우
- 요청한 날짜에 관측 데이터가 없을 수 있음
- 날짜 범위를 넓혀서 재시도

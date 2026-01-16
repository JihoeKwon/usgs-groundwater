# Agent Role: USGS Groundwater Specialist

## Identity

**Name**: USGS Groundwater Agent
**Type**: Data Collection & Geospatial Analysis Specialist
**Domain**: Hydrology, Groundwater Monitoring, Spatial Interpolation

## Purpose

미국 전역의 지하수 데이터를 수집하고 Kriging 공간 보간을 통해 지하수위 분포를 분석/시각화하는 전문 에이전트입니다.

## Core Capabilities

### 1. 데이터 수집 (USGS NWIS API)
- 경계 상자 기반 지하수 관측소 조회
- 단일/다중 날짜 지하수위 데이터 수집
- 대수층 타입별 필터링 (Unconfined/Confined/Mixed)
- CSV 형식 데이터 출력 (Wide Format)

### 2. 공간 보간 (Kriging)
- Ordinary Kriging 알고리즘
- Variogram 모델: spherical, gaussian, exponential, linear
- 단일 날짜: Grid CSV + GeoTIFF 출력
- 다중 날짜: 섹션 형식 .dat 파일 출력
- 불확실성(분산) 계산

### 3. 시각화
- 등고선 맵 (Contour Map)
- 불확실성 분포 맵
- 관측 포인트 오버레이
- 시계열 GIF 애니메이션
- 두 데이터셋 비교 플롯

## MCP Architecture

이 프로젝트는 3개의 연동된 MCP 서버로 구성됩니다:

```
[usgs-gwinfo] → [kriging] → [visualize-kriging]
```

**Pipeline 흐름:**
1. `usgs-gwinfo`: 데이터 수집 및 CSV 출력
2. `kriging`: CSV 입력 → Kriging 보간 → .dat/.csv + GeoTIFF
3. `visualize-kriging`: Grid 입력 → PNG/GIF 시각화

## Available Tools

### usgs-gwinfo MCP
| 도구 | 설명 |
|------|------|
| `get_groundwater_sites` | 관측소 목록 조회 |
| `get_groundwater_data` | 다중 날짜 지하수위 데이터 수집 및 CSV 생성 |
| `get_groundwater_data_single_date` | 단일 날짜 데이터 수집 |
| `get_site_history` | 특정 관측소 이력 |

### kriging MCP
| 도구 | 설명 |
|------|------|
| `kriging_interpolate` | 단일 날짜 Kriging 보간 수행 |
| `kriging_interpolate_multiple` | 다중 날짜 Kriging (시계열 분석) |
| `get_variogram_models` | 사용 가능한 베리오그램 모델 목록 |

### visualize-kriging MCP
| 도구 | 설명 |
|------|------|
| `visualize_kriging_result` | PNG (단일) 또는 GIF (다중) 시각화 생성 |
| `get_available_colormaps` | 컬러맵 옵션 |
| `create_comparison_plot` | 비교 플롯 |

## Input/Output Specifications

### Inputs
- **Bounding Box**: `west,south,east,north` (WGS84)
- **Date Range**: ISO 8601 (`2026-01-13` 또는 `P7D`)
- **Aquifer Type**: `U`(Unconfined), `C`(Confined), `X`(Mixed), `M`(Multiple)

### Outputs
- **CSV**: 지하수위 데이터 (UTF-8-sig, Excel 호환)
- **DAT**: 다중 날짜 Kriging 결과 (섹션 형식)
- **GeoTIFF**: Kriging 결과 래스터 (선택적)
- **PNG**: 단일 프레임 시각화 (깊이 맵 + 분산 맵)
- **GIF**: 시계열 애니메이션

## Geographic Coverage

주요 지역 경계 상자:
| 지역 | bbox |
|------|------|
| San Diego | `-117.5,32.5,-116.5,33.5` |
| Southern California | `-120.5,32.5,-114.5,35.5` |
| Los Angeles | `-118.7,33.7,-117.5,34.4` |
| Central Valley | `-122.5,35.0,-118.5,40.5` |
| San Francisco Bay | `-123.0,37.0,-121.5,38.5` |

미국 전역 커버리지 (USGS 네트워크 범위 내)

## Technical Stack

- **Language**: Python 3.10+
- **Core Libraries**: requests, pandas, pykrige, matplotlib
- **Optional**: rasterio (GeoTIFF 출력), pillow (GIF 생성)
- **API**: USGS NWIS REST API
- **Protocol**: Model Context Protocol (MCP)

## Current Project State

**Location**: `D:\Claude\usgs-groundwater\`
**Status**: Production-ready MCP servers
**Last Updated**: 2026-01-16

**Key Files:**
| 파일 | 설명 |
|------|------|
| `CLAUDE.md` | 프로젝트 작업 시 읽어야 할 메인 가이드 |
| `README.md` | 사용자 문서 |
| `USGS_NWIS_GUIDE.md` | USGS API 레퍼런스 |
| `USGS_GWINFO_MCP_GUIDE.md` | MCP 도구 상세 가이드 |
| `USGS_MCP_SERVER.md` | MCP 서버 통합 가이드 |
| `.mcp.json` | MCP 설정 파일 |

## When to Call This Agent

Desktop Orchestrator에서 다음 요청 시 이 프로젝트를 호출해야 합니다:

**지하수 관련 키워드**
- "지하수", "groundwater", "water level", "수위"
- "USGS", "관측소", "monitoring well"

**공간 분석 요청**
- "kriging", "보간", "interpolation", "spatial analysis"
- "분포도", "contour", "등고선", "distribution"

**지역 언급 포함**
- "San Diego", "California", "미국 지하수" + 지하수 관련
- 경계 상자(bbox) 좌표 포함 시

**시각화 요청**
- "지하수 맵", "groundwater map", "시각화"
- "불확실성", "variance", "uncertainty map"
- "애니메이션", "시계열", "추이"

## Example Workflows

### 단일 날짜 분석
```
User: San Diego 지역의 1월 13일 지하수 데이터를 시각화해줘

Pipeline:
1. get_groundwater_data_single_date(bbox, date)
2. kriging_interpolate(csv)
3. visualize_kriging_result(grid_csv) → PNG
```

### 시계열 분석
```
User: 남부 캘리포니아의 1월 1일~13일 지하수위 추이를 보여줘

Pipeline:
1. get_groundwater_data(bbox, start_date, end_date)
2. kriging_interpolate_multiple(csv)
3. visualize_kriging_result(dat_file) → GIF
```

## Constraints & Limitations

### API 제한
- **Timeout**: 영역 전체 쿼리 120초, 관측소별 60초
- **Data Lag**: 실시간이 아님 (며칠 지연)
- **Rate Limiting**: USGS API 과부하 시 지연

### 데이터 품질
- 관측소 밀도에 따라 보간 정확도 달라짐
- 일부 관측소는 결측치 발생 가능
- Kriging은 outlier에 민감

### 지리적 범위
- 미국 본토 한정 (USGS 네트워크)
- 하와이, 알래스카는 관측소 밀도 낮음

## Error Handling

이 프로젝트는 다음 상황에서 명확한 에러 메시지를 반환합니다:

| 에러 | 원인 | 해결책 |
|------|------|--------|
| "No sites found" | 범위 내 관측소 없음 | bbox 확대 또는 대수층 유형 변경 |
| "No data available" | 해당 날짜에 데이터 없음 | 날짜 범위 확대 |
| "Need at least 3 data points" | Kriging용 데이터 부족 | bbox 확대 또는 날짜 변경 |
| "pykrige not installed" | 의존성 미설치 | `pip install pykrige` |
| "Timeout" | API 응답 지연 | bbox 축소 또는 날짜 범위 축소 |

## Version & Maintenance

**Current Version**: 2.0 (MCP 기반 파이프라인)
**Maintained by**: CLI 환경에서 직접 수정/테스트
**Update Frequency**: API 변경 시 또는 기능 추가 요청 시

**Backup**: `backup/` 폴더에 레거시 standalone 스크립트 보관

---

## Summary for Desktop Agent

**한 줄 요약**: 미국 지하수 데이터 수집 및 Kriging 보간 후 시각화 자동 파이프라인

**호출 조건**: 지하수, USGS, 공간 분석, 분포 시각화 관련 요청

**필수 읽기**: `CLAUDE.md` (프로젝트 컨텍스트)

**출력물**: CSV 데이터 + PNG/GIF 시각화 이미지

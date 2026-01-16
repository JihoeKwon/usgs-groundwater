# USGS Groundwater MCP Server 통합 가이드

USGS 지하수 데이터를 MCP (Model Context Protocol)를 통해 제공하는 3개의 연동 서버입니다.

## 설치

### 1. Python 의존성 설치

```bash
# 기본 의존성
pip install requests pandas mcp

# Kriging + 시각화 (전체 기능)
pip install pykrige matplotlib

# GeoTIFF 출력 (선택)
pip install rasterio
```

### 2. MCP 서버 설정

`.mcp.json` 파일에 3개의 서버가 설정되어 있습니다:

```json
{
  "mcpServers": {
    "usgs-gwinfo": {
      "type": "stdio",
      "command": "python",
      "args": ["usgs_gwinfo_mcp.py"]
    },
    "kriging": {
      "type": "stdio",
      "command": "python",
      "args": ["kriging_mcp.py"]
    },
    "visualize-kriging": {
      "type": "stdio",
      "command": "python",
      "args": ["visualize_mcp.py"]
    }
  }
}
```

## MCP 서버 구성

### 1. usgs-gwinfo (데이터 수집)

USGS NWIS API에서 지하수 데이터를 수집합니다.

**제공 도구:**
| 도구 | 설명 |
|------|------|
| `get_groundwater_sites` | 경계 상자 내 관측소 목록 조회 |
| `get_groundwater_data` | 다중 날짜 범위의 지하수위 데이터 조회 |
| `get_groundwater_data_single_date` | 단일 날짜 지하수위 데이터 조회 |
| `get_site_history` | 특정 관측소의 과거 데이터 조회 |

**사용 예시:**
```
남부 캘리포니아의 2026년 1월 1일부터 13일까지 지하수 데이터를 조회해줘
```

---

### 2. kriging (공간 보간)

Ordinary Kriging을 사용한 공간 보간을 수행합니다.

**제공 도구:**
| 도구 | 설명 |
|------|------|
| `kriging_interpolate` | 단일 날짜 Kriging 보간 (Grid CSV + GeoTIFF 출력) |
| `kriging_interpolate_multiple` | 다중 날짜 Kriging 보간 (섹션 형식 .dat 출력) |
| `get_variogram_models` | 사용 가능한 베리오그램 모델 목록 |

**베리오그램 모델:**
- `spherical` (기본값) - 유한 범위에서 실에 도달
- `gaussian` - 부드러운 연속 현상에 적합
- `exponential` - 점근적으로 실에 도달
- `linear` - 단순 선형 모델

**출력 형식:**

단일 날짜:
- Grid CSV: 격자점별 Lon, Lat, Depth_ft, Variance
- GeoTIFF: 지리참조된 래스터 (rasterio 필요)
- Metadata: 통계 요약

다중 날짜 (.dat 섹션 형식):
```
[HEADER]
NFRAME=13
GRID=50x50
POINTS=2500
DATES=2026-01-01,2026-01-02,...

[COORDINATES]
Lon,Lat
-117.421197,32.514499
...

[DEPTH]
35.47,35.46,...
...

[VARIANCE]
1234.56,1234.78,...
...
```

---

### 3. visualize-kriging (시각화)

Kriging 결과를 시각화합니다.

**제공 도구:**
| 도구 | 설명 |
|------|------|
| `visualize_kriging_result` | 단일 프레임 PNG 또는 다중 프레임 GIF 애니메이션 생성 |
| `get_available_colormaps` | 사용 가능한 컬러맵 목록 |
| `create_comparison_plot` | 두 데이터셋 비교 플롯 |

**출력:**
- 단일 프레임: 지하수 깊이 맵 + 불확실성(분산) 맵 (side-by-side PNG)
- 다중 프레임: 시계열 GIF 애니메이션

**추천 컬러맵:**
- `viridis_r` (기본값) - 노란색(얕음) → 보라색(깊음)
- `plasma_r` - 비슷하지만 더 따뜻한 색상
- `RdYlBu` - 빨강-노랑-파랑 발산형
- `terrain` - 지형 색상

---

## 전체 파이프라인 예시

### Claude Code에서 사용

**시계열 분석 요청:**
```
남부 캘리포니아 지역의 2026-01-01부터 2026-01-13까지의 지하수위 분포 추이를 시각화해줘
```

Claude Code가 자동으로:
1. `get_groundwater_data` → CSV 생성
2. `kriging_interpolate_multiple` → .dat 파일 생성
3. `visualize_kriging_result` → GIF 애니메이션 생성

**단일 날짜 분석 요청:**
```
San Diego 지역의 1월 13일 지하수 데이터를 가져와서 시각화해줘
```

---

## 아키텍처

### 데이터 흐름

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│ usgs-gwinfo │───▶│   kriging   │───▶│ visualize-kriging│
│             │    │             │    │                 │
│  USGS API   │    │  PyKrige    │    │  Matplotlib     │
│  requests   │    │  numpy      │    │  pillow (GIF)   │
└─────────────┘    └─────────────┘    └─────────────────┘
      │                  │                    │
      ▼                  ▼                    ▼
   CSV 파일          .dat/.csv            PNG/GIF
   (Wide Format)    (Grid Data)          (시각화)
```

### API 엔드포인트

- `https://waterservices.usgs.gov/nwis/site/` - 관측소 메타데이터 (RDB)
- `https://waterservices.usgs.gov/nwis/dv/` - 일별 데이터 (JSON)
- `https://waterservices.usgs.gov/nwis/gwlevels/` - 지하수 측정값 (JSON)

### 타임아웃 설정

- 영역 전체 조회: 120초
- 특정 관측소 조회: 60초

---

## 주의사항

1. **API 제한**: USGS API는 공개 API이지만 대량 요청 시 타임아웃 발생 가능
2. **데이터 지연**: 실시간 데이터가 아니며, 며칠의 지연이 있을 수 있음
3. **관측소 가용성**: 모든 관측소가 활성 상태는 아님
4. **네트워크**: 인터넷 연결 필요
5. **Kriging 제한**: 최소 3개 이상의 데이터 포인트 필요

---

## 문제 해결

### "pykrige not installed" 에러
```bash
pip install pykrige
```

### "matplotlib not installed" 에러
```bash
pip install matplotlib
```

### 타임아웃 에러
- 경계 상자 범위를 줄이세요
- 날짜 범위를 줄이세요

### "No sites found" 에러
- 경계 상자 좌표 확인 (서,남,동,북 순서)
- 대수층 유형을 다른 값으로 변경 (`U`, `C`, `X`)

### GIF 생성 실패
```bash
pip install pillow
```

---

## 참고 자료

- [USGS NWIS API 공식 문서](https://waterservices.usgs.gov/rest/)
- [MCP 프로토콜 사양](https://modelcontextprotocol.io/)
- [PyKrige Documentation](https://geostat-framework.readthedocs.io/projects/pykrige/)
- `USGS_NWIS_GUIDE.md` - 상세 API 사용 가이드
- `USGS_GWINFO_MCP_GUIDE.md` - MCP 도구 상세 가이드

# USGS NWIS MCP Server

USGS 지하수 데이터를 MCP (Model Context Protocol)를 통해 제공하는 서버입니다.

## 설치

### 1. Python 의존성 설치

```bash
pip install -r requirements_usgs.txt
pip install mcp
```

### 2. MCP 서버 설정

`.mcp.json` 파일에 이미 서버가 추가되어 있습니다:

```json
{
  "mcpServers": {
    "usgs-nwis": {
      "type": "stdio",
      "command": "python",
      "args": ["usgs_nwis_mcp_server.py"]
    }
  }
}
```

## 제공되는 도구 (Tools)

### 1. `get_recent_groundwater_data`

특정 주(state)의 최근 지하수 데이터를 조회합니다.

**파라미터:**
- `state_code` (string, 기본값: "CA"): 두 글자 주 코드 (예: "CA", "TX")
- `days` (integer, 기본값: 7): 최근 며칠간의 데이터 (1-365)
- `max_sites` (integer, 기본값: 50): 최대 측정소 수 (1-200)

**반환값:**
```json
{
  "sites": [
    {
      "site_no": "측정소 번호",
      "site_name": "측정소 이름",
      "latitude": 위도,
      "longitude": 경도
    }
  ],
  "data": [
    {
      "site_no": "측정소 번호",
      "datetime": "ISO 8601 날짜시간",
      "water_level_ft": 지하수 깊이 (feet)
    }
  ],
  "summary": {
    "total_sites": 총 측정소 수,
    "total_data_points": 총 데이터 포인트 수,
    "state": "주 코드",
    "period_days": 조회 기간
  }
}
```

**사용 예시:**
```
캘리포니아의 최근 30일 지하수 데이터를 조회해줘 (최대 100개 측정소)
```

### 2. `get_site_history`

특정 측정소의 과거 지하수 수위 데이터를 조회합니다.

**파라미터:**
- `site_no` (string, 필수): USGS 측정소 번호 (예: "365543119204701")
- `start_date` (string, 선택): 시작일 YYYY-MM-DD 형식
- `end_date` (string, 선택): 종료일 YYYY-MM-DD 형식
- `days` (integer, 기본값: 365): 최근 며칠 (start_date/end_date 미지정시)

**반환값:**
```json
{
  "site_info": {
    "site_no": "측정소 번호",
    "site_name": "측정소 이름",
    "latitude": 위도,
    "longitude": 경도
  },
  "data": [
    {
      "datetime": "ISO 8601 날짜시간",
      "water_level_ft": 지하수 깊이 (feet)
    }
  ],
  "summary": {
    "total_records": 총 레코드 수,
    "min_water_level_ft": 최소 수위,
    "max_water_level_ft": 최대 수위,
    "avg_water_level_ft": 평균 수위
  }
}
```

**사용 예시:**
```
측정소 365543119204701의 최근 1년 데이터를 조회해줘
```

```
측정소 365543119204701의 2024-01-01부터 2024-12-31까지 데이터를 조회해줘
```

## 테스트

### MCP 서버 직접 실행

```bash
python usgs_nwis_mcp_server.py
```

서버가 stdio 모드로 실행되며, MCP 프로토콜 메시지를 수신 대기합니다.

### Claude Code에서 사용

Claude Code를 재시작한 후, 다음과 같이 요청할 수 있습니다:

```
캘리포니아의 최근 7일 지하수 데이터를 조회해줘
```

```
USGS 측정소 365543119204701의 최근 1년 데이터를 분석해줘
```

## 아키텍처

### 컴포넌트 구조

```
usgs_nwis_mcp_server.py
├── USGSGroundwaterAPI (클래스)
│   ├── get_recent_groundwater_data()
│   └── get_site_history()
├── MCP Server 인스턴스
│   ├── @list_tools() - 도구 목록 제공
│   ├── @call_tool() - 도구 실행
│   └── stdio 통신
```

### 데이터 흐름

1. **MCP 클라이언트 (Claude)** → 도구 호출 요청
2. **MCP Server** → USGSGroundwaterAPI 메서드 실행
3. **USGSGroundwaterAPI** → USGS NWIS API HTTP 요청
4. **USGS API** → JSON 응답 반환
5. **USGSGroundwaterAPI** → 데이터 파싱 및 구조화
6. **MCP Server** → 포맷된 텍스트 + JSON 응답
7. **MCP 클라이언트** → 결과 수신 및 표시

### API 엔드포인트

- `https://waterservices.usgs.gov/nwis/gwlevels/` - 지하수 수위 데이터

### 타임아웃 설정

- 주 전체 조회: 120초
- 특정 측정소 조회: 60초

## 주의사항

1. **API 제한**: USGS API는 공개 API이지만 대량 요청 시 타임아웃 발생 가능
2. **데이터 지연**: 실시간 데이터가 아니며, 며칠의 지연이 있을 수 있음
3. **측정소 가용성**: 모든 측정소가 활성 상태는 아님
4. **네트워크**: 인터넷 연결 필요

## 문제 해결

### "Module 'mcp' not found" 에러

```bash
pip install mcp
```

### 타임아웃 에러

- `days` 파라미터를 줄이세요 (예: 30 → 7)
- `max_sites` 파라미터를 줄이세요 (예: 100 → 50)

### 데이터 없음

- 해당 기간에 측정된 데이터가 없을 수 있습니다
- 다른 측정소를 시도하거나 기간을 늘려보세요

## 참고 자료

- [USGS NWIS API 공식 문서](https://waterservices.usgs.gov/rest/)
- [MCP 프로토콜 사양](https://modelcontextprotocol.io/)
- `USGS_NWIS_GUIDE.md` - 상세 API 사용 가이드

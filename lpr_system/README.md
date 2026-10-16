# lpr_system — 데스크톱 앱과 인식 모듈

번호판 검출(YOLOv11m)·추적(ByteTrack)·인식(PaddleOCR)·기록(SQLite)을 담당하는 PyQt6 앱.
프로젝트 전체 설명과 검증 결과는 [저장소 README](../README.md)를 참고한다.

---

## 파일 구성

| 파일 | 역할 |
|---|---|
| `app_pyqt6.py` | PyQt6 앱. 영상 파일 처리, 카메라 관리, 조회·통계·요금·내보내기 화면 |
| `detector.py` | YOLO 검출·추적, 폴백 검출(이진화·윤곽선·색상), 번호판 색 분류, OCR 전처리 |
| `ocr_engine.py` | PaddleOCR 래퍼, 신뢰도 기준 적용, 정규식 번호 추출 |
| `pipeline.py` | 화이트리스트 교정, 중복 제거, DB 기록, 카메라 스트림 처리 |
| `database.py` | SQLite 스키마 생성, 기록·조회 함수 |
| `parking_fee.py` | 입·출차 시각 기반 요금 계산 |
| `excel_export.py` | 기간별 기록 엑셀 내보내기 |
| `query_tool.py` | DB 조회·차량 관리 명령줄 도구 |
| `load_demo_data.py` | `analytics/demo_parking.db`의 데모 데이터를 앱 DB(`parking.db`)로 복사 |
| `clear_db.py`, `clear_db.bat` | 인식 기록·입출차 기록 삭제 |
| `demo.py` | 합성 번호판 이미지로 색 분류·전처리·DB 저장 확인 (OCR 미사용) |
| `models/plate_yolo11m.pt` | 학습한 번호판 검출 모델 |

실행 중 생성되는 `parking.db`, `plate_images/`(인식한 번호판 이미지), `logs/`, `exports/`는 `.gitignore`로 제외했다.

---

## 실행

```bash
pip install -r requirements.txt
python app_pyqt6.py
```

앱 화면: 대시보드, 번호판 검색, 등록차량, 블랙리스트, 카메라, 영상파일, 설정관리, 통계, 엑셀내보내기, 요금계산

### 명령줄 조회 (`query_tool.py`)

```bash
python query_tool.py all                      # 최근 100건 조회
python query_tool.py plate 123가4567          # 번호 검색 (부분 일치)
python query_tool.py date 2026-09-01          # 날짜별 조회
python query_tool.py stats                    # 색상·종류별 통계
python query_tool.py add 123가4567 홍길동 "검정 소나타"   # 화이트리스트 등록
python query_tool.py block 999다8888 "도주 차량"         # 블랙리스트 등록
python query_tool.py unblock 999다8888        # 블랙리스트 해제
python query_tool.py blocklist                # 블랙리스트 전체 조회
python query_tool.py purge 90                 # 90일 이전 기록 삭제
```

---

## 주요 설정값

| 파일 | 상수 | 값 | 의미 |
|---|---|---|---|
| `detector.py` | `_YOLO_CONF` | 0.07 | YOLO 검출 신뢰도 기준 |
| `detector.py` | `MIN_AREA` / `MAX_AREA` | 700 / 45000 | 폴백 검출 윤곽선 면적 범위 |
| `detector.py` | `MIN_ASPECT` / `MAX_ASPECT` | 1.1 / 9.0 | 폴백 검출 가로:세로 비율 범위 |
| `ocr_engine.py` | `CONFIDENCE_THRESHOLD` | 0.20 | OCR 결과 신뢰도 기준 (일반) |
| `ocr_engine.py` | `CONFIDENCE_THRESHOLD_OLD` | 0.15 | OCR 결과 신뢰도 기준 (녹색 구형) |
| `pipeline.py` | `SNAP_DIST` | 1 | 화이트리스트 교정 편집거리 (같은 길이, 후보 1개일 때만) |
| `pipeline.py` | `PLATE_HANGUL` | 40자 | 번호판 용도 기호. 이 밖의 한글이 나오면 결과를 버림 (등록 차량 번호와 같으면 통과) |
| `pipeline.py` | `COOLDOWN_SEC` | 120 | 같은 차량으로 보는 시간 범위(초) |
| `pipeline.py` | `FUZZY_DIST` | 2 | 같은 차량으로 보는 편집거리 |
| `pipeline.py` | `VOTE_FRAMES` | 1 | 번호 확정에 필요한 프레임 수 (1 = 첫 인식 즉시 확정) |
| `pipeline.py` | `PROCESS_EVERY_N` | 1 | 카메라 모드 처리 간격(프레임) |

---

## 처리 흐름

**영상 파일 모드** (`app_pyqt6.py` VideoWorker)

1. 매 프레임 `detector.track_plates`로 번호판 위치와 추적 ID 갱신
2. 번호가 확정되지 않은 추적 ID 중 가장 큰 번호판을 OCR 작업 스레드로 전달 (같은 ID 재시도 간격 0.5초)
3. `pipeline.process_tracked_roi`: 색 분류, OCR, 화이트리스트 교정, 한글 검사, 중복 제거, DB 기록
4. 블랙리스트 차량이면 경보 팝업

**카메라 모드** (`pipeline.process_stream`)

1. 프레임마다 `detect_plate_candidates` 실행 (YOLO 결과가 없거나 녹색 번호판 색이 보이면 폴백 검출 추가)
2. `process_frame`: 크기 필터, 색 분류, OCR, 화이트리스트 교정, 한글 검사, 중복 제거, DB 기록
3. 연결이 끊기면 5초 간격으로 재연결

DB 테이블 구성은 [저장소 README 4장](../README.md#4-db-설계)에 정리했다.

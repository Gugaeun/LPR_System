"""검출·OCR 검증 실험 6단계: 한글 검사 적용 전후 비교 (lpr_env에서 실행)

폴백 영역 중 PaddleOCR이 번호판 형식 문자열을 낸 영역을 앱의 process_tracked_roi로 한 건씩 처리한다.
같은 입력으로 한글 검사(pipeline._valid_plate_hangul)를 끈 경우와 켠 경우의 저장 결과를 비교한다.

  - 기록은 임시 DB에만 저장한다(앱의 parking.db는 건드리지 않는다).
  - 영역마다 중복 제거·투표 버퍼를 비워 서로 영향을 주지 않게 한다.
  - 정답은 data/crops_checked.csv(눈으로 확인한 번호)를 쓴다.

사용법:
  python step6_hangul_check.py --data data
"""
import argparse
import csv
import os
import sys
import tempfile

import cv2

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lpr_system")
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
args = ap.parse_args()
data = os.path.abspath(args.data)

# 앱 모듈을 수정 없이 불러오고, DB 경로만 임시 파일로 바꾼다
sys.path.insert(0, os.path.abspath(APP))
import database  # noqa: E402
database.DB_PATH = os.path.join(tempfile.mkdtemp(), "hangul_check.db")
database.init_db()
# torch(pipeline이 불러오는 detector)를 paddle(ocr_engine)보다 먼저 불러와야 DLL 로드 오류가 없다
import pipeline as pl  # noqa: E402
import ocr_engine as O  # noqa: E402

O.init_reader(gpu=False)
conn = database.get_connection()
truth = {r["crop"]: r["truth"] for r in csv.DictReader(open(os.path.join(data, "crops_checked.csv"), encoding="utf-8-sig"))}
rows = [r for r in csv.DictReader(open(os.path.join(data, "crops_both.csv"), encoding="utf-8-sig")) if r["paddle"]]
check = pl._valid_plate_hangul


def run(check_on):
    pl._valid_plate_hangul = check if check_on else (lambda text, c: True)
    saved = []
    for r in rows:
        pl._last_seen.clear()
        pl._release_vote_buffer("TEST")
        roi = cv2.imread(os.path.join(data, "crops", r["crop"] + ".png"))
        bbox = tuple(int(r[k]) for k in ("x", "y", "w", "h"))
        res = pl.process_tracked_roi(roi, bbox, "TEST", "ENTRY", "hangul_check", conn, 0)
        if res:
            saved.append(res["plate_text"] == truth[r["crop"]])
    return len(saved), sum(saved), len(saved) - sum(saved)


print(f"입력: PaddleOCR이 번호판 형식 문자열을 낸 폴백 영역 {len(rows)}개")
print("| 조건 | 저장 | 정답 | 오답 |")
print("|---|---|---|---|")
for on, name in ((False, "한글 검사 없음"), (True, "한글 검사 적용")):
    n, ok, bad = run(on)
    print(f"| {name} | {n} | {ok} | {bad} |")

"""검출·OCR 검증 실험 2단계 (lpr_env에서 실행)

검출 결과와 상관없이 OCR 성능만 보기 위해, 영상 프레임에서 번호판 영역을 직접 지정해 자른
정답 이미지(gt/plates/*.png)를 PaddleOCR로 읽는다.
전처리·인식은 앱의 YOLO 경로(pipeline.process_tracked_roi)와 같다: 색 분류 → OCR 전처리 → read_plate

입력 gt_labels.csv 칼럼: crop(이미지 파일명), label(정답 번호), kind(번호판 종류)

사용법:
  python step2_gt_paddle.py --gt data/gt
  결과: data/gt/gt_paddle.csv
"""
import argparse
import csv
import os
import sys

import cv2

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lpr_system")
ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True, help="gt_labels.csv와 plates 폴더가 있는 폴더")
args = ap.parse_args()
gt = os.path.abspath(args.gt)

# 앱 모듈(detector, ocr_engine)을 수정 없이 그대로 불러온다
sys.path.insert(0, os.path.abspath(APP))
import detector as D  # noqa: E402
import ocr_engine as O  # noqa: E402

O.init_reader(gpu=False)
rows = list(csv.DictReader(open(os.path.join(gt, "gt_labels.csv"), encoding="utf-8-sig")))
for r in rows:
    roi = cv2.imread(os.path.join(gt, "plates", r["crop"] + ".png"))
    color, ptype = D.classify_plate_color(roi)
    proc = D.preprocess_for_ocr(roi, color, None)
    text, conf = O.read_plate(proc, ptype, raw_roi=roi)
    r.update(color=color, ptype=ptype, paddle=text, paddle_conf=round(conf, 3))

with open(os.path.join(gt, "gt_paddle.csv"), "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
print("paddle", sum(r["paddle"] == r["label"] for r in rows), "/", len(rows))

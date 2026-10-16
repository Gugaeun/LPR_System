"""검출·OCR 검증 실험 3단계 (EasyOCR 전용 가상환경에서 실행)

1단계 crop(crops_paddle.csv)과 2단계 정답 영역(gt_paddle.csv)을 EasyOCR(한국어)로 같은 조건에서 읽는다.
결과 정리 규칙은 앱의 ocr_engine.read_plate와 같다:
  위→아래 줄 정렬, 신뢰도 문턱(일반 0.20 / 구형 0.15), 한글·숫자만 남김, 번호판 정규식 추출

EasyOCR 1.7.2 설치 시 numpy 2.x, opencv-python-headless 5.x가 함께 설치된다.
lpr_env는 paddlepaddle 2.6 때문에 numpy 1.26, opencv 4.6으로 고정되어 있어 별도 가상환경에서 실행한다.
  python -m venv easy_env && easy_env\\Scripts\\pip install easyocr

사용법:
  python step3_easyocr.py --csv data/crops_paddle.csv --images data/crops --out data/crops_both.csv
  python step3_easyocr.py --csv data/gt/gt_paddle.csv --images data/gt/plates --out data/gt/gt_both.csv
"""
import argparse
import csv
import os
import re

import cv2
import easyocr

P_NEW = re.compile(r"(\d{2,3}[가-힣]\d{4})")
P_OLD = re.compile(r"([가-힣]{2}\d{2}[가-힣]\d{4})")
P_SMALL = re.compile(r"(\d{2}[가-힣]\d{4})")


def extract(cleaned):
    for p in (P_NEW, P_SMALL):
        m = p.search(cleaned)
        if m:
            return m.group()
    m = P_OLD.search(cleaned)
    return re.sub(r"^[가-힣]{2}", "", m.group()) if m else ""


def read(reader, img, ptype):
    res = sorted(reader.readtext(img), key=lambda t: t[0][0][1]) if img is not None else []
    thr = 0.15 if ptype == "OLD" else 0.20
    kept = [(t, c) for _, t, c in res if c >= thr]
    if not kept:
        return ""
    text = extract(re.sub(r"[^가-힣0-9]", "", "".join(t for t, _ in kept)))
    if not text and len(kept) >= 2:   # 2줄 번호판 줄 순서가 뒤집힌 경우
        text = extract(re.sub(r"[^가-힣0-9]", "", "".join(t for t, _ in reversed(kept))))
    return text


ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

reader = easyocr.Reader(["ko"], gpu=False, verbose=False)
rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
for r in rows:
    r["easy"] = read(reader, cv2.imread(os.path.join(args.images, r["crop"] + ".png")), r.get("ptype", "NEW"))
with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
print("done", len(rows))

"""검출·OCR 검증 실험 5단계: 집계 (번호 원문 없이 건수만 출력)

입력
  data/crops_both.csv    1·3단계 결과 (폴백 crop별 PaddleOCR / EasyOCR 출력)
  data/crops_checked.csv 번호판 형식 출력이 나온 crop을 눈으로 확인한 정답 (crop, truth)
  data/gt/gt_both.csv    2·3단계 결과 (직접 지정한 번호판 영역의 정답 대비 출력)
  --frames               1단계에서 처리한 프레임 수 (1단계 로그의 프레임 합계)

사용법:
  python step5_summary.py --data data --frames 185
  (4단계 결과 data/mosaic_check.csv가 있으면 함께 집계)
"""
import argparse
import csv
import os
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--frames", type=int, required=True)
args = ap.parse_args()


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


# 번호판 용도 기호로 쓰는 한글 (비사업용·사업용·택배·렌터카)
PLATE_HANGUL = set("가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주" "바사아자배" "하허호")


def valid_hangul(text):
    return all(ch in PLATE_HANGUL for ch in text if not ch.isdigit())


def hangul_only_error(pred, truth):
    # 길이와 숫자 자리가 모두 같고 한글 한 글자만 다른 경우
    if len(pred) != len(truth):
        return False
    diff = [(a, b) for a, b in zip(pred, truth) if a != b]
    return len(diff) == 1 and not diff[0][1].isdigit()


crops = load(os.path.join(args.data, "crops_both.csv"))
truth = {r["crop"]: r["truth"] for r in load(os.path.join(args.data, "crops_checked.csv"))}
yolo = [r for r in crops if r["method"] == "yolo"]
fb = [r for r in crops if r["method"] == "fallback"]

print("## 1. 검출 단계")
print(f"- 처리 프레임: {args.frames}")
print(f"- YOLOv11m 검출 영역: {len(yolo)}")
print(f"- 폴백 후보 영역(크기 필터 후): {len(fb)} (프레임당 평균 {len(fb) / args.frames:.1f})")
hit = [r for r in fb if r["paddle"] or r["easy"]]
print(f"- 두 OCR 중 하나라도 번호판 형식 문자열이 나온 영역: {len(hit)} ({len(hit) / len(fb):.1%})")
print()
print("## 2. 폴백 영역 OCR 결과 (번호판 형식 출력만 눈으로 확인)")
print("| 항목 | PaddleOCR | EasyOCR |")
print("|---|---|---|")
stat = {}
for eng in ("paddle", "easy"):
    out = [r for r in fb if r[eng]]
    ok = [r for r in out if r[eng] == truth.get(r["crop"])]
    bad = [r for r in out if r[eng] != truth.get(r["crop"])]
    stat[eng] = dict(out=len(out), ok=len(ok), uniq=len({truth[r["crop"]] for r in ok}),
                     bad=len(bad), hangul=sum(hangul_only_error(r[eng], truth[r["crop"]]) for r in bad),
                     invalid_bad=sum(not valid_hangul(r[eng]) for r in bad),
                     invalid_ok=sum(not valid_hangul(r[eng]) for r in ok))
for key, name in (("out", "번호판 형식 출력"), ("ok", "그중 정답"), ("bad", "그중 오답"),
                  ("hangul", "오답 중 한글 한 글자만 틀림"),
                  ("invalid_bad", "오답 중 번호판에 쓰지 않는 한글 포함"),
                  ("invalid_ok", "정답 중 위 검사에 걸리는 경우"), ("uniq", "정답 차량 수(중복 제외)")):
    print(f"| {name} | {stat['paddle'][key]} | {stat['easy'][key]} |")
both_ok = {truth[r["crop"]] for r in fb if r["paddle"] and r["paddle"] == truth.get(r["crop"])} | \
          {truth[r["crop"]] for r in fb if r["easy"] and r["easy"] == truth.get(r["crop"])}
print(f"\n- 두 OCR 합산 정답 차량 수: {len(both_ok)}")
print()

gt = load(os.path.join(args.data, "gt", "gt_both.csv"))
print("## 3. 직접 지정한 번호판 영역 OCR 정확도")
print(f"- 영역 {len(gt)}개, 차량 {len({r['label'] for r in gt})}대")
print("| 번호판 종류 | 영역 수 | PaddleOCR | EasyOCR |")
print("|---|---|---|---|")
by = defaultdict(Counter)
for r in gt:
    by[r["kind"]]["n"] += 1
    by[r["kind"]]["paddle"] += r["paddle"] == r["label"]
    by[r["kind"]]["easy"] += r["easy"] == r["label"]
for kind, c in by.items():
    print(f"| {kind} | {c['n']} | {c['paddle']} | {c['easy']} |")
tp = sum(c["paddle"] for c in by.values())
te = sum(c["easy"] for c in by.values())
print(f"| 합계 | {len(gt)} | {tp} ({tp / len(gt):.1%}) | {te} ({te / len(gt):.1%}) |")

mpath = os.path.join(args.data, "mosaic_check.csv")
if os.path.exists(mpath):
    m = load(mpath)
    print()
    print("## 4. 번호판 모자이크 여부에 따른 YOLO 검출 (4단계)")
    print(f"- 번호판 위치를 확인한 영역: {len(m)}")
    print("| 입력 | 원본 번호판 | 번호판 모자이크 | 대조군: 번호판 아닌 영역 모자이크 |")
    print("|---|---|---|---|")
    for k, name in (("frame", "전체 프레임"), ("crop", "차량 중심 영역")):
        ctrl = [o[k + "_control"] for o in m if o[k + "_control"] != ""]
        print(f"| {name} | {sum(o[k + '_orig'] == 'True' for o in m)} / {len(m)} "
              f"| {sum(o[k + '_mosaic'] == 'True' for o in m)} / {len(m)} "
              f"| {sum(c == 'True' for c in ctrl)} / {len(ctrl)} |")

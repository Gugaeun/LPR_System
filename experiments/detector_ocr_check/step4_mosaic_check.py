"""검출·OCR 검증 실험 4단계: 번호판 모자이크 여부에 따른 YOLO 검출 비교 (lpr_env에서 실행)

학습 데이터(AIHub)의 번호판은 개인정보 보호를 위해 모자이크 처리되어 있다.
실제 영상에서 YOLO가 번호판을 찾지 못한 원인이 이 차이인지 확인하기 위해,
같은 프레임의 번호판 영역만 모자이크한 뒤 검출 결과를 비교한다.

  1. 1·3단계 결과(crops_both.csv) 중 번호판 형식 문자열이 나온 영역에서 PaddleOCR 글자 상자로 번호판 위치를 찾는다.
  2. 원본 프레임과 번호판 모자이크 프레임에 앱과 같은 설정(conf 0.07)으로 YOLO를 실행한다.
  3. 검출 상자 중심이 번호판 영역 안에 있으면 검출 성공으로 센다.
  4. 대조군: 번호판은 원본으로 두고, 같은 크기의 번호판이 아닌 영역(번호판 위쪽 차체)만 모자이크해
     그 위치에서 검출되는지 확인한다. 검출되면 모델이 번호판이 아닌 모자이크 무늬 자체에 반응한다는 뜻이다.

사용법:
  python step4_mosaic_check.py --videos 영상폴더 --data data [--flip 회전필요.mp4]
"""
import argparse
import csv
import os

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--videos", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--files", nargs="*", default=None, help="사용할 영상 파일명(순서대로 v0, v1, ...). 없으면 폴더 전체를 이름순으로 사용")
ap.add_argument("--flip", nargs="*", default=[])
ap.add_argument("--blocks", type=int, default=8, help="모자이크 가로 블록 수")
args = ap.parse_args()
args.videos, args.data = os.path.abspath(args.videos), os.path.abspath(args.data)
C.wait_yolo()

rows = list(csv.DictReader(open(os.path.join(args.data, "crops_both.csv"), encoding="utf-8-sig")))
rows = [r for r in rows if r["paddle"] or r["easy"]]
videos = args.files or sorted(f for f in os.listdir(args.videos) if f.lower().endswith((".mp4", ".mov", ".avi")))
out = []
for r in rows:
    name = videos[int(r["video"][1:])]
    fr = C.load_frame(os.path.join(args.videos, name), int(r["frame"]), args.width, name in args.flip)
    if fr is None:
        continue
    x, y, cw, ch = (int(r[k]) for k in ("x", "y", "w", "h"))
    crop = fr[y:y + ch, x:x + cw]
    pb = C.plate_box(crop)
    if pb is None:
        continue
    fbox = (x + pb[0], y + pb[1], x + pb[2], y + pb[3])   # 프레임 좌표
    row = dict(crop=r["crop"],
               frame_orig=C.hits(fr, fbox), frame_mosaic=C.hits(C.mosaic(fr, fbox, args.blocks), fbox),
               crop_orig=C.hits(crop, pb), crop_mosaic=C.hits(C.mosaic(crop, pb, args.blocks), pb))
    cb = C.shifted(pb, crop.shape[0])
    row["crop_control"] = C.hits(C.mosaic(crop, cb, args.blocks), cb) if cb else ""
    fb = C.shifted(fbox, fr.shape[0])
    row["frame_control"] = C.hits(C.mosaic(fr, fb, args.blocks), fb) if fb else ""
    out.append(row)

with open(os.path.join(args.data, "mosaic_check.csv"), "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    wr.writeheader()
    wr.writerows(out)
print(f"번호판 위치 확인 영역: {len(out)}")
for k, name in (("frame", "전체 프레임"), ("crop", "차량 중심 영역")):
    ctrl = [o[k + "_control"] for o in out if o[k + "_control"] != ""]
    print(f"{name}: 원본 {sum(o[k + '_orig'] for o in out)} / 번호판 모자이크 {sum(o[k + '_mosaic'] for o in out)}"
          f" / 대조군(번호판 아닌 영역 모자이크) {sum(ctrl)} of {len(ctrl)}")

"""검출·OCR 검증 실험 1단계 (lpr_env에서 실행)

같은 영상 프레임에서 번호판 후보 영역을 두 경로로 잘라 저장하고 PaddleOCR로 읽는다.
  - yolo     : 학습한 YOLOv11m 검출 (앱의 detector._detect_yolo)
  - fallback : YOLO 없이 이진화·윤곽선·색상 기반 후보 검출 (앱의 폴백 경로)
처리 순서는 앱 pipeline.process_frame과 같다: 크기 필터 → 색 분류 → OCR 전처리 → read_plate

사용법:
  python step1_crops_paddle.py --videos 영상폴더 --out data [--flip 회전필요.mp4]
  결과: data/crops/*.png, data/crops_paddle.csv
"""
import argparse
import csv
import os
import sys
import time

import cv2

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lpr_system")
ap = argparse.ArgumentParser()
ap.add_argument("--videos", required=True, help="실험 영상 폴더")
ap.add_argument("--out", required=True, help="crop 이미지와 결과 CSV를 저장할 폴더")
ap.add_argument("--step", type=float, default=1.0, help="프레임 추출 간격(초)")
ap.add_argument("--width", type=int, default=1280, help="처리 전 프레임 가로 크기")
ap.add_argument("--files", nargs="*", default=None, help="사용할 영상 파일명(순서대로 v0, v1, ...). 없으면 폴더 전체를 이름순으로 사용")
ap.add_argument("--flip", nargs="*", default=[], help="180도 회전이 필요한 영상 파일명")
args = ap.parse_args()
args.videos, args.out = os.path.abspath(args.videos), os.path.abspath(args.out)
os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)

# 앱 모듈(detector, ocr_engine)을 수정 없이 그대로 불러온다
sys.path.insert(0, os.path.abspath(APP))
import detector as D  # noqa: E402
import ocr_engine as O  # noqa: E402

for _ in range(120):                       # YOLO는 import 시 별도 스레드에서 로드된다
    if D._YOLO_MODEL is not None:
        break
    time.sleep(0.5)
print("YOLO loaded:", D._YOLO_MODEL is not None, D._YOLO_DEVICE)
O.init_reader(gpu=False)
yolo_fn = D._detect_yolo


def size_filter(cands):
    # 앱과 같은 게이트 필터: 가장 큰 후보의 30% 미만 크기는 배경 차량으로 보고 제외
    if not cands:
        return cands
    m = max(b[2] * b[3] for _, b, _ in cands)
    return [c for c in cands if c[1][2] * c[1][3] >= m * 0.3]


rows = []
frames = 0
videos = args.files or sorted(f for f in os.listdir(args.videos) if f.lower().endswith((".mp4", ".mov", ".avi")))
for vi, name in enumerate(videos):
    vid = f"v{vi}"
    cap = cv2.VideoCapture(os.path.join(args.videos, name))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps * args.step)))
    for fi in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        frames += 1
        if name in args.flip:
            fr = cv2.rotate(fr, cv2.ROTATE_180)
        h, w = fr.shape[:2]
        fr = cv2.resize(fr, (args.width, int(h * args.width / w)))
        for method in ("yolo", "fallback"):
            if method == "yolo":
                cands = D._nms(yolo_fn(fr))
            else:
                D._detect_yolo = lambda f: []
                try:
                    cands = D.detect_plate_candidates(fr)
                finally:
                    D._detect_yolo = yolo_fn
            for k, (roi, bbox, cnt) in enumerate(size_filter(cands)):
                cid = f"{vid}_f{fi:05d}_{method}_{k}"
                cv2.imwrite(os.path.join(args.out, "crops", cid + ".png"), roi)
                color, ptype = D.classify_plate_color(roi)
                proc = D.preprocess_for_ocr(roi, color, cnt)
                text, conf = O.read_plate(proc, ptype, raw_roi=roi)
                rows.append(dict(crop=cid, video=vid, frame=fi, method=method,
                                 x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3],
                                 color=color, ptype=ptype, paddle=text, paddle_conf=round(conf, 3)))
    print(vid, name, "crops:", sum(1 for r in rows if r["video"] == vid))

with open(os.path.join(args.out, "crops_paddle.csv"), "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
print("frames", frames, "total crops", len(rows))

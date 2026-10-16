"""README용 비교 그림 생성 (lpr_env에서 실행, 1~4단계 결과 필요)

공개용 그림이라 읽을 수 있는 번호판은 끝 4자리(일련번호)를 검은 상자로 가린다.
가림은 YOLO 검출이 끝난 뒤 그림에만 적용하므로 실험 결과에는 영향이 없다.

사용법:
  python make_figures.py --videos 영상폴더 --data data --files 영상1.mp4 ... [--flip 회전필요.mp4]
  결과: images/*.png
"""
import argparse
import csv
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--videos", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--files", nargs="*", default=None)
ap.add_argument("--flip", nargs="*", default=[])
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"))
ap.add_argument("--crop", nargs="*", default=["v1_f00250_fallback_0"], help="차량 중심 영역 그림에 쓸 영역")
ap.add_argument("--frame", default="v4_f00250_fallback_1", help="전체 프레임 그림에 쓸 영역")
ap.add_argument("--fallback-frame", default="v3_f00075", help="폴백 후보 그림에 쓸 프레임(video_fNNNNN)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
C.wait_yolo()

FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22)
FONT_S = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 19)
GREEN, RED, GRAY = (34, 160, 70), (210, 50, 50), (90, 90, 90)
rows = {r["crop"]: r for r in csv.DictReader(open(os.path.join(args.data, "crops_both.csv"), encoding="utf-8-sig"))}
videos = args.files or sorted(f for f in os.listdir(args.videos) if f.lower().endswith((".mp4", ".mov", ".avi")))


def frame_of(vid, fno):
    name = videos[int(vid[1:])]
    return C.load_frame(os.path.join(args.videos, name), int(fno), 1280, name in args.flip)


def hide_serials(img, scale=2):
    # 공개용: 읽을 수 있는 번호의 끝 4자리를 검은 상자로 가림
    # 멀리 있는 작은 번호판까지 찾도록 2배로 키운 이미지에서 글자 상자를 찾는다
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    out = img.copy()
    for x0, y0, x1, y1 in C.serial_boxes(big):
        cv2.rectangle(out, (x0 // scale - 2, y0 // scale - 2), (x1 // scale + 2, y1 // scale + 2), (0, 0, 0), -1)
    return out


def view_box(img, boxes, mx=1.0, my=0.8):
    # 번호판과 대조군 영역이 함께 보이도록 둘을 합친 범위에 여백을 더한다
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    pw, ph = boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]
    h, w = img.shape[:2]
    return (max(0, int(x0 - pw * mx)), max(0, int(y0 - ph * my)), min(w, int(x1 + pw * mx)), min(h, int(y1 + ph * my)))


def draw_boxes(img, boxes, color=GREEN):
    out = img.copy()
    t = max(2, img.shape[1] // 300)
    for b in boxes:
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), color[::-1], t)
    return out


def panel(img, title, result, color, height=300):
    h, w = img.shape[:2]
    s = height / h
    body = Image.fromarray(cv2.cvtColor(cv2.resize(img, (int(w * s), height)), cv2.COLOR_BGR2RGB))
    pw = max(body.width, 260)
    p = Image.new("RGB", (pw, height + 84), "white")
    p.paste(body, ((pw - body.width) // 2, 40))
    d = ImageDraw.Draw(p)
    d.text((pw // 2, 20), title, font=FONT, fill=(30, 30, 30), anchor="mm")
    d.text((pw // 2, height + 62), result, font=FONT_S, fill=color, anchor="mm")
    return p


def row_of(panels, gap=16):
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    out = Image.new("RGB", (w, max(p.height for p in panels)), "white")
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def result_text(boxes, target, what):
    hit = [b for b in boxes if target[0] <= (b[0] + b[2]) / 2 <= target[2] and target[1] <= (b[1] + b[3]) / 2 <= target[3]]
    if hit:
        return f"{what} 검출 (신뢰도 {max(b[4] for b in hit):.2f})", GREEN, hit
    return "검출 없음", RED, []


def mosaic_figure(img, name):
    # 같은 이미지에 세 조건을 적용하고 YOLO 결과를 나란히 비교
    pb = C.plate_box(img)
    cb = C.shifted(pb, img.shape[0])
    conds = [("원본 번호판", img, pb, "번호판"),
             ("번호판만 모자이크", C.mosaic(img, pb), pb, "번호판 위치")]
    if cb:
        conds.append(("차체 일부만 모자이크", C.mosaic(img, cb), cb, "모자이크 위치"))
    vx0, vy0, vx1, vy1 = view_box(img, [pb] + ([cb] if cb else []))
    panels = []
    for title, im, target, what in conds:
        boxes = C.yolo_boxes(im)                      # 검출은 실험과 같은 영역 전체에서 실행
        text, color, hit = result_text(boxes, target, what)
        shown = draw_boxes(hide_serials(im), hit)[vy0:vy1, vx0:vx1]   # 표시만 확대
        panels.append(panel(shown, title, text, color))
    row_of(panels).save(os.path.join(args.out, name))
    print("saved", name)


# 1. 번호판 모자이크 비교: 차량 중심 영역
for i, cid in enumerate(args.crop):
    r = rows[cid]
    fr = frame_of(r["video"], r["frame"])
    x, y, w, h = (int(r[k]) for k in ("x", "y", "w", "h"))
    mosaic_figure(fr[y:y + h, x:x + w], f"mosaic_crop{'' if i == 0 else i}.png")

# 2. 번호판 모자이크 비교: 전체 프레임
r = rows[args.frame]
fr = frame_of(r["video"], r["frame"])
x, y, w, h = (int(r[k]) for k in ("x", "y", "w", "h"))
pb = C.plate_box(fr[y:y + h, x:x + w])
fbox = (x + pb[0], y + pb[1], x + pb[2], y + pb[3])
fcb = C.shifted(fbox, fr.shape[0])
panels = []
for title, im, target, what in (("원본 번호판", fr, fbox, "번호판"),
                                ("번호판만 모자이크", C.mosaic(fr, fbox), fbox, "번호판 위치"),
                                ("차체 일부만 모자이크", C.mosaic(fr, fcb), fcb, "모자이크 위치")):
    boxes = C.yolo_boxes(im)
    text, color, hit = result_text(boxes, target, what)
    panels.append(panel(draw_boxes(hide_serials(im), hit), title, text, color, height=240))
row_of(panels).save(os.path.join(args.out, "mosaic_frame.png"))
print("saved mosaic_frame.png")

# 3. 폴백 후보 영역: 한 프레임의 후보와 OCR 결과
vid, fno = args.fallback_frame.split("_f")
cands = [r for r in rows.values() if r["video"] == vid and int(r["frame"]) == int(fno) and r["method"] == "fallback"]
fr = frame_of(vid, int(fno))
shown = hide_serials(fr)
for r in cands:
    x, y, w, h = (int(r[k]) for k in ("x", "y", "w", "h"))
    ok = bool(r["paddle"] or r["easy"])
    cv2.rectangle(shown, (x, y), (x + w, y + h), (GREEN if ok else RED)[::-1], 3)
title = f"폴백 후보 영역 {len(cands)}개"
result = f"초록: 번호판 형식 문자열 나옴 {sum(bool(r['paddle'] or r['easy']) for r in cands)}개 / 빨강: 없음"
panel(shown, title, result, GRAY, height=420).save(os.path.join(args.out, "fallback_candidates.png"))
print("saved fallback_candidates.png")

# 4. OCR 비교 예시: 직접 지정한 번호판 영역 (가림 위치는 눈으로 확인해 비율로 지정)
OCR_EXAMPLES = {   # 파일명: (정답 영역 이미지, 가림 영역 x0, y0, x1, y1 비율)
    "ocr_white.png":   ("gt11", (0.55, 0.20, 0.91, 0.90)),
    "ocr_white2.png":  ("gt03", (0.39, 0.20, 0.86, 0.92)),
    "ocr_yellow.png":  ("gt17", (0.53, 0.10, 0.90, 0.85)),
    "ocr_yellow2.png": ("gt16", (0.30, 0.36, 0.90, 0.92)),
    "ocr_green2.png":  ("gt05", (0.27, 0.40, 0.86, 0.95)),
}
for name, (gid, (a, b, c, d)) in OCR_EXAMPLES.items():
    im = cv2.imread(os.path.join(args.data, "gt", "plates", gid + ".png"))
    h, w = im.shape[:2]
    cv2.rectangle(im, (int(w * a), int(h * b)), (int(w * c), int(h * d)), (0, 0, 0), -1)
    s = 110 / h
    cv2.imwrite(os.path.join(args.out, name), cv2.resize(im, (int(w * s), 110), interpolation=cv2.INTER_AREA))
    print("saved", name)

# 5. 한글 검사 예시: '사'를 '시'로 읽은 번호판 (두 OCR 모두 같은 오답)
r = rows["v4_f00025_fallback_0"]
fr = frame_of(r["video"], r["frame"])
x, y, w, h = (int(r[k]) for k in ("x", "y", "w", "h"))
crop = fr[y:y + h, x:x + w]
pb = C.plate_box(crop, margin=0.08)
plate = cv2.resize(crop[pb[1]:pb[3], pb[0]:pb[2]], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(args.out, "hangul_example.png"), hide_serials(plate))
print("saved hangul_example.png")

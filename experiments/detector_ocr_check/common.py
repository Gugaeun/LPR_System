"""4단계 실험과 그림 생성에서 함께 쓰는 함수 (lpr_env에서 실행)"""
import os
import re
import sys

import cv2
import numpy as np

APP = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lpr_system"))
# 앱 모듈(detector)을 수정 없이 그대로 불러온다
sys.path.insert(0, APP)
import detector as D  # noqa: E402

_OCR = None
_OCR_MASK = None


def wait_yolo(timeout_sec=60):
    # detector는 import 시 별도 스레드에서 YOLO를 불러온다
    import time
    for _ in range(int(timeout_sec * 2)):
        if D._YOLO_MODEL is not None:
            return
        time.sleep(0.5)


def ocr():
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(lang="korean", use_angle_cls=False, use_gpu=False, show_log=False)
    return _OCR


def ocr_mask():
    # 공개용 가림 전용: 기본값(긴 변 960px로 축소) 대신 원본 크기에 가깝게 검사해 작은 번호판까지 찾는다
    global _OCR_MASK
    if _OCR_MASK is None:
        from paddleocr import PaddleOCR
        _OCR_MASK = PaddleOCR(lang="korean", use_angle_cls=False, use_gpu=False, show_log=False,
                              det_limit_side_len=2600, det_db_box_thresh=0.3)
    return _OCR_MASK


def load_frame(video_path, frame_no, width=1280, flip=False):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, fr = cap.read()
    if not ok:
        return None
    if flip:
        fr = cv2.rotate(fr, cv2.ROTATE_180)
    h, w = fr.shape[:2]
    return cv2.resize(fr, (width, int(h * width / w)))


def text_boxes(img, engine=None):
    """PaddleOCR 글자 상자 [(x0, y0, x1, y1, text), ...]"""
    res = (engine or ocr()).ocr(img, cls=False)[0] or []
    out = []
    for b, (t, _) in res:
        p = np.array(b)
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        out.append((int(x0), int(y0), int(x1), int(y1), t))
    return out


def plate_box(img, margin=0.15):
    # 숫자가 들어간 글자 상자를 합쳐 번호판 위치로 본다(여백 15% 추가)
    boxes = [b for b in text_boxes(img) if any(ch.isdigit() for ch in b[4])]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    h, w = img.shape[:2]
    return (int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my)))


def mosaic(img, box, blocks=8):
    x0, y0, x1, y1 = box
    roi = img[y0:y1, x0:x1]
    bh = max(1, round(blocks * (y1 - y0) / max(1, x1 - x0)))
    small = cv2.resize(roi, (blocks, bh), interpolation=cv2.INTER_AREA)
    out = img.copy()
    out[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return out


def shifted(box, limit_h):
    # 번호판 높이의 2배만큼 위로 옮긴 같은 크기 영역(벗어나면 아래로)
    x0, y0, x1, y1 = box
    d = 2 * (y1 - y0)
    if y0 - d >= 0:
        return (x0, y0 - d, x1, y1 - d)
    if y1 + d <= limit_h:
        return (x0, y0 + d, x1, y1 + d)
    return None


def yolo_boxes(img):
    """앱과 같은 설정(conf 0.07)의 YOLO 검출 [(x0, y0, x1, y1, conf), ...]"""
    r = D._YOLO_MODEL(img, conf=D._YOLO_CONF, verbose=False, device=D._YOLO_DEVICE)[0]
    return [(*map(int, b), float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]


def hits(img, box):
    x0, y0, x1, y1 = box
    return any(x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1 for b in yolo_boxes(img))


def serial_boxes(img):
    """공개용 가림 영역
    - 번호 형식으로 읽힌 글자 상자: 끝 4자리(일련번호)가 있는 오른쪽 부분
    - 그 밖에 숫자가 3개 이상 읽힌 글자 상자(흐린 번호판 등): 상자 전체
    """
    out = []
    for x0, y0, x1, y1, t in text_boxes(img, ocr_mask()):
        clean = re.sub(r"\s", "", t)
        if re.search(r"\d{4}$", clean):
            ratio = min(1.0, 4 / len(clean) * 1.12)
            out.append((int(x1 - (x1 - x0) * ratio), y0, x1, y1))
        elif sum(ch.isdigit() for ch in clean) >= 3:
            out.append((x0, y0, x1, y1))
    return out

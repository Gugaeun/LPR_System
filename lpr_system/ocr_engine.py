"""
ocr_engine.py — PaddleOCR 기반 한국 번호판 문자 인식

PaddleOCR 2.7.x (lang='korean') 사용. GPU/CPU 런타임 전환 지원.
검출은 detector.py의 YOLO가 담당하고, 여기서는 크롭된 번호판 이미지의
문자만 읽는다.
"""

import re
import threading
from typing import Optional, Tuple

import numpy as np

try:
    from paddleocr import PaddleOCR as _PaddleOCR
    _PADDLE_AVAILABLE = True
except ImportError:
    _PaddleOCR        = None
    _PADDLE_AVAILABLE = False

# ── 번호판 정규식 ─────────────────────────────────────────────
PATTERN_NEW   = re.compile(r"(\d{2,3}[가-힣]\d{4})")
PATTERN_OLD   = re.compile(r"([가-힣]{2}\d{2}[가-힣]\d{4})")
PATTERN_SMALL = re.compile(r"(\d{2}[가-힣]\d{4})")

CONFIDENCE_THRESHOLD     = 0.20   # 일반 번호판
CONFIDENCE_THRESHOLD_OLD = 0.15   # 녹색(구형) 번호판

_READER:      Optional[object] = None
_INIT_LOCK:   threading.Lock   = threading.Lock()
_CURRENT_GPU: bool             = False   # 현재 로드된 Reader의 GPU 모드


def is_gpu_available() -> bool:
    """PaddlePaddle CUDA 사용 가능 여부."""
    try:
        import paddle
        return paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def init_reader(gpu: bool = False) -> None:
    """PaddleOCR Reader를 최초 1회 초기화 (멀티스레드 안전)."""
    global _READER, _CURRENT_GPU
    if not _PADDLE_AVAILABLE:
        print("[OCR] paddleocr 미설치 - pip install paddleocr")
        return
    with _INIT_LOCK:
        if _READER is None:
            mode = "GPU" if gpu else "CPU"
            print(f"[OCR] PaddleOCR 모델 로딩 중... ({mode})")
            _READER = _PaddleOCR(
                use_angle_cls=False, lang="korean",
                use_gpu=gpu, show_log=False,
            )
            _CURRENT_GPU = gpu
            print(f"[OCR] 준비 완료 ({mode})")


def reinit_reader(gpu: bool) -> None:
    """GPU 모드를 런타임에 전환. 기존 Reader를 해제하고 재초기화."""
    global _READER, _CURRENT_GPU
    with _INIT_LOCK:
        if _CURRENT_GPU == gpu and _READER is not None:
            return   # 이미 같은 모드
        _READER = None
        _CURRENT_GPU = gpu
    init_reader(gpu=gpu)


def read_plate(
    processed_img: np.ndarray,
    plate_type: str = "NEW",
    raw_roi: Optional[np.ndarray] = None,
) -> Tuple[str, float]:
    if _READER is None:
        return "", 0.0

    # PaddleOCR은 자체 전처리가 강력하므로 원본 crop을 우선 사용.
    # 단, crop이 극단적으로 작으면(예: 먼 차량) 원본엔 글자 정보가 거의 없어
    # PaddleOCR 자체 전처리로도 못 살리므로, 이 경우만 업스케일된 processed_img를 쓴다.
    _MIN_RAW_H = 32
    use_raw = (raw_roi is not None and raw_roi.size > 0
              and raw_roi.shape[0] >= _MIN_RAW_H)
    img = raw_roi if use_raw else processed_img
    if img is None or img.size == 0:
        return "", 0.0

    try:
        result = _READER.ocr(img, cls=False)
    except Exception as e:
        print(f"[OCR] 인식 오류: {e}")
        return "", 0.0

    if not result or not result[0]:
        return "", 0.0

    lines = result[0]   # [[box, (text, score)], ...]

    # bbox y좌표 기준으로 위→아래 정렬 (구식 2줄 번호판 대응)
    lines_sorted = sorted(lines, key=lambda ln: ln[0][0][1])

    thr = CONFIDENCE_THRESHOLD_OLD if plate_type == "OLD" else CONFIDENCE_THRESHOLD
    filtered = [
        (text, conf)
        for _, (text, conf) in lines_sorted
        if conf >= thr
    ]

    if not filtered:
        return "", 0.0

    avg_conf = sum(c for _, c in filtered) / len(filtered)
    cleaned  = re.sub(r"[^가-힣0-9]", "", "".join(t for t, _ in filtered).upper())

    result_new = _extract_plate_number(cleaned, plate_type, avg_conf)
    if result_new[0]:
        return result_new

    # 구식 번호판: 줄 순서 뒤집어 재시도 (서울12 + 가3456 순서가 뒤집힌 경우)
    if len(filtered) >= 2:
        reversed_cleaned = re.sub(
            r"[^가-힣0-9]", "",
            "".join(t for t, _ in reversed(filtered)).upper()
        )
        result_old = _extract_plate_number(reversed_cleaned, "OLD", avg_conf)
        if result_old[0]:
            return result_old

    return "", 0.0


def _extract_plate_number(
    cleaned: str,
    plate_type: str,
    confidence: float,
) -> Tuple[str, float]:
    # 지역명(서울·경기 등)은 제외하고 '핵심 번호'만 추출.
    for pattern in (PATTERN_NEW, PATTERN_SMALL):
        m = pattern.search(cleaned)
        if m:
            return m.group(), confidence

    # 구식 전체형만 잡히면 지역 2글자를 떼고 반환
    m = PATTERN_OLD.search(cleaned)
    if m:
        core = re.sub(r"^[가-힣]{2}", "", m.group())
        return core, confidence

    return "", 0.0

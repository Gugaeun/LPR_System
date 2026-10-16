"""
pipeline.py — 번호판 인식 파이프라인

  - 화이트리스트 교정: 같은 길이·편집거리 1·후보 1개일 때만 등록 번호로 교정
  - 한글 검사: 번호판 용도 기호(PLATE_HANGUL)에 없는 한글이 나오면 결과를 버림
  - 중복 제거: COOLDOWN_SEC(120초) 안에 편집거리 FUZZY_DIST(2) 이하 번호는 같은 차량으로 보고
    첫 기록만 유지
  - 투표 버퍼: VOTE_FRAMES 프레임을 모아 최다 득표 번호 확정 (현재 1 = 첫 인식 즉시 확정)
  - 스트림 자동 재연결, on_progress 예외 처리, None 프레임 가드
"""

import os
import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import Optional, Callable

import cv2
import numpy as np

from database import (
    get_connection, insert_plate_log,
    is_whitelisted, is_blacklisted, insert_entry_exit,
    get_whitelist_plates,
)
from detector import (
    has_motion, detect_plate_candidates,
    classify_plate_color, preprocess_for_ocr,
    release_bg_subtractor,
)
from ocr_engine import read_plate

# ── 파라미터 ──────────────────────────────────────────────────
PROCESS_EVERY_N = 1      # 매 프레임 OCR (옆으로 지나가며 찍는 정면 창이 짧아 1로)
COOLDOWN_SEC    = 120    # 동일 번호판 재저장 방지 (30→120초)
VOTE_FRAMES     = 1      # 투표 프레임 수 (1=첫 성공 즉시 확정, recall 우선)
VOTE_WIN_RATIO  = 0.4    # 이 비율 이상 득표 시 확정
VOTE_IOU_TH     = 0.3    # 같은 번호판으로 볼 bbox 겹침 임계값
FUZZY_DIST      = 2      # 편집거리 이내 → 같은 차량으로 간주
SAVE_IMAGES     = True

IMAGE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plate_images")
LOG_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
ENTRY_LOG  = os.path.join(LOG_DIR, "entry_timestamps.txt")

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


def _save_entry_log(plate_text: str, timestamp: str, camera_id: str, event: str) -> None:
    try:
        with open(ENTRY_LOG, "a", encoding="utf-8") as f:
            f.write(f"{timestamp}\t{event}\t{plate_text}\t{camera_id}\n")
    except Exception:
        pass


def _get_date_image_dir(timestamp: str) -> str:
    date_str = timestamp[:10]  # "YYYY-MM-DD"
    d = os.path.join(IMAGE_DIR, date_str)
    os.makedirs(d, exist_ok=True)
    return d

logging.basicConfig(
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f"system_{datetime.now():%Y%m%d}.log"),
            encoding="utf-8"),
        logging.StreamHandler(),
    ],
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  투표(Voting) 버퍼
# ══════════════════════════════════════════════════════════════

def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


class VoteBuffer:
    """
    같은 위치의 번호판 후보를 VOTE_FRAMES개 모아서
    최다 득표 텍스트를 확정 결과로 반환.

    위치 쿨다운:
      결과 확정 후 COOLDOWN_SEC 동안 같은 bbox 위치에서
      새 투표 슬롯 생성을 차단한다. 느린 카메라에서 동일 차량이
      2사이클 이상 발화되는 문제를 원천 차단.
    """
    def __init__(self):
        self._slots:     list[dict] = []
        self._confirmed: list[dict] = []  # [{"bbox": ..., "time": datetime}]
        self._lock = threading.Lock()

    def _find_slot(self, bbox) -> Optional[int]:
        for i, slot in enumerate(self._slots):
            if _bbox_iou(bbox, slot["bbox"]) > VOTE_IOU_TH:
                return i
        return None

    def _in_cooldown(self, bbox) -> bool:
        """bbox 위치가 최근 확정된 위치와 겹치면 True."""
        now = datetime.now()
        # 만료된 항목 정리
        self._confirmed = [
            c for c in self._confirmed
            if (now - c["time"]).total_seconds() < COOLDOWN_SEC
        ]
        return any(_bbox_iou(bbox, c["bbox"]) > VOTE_IOU_TH for c in self._confirmed)

    def feed(self, bbox, text: str, conf: float,
             roi: np.ndarray) -> Optional[dict]:
        with self._lock:
            # 이미 확정된 위치면 투표 슬롯 자체를 만들지 않음
            if self._in_cooldown(bbox):
                return None

            idx = self._find_slot(bbox)
            if idx is None:
                self._slots.append({
                    "bbox":  bbox,
                    "votes": defaultdict(int),
                    "rois":  {},
                    "confs": defaultdict(list),
                })
                idx = len(self._slots) - 1

            slot = self._slots[idx]
            slot["bbox"] = bbox
            slot["votes"][text] += 1
            slot["confs"][text].append(conf)
            if text not in slot["rois"]:
                slot["rois"][text] = roi

            total = sum(slot["votes"].values())
            if total < VOTE_FRAMES:
                return None

            best_text  = max(slot["votes"], key=slot["votes"].get)
            best_ratio = slot["votes"][best_text] / total
            self._slots.pop(idx)

            if best_ratio < VOTE_WIN_RATIO:
                logger.debug(f"투표 미달 ({best_ratio:.0%}): {dict(slot['votes'])}")
                return None

            # 확정 — 이 위치를 쿨다운에 등록
            self._confirmed.append({"bbox": bbox, "time": datetime.now()})

            avg_conf = sum(slot["confs"][best_text]) / len(slot["confs"][best_text])
            return {
                "text":       best_text,
                "confidence": avg_conf,
                "bbox":       bbox,
                "roi":        slot["rois"].get(best_text),
            }

    def clear(self):
        with self._lock:
            self._slots.clear()
            self._confirmed.clear()


_vote_buffers: dict[str, VoteBuffer] = {}
_vb_lock = threading.Lock()

def _get_vote_buffer(camera_id: str) -> VoteBuffer:
    with _vb_lock:
        if camera_id not in _vote_buffers:
            _vote_buffers[camera_id] = VoteBuffer()
        return _vote_buffers[camera_id]

def _release_vote_buffer(camera_id: str):
    with _vb_lock:
        _vote_buffers.pop(camera_id, None)


# ══════════════════════════════════════════════════════════════
#  Levenshtein 편집거리 기반 fuzzy 중복 제거
# ══════════════════════════════════════════════════════════════

def _levenshtein(a: str, b: str) -> int:
    if a == b:   return 0
    if not a:    return len(b)
    if not b:    return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j]+1, curr[j-1]+1, prev[j-1]+(ca != cb)))
        prev = curr
    return prev[-1]


# ══════════════════════════════════════════════════════════════
#  등록차량(whitelist) 스냅 보정
#  OCR 오독(225→25, 루→로 등)을, 등록된 번호와 편집거리 1 이내로
#  '유일하게' 일치하면 그 번호로 교정한다. 미등록 차량은 건드리지 않음.
# ══════════════════════════════════════════════════════════════
SNAP_DIST = 1                     # 이 편집거리 이내면 등록번호로 스냅
_WL_CACHE: dict = {"plates": [], "time": None}
_WL_TTL_SEC = 60                  # 등록목록 캐시 갱신 주기
_wl_lock = threading.Lock()

# 같은 교정 로그가 매 프레임 도배되지 않도록 throttle (초)
SNAP_LOG_COOLDOWN = 60
_snap_log_seen: dict = {}         # {(오독, 교정): 마지막 로그 시각}


def _get_whitelist_cached(conn) -> list:
    now = datetime.now()
    with _wl_lock:
        t = _WL_CACHE["time"]
        if t is None or (now - t).total_seconds() > _WL_TTL_SEC:
            try:
                _WL_CACHE["plates"] = get_whitelist_plates(conn)
                _WL_CACHE["time"]   = now
            except Exception:
                pass
        return _WL_CACHE["plates"]


def _snap_to_whitelist(plate_text: str, conn) -> str:
    """
    등록번호와 '같은 길이의 한 글자 치환'(루↔로 등)일 때만 그 번호로 교정.

    길이가 다른 경우(225↔25처럼 숫자 추가/삭제)는 교정하지 않는다:
      진짜 2자리 구형판이 등록된 3자리 신형판으로 오스냅되어 엉뚱한 차로
      인식되는 사고를 막기 위함. (치환 오독은 다른 실차와 충돌 위험이 낮음)
    """
    plates = _get_whitelist_cached(conn)
    if not plates or plate_text in plates:
        return plate_text
    cands = [
        p for p in plates
        if len(p) == len(plate_text) and _levenshtein(plate_text, p) <= SNAP_DIST
    ]
    if len(cands) == 1:           # 후보가 둘 이상이면 모호 → 교정 안 함
        # 같은 교정은 SNAP_LOG_COOLDOWN 초에 한 번만 로그 (교정 자체는 매번 수행)
        key  = (plate_text, cands[0])
        now  = datetime.now()
        last = _snap_log_seen.get(key)
        if last is None or (now - last).total_seconds() >= SNAP_LOG_COOLDOWN:
            logger.info(f"[SNAP] '{plate_text}' → 등록번호 '{cands[0]}' 교정")
            _snap_log_seen[key] = now
        return cands[0]
    return plate_text


# ══════════════════════════════════════════════════════════════
#  번호판 한글 검사
#  번호판 용도 기호로 쓰지 않는 한글('시', '리' 등)이 나오면 오인식으로 보고 버린다.
#  영상 모드는 같은 추적 ID를 0.5초 뒤 다시 읽고, 카메라 모드는 다음 프레임에서 다시 읽는다.
#  실험 결과: 오답 12건 중 10건 제외, 정답 제외 0건
#  (experiments/detector_ocr_check)
# ══════════════════════════════════════════════════════════════
PLATE_HANGUL = frozenset(
    "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주"  # 비사업용
    "바사아자배"                                                        # 사업용·택배
    "하허호"                                                            # 렌터카
)


def _valid_plate_hangul(plate_text: str, conn) -> bool:
    """번호의 한글이 용도 기호 목록에 있거나, 등록차량 번호와 정확히 같으면 True."""
    if all(ch in PLATE_HANGUL for ch in plate_text if not ch.isdigit()):
        return True
    return plate_text in _get_whitelist_cached(conn)


# {키: {"time": datetime, "conf": float}}
_last_seen: dict[str, dict] = {}
_ls_lock   = threading.Lock()


def _check_and_update(plate_text: str, confidence: float) -> tuple[bool, str, str]:
    """
    (중복여부, 최종채택번호판, 교체된_이전번호판) 반환.
    - 신규          → (False, plate_text, "")            저장
    - 기존보다 신뢰도 높음 → (False, plate_text, old_key)  교체 저장
    - 기존보다 신뢰도 낮음 → (True,  "", "")              무시
    """
    now = datetime.now()
    with _ls_lock:
        # 만료 항목 정리
        if len(_last_seen) > 200:
            expired = [k for k, v in _last_seen.items()
                       if (now - v["time"]).total_seconds() > COOLDOWN_SEC * 5]
            for k in expired:
                del _last_seen[k]

        # 편집거리 FUZZY_DIST 이내 + 쿨다운 이내 항목 탐색
        similar_key = None
        for key, info in _last_seen.items():
            if (now - info["time"]).total_seconds() >= COOLDOWN_SEC:
                continue
            if _levenshtein(plate_text, key) <= FUZZY_DIST:
                similar_key = key
                break

        if similar_key is None:
            _last_seen[plate_text] = {"time": now, "conf": confidence}
            return False, plate_text, ""

        # 쿨다운 내 동일/유사 번호판은 신뢰도 무관하게 중복 처리
        # (신뢰도가 높아도 새 로그를 찍지 않음 — 첫 인식이 기준)
        prev_conf = _last_seen[similar_key]["conf"]
        if confidence > prev_conf:
            _last_seen[similar_key]["conf"] = confidence  # 신뢰도만 갱신
        logger.debug(f"[DEDUP] '{plate_text}'({confidence:.2f}) 쿨다운 내 중복 → 무시")
        return True, "", ""


# ── 중복 실행 방지 ─────────────────────────────────────────────
_video_lock    = threading.Lock()
_video_running: dict[str, bool] = {}


# ══════════════════════════════════════════════════════════════
#  단일 프레임 처리 (투표 버퍼 경유)
# ══════════════════════════════════════════════════════════════

def process_frame(
    frame:        np.ndarray,
    frame_number: int,
    camera_id:    str = "CAM_01",
    direction:    str = "BOTH",
    source_file:  str = "",
    conn=None,
    on_detected:  Optional[Callable] = None,
) -> list[dict]:

    if frame is None or frame.size == 0:
        return []

    _close_conn = conn is None
    if _close_conn:
        conn = get_connection()

    vbuf        = _get_vote_buffer(camera_id)
    results_out = []

    try:
        # ── 게이트 모드: 가장 큰(가까운) 번호판 위주로 처리 ──────────
        #   입구 카메라는 대상 차가 화면에서 제일 크게 잡힌다. 배경에 주차된
        #   다른 차들은 작게 잡히므로, 최대 박스의 30% 미만 크기는 버려서
        #   배경 차 오인식을 제거한다. (고정 크기 문턱이 아니라 프레임 내
        #   상대 크기라 멀리서 오는 흰판 recall은 유지)
        cands = detect_plate_candidates(frame)
        if cands:
            max_area = max(b[2] * b[3] for _, b, _ in cands)
            before = len(cands)
            cands = [c for c in cands if (c[1][2] * c[1][3]) >= max_area * 0.3]
            if len(cands) < before:
                logger.debug(f"[{camera_id}] 크기 필터: {before} → {len(cands)}개")

        for plate_roi, bbox, contour in cands:
            plate_color, plate_type = classify_plate_color(plate_roi)
            logger.debug(f"[{camera_id}] 후보 bbox={bbox} color={plate_color} type={plate_type}")
            processed               = preprocess_for_ocr(plate_roi, plate_color, contour)
            plate_text, confidence  = read_plate(processed, plate_type, raw_roi=plate_roi)
            logger.debug(f"[{camera_id}] OCR 결과: '{plate_text}' conf={confidence:.2f}")

            if not plate_text:
                continue

            # ── 등록차량 스냅 보정 (225→25, 루→로 등 OCR 오독 교정) ──
            plate_text = _snap_to_whitelist(plate_text, conn)

            # ── 번호판에 쓰지 않는 한글이면 버림 ──────────────────
            if not _valid_plate_hangul(plate_text, conn):
                logger.debug(f"[{camera_id}] 한글 검사 제외: '{plate_text}'")
                continue

            # ── 투표 버퍼 ──────────────────────────────────────
            confirmed = vbuf.feed(bbox, plate_text, confidence, plate_roi)
            if confirmed is None:
                continue

            final_text = confirmed["text"]
            final_conf = confirmed["confidence"]
            final_roi  = confirmed["roi"]
            final_bbox = confirmed["bbox"]

            # ── fuzzy 중복 체크 ────────────────────────────────
            is_dup, accepted, replaced = _check_and_update(final_text, final_conf)
            if is_dup:
                continue
            final_text = accepted
            if replaced:
                # 쿨다운 창 이내 레코드만 삭제 (과거 정상 기록은 보존)
                conn.execute(
                    "DELETE FROM plate_logs WHERE plate_text=? "
                    "AND timestamp >= datetime('now', 'localtime', ? || ' seconds')",
                    (replaced, f"-{COOLDOWN_SEC * 2}"),
                )
                conn.commit()
                logger.debug(f"[DEDUP] DB에서 최근 '{replaced}' 레코드 삭제")

            timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            image_path = ""

            if SAVE_IMAGES and final_roi is not None:
                safe_ts    = timestamp.replace(":", "-").replace(" ", "_")
                date_dir   = _get_date_image_dir(timestamp)
                image_path = os.path.join(date_dir, f"{final_text}_{safe_ts}.jpg")
                cv2.imwrite(image_path, final_roi)

            insert_plate_log(
                conn, final_text, plate_color, plate_type, final_conf,
                timestamp, camera_id, image_path, frame_number, source_file,
            )

            whitelisted = is_whitelisted(conn, final_text)
            blacklisted = is_blacklisted(conn, final_text)

            if direction == "BOTH":
                # 마지막 이벤트가 ENTRY면 EXIT, 아니면(없거나 EXIT이면) ENTRY
                last = conn.execute(
                    "SELECT event_type FROM entry_exit_log WHERE plate_text=? ORDER BY id DESC LIMIT 1",
                    (final_text,)
                ).fetchone()
                event = "EXIT" if last and last[0] == "ENTRY" else "ENTRY"
            else:
                event = direction  # "ENTRY" or "EXIT"
            insert_entry_exit(conn, final_text, event, camera_id, image_path)
            _save_entry_log(final_text, timestamp, camera_id, event)

            status = "BLACKLIST" if blacklisted else ("GRANTED" if whitelisted else "UNKNOWN")
            logger.info(
                f"[{camera_id}] #{frame_number} | {final_text} | "
                f"{plate_type}({plate_color}) | conf={final_conf:.2f} | vote={VOTE_FRAMES}f | {status}"
            )

            result = dict(
                plate_text=final_text, plate_color=plate_color, plate_type=plate_type,
                confidence=final_conf, timestamp=timestamp,
                bbox=final_bbox, roi=final_roi,
                whitelisted=whitelisted, blacklisted=blacklisted,
                status=status, image_path=image_path, camera_id=camera_id,
            )
            results_out.append(result)

            if on_detected:
                try:
                    on_detected(result)
                except Exception as e:
                    logger.warning(f"UI 콜백 오류: {e}")

    finally:
        if _close_conn:
            conn.close()

    return results_out


# ══════════════════════════════════════════════════════════════
#  Track 기반 단일 crop 처리 (영상 UI 실시간 추적용)
#    검출/추적은 detector.track_plates가 담당하고, 여기서는 그 crop
#    하나를 OCR→DB 처리한다. process_frame의 후반 로직과 동일하되
#    단일 roi 대상. (박스 렌더링과 OCR을 분리하기 위한 진입점)
# ══════════════════════════════════════════════════════════════

def process_tracked_roi(
    plate_roi:    np.ndarray,
    bbox:         tuple,
    camera_id:    str = "CAM_01",
    direction:    str = "BOTH",
    source_file:  str = "",
    conn=None,
    frame_number: int = 0,
) -> Optional[dict]:
    """track으로 얻은 단일 번호판 crop을 OCR→DB 처리. 결과 dict 또는 None."""
    if plate_roi is None or plate_roi.size == 0:
        return None

    _close_conn = conn is None
    if _close_conn:
        conn = get_connection()

    vbuf = _get_vote_buffer(camera_id)
    try:
        plate_color, plate_type = classify_plate_color(plate_roi)
        processed               = preprocess_for_ocr(plate_roi, plate_color, None)
        plate_text, confidence  = read_plate(processed, plate_type, raw_roi=plate_roi)
        if not plate_text:
            return None

        plate_text = _snap_to_whitelist(plate_text, conn)
        if not _valid_plate_hangul(plate_text, conn):
            logger.debug(f"[{camera_id}] 한글 검사 제외: '{plate_text}'")
            return None

        confirmed = vbuf.feed(bbox, plate_text, confidence, plate_roi)
        if confirmed is None:
            return None

        final_text = confirmed["text"]
        final_conf = confirmed["confidence"]
        final_roi  = confirmed["roi"]
        final_bbox = confirmed["bbox"]

        is_dup, accepted, replaced = _check_and_update(final_text, final_conf)
        if is_dup:
            return None
        final_text = accepted
        if replaced:
            conn.execute(
                "DELETE FROM plate_logs WHERE plate_text=? "
                "AND timestamp >= datetime('now', 'localtime', ? || ' seconds')",
                (replaced, f"-{COOLDOWN_SEC * 2}"),
            )
            conn.commit()

        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_path = ""
        if SAVE_IMAGES and final_roi is not None:
            safe_ts    = timestamp.replace(":", "-").replace(" ", "_")
            date_dir   = _get_date_image_dir(timestamp)
            image_path = os.path.join(date_dir, f"{final_text}_{safe_ts}.jpg")
            cv2.imwrite(image_path, final_roi)

        insert_plate_log(
            conn, final_text, plate_color, plate_type, final_conf,
            timestamp, camera_id, image_path, frame_number, source_file,
        )

        whitelisted = is_whitelisted(conn, final_text)
        blacklisted = is_blacklisted(conn, final_text)

        if direction == "BOTH":
            last = conn.execute(
                "SELECT event_type FROM entry_exit_log WHERE plate_text=? ORDER BY id DESC LIMIT 1",
                (final_text,)
            ).fetchone()
            event = "EXIT" if last and last[0] == "ENTRY" else "ENTRY"
        else:
            event = direction
        insert_entry_exit(conn, final_text, event, camera_id, image_path)
        _save_entry_log(final_text, timestamp, camera_id, event)

        status = "BLACKLIST" if blacklisted else ("GRANTED" if whitelisted else "UNKNOWN")
        logger.info(
            f"[{camera_id}] #{frame_number} | {final_text} | "
            f"{plate_type}({plate_color}) | conf={final_conf:.2f} | track | {status}"
        )

        return dict(
            plate_text=final_text, plate_color=plate_color, plate_type=plate_type,
            confidence=final_conf, timestamp=timestamp,
            bbox=final_bbox, roi=final_roi,
            whitelisted=whitelisted, blacklisted=blacklisted,
            status=status, image_path=image_path, camera_id=camera_id,
        )
    finally:
        if _close_conn:
            conn.close()


# ══════════════════════════════════════════════════════════════
#  영상 파일 처리
# ══════════════════════════════════════════════════════════════

def process_video(
    video_path:  str,
    camera_id:   str = "CAM_01",
    direction:   str = "BOTH",
    stop_event:  Optional[threading.Event] = None,
    on_detected: Optional[Callable] = None,
    on_progress: Optional[Callable] = None,
) -> int:
    with _video_lock:
        if _video_running.get(camera_id):
            logger.warning(f"[{camera_id}] 이미 처리 중")
            return 0
        _video_running[camera_id] = True

    if not os.path.isfile(video_path):
        logger.error(f"파일 없음: {video_path}")
        with _video_lock:
            _video_running[camera_id] = False
        return 0

    conn = get_connection()
    cap  = None
    try:
        cap   = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"영상 열기 실패: {video_path}")

        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fc       = 0
        detected = 0

        logger.info(f"영상 처리 시작: {os.path.basename(video_path)} | {total}프레임")

        while True:
            if stop_event and stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break

            fc += 1
            if fc % PROCESS_EVERY_N != 0:
                continue

            results  = process_frame(frame, fc, camera_id, direction,
                                     os.path.basename(video_path), conn, on_detected)
            detected += len(results)

            if on_progress:
                try:
                    on_progress(fc, total, detected)
                except Exception as e:
                    logger.warning(f"progress 콜백 오류: {e}")

        logger.info(f"영상 처리 완료: {detected}건")
        return detected

    except Exception as e:
        logger.exception(f"영상 처리 오류: {e}")
        return 0
    finally:
        if cap:
            cap.release()
        conn.close()
        with _video_lock:
            _video_running[camera_id] = False


# ══════════════════════════════════════════════════════════════
#  실시간 스트림 (자동 재연결 포함)
# ══════════════════════════════════════════════════════════════
_STREAM_RETRY_SEC = 5


def process_stream(
    source:      str,
    camera_id:   str = "CAM_01",
    direction:   str = "BOTH",
    stop_event:  Optional[threading.Event] = None,
    on_detected: Optional[Callable] = None,
    on_frame:    Optional[Callable] = None,
) -> None:
    import time
    src = int(source) if source.isdigit() else source
    logger.info(f"스트림 시작: {source} ({camera_id})")

    while not (stop_event and stop_event.is_set()):
        conn = get_connection()
        cap  = cv2.VideoCapture(src)

        # 웹캠은 기본 640x480으로 열림 → 번호판 글자 해상도 확보 위해 1080p 요청.
        #   (지원 안 하면 카메라가 가능한 최대 근사치로 맞춰줌. RTSP는 무시됨.)
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not cap.isOpened():
            logger.error(f"스트림 열기 실패: {source} — {_STREAM_RETRY_SEC}초 후 재시도")
            cap.release()
            conn.close()
            time.sleep(_STREAM_RETRY_SEC)
            continue

        _aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        _ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"[{camera_id}] 캡처 해상도: {_aw}x{_ah}")

        fc = 0
        pending = None
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=1)

        def _yolo_task(f, f_num):
            process_frame(f, f_num, camera_id, direction, source, conn, on_detected)

        try:
            while not (stop_event and stop_event.is_set()):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[{camera_id}] 프레임 끊김 — 재연결 시도")
                    break

                fc += 1

                # UI 콜백 — 매 프레임, 비블로킹
                if on_frame:
                    try:
                        on_frame(frame.copy())
                    except Exception:
                        pass

                # YOLO 추론 — 이전 추론 끝났을 때만 새로 제출 (드랍 방식)
                if fc % PROCESS_EVERY_N == 0:
                    if pending is None or pending.done():
                        pending = executor.submit(_yolo_task, frame.copy(), fc)

        finally:
            if pending is not None:
                pending.cancel()
            executor.shutdown(wait=False)
            cap.release()
            conn.close()

        if stop_event and stop_event.is_set():
            break
        time.sleep(_STREAM_RETRY_SEC)

    _get_vote_buffer(camera_id).clear()
    _release_vote_buffer(camera_id)
    release_bg_subtractor(camera_id)
    logger.info(f"스트림 종료: {source}")


def start_camera_thread(
    source:      str,
    camera_id:   str = "CAM_01",
    direction:   str = "BOTH",
    on_detected: Optional[Callable] = None,
    on_frame:    Optional[Callable] = None,
) -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    t = threading.Thread(
        target=process_stream,
        args=(source, camera_id, direction, stop_event, on_detected, on_frame),
        daemon=True, name=camera_id,
    )
    t.start()
    return t, stop_event

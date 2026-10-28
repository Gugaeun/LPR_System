"""
app_pyqt6.py - PyQt6 기반 번호판 인식 시스템 UI (디자인 개선판)
app_pyqt6.py - PyQt6 기반 번호판 인식 시스템 UI (디자인 개선판)
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import qtawesome as qta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import (
    init_db, get_connection,
    query_all, query_by_plate, query_by_date,
    query_total_count, query_today_count,
    query_by_date_range, query_entry_exit_range,
    query_hourly_stats, query_entry_exit_hourly,
    get_all_whitelist, add_whitelist, delete_whitelist,
    get_all_blacklist, add_blacklist, delete_blacklist,
    get_all_cameras, upsert_camera, delete_camera,
    get_current_parked,
    delete_plate_log,
    delete_entry_exit_by_text, manual_exit, get_session_times,
    is_whitelisted,
    query_overstay, has_overstay_log, insert_overstay_log, get_all_overstay,
)
from pipeline import process_frame as _pf, _release_vote_buffer, start_camera_thread
import pipeline as pl
from ocr_engine import init_reader, is_gpu_available, reinit_reader
from excel_export import export_plate_logs, export_entry_exit
from parking_fee import calc_fee, FEE_CONFIG

# 불법 주정차 감지 기준 — 미등록(비회원) 차량이 이 시간(분) 이상 연속 체류하면 의심 등록
OVERSTAY_THRESHOLD_MIN = 60

# ── 색상 팔레트 ─────────────────────────────────────────────────────────────────────
C_BG      = "#F1F5F9"
C_SURFACE = "#FFFFFF"
C_ACCENT  = "#3B82F6"
C_GREEN   = "#10B981"
C_RED     = "#EF4444"
C_YELLOW  = "#F59E0B"
C_TEXT    = "#1E293B"
C_MUTED   = "#64748B"
C_BORDER  = "#E2E8F0"
C_SIDEBAR = "#0F2444"
C_SIDEBAR_ACTIVE = "#1D4ED8"

STYLE_SHEET = f"""
QMainWindow, QWidget#MainBG {{ background-color: {C_BG}; }}
QWidget {{ background-color: transparent; }}
QFrame#Card {{
    background-color: {C_SURFACE};
    border: none;
    border-radius: 10px;
}}
QFrame#RedCard {{
    background-color: #FEF2F2;
    border: none;
    border-radius: 10px;
}}
QFrame#AlertItem {{
    background-color: #FEF2F2;
    border: none;
    border-radius: 8px;
}}
QFrame#InfoBanner {{
    background-color: #EFF6FF;
    border: none;
    border-radius: 8px;
}}
QLabel {{ color: {C_TEXT}; font-size: 19px; }}
QLabel#HeaderTitle {{ font-size: 30px; font-weight: bold; color: {C_TEXT}; }}
QLabel#SectionTitle {{ font-size: 25px; font-weight: bold; color: {C_TEXT}; }}
QLabel#Muted {{ color: {C_MUTED}; font-size: 19px; }}
QLabel#ColHeader {{ color: {C_MUTED}; font-size: 19px; font-weight: bold; }}
QLabel#StatValue {{ font-size: 38px; font-weight: bold; color: {C_TEXT}; }}
QLabel#FeeAmount {{ font-size: 32px; font-weight: bold; }}
QPushButton {{
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
    font-size: 19px;
    border: none;
}}
QPushButton#Primary {{ background-color: {C_ACCENT}; color: white; }}
QPushButton#Primary:hover {{ background-color: #2563EB; }}
QPushButton#Primary:disabled {{ background-color: {C_MUTED}; }}
QPushButton#Success {{ background-color: {C_GREEN}; color: white; }}
QPushButton#Success:hover {{ background-color: #059669; }}
QPushButton#Danger  {{ background-color: {C_RED}; color: white; }}
QPushButton#Danger:hover {{ background-color: #DC2626; }}
QPushButton#Warning {{ background-color: {C_YELLOW}; color: white; }}
QPushButton#Muted {{ background-color: {C_MUTED}; color: white; }}
QPushButton#Outline {{
    background-color: {C_SIDEBAR_ACTIVE};
    color: white;
    border: none;
}}
QPushButton#Outline:hover {{ background-color: #0284C7; }}
QPushButton#Ghost {{
    background-color: transparent;
    color: {C_ACCENT};
    border: none;
    font-weight: normal;
}}
QPushButton#Ghost:hover {{ background-color: {C_ACCENT}18; }}
QLineEdit, QComboBox {{
    padding: 8px 12px;
    border: none;
    border-radius: 6px;
    background-color: {C_BG};
    color: {C_TEXT};
    font-size: 19px;
}}
QLineEdit:focus, QComboBox:focus {{
    background-color: #E8F0FE;
    outline: none;
}}
QComboBox::drop-down {{ border: none; padding-right: 8px; }}
QScrollArea {{ border: none; background-color: transparent; }}
QScrollBar:vertical {{
    background: {C_BG}; width: 8px; border-radius: 4px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C_MUTED}55; border-radius: 4px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {C_MUTED}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QTableWidget {{
    border: none;
    border-radius: 8px;
    background-color: {C_SURFACE};
    gridline-color: {C_BG};
    selection-background-color: {C_ACCENT}22;
    font-size: 19px;
}}
QHeaderView::section {{
    background-color: #0F2444;
    color: #CBD5E1;
    border: none;
    font-weight: bold;
    font-size: 19px;
    padding: 10px 14px;
}}
QTableWidget::item {{ padding: 10px 14px; color: {C_TEXT}; }}
QTableWidget::item:selected {{ color: {C_TEXT}; background-color: {C_ACCENT}22; }}
QProgressBar {{
    border: none;
    border-radius: 4px;
    background-color: {C_BG};
    height: 10px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {C_ACCENT};
    border-radius: 4px;
}}
QCheckBox {{ color: {C_TEXT}; font-size: 19px; spacing: 8px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 4px; border: none;
    background-color: {C_BG};
}}
QCheckBox::indicator:checked {{
    background-color: {C_ACCENT};
}}
QStatusBar {{ background-color: {C_SURFACE}; border: none; font-size: 19px; }}
QFrame#Sidebar {{
    background-color: {C_SIDEBAR};
    border: none;
}}
QToolButton {{
    background-color: transparent;
    color: #475569;
    border: none;
    border-radius: 8px;
    font-size: 18px;
    padding: 6px 0px;
    qproperty-toolButtonStyle: ToolButtonTextBesideIcon;
}}
QToolButton:hover {{
    background-color: rgba(255,255,255,0.10);
    color: #CBD5E1;
}}
"""


# ── PIL 한국어 텍스트 오버레이 ──────────────────────────────────────────────────────
_KR_FONT_CACHE: dict = {}

def _get_kr_font(size: int):
    """크기별로 단 한번만 로드하고 캐싱."""
    if size in _KR_FONT_CACHE:
        return _KR_FONT_CACHE[size]
    try:
        from PIL import ImageFont
        candidates = [
            os.path.join(os.path.dirname(__file__), "assets", "fonts", "Pretendard-Bold.otf"),
            "C:/Windows/Fonts/Pretendard-Bold.otf",
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/gulim.ttc",
            "C:/Windows/Fonts/batang.ttc",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        ]
        for path in candidates:
            try:
                font = ImageFont.truetype(path, size)
                _KR_FONT_CACHE[size] = font
                return font
            except Exception:
                pass
        font = ImageFont.load_default()
        _KR_FONT_CACHE[size] = font
        return font
    except ImportError:
        return None


def put_text_kr(
    img: np.ndarray,
    text: str,
    x: int, y: int,
    font_size: int = 26,
    text_color=(255, 255, 255),   # BGR
    bg_color=(34, 197, 94),       # BGR
) -> np.ndarray:
    """
    PIL을 사용한 한국어 포함 텍스트를 프레임에 오버레이.
    Pillow 미설치 시 cv2.putText(ASCII만)로 대체.
    """
    try:
        from PIL import Image, ImageDraw

        font = _get_kr_font(font_size)
        if font is None:
            raise ImportError

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(pil_img)

        bbox = draw.textbbox((0, 0), text, font=font)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]

        # 텍스트 크기 (텍스트 배경 계산)
        bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
        draw.rectangle([x, y - th - 10, x + tw + 10, y], fill=bg_rgb)

        fg_rgb = (text_color[2], text_color[1], text_color[0])
        draw.text((x + 5, y - th - 5), text, font=font, fill=fg_rgb)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    except Exception:
        # PIL 사용 불가 시 ASCII fallback
        cv2.putText(img, text, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (text_color[2], text_color[1], text_color[0]), 2, cv2.LINE_AA)
        return img


# ── 헬퍼 함수 ───────────────────────────────────────────────────────────────────────
def status_color(status):
    return {"GRANTED": C_GREEN, "BLACKLIST": C_RED, "UNKNOWN": C_YELLOW}.get(status, C_MUTED)

def create_shadow(blur=8, opacity=18, dy=2):
    s = QGraphicsDropShadowEffect()
    s.setBlurRadius(blur)
    s.setColor(QColor(0, 0, 0, opacity))
    s.setOffset(0, dy)
    return s

def cv2_to_qimage(img):
    if img is None or img.size == 0:
        return QImage()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().setParent(None)
        elif item.layout():
            clear_layout(item.layout())

_TAG_WIDTH  = 84   # 미등록/등록/차단/입차/출차/신형/구형 등 모든 뱃지·태그 공통 폭
                   # (텍스트 글자 수와 무관하게 같은 크기로 보이도록 고정)
_TAG_HEIGHT = 36   # "삭제"/"중지"/"시작" 등 옆에 나란히 오는 버튼과 높이를 맞춤

def make_tag(text, color):
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background-color: {color}; color: white; border-radius: 4px; "
        f"font-weight: bold; font-size: 18px;"
    )
    lbl.setFixedSize(_TAG_WIDTH, _TAG_HEIGHT)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return lbl

def make_badge(text, color):
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background-color: {color}; color: white; border-radius: 4px; "
        f"font-size: 18px; font-weight: bold;"
    )
    lbl.setFixedSize(_TAG_WIDTH, _TAG_HEIGHT)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return lbl

# ── 사이드바 네비 버튼 (아이콘 좌측 고정 + 텍스트 우측, 수동 레이아웃) ──────────
# QToolButton의 BesideIcon 스타일은 Qt 스타일시트와 결합하면 아이콘·텍스트가
# 겹치거나 잘리는 문제가 있어, 직접 QHBoxLayout으로 배치해 안정적으로 제어한다.
class NavButton(QFrame):
    clicked = pyqtSignal()

    def __init__(self, text, icon_name):
        super().__init__()
        self.icon_name = icon_name
        self.setFixedSize(176, 56)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lo = QHBoxLayout(self)
        lo.setContentsMargins(14, 0, 10, 0)
        lo.setSpacing(10)

        self.icon_lbl = QLabel()
        self.icon_lbl.setFixedSize(20, 20)
        self.icon_lbl.setStyleSheet("background: transparent; border: none;")
        lo.addWidget(self.icon_lbl)

        self.text_lbl = QLabel(text)
        self.text_lbl.setStyleSheet(
            "background: transparent; border: none; font-size: 19px;"
        )
        lo.addWidget(self.text_lbl, 1)

        self.set_active(False)

    def set_active(self, active: bool):
        color = "white" if active else "#475569"
        self.icon_lbl.setPixmap(qta.icon(self.icon_name, color=color).pixmap(20, 20))
        if active:
            self.setStyleSheet(
                "NavButton { background-color: #1D4ED8; border-radius: 8px; border: none; }"
            )
            self.text_lbl.setStyleSheet(
                "background: transparent; border: none; font-size: 19px; "
                "font-weight: bold; color: white;"
            )
        else:
            self.setStyleSheet(
                "NavButton { background-color: transparent; border-radius: 8px; border: none; } "
                "NavButton:hover { background-color: rgba(0,0,0,0.06); }"
            )
            self.text_lbl.setStyleSheet(
                "background: transparent; border: none; font-size: 19px; color: #475569;"
            )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


def col_header_row(*labels_widths):
    """(label, width_or_None) 튜플을 받아 컬럼 헤더 행 (QFrame) 반환"""
    frame = QFrame()
    frame.setStyleSheet(
        f"background-color: {C_SIDEBAR}; border-radius: 6px;"
    )
    lo = QHBoxLayout(frame)
    lo.setContentsMargins(14, 6, 14, 6)
    lo.setSpacing(10)
    for label, w in labels_widths:
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #CBD5E1; font-size: 18px; font-weight: bold;")
        if w:
            lbl.setFixedWidth(w)
        else:
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        lo.addWidget(lbl)
    return frame


# ── 신호 버스 (전역 UI) ─────────────────────────────────────────────────────────────
class _CameraSignals(QObject):
    detected       = pyqtSignal(dict)
    blacklist_alert = pyqtSignal(dict)
    frame          = pyqtSignal(str, QImage)   # (camera_id, 프레임) 실시간 미리보기

camera_signals = _CameraSignals()


# ── 로그 행 위젯 ────────────────────────────────────────────────────────────────────
class LogRow(QFrame):
    delete_clicked = pyqtSignal(dict)

    def __init__(self, data, show_delete=False, roi_qimage=None, on_click=None):
        super().__init__()
        self.data = data
        self._roi_qimage = roi_qimage
        self._on_click = on_click
        self.setMinimumHeight(56)

        status = ("BLACKLIST" if data.get("is_black")
                  else ("GRANTED" if data.get("is_white") else "UNKNOWN"))
        color = status_color(status)
        accent_color  = {"GRANTED": C_GREEN, "BLACKLIST": C_RED, "UNKNOWN": C_ACCENT}.get(status, C_ACCENT)
        bg            = {"GRANTED": "#F0FDF4", "BLACKLIST": "#FEF2F2", "UNKNOWN": "#FFFFFF"}.get(status, "#FFFFFF")
        border_color  = {"GRANTED": "#BBF7D0", "BLACKLIST": "#FECACA", "UNKNOWN": "#CBD5E1"}.get(status, "#CBD5E1")
        label_map     = {"GRANTED": "등록", "BLACKLIST": "차단", "UNKNOWN": "미등록"}
        label_text    = label_map.get(status, "미등록")

        self.setStyleSheet(
            f"QFrame#LogRow {{ background-color: {bg}; border: 1px solid {border_color}; border-radius: 10px; }}"
            f"QFrame#LogRow:hover {{ border: 1px solid {accent_color}; }}"
        )
        self.setObjectName("LogRow")
        self.setGraphicsEffect(create_shadow())
        if on_click:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

        lo = QHBoxLayout(self)
        lo.setContentsMargins(12, 8, 14, 8)
        lo.setSpacing(12)

        # 왼쪽 컬러 막대 (박스 안에 여백 두고 작게 — 둥근 모서리 밖으로 안 튀어나오게)
        bar = QFrame()
        bar.setObjectName("AccentBar")
        bar.setFixedWidth(6)
        bar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        bar.setStyleSheet(
            f"QFrame#AccentBar {{ background-color: {accent_color}; border-radius: 3px; border: none; }}"
        )
        lo.addWidget(bar)

        # 번호판 (굵게, 크게)
        pl_lbl = QLabel(data.get("plate_text", ""))
        pl_lbl.setStyleSheet(
            f"font-weight: bold; font-size: 19px; color: {C_TEXT}; "
            f"border: none; background: transparent;"
        )
        pl_lbl.setFixedWidth(136)
        lo.addWidget(pl_lbl)

        # 신형/구형 뱃지
        is_new = data.get("plate_type", "NEW") == "NEW"
        badge = make_badge("신형" if is_new else "구형", C_ACCENT if is_new else "#94A3B8")
        badge.setFixedHeight(26)
        lo.addWidget(badge)

        # 카메라 ID
        cam_lbl = QLabel(data.get("camera_id", ""))
        cam_lbl.setFixedWidth(104)
        cam_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 18px; border: none; background: transparent;")
        lo.addWidget(cam_lbl)

        # 신뢰도
        try:
            conf_val = f"{float(data.get('confidence', 0)):.0%}"
        except Exception:
            conf_val = "-"
        conf_lbl = QLabel(conf_val)
        conf_lbl.setFixedWidth(66)
        conf_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 18px; border: none; background: transparent;")
        lo.addWidget(conf_lbl)

        # 타임스탬프
        ts_lbl = QLabel(data.get("timestamp", ""))
        ts_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 18px; border: none; background: transparent;")
        lo.addWidget(ts_lbl, 1)

        # 입차 / 출차 시간 (data에 entry_time 키가 있을 때만)
        if "entry_time" in data:
            et = data.get("entry_time"); xt = data.get("exit_time")
            ein  = et[5:16] if et else "-:-"      # MM-DD HH:MM
            xout = xt[5:16] if xt else "-:-"
            in_lbl = QLabel(f"입차 {ein}")
            in_lbl.setStyleSheet(f"color: {C_GREEN}; font-size: 18px; font-weight: bold; border: none; background: transparent;")
            out_lbl = QLabel(f"출차 {xout}")
            out_lbl.setStyleSheet(
                f"color: {C_RED if xt else C_MUTED}; font-size: 18px; font-weight: bold; "
                f"border: none; background: transparent;"
            )
            lo.addWidget(in_lbl); lo.addWidget(out_lbl)

        # 입/출차 뱃지
        ev = data.get("event_type", "")
        if ev == "ENTRY":
            ev_badge = make_badge("입차", C_GREEN); ev_badge.setFixedHeight(26); lo.addWidget(ev_badge)
        elif ev == "EXIT":
            ev_badge = make_badge("출차", C_RED); ev_badge.setFixedHeight(26); lo.addWidget(ev_badge)

        # 등록 상태 태그
        tag = make_tag(label_text, color); tag.setFixedHeight(26); lo.addWidget(tag)

        # 삭제 버튼
        if show_delete:
            btn = QPushButton("삭제")
            btn.setFixedSize(84, 36)
            btn.setStyleSheet(
                f"background-color: {C_RED}; color: white; border-radius: 6px; "
                f"font-weight: bold; font-size: 17px; border: none;"
            )
            btn.clicked.connect(lambda: self.delete_clicked.emit(self.data))
            lo.addWidget(btn)

    def mousePressEvent(self, event):
        try:
            if self._on_click and event.button() == Qt.MouseButton.LeftButton:
                self._on_click(self._roi_qimage, self.data)
        except Exception:
            pass
        super().mousePressEvent(event)


# ── 막대 차트 위젯 ──────────────────────────────────────────────────────────────────
class SimpleBarChart(QWidget):
    def __init__(self, rows, value_keys, colors_hex, title="", parent=None):
        super().__init__(parent)
        self.rows = [dict(r) for r in rows]
        self.value_keys = value_keys
        self.colors = [QColor(c) for c in colors_hex]
        self.title = title
        self.setMinimumHeight(240)
        self.setToolTip(title)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(C_SURFACE))
        if not self.rows:
            p.setPen(QColor(C_MUTED))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "데이터 없음")
            return
        W, H = self.width(), self.height()
        pl, pr, pt, pb = 10, 10, 30, 36
        cw, ch = W - pl - pr, H - pt - pb

        max_val = max(
            (sum(int(r.get(k) or 0) for k in self.value_keys) for r in self.rows),
            default=1
        ) or 1

        n = len(self.rows)
        slot_w = cw / n
        bar_group = slot_w * 0.75
        nk = len(self.value_keys)
        bar_w = bar_group / nk

        small_font = QFont()
        small_font.setPointSize(11)
        p.setFont(small_font)

        for i, row in enumerate(self.rows):
            cx = pl + (i + 0.5) * slot_w

            # X axis label
            p.setPen(QColor(C_MUTED))
            label = str(row.get("hour", i))
            p.drawText(int(cx - 16), H - pb + 4, 32, 24,
                       Qt.AlignmentFlag.AlignCenter, label)

            x0 = cx - bar_group / 2
            total = sum(int(row.get(k) or 0) for k in self.value_keys)

            for j, (key, color) in enumerate(zip(self.value_keys, self.colors)):
                val = int(row.get(key) or 0)
                bh = int((val / max_val) * ch)
                bx = int(x0 + j * bar_w)
                bw = max(2, int(bar_w) - 2)
                by = pt + ch - bh

                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(color)
                path = QPainterPath()
                path.addRoundedRect(bx, by, bw, bh, 3, 3)
                p.drawPath(path)

            if total > 0:
                total_h = int((total / max_val) * ch)
                p.setPen(QColor(C_MUTED))
                p.drawText(int(cx - 15), pt + ch - total_h - 17, 30, 14,
                           Qt.AlignmentFlag.AlignCenter, str(total))

        p.end()


# ── 영상 처리 워커 스레드 ───────────────────────────────────────────────────────────
class VideoWorker(QThread):
    frame_ready     = pyqtSignal(QImage)
    progress_ready  = pyqtSignal(int, int, int)
    crop_ready      = pyqtSignal(QImage, str, str, str)
    log_ready       = pyqtSignal(dict)
    finished_signal = pyqtSignal()
    alert_signal    = pyqtSignal(dict)

    def __init__(self, video_path, camera_id, direction="ENTRY"):
        super().__init__()
        self.video_path = video_path
        self.camera_id  = camera_id
        self.direction  = direction
        self.is_running = True

    def run(self):
        # ── 실시간 추적(track) + 백그라운드 OCR 구조 ─────────────────
        #   YOLO track이 매 프레임 박스를 그려 대상을 실시간으로 따라가고
        #   (사라지면 자동 소멸), OCR은 track_id별로 백그라운드에서 한 번만
        #   수행해 번호판 텍스트를 채운다. → 박스 지연·프레임 끊김 해소.
        from detector import track_plates

        _release_vote_buffer(self.camera_id)
        cap = cv2.VideoCapture(self.video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_time = 1.0 / fps
        executor    = ThreadPoolExecutor(max_workers=1)
        pending     = None
        pending_tid = None
        db_conn     = get_connection()
        fc = 0; detected = 0
        track_texts = {}   # {track_id: {"plate_text","whitelisted","blacklisted"}}
        track_tried = {}   # {track_id: 마지막 OCR 시도 프레임번호}

        def _ocr_task(roi, bbox, fc_num):
            return pl.process_tracked_roi(
                roi, bbox, self.camera_id, self.direction,
                os.path.basename(self.video_path), db_conn, fc_num)

        try:
            while self.is_running:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                fc += 1

                # 1) YOLO track — 매 프레임 실시간 박스/추적
                tracks  = track_plates(frame)          # [(x,y,w,h,tid,roi), ...]
                cur_ids = {t[4] for t in tracks}

                # 2) 완료된 OCR 결과 수거
                if pending is not None and pending.done():
                    try:
                        res = pending.result()
                    except Exception:
                        res = None
                    if res:
                        detected += 1
                        track_texts[pending_tid] = {
                            "plate_text":  res["plate_text"],
                            "whitelisted": res.get("whitelisted", False),
                            "blacklisted": res.get("blacklisted", False),
                        }
                        roi = res.get("roi")
                        if roi is not None and hasattr(roi, "size") and roi.size > 0:
                            self.crop_ready.emit(
                                cv2_to_qimage(roi), res["plate_text"],
                                f"신뢰도 {res['confidence']:.1%}", res["plate_type"])
                        rd = {**res,
                              "is_white": res.get("whitelisted", False),
                              "is_black": res.get("blacklisted", False)}
                        self.log_ready.emit(rd)
                        if res.get("blacklisted"):
                            self.alert_signal.emit({
                                "msg":        f"차단 차량: {res['plate_text']}",
                                "ts":         res["timestamp"],
                                "camera":     res["camera_id"],
                                "plate_text": res["plate_text"],
                            })
                    pending = None; pending_tid = None

                # 3) 다음 OCR 제출 — 텍스트 미확정 track 중 가장 큰 것(0.5초 쿨다운)
                if pending is None:
                    best = None; best_area = 0
                    for (x, y, w, h, tid, roi) in tracks:
                        if tid in track_texts:
                            continue
                        if fc - track_tried.get(tid, -10**9) < int(fps * 0.5):
                            continue
                        if roi is None or roi.size == 0:
                            continue
                        if w * h > best_area:
                            best_area = w * h
                            best = (tid, roi, (x, y, w, h))
                    if best is not None:
                        tid, roi, bbox = best
                        track_tried[tid] = fc
                        pending_tid = tid
                        pending = executor.submit(_ocr_task, roi.copy(), bbox, fc)

                # 4) 박스 렌더링 — 송출용 축소본에 그린다(렌더/전송 부하 ↓)
                #    원본 1920×1080에 한글 렌더(put_text_kr)를 돌리면 프레임당
                #    수백ms라 렉이 생김 → 960폭으로 줄인 뒤 박스·텍스트를 그린다.
                #    track 좌표도 같은 배율로 축소.
                DISP_W = 960
                sc   = DISP_W / max(1, frame.shape[1])
                disp = cv2.resize(frame, (DISP_W, int(frame.shape[0] * sc)))
                for (x, y, w, h, tid, roi) in tracks:
                    info = track_texts.get(tid)
                    if info:
                        label = info["plate_text"]
                        color = (0, 0, 255) if info["blacklisted"] else (34, 197, 94)
                    else:
                        label = "인식중…"
                        color = (150, 150, 150)
                    sx, sy = int(x * sc), int(y * sc)
                    sw, shh = int(w * sc), int(h * sc)
                    cv2.rectangle(disp, (sx, sy), (sx + sw, sy + shh), color, 2)
                    ly = sy if sy > 30 else sy + shh + 30
                    disp = put_text_kr(disp, label, sx, ly, font_size=20,
                                       text_color=(255, 255, 255), bg_color=color)

                # 5) emit (축소본)
                if fc % max(1, int(fps / 30)) == 0:
                    self.frame_ready.emit(cv2_to_qimage(disp))
                if fc % 5 == 0:
                    self.progress_ready.emit(fc, total, detected)

                # 6) 메모리: 사라진 track 기록 정리
                if len(track_texts) > 300:
                    for tid in [t for t in track_texts if t not in cur_ids][:100]:
                        track_texts.pop(tid, None)
                        track_tried.pop(tid, None)

                elapsed = time.time() - t0
                if frame_time - elapsed > 0:
                    time.sleep(frame_time - elapsed)

        except Exception as e:
            print(f"[Video Error] {e}")
            import traceback; traceback.print_exc()
        finally:
            executor.shutdown(wait=False)
            cap.release()
            db_conn.close()
            self.finished_signal.emit()

    def stop(self):
        self.is_running = False


# ════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("번호판 인식 시스템")
        self.setMinimumSize(1360, 820)
        self.setStyleSheet(STYLE_SHEET)

        init_db()
        # OCR(PaddleOCR)은 CPU 전용 paddle 사용 — YOLO(torch)-GPU와의 CUDA 심볼
        # 충돌을 피하기 위함. 번호판 crop만 처리하므로 CPU로도 충분히 빠르다.
        # (YOLO 검출은 detector.py에서 자동으로 GPU 사용)
        self.gpu_avail      = is_gpu_available()   # OCR GPU 가용 여부(현재 항상 False)
        init_reader(gpu=self.gpu_avail)
        self.conn           = get_connection()
        self.alert_list     = []
        self.worker         = None
        self.camera_threads = {}   # {camera_id: (thread, stop_event)}

        camera_signals.detected.connect(self._on_camera_detected)
        camera_signals.blacklist_alert.connect(self.show_alert_popup)

        main_widget = QWidget()
        main_widget.setObjectName("MainBG")
        self.setCentralWidget(main_widget)
        root = QHBoxLayout(main_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 사이드바
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(188)
        self.sidebar_lo = QVBoxLayout(self.sidebar)
        self.sidebar_lo.setContentsMargins(6, 16, 6, 16)
        self.sidebar_lo.setSpacing(10)
        self.sidebar_lo.setAlignment(Qt.AlignmentFlag.AlignTop)
        root.addWidget(self.sidebar)

        # 헤더 배너
        right = QWidget()
        right_lo = QVBoxLayout(right)
        right_lo.setContentsMargins(0, 0, 0, 0)
        right_lo.setSpacing(0)

        # ?ㅻ뜑
        header = QFrame()
        header.setStyleSheet(
            f"background-color: {C_SURFACE}; border-bottom: 1px solid {C_BORDER};"
        )
        header.setFixedHeight(52)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 0, 20, 0)
        hl.setSpacing(12)

        # ── 로고 (프로젝트 루트의 header.png) ─────────────────────
        _logo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "header.png"
        )
        if os.path.exists(_logo_path):
            logo_lbl = QLabel()
            pm = QPixmap(_logo_path).scaledToHeight(
                34, Qt.TransformationMode.SmoothTransformation
            )
            logo_lbl.setPixmap(pm)
            logo_lbl.setStyleSheet("border: none; background: transparent;")
            hl.addWidget(logo_lbl)

        title_lbl = QLabel("번호판 인식 시스템")
        title_lbl.setObjectName("HeaderTitle")
        hl.addWidget(title_lbl)
        hl.addStretch()
        self.header_time = QLabel()
        self.header_time.setObjectName("Muted")
        hl.addWidget(self.header_time)
        right_lo.addWidget(header)

        self.stacked = QStackedWidget()
        right_lo.addWidget(self.stacked, 1)
        root.addWidget(right, 1)

        self.init_pages()
        self.init_sidebar()

        timer = QTimer(self)
        timer.timeout.connect(self.update_time)
        timer.start(1000)
        self.update_time()

        self.overstay_timer = QTimer(self)
        self.overstay_timer.timeout.connect(self.check_overstay)
        self.overstay_timer.start(60_000)   # 1분마다 불법 주정차(장시간 체류) 점검

        self.refresh_dashboard()
        self.check_overstay()

    # ── 공통 헬퍼 ──────────────────────────────────────────────────────────────────
    def update_time(self):
        self.header_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def show_toast(self, msg, is_error=False):
        bar = self.statusBar()
        bar.showMessage(msg, 3000)
        bar.setStyleSheet(
            f"color: {'#EF4444' if is_error else '#10B981'}; "
            f"font-weight: bold; padding: 4px 12px;"
        )

    def _section_header(self, title, right_widget=None):
        row = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setObjectName("SectionTitle")
        row.addWidget(lbl)
        row.addStretch()
        if right_widget:
            row.addWidget(right_widget)
        return row

    def _scroll_area(self):
        sa = QScrollArea(widgetResizable=True)
        sa.setFrameShape(QScrollArea.Shape.NoFrame)
        w  = QWidget()
        w.setStyleSheet("background-color: transparent;")
        vb = QVBoxLayout(w)
        vb.setAlignment(Qt.AlignmentFlag.AlignTop)
        vb.setSpacing(4)
        sa.setWidget(w)
        return sa, vb

    def _bordered_list(self, header, scroll):
        """컬럼헤더 + 행 스크롤을 하나의 외곽 테두리 박스로 감싼다.
        (안쪽 행 카드는 테두리 없음 → 큰 틀만 테두리)"""
        box = QFrame()
        box.setObjectName("ListBox")
        # objectName으로 범위 한정 → 자식 QFrame(헤더·행)에 테두리가 번지지 않음
        box.setStyleSheet(
            f"QFrame#ListBox {{ background-color: {C_SURFACE}; border: 1px solid {C_BORDER}; "
            f"border-radius: 10px; }}"
        )
        bl = QVBoxLayout(box)
        bl.setContentsMargins(8, 8, 8, 8)
        bl.setSpacing(6)
        bl.addWidget(header)
        bl.addWidget(scroll, 1)
        return box

    # ── 사이드바 ────────────────────────────────────────────────────────────────────
    def init_sidebar(self):
        menus = [
            ("대시보드",    "fa5s.tachometer-alt"),
            ("번호판 검색", "fa5s.search"),
            ("등록차량",    "fa5s.check-circle"),
            ("블랙리스트",  "fa5s.ban"),
            ("카메라",      "fa5s.video"),
            ("영상파일",    "fa5s.file-video"),
            ("설정관리",    "fa5s.sliders-h"),
            ("통계",        "fa5s.chart-bar"),
            ("엑셀내보내기", "fa5s.file-excel"),
            ("요금계산",    "fa5s.calculator"),
        ]
        self.nav_btns = []
        for i, (text, icon_name) in enumerate(menus):
            btn = NavButton(text, icon_name)
            btn.clicked.connect(lambda idx=i: self.switch_page(idx))
            self.sidebar_lo.addWidget(btn)
            self.nav_btns.append(btn)
        self.switch_page(0)

    def switch_page(self, idx):
        self.stacked.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_btns):
            btn.set_active(i == idx)

        refresh_map = {
            0: self.refresh_dashboard,
            1: lambda: self.search_load("all"),
            2: self.wl_refresh,
            3: self.bl_refresh,
            4: self.cam_refresh,
            7: self.stats_refresh,
        }
        if idx in refresh_map:
            try:
                refresh_map[idx]()
            except Exception:
                import traceback; traceback.print_exc()
        if idx == 5:
            try:
                self._refresh_video_cam_combo()
            except Exception:
                pass

    def init_pages(self):
        for attr, builder in [
            ("page_dash",     self.build_dashboard),
            ("page_search",   self.build_search),
            ("page_wl",       self.build_whitelist),
            ("page_bl",       self.build_blacklist),
            ("page_cam",      self.build_cameras),
            ("page_video",    self.build_video),
            ("page_settings", self.build_settings),
            ("page_stats",    self.build_stats),
            ("page_export",   self.build_export),
            ("page_fee",      self.build_parking_fee),
        ]:
            page = QWidget()
            setattr(self, attr, page)
            self.stacked.addWidget(page)
            try:
                builder(page)
            except Exception as _e:
                import traceback
                traceback.print_exc()

    # ════════════════════════════════════════════════════════════
    # 대시보드
    # ════════════════════════════════════════════════════════════
    def build_dashboard(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16)
        lo.setSpacing(14)

        rb = QPushButton("전체보기")
        rb.setObjectName("Outline")
        rb.setIcon(qta.icon("fa5s.sync-alt", color=C_ACCENT))
        rb.clicked.connect(self.refresh_dashboard)
        lo.addLayout(self._section_header("대시보드", rb))

        # 통계 카드 4개
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        self.lbl_total  = self._stat_card(cards_row, "총 인식",     "fa5s.camera",       "#1f1e1e")
        self.lbl_today  = self._stat_card(cards_row, "오늘 인식",   "fa5s.calendar-day", C_ACCENT)
        self.lbl_parked = self._stat_card(cards_row, "주차 중",      "fa5s.parking",      C_RED)
        self.lbl_cam_on = self._stat_card(cards_row, "활성 카메라",  "fa5s.video",        C_GREEN)
        lo.addLayout(cards_row)

        # 하단 2분할
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        # 최근 인식 로그
        log_card = QFrame(); log_card.setObjectName("Card")
        log_lo = QVBoxLayout(log_card); log_lo.setContentsMargins(14, 12, 14, 12)
        log_lo.setSpacing(8)
        lbl = QLabel("최근 인식 로그")
        lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 20px; font-weight: bold;")
        log_lo.addWidget(lbl)
        self.dash_scroll = QScrollArea(widgetResizable=True)
        self.dash_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.dash_content = QWidget()
        self.dash_content.setStyleSheet("background-color: transparent;")
        self.dash_vbox = QVBoxLayout(self.dash_content)
        self.dash_vbox.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.dash_vbox.setSpacing(4)
        self.dash_scroll.setWidget(self.dash_content)
        log_lo.addWidget(self.dash_scroll)
        bottom.addWidget(log_card, 3)

        # 오른쪽 컬럼
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        # 주차 중인 차량
        park_card = QFrame(); park_card.setObjectName("Card")
        park_lo = QVBoxLayout(park_card); park_lo.setContentsMargins(14, 12, 14, 12)
        park_title = QLabel("주차 중인 차량")
        park_title.setStyleSheet(f"color: {C_ACCENT}; font-weight: bold; font-size: 20px;")
        park_title.setContentsMargins(0, 0, 0, 4)
        park_lo.addWidget(park_title)
        self.dash_park_sa = QScrollArea(widgetResizable=True)
        self.dash_park_sa.setFrameShape(QScrollArea.Shape.NoFrame)
        park_inner = QWidget()
        park_inner.setStyleSheet("background-color: transparent;")
        self.dash_parked_vbox = QVBoxLayout(park_inner)
        self.dash_parked_vbox.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.dash_parked_vbox.setSpacing(4)
        self.dash_park_sa.setWidget(park_inner)
        park_lo.addWidget(self.dash_park_sa)
        right_col.addWidget(park_card, 1)

        # 경보 알림 카드
        alert_card = QFrame()
        alert_card.setStyleSheet(
            "QFrame { background-color: #FFFFFF; border: none; border-radius: 10px; }"
        )
        alert_card_lo = QVBoxLayout(alert_card)
        alert_card_lo.setContentsMargins(0, 0, 0, 0)
        alert_card_lo.setSpacing(0)

        # 헤더 (다크 카드 그라디언트)
        alert_header = QFrame()
        alert_header.setStyleSheet(
            "QFrame { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #7F1D1D, stop:1 #B91C1C); "
            "border-radius: 9px 9px 0px 0px; border: none; }"
        )
        alert_header.setFixedHeight(42)
        ah_lo = QHBoxLayout(alert_header)
        ah_lo.setContentsMargins(12, 0, 12, 0)
        ah_lo.setSpacing(8)

        bell_lbl = QLabel()
        bell_lbl.setPixmap(qta.icon("fa5s.bell", color="white").pixmap(15, 15))
        bell_lbl.setStyleSheet("border: none; background: transparent;")
        ah_lo.addWidget(bell_lbl)

        alert_hdr_title = QLabel("경보 알림")
        alert_hdr_title.setStyleSheet(
            "color: white; font-weight: bold; font-size: 20px; "
            "border: none; background: transparent;"
        )
        ah_lo.addWidget(alert_hdr_title)
        ah_lo.addStretch()

        self.dash_alert_count = QLabel("0")
        self.dash_alert_count.setStyleSheet(
            "background-color: white; color: #991B1B; border-radius: 9px; "
            "padding: 0px 8px; font-size: 16px; font-weight: bold; border: none; min-width: 18px;"
        )
        self.dash_alert_count.setFixedHeight(20)
        self.dash_alert_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ah_lo.addWidget(self.dash_alert_count)
        alert_card_lo.addWidget(alert_header)

        # 스크롤 영역
        self.dash_alert_sa = QScrollArea(widgetResizable=True)
        self.dash_alert_sa.setStyleSheet("border: none; background: transparent;")
        alert_inner = QWidget()
        alert_inner.setStyleSheet("background: transparent;")
        self.dash_alert_vbox = QVBoxLayout(alert_inner)
        self.dash_alert_vbox.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.dash_alert_vbox.setContentsMargins(8, 8, 8, 8)
        self.dash_alert_vbox.setSpacing(6)
        self.dash_alert_sa.setWidget(alert_inner)
        alert_card_lo.addWidget(self.dash_alert_sa)

        right_col.addWidget(alert_card, 1)

        bottom.addLayout(right_col, 1)
        lo.addLayout(bottom, 1)

    def _stat_card(self, parent_layout, title, icon_name, icon_color):
        frame = QFrame(); frame.setObjectName("Card")
        frame.setGraphicsEffect(create_shadow())
        lo = QVBoxLayout(frame); lo.setContentsMargins(16, 14, 16, 14); lo.setSpacing(8)

        top_row = QHBoxLayout()
        icon_lbl = QLabel()
        icon_lbl.setPixmap(qta.icon(icon_name, color=icon_color).pixmap(22, 22))
        top_row.addWidget(icon_lbl)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 19px; font-weight: bold;")
        top_row.addWidget(title_lbl)
        top_row.addStretch()
        lo.addLayout(top_row)

        val = QLabel("0"); val.setObjectName("StatValue")
        val.setAlignment(Qt.AlignmentFlag.AlignRight)
        lo.addWidget(val)

        parent_layout.addWidget(frame)
        return val

    def refresh_dashboard(self):
        try:
            self.lbl_total.setText(str(query_total_count(self.conn)))
            self.lbl_today.setText(str(query_today_count(self.conn)))
            parked = get_current_parked(self.conn)
            self.lbl_parked.setText(str(len(parked)))
            self.lbl_cam_on.setText(str(
                sum(1 for c in get_all_cameras(self.conn) if c["enabled"])
            ))

            # 인식 로그 각 행(촬영 시각) 자체를 entry_exit_log와 "같은 시각"으로
            # 정확히 매칭해 그 순간이 입차 촬영인지 출차 촬영인지 판별.
            # (번호판의 "최종 상태"를 모든 행에 똑같이 바르면 입차 사진 행도
            #  출차로 보이는 문제가 있었음 — 행별 매칭으로 수정)
            try:
                ee_cur = self.conn.execute(
                    "SELECT plate_text, event_type, timestamp FROM entry_exit_log "
                    "ORDER BY id DESC LIMIT 1000"
                )
                ee_exact: dict = {}
                for _ee in ee_cur.fetchall():
                    ee_exact[(_ee[0], _ee[2])] = _ee[1]
            except Exception:
                ee_exact = {}

            clear_layout(self.dash_vbox)
            for r in query_all(self.conn, limit=30):
                d = dict(r)
                plate_text = d.get("plate_text", "")
                ts         = d.get("timestamp", "")
                ev = ee_exact.get((plate_text, ts))
                if ev is None:
                    # 정확히 같은 초가 없으면 가장 가까운 시각의 입출차 이벤트로 보정
                    nearest = self.conn.execute(
                        "SELECT event_type FROM entry_exit_log WHERE plate_text=? "
                        "ORDER BY ABS(strftime('%s', timestamp) - strftime('%s', ?)) ASC "
                        "LIMIT 1",
                        (plate_text, ts)
                    ).fetchone()
                    ev = nearest[0] if nearest else ""
                d["event_type"] = ev
                self.dash_vbox.addWidget(LogRow(d))

            clear_layout(self.dash_parked_vbox)
            for p in parked:
                d = dict(p)
                row = QFrame()
                row.setStyleSheet(
                    "QFrame { background-color: #FFFFFF; border: 1px solid transparent; "
                    "border-radius: 8px; } "
                    "QFrame:hover { border: 1px solid #3B82F6; background-color: #F8FAFF; }"
                )
                rl = QHBoxLayout(row); rl.setContentsMargins(10, 6, 10, 6); rl.setSpacing(8)
                icon_lbl = QLabel()
                icon_lbl.setPixmap(qta.icon("fa5s.parking", color=C_ACCENT).pixmap(16, 16))
                icon_lbl.setStyleSheet("background: transparent; border: none;")
                rl.addWidget(icon_lbl)
                pl_lbl = QLabel(d["plate_text"])
                pl_lbl.setStyleSheet(f"font-weight: bold; color: {C_TEXT}; background: transparent; border: none;")
                pl_lbl.setFixedWidth(122)
                rl.addWidget(pl_lbl)
                owner_lbl = QLabel(d.get("owner_name") or "-")
                owner_lbl.setStyleSheet(f"color: {C_MUTED}; background: transparent; border: none;")
                rl.addWidget(owner_lbl, 1)
                time_lbl = QLabel(d.get("entry_time", "")[-8:])
                time_lbl.setStyleSheet(f"color: {C_MUTED}; background: transparent; border: none;")
                rl.addWidget(time_lbl)
                ob = QPushButton("출차"); ob.setObjectName("Primary"); ob.setFixedSize(66, 34)
                ob.clicked.connect(lambda _, pt=d["plate_text"]: self._dash_exit(pt))
                rl.addWidget(ob)
                self.dash_parked_vbox.addWidget(row)

            clear_layout(self.dash_alert_vbox)
            _alert_cnt = len(self.alert_list)
            if hasattr(self, "dash_alert_count"):
                self.dash_alert_count.setText(str(min(_alert_cnt, 99)))

            if not self.alert_list:
                # 경보 없음 상태
                _empty_w = QWidget()
                _empty_w.setStyleSheet("background: transparent;")
                _ev_lo = QVBoxLayout(_empty_w)
                _ev_lo.setAlignment(Qt.AlignmentFlag.AlignCenter)
                _ev_lo.setSpacing(6)
                _shield = QLabel()
                _shield.setPixmap(qta.icon("fa5s.shield-alt", color=C_GREEN).pixmap(30, 30))
                _shield.setAlignment(Qt.AlignmentFlag.AlignCenter)
                _shield.setStyleSheet("background: transparent; border: none;")
                _ev_lo.addWidget(_shield)
                _ok_lbl = QLabel("경보 없음")
                _ok_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                _ok_lbl.setStyleSheet(
                    f"color: {C_GREEN}; font-size: 19px; font-weight: bold; "
                    "background: transparent; border: none;"
                )
                _ev_lo.addWidget(_ok_lbl)
                self.dash_alert_vbox.addStretch()
                self.dash_alert_vbox.addWidget(_empty_w)
                self.dash_alert_vbox.addStretch()
            else:
                for a in self.alert_list[:8]:
                    item = QFrame()
                    item.setStyleSheet(
                        "QFrame { background-color: #FEF2F2; "
                        "border: none; "
                        "border-left: 4px solid #EF4444; "
                        "border-radius: 6px; }"
                    )
                    il = QVBoxLayout(item)
                    il.setContentsMargins(10, 8, 10, 8)
                    il.setSpacing(3)

                    top_row = QHBoxLayout(); top_row.setSpacing(6)
                    warn_ic = QLabel()
                    warn_ic.setPixmap(
                        qta.icon("fa5s.exclamation-triangle", color="#DC2626").pixmap(13, 13)
                    )
                    warn_ic.setStyleSheet("background: transparent; border: none;")
                    top_row.addWidget(warn_ic)
                    plate_lbl = QLabel(a.get("plate_text", a.get("msg", "")))
                    plate_lbl.setStyleSheet(
                        "color: #7F1D1D; font-weight: bold; font-size: 18px; "
                        "background: transparent; border: none;"
                    )
                    top_row.addWidget(plate_lbl)
                    top_row.addStretch()
                    il.addLayout(top_row)

                    sub_lbl = QLabel(
                        f"{a.get('camera', '')}  |  {a.get('ts', '')}"
                    )
                    sub_lbl.setStyleSheet(
                        "color: #9B1C1C; font-size: 16px; "
                        "background: transparent; border: none;"
                    )
                    il.addWidget(sub_lbl)
                    self.dash_alert_vbox.addWidget(item)
        except Exception:
            pass

    def _dash_exit(self, plate_text):
        manual_exit(self.conn, plate_text)
        self.show_toast(f"{plate_text} 출차 완료")
        self.refresh_dashboard()

    def check_overstay(self):
        """미등록 차량이 기준 시간 이상 연속 주차 중이면 불법 주정차 의심으로
        시각·사진과 함께 기록하고 경보 알림에 추가한다."""
        try:
            rows = query_overstay(self.conn, OVERSTAY_THRESHOLD_MIN)
            new_found = False
            for r in rows:
                d = dict(r)
                if has_overstay_log(self.conn, d["plate_text"], d["entry_time"]):
                    continue
                img_row = self.conn.execute(
                    "SELECT image_path FROM plate_logs WHERE plate_text=? "
                    "ORDER BY id DESC LIMIT 1",
                    (d["plate_text"],)
                ).fetchone()
                img_path = img_row[0] if img_row and img_row[0] else ""
                insert_overstay_log(
                    self.conn, d["plate_text"], d.get("camera_id", ""),
                    d["entry_time"], d["duration_min"], img_path
                )
                self.alert_list.insert(0, {
                    "plate_text": d["plate_text"],
                    "camera": d.get("camera_id", ""),
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "msg": f"불법 주정차 의심 ({d['duration_min']}분 경과)",
                })
                self.show_toast(
                    f"불법 주정차 의심: {d['plate_text']} ({d['duration_min']}분 경과)"
                )
                new_found = True
            if new_found:
                self.refresh_dashboard()
        except Exception:
            import traceback; traceback.print_exc()

    def _on_camera_detected(self, result):
        try:
            self.refresh_dashboard()
        except Exception:
            pass
        # 카메라 페이지 우측 패널 갱신: 크롭 번호판 + 실시간 인식 기록
        try:
            roi = result.get("roi")
            roi_qimg = cv2_to_qimage(roi) if (roi is not None and getattr(roi, "size", 0)) else None

            # 크롭 카드
            self.cam_crop_text.setText(result.get("plate_text", ""))
            self.cam_crop_conf.setText(f"신뢰도 {float(result.get('confidence', 0) or 0):.1%}")
            if roi_qimg is not None and not roi_qimg.isNull():
                self.cam_crop_lbl.setPixmap(QPixmap.fromImage(roi_qimg).scaled(
                    self.cam_crop_lbl.width(), self.cam_crop_lbl.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))

            # 인식 기록 행 추가 (최근이 위로, 최대 30개)
            self.cam_hist_vbox.insertWidget(0, LogRow(result, roi_qimage=roi_qimg))
            while self.cam_hist_vbox.count() > 30:
                item = self.cam_hist_vbox.itemAt(self.cam_hist_vbox.count() - 1)
                if item and item.widget():
                    item.widget().setParent(None)
                else:
                    break
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════
    # 기록 조회
    # ════════════════════════════════════════════════════════════
    def build_search(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16)
        lo.setSpacing(12)
        lo.addLayout(self._section_header("기록 조회"))

        # 필터 카드
        ff = QFrame(); ff.setObjectName("Card")
        fl = QHBoxLayout(ff); fl.setContentsMargins(14, 12, 14, 12); fl.setSpacing(8)

        self.s_plate_tf = QLineEdit()
        self.s_plate_tf.setPlaceholderText("번호판 입력 (예: 12가3456)")
        self.s_plate_tf.setFixedWidth(220)
        self.s_plate_tf.returnPressed.connect(lambda: self.search_load("plate"))

        bp = QPushButton("조회"); bp.setObjectName("Primary")
        bp.setIcon(qta.icon("fa5s.search", color="white"))
        bp.clicked.connect(lambda: self.search_load("plate"))

        self.s_date_tf = QLineEdit(datetime.now().strftime("%Y-%m-%d"))
        self.s_date_tf.setFixedWidth(156)
        bd = QPushButton("날짜 조회"); bd.setObjectName("Primary")
        bd.setIcon(qta.icon("fa5s.calendar", color="white"))
        bd.clicked.connect(lambda: self.search_load("date"))

        self.s_cam_combo = QComboBox(); self.s_cam_combo.setFixedWidth(136)
        self.s_cam_combo.addItem("전체")
        for c in get_all_cameras(self.conn):
            self.s_cam_combo.addItem(c["camera_id"])
        self.s_cam_combo.currentIndexChanged.connect(lambda: self.search_load("all"))

        ba = QPushButton("전체보기"); ba.setObjectName("Outline")
        ba.clicked.connect(lambda: self.search_load("all"))

        self.s_exit_btn = QPushButton("수동 출차"); self.s_exit_btn.setObjectName("Danger")
        self.s_exit_btn.setIcon(qta.icon("fa5s.sign-out-alt", color="white"))
        self.s_exit_btn.clicked.connect(self._search_manual_exit)

        fl.addWidget(self.s_plate_tf); fl.addWidget(bp)
        fl.addWidget(self.s_date_tf);  fl.addWidget(bd)
        fl.addWidget(QLabel("카메라")); fl.addWidget(self.s_cam_combo)
        fl.addStretch()
        fl.addWidget(ba); fl.addWidget(self.s_exit_btn)
        lo.addWidget(ff)

        # 결과 정보 라벨
        self.s_info_lbl = QLabel("")
        self.s_info_lbl.setObjectName("Muted")
        lo.addWidget(self.s_info_lbl)

        # 열 헤더 행
        lo.addWidget(col_header_row(
            ("", 14), ("번호판", 136), ("구분", 84), ("카메라", 104),
            ("신뢰도", 66), ("인식 시각", None), ("입차", 110), ("출차", 110),
            ("", 84), ("삭제", 84),
        ))

        self.s_scroll, self.s_vbox = self._scroll_area()
        lo.addWidget(self.s_scroll, 1)

        # 페이지네이션 네비게이션
        nav = QHBoxLayout()
        self.s_prev_btn = QPushButton("◀ 이전"); self.s_prev_btn.setObjectName("Ghost")
        self.s_next_btn = QPushButton("다음 ▶"); self.s_next_btn.setObjectName("Ghost")
        self._s_offset = 0
        self.s_prev_btn.clicked.connect(self._search_prev)
        self.s_next_btn.clicked.connect(self._search_next)
        nav.addStretch(); nav.addWidget(self.s_prev_btn); nav.addWidget(self.s_next_btn)
        lo.addLayout(nav)

    def search_load(self, mode="all"):
        PAGE = 50
        cam_filter = self.s_cam_combo.currentText() if hasattr(self, "s_cam_combo") else "전체"

        if mode == "plate":
            rows = query_by_plate(self.conn, self.s_plate_tf.text().strip())
            self._s_offset = 0
        elif mode == "date":
            rows = query_by_date(self.conn, self.s_date_tf.text().strip())
            self._s_offset = 0
        else:
            rows = query_all(self.conn, limit=PAGE, offset=self._s_offset)

            if cam_filter and cam_filter != "전체":
                rows = [r for r in rows if dict(r).get("camera_id") == cam_filter]

        clear_layout(self.s_vbox)
        for r in rows:
            d = dict(r)
            # 입차/출차 시간 부착
            et, xt = get_session_times(self.conn, d.get("plate_text", ""))
            d["entry_time"] = et
            d["exit_time"]  = xt
            w = LogRow(d, show_delete=True)
            w.delete_clicked.connect(self.search_delete)
            self.s_vbox.addWidget(w)

        self.s_info_lbl.setText(
            f"검색 결과  {len(rows)}건" if mode != "all"
            else f"전체  {self._s_offset + 1}~{self._s_offset + len(rows)}건"
        )

    def _search_prev(self):
        self._s_offset = max(0, self._s_offset - 50)
        self.search_load("all")

    def _search_next(self):
        self._s_offset += 50
        self.search_load("all")

    def _search_manual_exit(self):
        val = self.s_plate_tf.text().strip()
        if not val:
            self.show_toast("번호판을 먼저 입력하세요", True); return
        rows = query_by_plate(self.conn, val)
        if not rows:
            self.show_toast("해당 번호판을 찾을 수 없습니다", True); return
        plate = dict(rows[0])["plate_text"]
        # 현재 주차 중(마지막 이벤트가 ENTRY)일 때만 출차 처리
        last = self.conn.execute(
            "SELECT event_type FROM entry_exit_log WHERE plate_text=? ORDER BY id DESC LIMIT 1",
            (plate,)
        ).fetchone()
        if last is None:
            self.show_toast(f"{plate}: 입차 기록이 없습니다", True); return
        if last[0] != "ENTRY":
            self.show_toast(f"{plate}: 이미 출차된 차량입니다", True); return
        manual_exit(self.conn, plate)
        self.show_toast(f"{plate} 수동 출차 완료")
        self.search_load("plate")          # 화면 갱신
        try:
            self.refresh_dashboard()
        except Exception:
            pass

    def search_delete(self, data):
        delete_plate_log(self.conn, data["id"])
        delete_entry_exit_by_text(self.conn, data["plate_text"])
        self.show_toast("삭제되었습니다.")
        self.search_load("all")

    # ════════════════════════════════════════════════════════════
    # 화이트리스트
    # ════════════════════════════════════════════════════════════
    def build_whitelist(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16); lo.setSpacing(12)
        lo.addLayout(self._section_header("화이트리스트 (등록 차량)"))

        form = QFrame(); form.setObjectName("Card")
        fl = QVBoxLayout(form); fl.setContentsMargins(16, 14, 16, 14); fl.setSpacing(10)
        fl.addWidget(QLabel("신규 등록", objectName="Muted"))

        row1 = QHBoxLayout(); row1.setSpacing(8)
        self.wl_plate = QLineEdit(); self.wl_plate.setPlaceholderText("번호판 *")
        self.wl_owner = QLineEdit(); self.wl_owner.setPlaceholderText("소유자")
        self.wl_desc  = QLineEdit(); self.wl_desc.setPlaceholderText("차량 설명")
        self.wl_phone = QLineEdit(); self.wl_phone.setPlaceholderText("전화번호")
        self.wl_memo  = QLineEdit(); self.wl_memo.setPlaceholderText("메모")
        ab = QPushButton("등록"); ab.setObjectName("Success")
        ab.clicked.connect(self.wl_add)
        for w in [self.wl_plate, self.wl_owner, self.wl_desc, self.wl_phone, self.wl_memo]:
            row1.addWidget(w)
        row1.addWidget(ab)
        fl.addLayout(row1)
        lo.addWidget(form)

        wl_header = col_header_row(
            ("번호판", 148), ("소유자", 120), ("차량 설명", 150),
            ("전화번호", 140), ("메모", None), ("등록일", 118), ("", 68),
        )
        self.wl_scroll, self.wl_vbox = self._scroll_area()
        lo.addWidget(self._bordered_list(wl_header, self.wl_scroll), 1)

    def wl_refresh(self):
        clear_layout(self.wl_vbox)
        for r in get_all_whitelist(self.conn):
            r = dict(r)
            row = QFrame(); row.setObjectName("WlRow")
            row.setMinimumHeight(52)
            # objectName으로 한정 → 호버 테두리가 내부 box(컬러 바 등)에 안 번짐
            row.setStyleSheet(
                "QFrame#WlRow { background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px; } "
                "QFrame#WlRow:hover { border: 1px solid #3B82F6; }"
            )
            row.setGraphicsEffect(create_shadow())
            rl = QHBoxLayout(row); rl.setContentsMargins(12, 8, 14, 8); rl.setSpacing(12)

            # 왼쪽 초록 액센트 바 (여백 + radius)
            accent = QFrame()
            accent.setObjectName("AccentBar")
            accent.setFixedWidth(6)
            accent.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            accent.setStyleSheet(
                f"QFrame#AccentBar {{ background-color: {C_GREEN}; border-radius: 3px; border: none; }}"
            )
            rl.addWidget(accent)

            pl_lbl = QLabel(r["plate_text"])
            pl_lbl.setStyleSheet(f"font-weight: bold; font-size: 19px; color: {C_TEXT}; border: none;")
            pl_lbl.setFixedWidth(148)
            rl.addWidget(pl_lbl)

            for text, w in [
                (r.get("owner_name", ""),   120),
                (r.get("vehicle_desc", ""), 150),
                (r.get("phone", ""),        140),
            ]:
                lbl = QLabel(text)
                lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 18px; border: none;")
                lbl.setFixedWidth(w)
                rl.addWidget(lbl)

            memo_lbl = QLabel(r.get("memo", ""))
            memo_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 18px; border: none;")
            rl.addWidget(memo_lbl, 1)

            date_lbl = QLabel(r.get("registered_at", "")[:10])
            date_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 17px; border: none;")
            date_lbl.setFixedWidth(118)
            rl.addWidget(date_lbl)

            db = QPushButton("삭제")
            db.setFixedSize(68, 34)
            db.setStyleSheet(
                f"background-color: {C_RED}; color: white; border-radius: 6px; "
                f"font-weight: bold; font-size: 17px; border: none;"
            )
            db.clicked.connect(lambda _, pt=r["plate_text"]: self.wl_del(pt))
            rl.addWidget(db)
            self.wl_vbox.addWidget(row)

    def wl_add(self):
        pt = self.wl_plate.text().strip()
        if not pt:
            self.show_toast("번호판을 먼저 입력하세요", True); return
        ok = add_whitelist(
            self.conn, pt,
            self.wl_owner.text(), self.wl_desc.text(),
            self.wl_phone.text(), self.wl_memo.text(),
        )
        if ok:
            self.show_toast(f"{pt} 등록 완료")
            for tf in [self.wl_plate, self.wl_owner, self.wl_desc, self.wl_phone, self.wl_memo]:
                tf.clear()
        else:
            self.show_toast("이미 등록된 번호판입니다", True)
        self.wl_refresh()

    def wl_del(self, pt):
        delete_whitelist(self.conn, pt)
        self.show_toast(f"{pt} 삭제 완료")
        self.wl_refresh()

    # ════════════════════════════════════════════════════════════
    # ──────────────────────────────────────────────────────────────
    # 블랙리스트 관리
    # ──────────────────────────────────────────────────────────────
    def build_blacklist(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16); lo.setSpacing(12)
        lo.addLayout(self._section_header("블랙리스트 관리 (차단 차량)"))

        form = QFrame(); form.setObjectName("Card")
        fl = QVBoxLayout(form); fl.setContentsMargins(16, 14, 16, 14); fl.setSpacing(10)
        fl.addWidget(QLabel("차단 차량 등록", objectName="Muted"))
        row1 = QHBoxLayout(); row1.setSpacing(8)
        self.bl_plate  = QLineEdit(); self.bl_plate.setPlaceholderText("번호판 *")
        self.bl_reason = QLineEdit(); self.bl_reason.setPlaceholderText("차단 사유")
        ab = QPushButton("차단 등록"); ab.setObjectName("Danger")
        ab.setIcon(qta.icon("fa5s.ban", color="white"))
        ab.clicked.connect(self.bl_add)
        row1.addWidget(self.bl_plate); row1.addWidget(self.bl_reason, 1); row1.addWidget(ab)
        fl.addLayout(row1)
        lo.addWidget(form)

        bl_header = col_header_row(
            ("", 4), ("", 18), ("번호판", 148),
            ("차단 사유", None), ("등록일", 118), ("", 104),
        )
        self.bl_scroll, self.bl_vbox = self._scroll_area()
        lo.addWidget(self._bordered_list(bl_header, self.bl_scroll), 1)

    def bl_refresh(self):
        clear_layout(self.bl_vbox)
        for r in get_all_blacklist(self.conn):
            r = dict(r)
            row = QFrame()
            row.setObjectName("BlRow")
            row.setMinimumHeight(52)
            # objectName으로 한정 → 호버 테두리가 내부 box(컬러 바 등)에 안 번짐
            row.setStyleSheet(
                "QFrame#BlRow { background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 10px; } "
                "QFrame#BlRow:hover { border: 1px solid #EF4444; }"
            )
            row.setGraphicsEffect(create_shadow())
            rl = QHBoxLayout(row); rl.setContentsMargins(12, 8, 12, 8); rl.setSpacing(10)

            # 빨간 왼쪽 강조 (여백 + radius)
            bar = QFrame()
            bar.setObjectName("AccentBar")
            bar.setFixedWidth(6)
            bar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            bar.setStyleSheet(
                f"QFrame#AccentBar {{ background-color: {C_RED}; border-radius: 3px; border: none; }}"
            )
            rl.addWidget(bar)

            icon_lbl = QLabel()
            icon_lbl.setPixmap(qta.icon("fa5s.ban", color=C_RED).pixmap(16, 16))
            icon_lbl.setContentsMargins(4, 8, 0, 8)
            rl.addWidget(icon_lbl)

            pl_lbl = QLabel(r["plate_text"])
            pl_lbl.setStyleSheet(f"font-weight: bold; font-size: 18px; color: {C_RED};")
            pl_lbl.setFixedWidth(148)
            pl_lbl.setContentsMargins(0, 8, 0, 8)
            rl.addWidget(pl_lbl)

            reason_lbl = QLabel(r.get("reason", ""))
            reason_lbl.setObjectName("Muted")
            rl.addWidget(reason_lbl, 1)

            date_lbl = QLabel(r.get("registered_at", "")[:10])
            date_lbl.setObjectName("Muted"); date_lbl.setFixedWidth(118)
            rl.addWidget(date_lbl)

            ub = QPushButton("차단 해제"); ub.setFixedSize(104, 34)
            ub.setStyleSheet(
                f"QPushButton {{ background-color: {C_GREEN}; color: white; border: none; "
                f"border-radius: 6px; font-weight: bold; font-size: 18px; }}"
                f"QPushButton:hover {{ background-color: #059669; }}"
            )
            ub.clicked.connect(lambda _, pt=r["plate_text"]: self.bl_del(pt))
            rl.addWidget(ub)
            self.bl_vbox.addWidget(row)

    def bl_add(self):
        pt = self.bl_plate.text().strip()
        if not pt:
            self.show_toast("번호판을 먼저 입력하세요", True); return
        ok = add_blacklist(self.conn, pt, self.bl_reason.text())
        if ok:
            self.show_toast(f"{pt} 차단 등록 완료")
            self.bl_plate.clear(); self.bl_reason.clear()
        else:
            self.show_toast("이미 차단된 번호판입니다", True)
        self.bl_refresh()

    def bl_del(self, pt):
        delete_blacklist(self.conn, pt)
        self.show_toast(f"{pt} 차단 해제 완료")
        self.bl_refresh()

    # ════════════════════════════════════════════════════════════
    # 카메라 설정
    # ════════════════════════════════════════════════════════════
    def build_cameras(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16); lo.setSpacing(12)
        lo.addLayout(self._section_header("카메라 설정"))

        form = QFrame(); form.setObjectName("Card")
        fl = QVBoxLayout(form); fl.setContentsMargins(16, 14, 16, 14); fl.setSpacing(10)
        fl.addWidget(QLabel("카메라 추가", objectName="Muted"))
        row1 = QHBoxLayout(); row1.setSpacing(8)
        self.cam_id   = QLineEdit(); self.cam_id.setPlaceholderText("ID (CAM_02)")
        self.cam_name = QLineEdit(); self.cam_name.setPlaceholderText("이름")
        self.cam_loc  = QLineEdit(); self.cam_loc.setPlaceholderText("위치")
        self.cam_dir  = QComboBox()
        self.cam_dir.addItems(["BOTH (양방향)", "ENTRY (입차)", "EXIT (출차)"])
        self.cam_src  = QLineEdit(); self.cam_src.setPlaceholderText("소스 (0 또는 rtsp://)")
        ab = QPushButton("추가"); ab.setObjectName("Primary")
        ab.setIcon(qta.icon("fa5s.plus", color="white"))
        ab.clicked.connect(self.cam_add)
        for w in [self.cam_id, self.cam_name, self.cam_loc, self.cam_dir]:
            row1.addWidget(w)
        row1.addWidget(self.cam_src, 2); row1.addWidget(ab)
        fl.addLayout(row1)
        lo.addWidget(form)

        # ── 중앙: 카메라 영상(좌, 크게) + 인식 패널(우, 작게) ──────
        mid = QHBoxLayout(); mid.setSpacing(12)

        # 좌: 실시간 미리보기 (최대 2대 좌우 분할)
        prev_split = QHBoxLayout(); prev_split.setSpacing(12)
        self.cam_previews = []
        for _i in range(2):
            pv = QLabel("카메라 대기 중")
            pv.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pv.setMinimumHeight(300)
            pv.setStyleSheet(
                f"background-color: #0F172A; color: {C_MUTED}; border: 1px solid {C_BORDER}; "
                f"border-radius: 10px; font-size: 19px;"
            )
            prev_split.addWidget(pv, 1)
            self.cam_previews.append(pv)
        mid.addLayout(prev_split, 3)

        # 우: 크롭 번호판 + 실시간 인식 기록 (작게)
        right_box = QVBoxLayout(); right_box.setSpacing(10)
        crop_card = QFrame(); crop_card.setObjectName("Card")
        crop_lo = QVBoxLayout(crop_card); crop_lo.setContentsMargins(12, 10, 12, 10); crop_lo.setSpacing(6)
        crop_lo.addWidget(QLabel("인식된 번호판", objectName="Muted"))
        self.cam_crop_lbl = QLabel(); self.cam_crop_lbl.setFixedHeight(70)
        self.cam_crop_lbl.setStyleSheet(f"background-color: {C_SIDEBAR}; border-radius: 6px;")
        self.cam_crop_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop_lo.addWidget(self.cam_crop_lbl)
        self.cam_crop_text = QLabel("인식 대기 중...")
        self.cam_crop_text.setStyleSheet(f"font-size: 21px; font-weight: bold; color: {C_TEXT};")
        self.cam_crop_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop_lo.addWidget(self.cam_crop_text)
        self.cam_crop_conf = QLabel(""); self.cam_crop_conf.setObjectName("Muted")
        self.cam_crop_conf.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop_lo.addWidget(self.cam_crop_conf)
        right_box.addWidget(crop_card)

        right_box.addWidget(QLabel("실시간 인식 기록", objectName="Muted"))
        self.cam_hist_scroll, self.cam_hist_vbox = self._scroll_area()
        right_box.addWidget(self.cam_hist_scroll, 1)
        mid.addLayout(right_box, 1)

        lo.addLayout(mid, 3)

        self._preview_slots = {}   # {camera_id: 슬롯(0/1)}
        self._cam_boxes     = {}   # {camera_id: (bbox, text, expire_time)} 미리보기 박스
        camera_signals.frame.connect(self._update_cam_preview)

        # ── 하단: 카메라 목록 (넓게) ─────────────────────────────
        cam_header = col_header_row(
            ("ID", 100), ("이름", 120), ("위치", 80), ("방향", 80),
            ("소스", None), ("상태", 80), ("", 188),
        )
        self.cam_scroll, self.cam_vbox = self._scroll_area()
        lo.addWidget(self._bordered_list(cam_header, self.cam_scroll), 2)

    def _update_cam_preview(self, camera_id, qimg):
        """카메라 스레드에서 온 프레임을 해당 슬롯 라벨에 표시."""
        if qimg.isNull():
            return
        slot = self._preview_slots.get(camera_id)
        if slot is None or slot >= len(self.cam_previews):
            return
        pv = self.cam_previews[slot]
        pv.setPixmap(QPixmap.fromImage(qimg).scaled(
            pv.width(), pv.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def cam_refresh(self):
        clear_layout(self.cam_vbox)
        for cam in get_all_cameras(self.conn):
            cam = dict(cam)
            cid = cam["camera_id"]
            running = cid in self.camera_threads

            row = QFrame()
            row.setObjectName("CamRow")
            if running:
                row.setStyleSheet(
                    "QFrame#CamRow { background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; }"
                    "QFrame#CamRow:hover { border: 1px solid #10B981; }"
                )
            else:
                row.setStyleSheet(
                    "QFrame#CamRow { background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px; }"
                    "QFrame#CamRow:hover { border: 1px solid #3B82F6; }"
                )
            row.setGraphicsEffect(create_shadow())
            rl = QHBoxLayout(row); rl.setContentsMargins(12, 8, 12, 8); rl.setSpacing(10)

            for text, w in [
                (cid,                   90),
                (cam.get("name", ""),   120),
                (cam.get("location", ""), 80),
                (cam.get("direction", ""), 80),
            ]:
                lbl = QLabel(text)
                lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 19px; border: none; background: transparent;")
                lbl.setFixedWidth(w)
                rl.addWidget(lbl)

            src_lbl = QLabel(cam.get("source") or "-")
            src_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 18px; border: none; background: transparent;")
            rl.addWidget(src_lbl, 1)

            # 상태 텍스트
            status_color_val = C_GREEN if running else C_RED
            status_text = "실행 중" if running else "중지"
            st_lbl = QLabel(status_text)
            st_lbl.setFixedWidth(74)
            st_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            st_lbl.setStyleSheet(
                f"color: white; background-color: {status_color_val}; border-radius: 5px; "
                f"font-size: 17px; font-weight: bold; border: none; padding: 2px 4px;"
            )
            rl.addWidget(st_lbl)

            # 시작/중지 버튼 (행 카드 안에서도 보이도록 색을 직접 지정)
            toggle_btn = QPushButton("중지" if running else "시작")
            _tg_color = C_RED if running else C_GREEN
            toggle_btn.setStyleSheet(
                f"QPushButton {{ background-color: {_tg_color}; color: white; border: none; "
                f"border-radius: 6px; font-weight: bold; font-size: 18px; }}"
                f"QPushButton:hover {{ background-color: {'#DC2626' if running else '#059669'}; }}"
            )
            toggle_btn.setFixedSize(84, 36)
            toggle_btn.clicked.connect(lambda _, c=cam: self.cam_toggle(c))
            rl.addWidget(toggle_btn)

            del_btn = QPushButton("삭제"); del_btn.setFixedSize(84, 36)
            del_btn.setStyleSheet(
                f"QPushButton {{ background-color: {C_MUTED}; color: white; border: none; "
                f"border-radius: 6px; font-weight: bold; font-size: 18px; }}"
                f"QPushButton:hover {{ background-color: #475569; }}"
            )
            del_btn.clicked.connect(lambda _, c=cid: self.cam_del(c))
            rl.addWidget(del_btn)
            self.cam_vbox.addWidget(row)

    def cam_add(self):
        cid = self.cam_id.text().strip()
        if not cid:
            self.show_toast("카메라 ID를 입력하세요", True); return
        dir_map = {"BOTH (양방향)": "BOTH", "ENTRY (입차)": "ENTRY", "EXIT (출차)": "EXIT"}
        direction = dir_map.get(self.cam_dir.currentText(), "BOTH")
        upsert_camera(self.conn, cid, self.cam_name.text(),
                      self.cam_loc.text() or "미설정", direction, self.cam_src.text())
        self.show_toast(f"{cid} 등록 완료")
        self.cam_id.clear(); self.cam_name.clear(); self.cam_loc.clear(); self.cam_src.clear()
        self.cam_refresh()

    def cam_toggle(self, cam):
        cid = cam["camera_id"]
        if cid in self.camera_threads:
            _, se = self.camera_threads.pop(cid)
            se.set()
            self.show_toast(f"{cid} 중지 완료")
            # 미리보기 슬롯 비우기
            slot = self._preview_slots.pop(cid, None)
            if slot is not None and slot < len(self.cam_previews):
                self.cam_previews[slot].clear()
                self.cam_previews[slot].setText("카메라 대기 중")
            self._cam_boxes.pop(cid, None)
        else:
            # 미리보기 슬롯 배정 (최대 2대)
            used = set(self._preview_slots.values())
            free = next((s for s in range(len(self.cam_previews)) if s not in used), None)
            if free is None:
                self.show_toast("미리보기는 최대 2대까지 가능합니다 (검출·기록은 정상)", True)
            else:
                self._preview_slots[cid] = free
            def _cb(result, _cid=cid):
                camera_signals.detected.emit(result)
                # 미리보기 박스용: 최근 검출 위치/번호를 1.5초간 표시
                bb = result.get("bbox")
                if bb:
                    self._cam_boxes[_cid] = (
                        bb, result.get("plate_text", ""), time.time() + 1.5,
                    )
                if result.get("blacklisted"):
                    camera_signals.blacklist_alert.emit({
                        "msg":        f"차단 차량: {result['plate_text']}",
                        "ts":         result["timestamp"],
                        "camera":     result["camera_id"],
                        "plate_text": result["plate_text"],
                    })

            # 실시간 미리보기용 프레임 콜백 (부하 절감 위해 매 3프레임만 전송)
            _fcnt = {"n": 0}
            def _on_frame(frame, _cid=cid):
                _fcnt["n"] += 1
                if _fcnt["n"] % 3:
                    return
                # 최근 검출 박스 오버레이 (만료 전까지)
                box = self._cam_boxes.get(_cid)
                if box and box[0] and box[2] > time.time():
                    (x, y, w, h), text, _exp = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (34, 197, 94), 3)
                    if text:
                        frame = put_text_kr(frame, text, x, y, font_size=26)
                qimg = cv2_to_qimage(cv2.resize(frame, (640, 360)))
                if not qimg.isNull():
                    camera_signals.frame.emit(_cid, qimg)

            t, se = start_camera_thread(
                cam["source"] or "0", cid,
                cam.get("direction", "BOTH"), on_detected=_cb, on_frame=_on_frame,
            )
            self.camera_threads[cid] = (t, se)
            self.show_toast(f"{cid} 시작 완료")
        self.cam_refresh()

    def cam_del(self, cid):
        if cid in self.camera_threads:
            _, se = self.camera_threads.pop(cid)
            se.set()
        delete_camera(self.conn, cid)
        self.show_toast(f"{cid} 삭제 완료")
        self.cam_refresh()

    # ════════════════════════════════════════════════════════════
    # 영상 처리
    # ════════════════════════════════════════════════════════════
    def build_video(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16); lo.setSpacing(12)
        lo.addLayout(self._section_header("영상 파일 처리"))

        # 콘트롤 카드
        ctrl = QFrame(); ctrl.setObjectName("Card")
        cl = QVBoxLayout(ctrl); cl.setContentsMargins(16, 14, 16, 14); cl.setSpacing(10)
        path_row = QHBoxLayout(); path_row.setSpacing(8)
        self.v_path = QLineEdit()
        self.v_path.setPlaceholderText(r"C:\Users\videos\parking.mp4")
        # 전체 경로는 툴팁으로도 확인 가능
        self.v_path.textChanged.connect(self.v_path.setToolTip)
        browse = QPushButton("찾아보기"); browse.setObjectName("Outline")
        browse.clicked.connect(self._browse_video)
        self.v_cam = QComboBox(); self.v_cam.setFixedWidth(170)
        self._refresh_video_cam_combo()
        self.v_dir = QComboBox(); self.v_dir.setFixedWidth(220)
        self.v_dir.addItems(["ENTRY (입차)", "EXIT (출차)", "BOTH (양방향)"])
        self.v_start = QPushButton("처리 시작"); self.v_start.setObjectName("Success")
        self.v_start.setIcon(qta.icon("fa5s.play", color="white"))
        self.v_stop = QPushButton("■  중단"); self.v_stop.setObjectName("Danger")
        self.v_stop.setEnabled(False)
        path_row.addWidget(self.v_path, 1); path_row.addWidget(browse)
        path_row.addWidget(QLabel("카메라")); path_row.addWidget(self.v_cam)
        path_row.addWidget(QLabel("방향:")); path_row.addWidget(self.v_dir)
        path_row.addWidget(self.v_start); path_row.addWidget(self.v_stop)
        cl.addLayout(path_row)

        self.v_prog_bar = QProgressBar()
        self.v_prog_bar.setRange(0, 1000)
        self.v_prog_bar.setValue(0)
        self.v_prog_bar.setFixedHeight(10)
        self.v_prog_bar.setTextVisible(False)
        cl.addWidget(self.v_prog_bar)
        self.v_prog_lbl = QLabel("대기 중...")
        self.v_prog_lbl.setObjectName("Muted")
        cl.addWidget(self.v_prog_lbl)
        lo.addWidget(ctrl)

        # 영상 + 크롭 패널
        split = QHBoxLayout(); split.setSpacing(12)

        # 영상 패널
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: #0a0a1a; border-radius: 8px;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        split.addWidget(self.video_label, 3)

        # 오른쪽 패널
        right_panel = QVBoxLayout(); right_panel.setSpacing(10)

        # 크롭 카드
        crop_card = QFrame(); crop_card.setObjectName("Card")
        crop_lo = QVBoxLayout(crop_card); crop_lo.setContentsMargins(14, 12, 14, 12); crop_lo.setSpacing(8)

        crop_title = QLabel("인식된 번호판"); crop_title.setObjectName("Muted")
        crop_lo.addWidget(crop_title)

        self.crop_lbl = QLabel()
        self.crop_lbl.setFixedHeight(90)
        self.crop_lbl.setStyleSheet(f"background-color: {C_SIDEBAR}; border-radius: 6px;")
        self.crop_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop_lo.addWidget(self.crop_lbl)

        self.crop_text = QLabel("인식 대기 중...")
        self.crop_text.setStyleSheet(f"font-size: 23px; font-weight: bold; color: {C_TEXT};")
        self.crop_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop_lo.addWidget(self.crop_text)

        badge_row = QHBoxLayout(); badge_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.crop_conf_lbl = QLabel("")
        self.crop_conf_lbl.setObjectName("Muted")
        self.crop_type_badge = QLabel("")
        self.crop_type_badge.setStyleSheet(
            f"background-color: {C_ACCENT}; color: white; border-radius: 4px; "
            f"padding: 2px 8px; font-size: 16px; font-weight: bold;"
        )
        self.crop_type_badge.setVisible(False)
        badge_row.addWidget(self.crop_conf_lbl); badge_row.addWidget(self.crop_type_badge)
        crop_lo.addLayout(badge_row)
        right_panel.addWidget(crop_card)

        # 인식 기록
        hist_lbl = QLabel("인식 기록 (클릭하면 이미지 보기)")
        hist_lbl.setObjectName("Muted")
        right_panel.addWidget(hist_lbl)
        self.v_scroll, self.v_vbox = self._scroll_area()
        right_panel.addWidget(self.v_scroll, 1)

        split.addLayout(right_panel, 1)
        lo.addLayout(split, 1)

        self.v_start.clicked.connect(self.start_video)
        self.v_stop.clicked.connect(self.stop_video)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "영상 파일 선택", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)"
        )
        if path:
            self.v_path.setText(path)

    def _refresh_video_cam_combo(self):
        if not hasattr(self, "v_cam"):
            return
        cur = self.v_cam.currentText()
        self.v_cam.clear()
        cams = [c["camera_id"] for c in get_all_cameras(self.conn)]
        self.v_cam.addItems(cams or ["CAM_01"])
        idx = self.v_cam.findText(cur)
        if idx >= 0:
            self.v_cam.setCurrentIndex(idx)

    def start_video(self):
        path = self.v_path.text().strip()
        if not path or not os.path.exists(path):
            self.show_toast("파일 경로를 확인하세요", True); return
        self.v_start.setEnabled(False)
        self.v_stop.setEnabled(True)
        clear_layout(self.v_vbox)
        self.crop_text.setText("처리 중...")
        self.crop_type_badge.setVisible(False)
        self.v_prog_bar.setValue(0)
        self.v_prog_lbl.setText("처리 시작...")
        dir_map = {"ENTRY (입차)": "ENTRY", "EXIT (출차)": "EXIT", "BOTH (양방향)": "BOTH"}
        direction = dir_map.get(self.v_dir.currentText(), "ENTRY")
        self.worker = VideoWorker(path, self.v_cam.currentText(), direction)
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.crop_ready.connect(self.update_crop_ui)
        self.worker.log_ready.connect(self.append_video_log)
        self.worker.progress_ready.connect(self.update_video_progress)
        self.worker.alert_signal.connect(self.show_alert_popup)
        self.worker.finished_signal.connect(self.on_video_finished)
        self.worker.start()

    def stop_video(self):
        if self.worker:
            self.worker.stop()
            self.v_prog_lbl.setText("중지 완료")

    def update_video_frame(self, qimg):
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def update_crop_ui(self, qimg, text, conf, ptype):
        self.crop_lbl.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.crop_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio
            )
        )
        self.crop_text.setText(text)
        self.crop_conf_lbl.setText(conf)
        self.crop_type_badge.setText("신형" if ptype == "NEW" else "구형")
        self.crop_type_badge.setStyleSheet(
            f"background-color: {C_ACCENT if ptype == 'NEW' else '#b3b3b3'}; "
            f"color: white; border-radius: 4px; padding: 2px 8px; "
            f"font-size: 16px; font-weight: bold;"
        )
        self.crop_type_badge.setVisible(True)

    def append_video_log(self, data):
        roi = data.get("roi")
        roi_qimg = None
        if roi is not None and hasattr(roi, "size") and roi.size > 0:
            roi_qimg = cv2_to_qimage(roi)

        def _click(qimg, d):
            if qimg and not qimg.isNull():
                self.crop_lbl.setPixmap(
                    QPixmap.fromImage(qimg).scaled(
                        self.crop_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio
                    )
                )
            self.crop_text.setText(d.get("plate_text", ""))

        self.v_vbox.insertWidget(0, LogRow(data, roi_qimage=roi_qimg, on_click=_click))
        if self.v_vbox.count() > 30:
            item = self.v_vbox.itemAt(self.v_vbox.count() - 1)
            if item and item.widget():
                item.widget().setParent(None)

    def update_video_progress(self, cur, total, det):
        self.v_prog_bar.setValue(int(cur / max(total, 1) * 1000))
        self.v_prog_lbl.setText(f"진행: {cur:,} / {total:,} 프레임 |  인식: {det}건")

    def show_alert_popup(self, data):
        try:
            self.alert_list.insert(0, data)
            dlg = QDialog(self)
            dlg.setWindowTitle("블랙리스트 차량 인식!")
            dlg.setMinimumSize(480, 260)
            dlg.setStyleSheet("background-color: #1a0000;")
            lo = QVBoxLayout(dlg)
            lo.setContentsMargins(32, 28, 32, 28)
            lo.setSpacing(16)

            title_lbl = QLabel("블랙리스트 차단 차량 인식!")
            title_lbl.setStyleSheet("color: #F59E0B; font-size: 23px; font-weight: bold;")
            title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lo.addWidget(title_lbl)

            plate_lbl = QLabel(data.get("plate_text", ""))
            plate_lbl.setStyleSheet("color: #FBBF24; font-size: 36px; font-weight: bold;")
            plate_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lo.addWidget(plate_lbl)

            info_lbl = QLabel(
                f"카메라: {data.get('camera', '')}     시각: {data.get('ts', '')}"
            )
            info_lbl.setStyleSheet("color: #FCD34D; font-size: 19px;")
            info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lo.addWidget(info_lbl)

            ok_btn = QPushButton("확인")
            ok_btn.setFixedHeight(42)
            ok_btn.setStyleSheet(
                "background-color: #EF4444; color: white; font-size: 19px; "
                "font-weight: bold; border-radius: 8px; border: none;"
            )
            ok_btn.clicked.connect(dlg.accept)
            lo.addWidget(ok_btn)

            dlg.exec()
            self.refresh_dashboard()
        except Exception:
            pass



















    def on_video_finished(self):
        self.v_start.setEnabled(True)
        self.v_stop.setEnabled(False)
        self.v_prog_bar.setValue(1000)
        self.v_prog_lbl.setText("처리 완료.")
        self.show_toast("영상 처리 완료 (결과 저장됨)")

    # ════════════════════════════════════════════════════════════
    # 인식 설정
    # ════════════════════════════════════════════════════════════
    def build_settings(self, page):
        sa = QScrollArea(widgetResizable=True)
        inner = QWidget()
        lo = QVBoxLayout(inner)
        lo.setContentsMargins(20, 16, 20, 16); lo.setSpacing(12)

        lo.addLayout(self._section_header("인식 파라미터 설정"))

        # 설명 배너
        banner = QFrame(); banner.setObjectName("InfoBanner")
        bl = QHBoxLayout(banner); bl.setContentsMargins(14, 10, 14, 10)
        info_lbl = QLabel(
            "투표(Voting): 여러 번호판을 여러 프레임에서 취합해 최다 득표 결과를 확정합니다\n"
            "Fuzzy 편집거리: 편집거리 이하의 유사 번호판은 같은 차량으로 통합. 0=정확히 일치만."
        )
        info_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 19px;")
        info_lbl.setWordWrap(True)
        bl.addWidget(info_lbl)
        lo.addWidget(banner)

        # GPU 패널
        from ocr_engine import _CURRENT_GPU
        gpu_card = QFrame(); gpu_card.setObjectName("Card")
        gpu_card.setStyleSheet(
            f"background-color: {'#F0FDF4' if (_CURRENT_GPU and self.gpu_avail) else C_SURFACE}; "
            f"border: none; border-radius: 10px;"
        )
        gl = QHBoxLayout(gpu_card); gl.setContentsMargins(16, 14, 16, 14); gl.setSpacing(12)
        gpu_text_col = QVBoxLayout(); gpu_text_col.setSpacing(3)
        gpu_title = QLabel("GPU 가속 (PaddleOCR)")
        gpu_title.setStyleSheet(f"font-weight: bold; font-size: 18px; color: {C_TEXT};")
        gpu_desc  = QLabel("OCR 추론을 GPU로 실행. CPU 대비 5~10배 빠름.")
        gpu_desc.setObjectName("Muted")
        gpu_rec   = QLabel("권장: CUDA GPU 사용 가능하면 ON")
        gpu_rec.setStyleSheet(f"color: {C_ACCENT}; font-size: 16px;")
        gpu_text_col.addWidget(gpu_title); gpu_text_col.addWidget(gpu_desc); gpu_text_col.addWidget(gpu_rec)
        gl.addLayout(gpu_text_col, 1)

        self.gpu_chk = QCheckBox("GPU 사용")
        self.gpu_chk.setChecked(self.gpu_avail)
        self.gpu_chk.setEnabled(True)
        if self.gpu_avail:
            _gpu_status_text = "GPU 사용 중"
            _gpu_status_color = C_GREEN
        else:
            _gpu_status_text = "GPU 미사용 (CPU 모드)"
            _gpu_status_color = C_MUTED
        self.gpu_status_lbl = QLabel(_gpu_status_text)
        self.gpu_status_lbl.setStyleSheet(
            f"color: {_gpu_status_color}; font-size: 19px;"
        )
        self.gpu_chk.stateChanged.connect(self._toggle_gpu)
        gpu_right = QVBoxLayout(); gpu_right.setAlignment(Qt.AlignmentFlag.AlignRight)
        gpu_right.addWidget(self.gpu_chk, alignment=Qt.AlignmentFlag.AlignRight)
        gpu_right.addWidget(self.gpu_status_lbl, alignment=Qt.AlignmentFlag.AlignRight)
        gl.addLayout(gpu_right)
        lo.addWidget(gpu_card)

        # 인식 파라미터 목록
        params = [
            ("투표 임계값",
             "해당 프레임 수 이상 득표한 결과 확정. 높을수록 안정적이나 느림.",
             "권장 7 / 최소 5", pl.VOTE_FRAMES),
            ("확정 득표율 (0~1)",
             "최다 득표 텍스트가 전체 표수의 이 비율 이상이어야 확정.",
             "0.4 (40%)", pl.VOTE_WIN_RATIO),
            ("쿨다운 (초)",
             "같은 번호판의 재검출 방지 시간. 짧을수록 중복 줄어듦.",
             "권장 30 / 입출차 60", pl.COOLDOWN_SEC),
            ("처리 주기 (N프레임)",
             "N 프레임마다 OCR 1회 실행. 낮을수록 투표 샘플 증가.",
             "3", pl.PROCESS_EVERY_N),
            ("Bbox IoU 임계값",
             "두 검출이 크게 겹칠 때 같은 번호판으로 병합.",
             "0.3", pl.VOTE_IOU_TH),
            ("Fuzzy 편집거리",
             "이 거리 이하의 텍스트는 같은 차량으로 간주. 0=정확히 일치만",
             "2", pl.FUZZY_DIST),
        ]
        self._set_fields = []
        for title, desc, rec, val in params:
            card, field = self._param_card(title, desc, rec, str(val))
            lo.addWidget(card)
            self._set_fields.append(field)

        # 버튼 행
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        ab = QPushButton("적용"); ab.setObjectName("Success")
        ab.setIcon(qta.icon("fa5s.check", color="white"))
        ab.clicked.connect(self.apply_settings)
        rb2 = QPushButton("기본값 복원"); rb2.setObjectName("Outline")
        rb2.clicked.connect(self.reset_settings)
        self.set_status_lbl = QLabel("")
        self.set_status_lbl.setStyleSheet(f"color: {C_GREEN}; font-size: 19px;")
        btn_row.addWidget(ab); btn_row.addWidget(rb2); btn_row.addWidget(self.set_status_lbl)
        btn_row.addStretch()
        lo.addLayout(btn_row)
        lo.addStretch()

        sa.setWidget(inner)
        outer = QVBoxLayout(page); outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(sa)

    def _param_card(self, title, desc, recommend, value):
        card = QFrame(); card.setObjectName("Card")
        lo = QHBoxLayout(card); lo.setContentsMargins(16, 12, 16, 12); lo.setSpacing(16)
        text_col = QVBoxLayout(); text_col.setSpacing(3)
        t = QLabel(title); t.setStyleSheet(f"font-weight: bold; font-size: 18px; color: {C_TEXT};")
        d = QLabel(desc);  d.setObjectName("Muted")
        r = QLabel(f"권장: {recommend}"); r.setStyleSheet(f"color: {C_ACCENT}; font-size: 16px;")
        text_col.addWidget(t); text_col.addWidget(d); text_col.addWidget(r)
        lo.addLayout(text_col, 1)
        field = QLineEdit(value); field.setFixedWidth(146)
        lo.addWidget(field)
        return card, field

    def _toggle_gpu(self, state):
        use_gpu = (state == Qt.CheckState.Checked.value)
        self.gpu_chk.setEnabled(False)
        self.gpu_status_lbl.setText("PaddleOCR 모델 재로딩 중..")
        self.gpu_status_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 19px;")

        def _reload():
            success = True
            try:
                reinit_reader(gpu=use_gpu)
            except Exception:
                success = False
            if use_gpu and not success:
                try:
                    reinit_reader(gpu=False)
                except Exception:
                    pass
                self.gpu_chk.setChecked(False)
                self.gpu_status_lbl.setText("GPU 초기화 실패 → CPU로 전환")
                self.gpu_status_lbl.setStyleSheet(f"color: {C_RED}; font-size: 19px;")
            else:
                actual = use_gpu and success
                self.gpu_status_lbl.setText("GPU 사용 중" if actual else "CPU 사용 중")
                self.gpu_status_lbl.setStyleSheet(
                    f"color: {C_GREEN if actual else C_MUTED}; font-size: 19px;"
                )
            self.gpu_chk.setEnabled(True)

        import threading
        threading.Thread(target=_reload, daemon=True).start()

    def apply_settings(self):
        try:
            vals = [f.text() for f in self._set_fields]
            pl.VOTE_FRAMES     = max(1,   int(vals[0]))
            pl.VOTE_WIN_RATIO  = max(0.1, min(1.0, float(vals[1])))
            pl.COOLDOWN_SEC    = max(1,   int(vals[2]))
            pl.PROCESS_EVERY_N = max(1,   int(vals[3]))
            pl.VOTE_IOU_TH     = max(0.1, min(0.9, float(vals[4])))
            pl.FUZZY_DIST      = max(0,   int(vals[5]))
            self.set_status_lbl.setText("설정 저장 완료 — 실행 중인 카메라에 즉시 반영됩니다")
            self.set_status_lbl.setStyleSheet(f"color: {C_GREEN}; font-size: 19px;")
        except ValueError:
            self.set_status_lbl.setText("숫자 형식 오류 — 숫자를 입력하세요")
            self.set_status_lbl.setStyleSheet(f"color: {C_RED}; font-size: 19px;")

    def reset_settings(self):
        defaults = ["7", "0.4", "30", "3", "0.3", "2"]
        for f, v in zip(self._set_fields, defaults):
            f.setText(v)
        self.set_status_lbl.setText("기본값으로 복원됨")
        self.set_status_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 19px;")

    # ════════════════════════════════════════════════════════════
    # 통계
    # ════════════════════════════════════════════════════════════
    def build_stats(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16); lo.setSpacing(12)

        top = QHBoxLayout(); top.setSpacing(8)
        lo.addLayout(self._section_header("통계 차트"))
        date_lbl = QLabel("날짜:")
        self.stat_date = QLineEdit(datetime.now().strftime("%Y-%m-%d"))
        self.stat_date.setFixedWidth(176)
        btn = QPushButton("조회"); btn.setObjectName("Primary")
        btn.setIcon(qta.icon("fa5s.chart-bar", color="white"))
        btn.clicked.connect(self.stats_refresh)
        top.addWidget(date_lbl); top.addWidget(self.stat_date)
        top.addWidget(btn); top.addStretch()
        lo.addLayout(top)

        # 차트 영역 (스크롤)
        self.stat_scroll, stat_inner_lo = self._scroll_area()
        self.stat_inner_lo = stat_inner_lo
        lo.addWidget(self.stat_scroll, 1)

    def _chart_card(self, title, chart_widget, legend_items=None):
        card = QFrame(); card.setObjectName("Card")
        cl = QVBoxLayout(card); cl.setContentsMargins(16, 14, 16, 14); cl.setSpacing(10)
        header = QHBoxLayout()
        t = QLabel(title); t.setStyleSheet(f"font-weight: bold; font-size: 21px; color: {C_TEXT};")
        header.addWidget(t); header.addStretch()
        if legend_items:
            for text, color in legend_items:
                dot = QLabel("●")
                dot.setStyleSheet(f"color: {color}; font-size: 19px;")
                lbl = QLabel(text); lbl.setObjectName("Muted")
                header.addWidget(dot); header.addWidget(lbl)
        cl.addLayout(header)
        cl.addWidget(chart_widget)
        return card

    def stats_refresh(self):
        try:
         d = self.stat_date.text().strip() or datetime.now().strftime("%Y-%m-%d")
        except Exception:
         d = datetime.now().strftime("%Y-%m-%d")
        try:
         clear_layout(self.stat_inner_lo)
        except Exception:
         return

        try:
         hourly    = query_hourly_stats(self.conn, d)
         ee_hourly = query_entry_exit_hourly(self.conn, d)
         type_rows = self.conn.execute(
            "SELECT plate_type, COUNT(*) AS cnt FROM plate_logs "
            "WHERE timestamp LIKE ? GROUP BY plate_type",
            (f"{d}%",)
         ).fetchall()
        except Exception:
         hourly = []; ee_hourly = []; type_rows = []

        # 시간대별 인식
        chart1 = SimpleBarChart(hourly, ["cnt"], [C_ACCENT])
        self.stat_inner_lo.addWidget(
            self._chart_card(f"{d} 시간대별 인식 건수", chart1)
        )

        # 시간대별 입출차
        chart2 = SimpleBarChart(ee_hourly, ["entries", "exits"], [C_GREEN, C_RED])
        self.stat_inner_lo.addWidget(
            self._chart_card(f"{d} 시간대별 입출차", chart2,
                             legend_items=[("입차", C_GREEN), ("출차", C_RED)])
        )

        # 번호판 유형 카드
        type_card = QFrame(); type_card.setObjectName("Card")
        tcl = QVBoxLayout(type_card); tcl.setContentsMargins(16, 14, 16, 14); tcl.setSpacing(12)
        t = QLabel("번호판 유형 분포"); t.setStyleSheet(f"font-weight: bold; font-size: 21px; color: {C_TEXT};")
        tcl.addWidget(t)
        type_row = QHBoxLayout(); type_row.setSpacing(0)
        type_map = {r["plate_type"]: r["cnt"] for r in [dict(x) for x in type_rows]}
        for label, key, color in [("신형", "NEW", C_ACCENT), ("구형", "OLD", C_GREEN)]:
            col = QVBoxLayout(); col.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cnt_lbl = QLabel(str(type_map.get(key, 0)))
            cnt_lbl.setStyleSheet(f"font-size: 42px; font-weight: bold; color: {color};")
            cnt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sub_lbl = QLabel(label); sub_lbl.setObjectName("Muted")
            sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(cnt_lbl); col.addWidget(sub_lbl)
            type_row.addLayout(col)
        tcl.addLayout(type_row)
        self.stat_inner_lo.addWidget(type_card)
        self.stat_inner_lo.addStretch()

    # ════════════════════════════════════════════════════════════
    # 엑셀 내보내기
    # ════════════════════════════════════════════════════════════
    def build_export(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16); lo.setSpacing(14)
        lo.addLayout(self._section_header("엑셀 내보내기"))

        form = QFrame(); form.setObjectName("Card")
        fl = QVBoxLayout(form); fl.setContentsMargins(20, 16, 20, 16); fl.setSpacing(14)
        fl.addWidget(QLabel("기간 선택", objectName="Muted"))

        today = datetime.now().strftime("%Y-%m-%d")
        date_row = QHBoxLayout(); date_row.setSpacing(8)
        self.ex_start = QLineEdit(today); self.ex_start.setFixedWidth(176)
        self.ex_end   = QLineEdit(today); self.ex_end.setFixedWidth(176)
        self.ex_preview_lbl = QLabel("")
        self.ex_preview_lbl.setObjectName("Muted")
        self.ex_start.textChanged.connect(self._ex_preview)
        self.ex_end.textChanged.connect(self._ex_preview)
        date_row.addWidget(self.ex_start)
        date_row.addWidget(QLabel("~"))
        date_row.addWidget(self.ex_end)
        date_row.addWidget(self.ex_preview_lbl)
        date_row.addStretch()
        fl.addLayout(date_row)

        self.ex_mask_chk = QCheckBox("개인정보 가림 처리 (번호판 끝 2자리·연락처 중간 자리 마스킹)")
        fl.addWidget(self.ex_mask_chk)

        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        b1 = QPushButton("인식 기록 내보내기"); b1.setObjectName("Primary")
        b1.setIcon(qta.icon("fa5s.table", color="white"))
        b1.clicked.connect(lambda: self.do_export("log"))
        b2 = QPushButton("입출차 기록 내보내기"); b2.setObjectName("Success")
        b2.setIcon(qta.icon("fa5s.exchange-alt", color="white"))
        b2.clicked.connect(lambda: self.do_export("ee"))
        btn_row.addWidget(b1); btn_row.addWidget(b2); btn_row.addStretch()
        fl.addLayout(btn_row)

        status_row = QHBoxLayout(); status_row.setSpacing(10)
        self.ex_status_lbl = QLabel("")
        self.ex_open_btn = QPushButton("폴더 열기"); self.ex_open_btn.setObjectName("Outline")
        self.ex_open_btn.setIcon(qta.icon("fa5s.folder-open", color=C_MUTED))
        self.ex_open_btn.setVisible(False)
        self.ex_open_btn.clicked.connect(self._open_export_folder)
        self._last_export_dir = None
        status_row.addWidget(self.ex_status_lbl)
        status_row.addWidget(self.ex_open_btn)
        status_row.addStretch()
        fl.addLayout(status_row)
        lo.addWidget(form)

        # 저장 위치 안내
        info = QFrame(); info.setObjectName("InfoBanner")
        il = QHBoxLayout(info); il.setContentsMargins(14, 10, 14, 10); il.setSpacing(8)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(qta.icon("fa5s.folder", color=C_MUTED).pixmap(14, 14))
        il.addWidget(icon_lbl)
        il.addWidget(QLabel("저장 위치: lpr_system/exports/ 폴더", objectName="Muted"))
        il.addStretch()
        lo.addWidget(info)
        lo.addStretch()

        self._ex_preview()

    def _ex_preview(self):
        try:
            s = self.ex_start.text().strip()
            e = self.ex_end.text().strip()
            r1 = query_by_date_range(self.conn, s, e)
            r2 = query_entry_exit_range(self.conn, s, e)
            self.ex_preview_lbl.setText(f"인식 기록 {len(r1)}건  |  입출차 기록 {len(r2)}건")
        except Exception:
            self.ex_preview_lbl.setText("")

    def do_export(self, mode):
        s = self.ex_start.text().strip()
        e = self.ex_end.text().strip()
        mask = self.ex_mask_chk.isChecked()
        try:
            if mode == "log":
                rows = query_by_date_range(self.conn, s, e)
                if not rows:
                    self.ex_status_lbl.setText("해당 날짜에 데이터가 없습니다")
                    self.ex_status_lbl.setStyleSheet(f"color: {C_YELLOW};"); return
                path = export_plate_logs(rows, s, e, mask_personal=mask)
            else:
                rows = query_entry_exit_range(self.conn, s, e)
                if not rows:
                    self.ex_status_lbl.setText("해당 날짜에 데이터가 없습니다")
                    self.ex_status_lbl.setStyleSheet(f"color: {C_YELLOW};"); return
                path = export_entry_exit(rows, s, e, mask_personal=mask)

            self._last_export_dir = os.path.dirname(path)
            self.ex_status_lbl.setText(f"저장 완료: {os.path.basename(path)}  ({len(rows)}건)")
            self.ex_status_lbl.setStyleSheet(f"color: {C_GREEN}; font-size: 19px;")
            self.ex_open_btn.setVisible(True)
            self.show_toast(f"파일 저장 완료 ({len(rows)}건)")
        except Exception as ex:
            self.ex_status_lbl.setText(f"오류: {ex}")
            self.ex_status_lbl.setStyleSheet(f"color: {C_RED}; font-size: 19px;")

    def _open_export_folder(self):
        if self._last_export_dir and os.path.isdir(self._last_export_dir):
            subprocess.Popen(f'explorer "{self._last_export_dir}"')

    # ════════════════════════════════════════════════════════════
    # 주차 요금 계산
    # ════════════════════════════════════════════════════════════
    def build_parking_fee(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16); lo.setSpacing(12)
        lo.addLayout(self._section_header("주차 요금 계산"))

        # 요금 설정 패널
        cfg_card = QFrame(); cfg_card.setObjectName("Card")
        cfg_lo = QVBoxLayout(cfg_card); cfg_lo.setContentsMargins(16, 14, 16, 14); cfg_lo.setSpacing(10)
        cfg_lo.addWidget(QLabel("요금 설정", objectName="Muted"))

        row1 = QHBoxLayout(); row1.setSpacing(8)
        row2 = QHBoxLayout(); row2.setSpacing(8)
        self.f_free  = QLineEdit(str(FEE_CONFIG["free_min"]));  self.f_free.setFixedWidth(136)
        self.f_base  = QLineEdit(str(FEE_CONFIG["base_fee"]));  self.f_base.setFixedWidth(136)
        self.f_basem = QLineEdit(str(FEE_CONFIG["base_min"]));  self.f_basem.setFixedWidth(136)
        self.f_unit  = QLineEdit(str(FEE_CONFIG["unit_fee"]));  self.f_unit.setFixedWidth(136)
        self.f_unitm = QLineEdit(str(FEE_CONFIG["unit_min"]));  self.f_unitm.setFixedWidth(136)
        self.f_max   = QLineEdit(str(FEE_CONFIG["daily_max"])); self.f_max.setFixedWidth(136)

        for label, field, parent_row in [
            ("무료 시간(분)", self.f_free,  row1),
            ("기본 요금(원)", self.f_base,  row1),
            ("기본 시간(분)", self.f_basem, row1),
            ("추가 단위(원)", self.f_unit,  row2),
            ("추가 시간(분)", self.f_unitm, row2),
            ("1일 최대(원)",  self.f_max,   row2),
        ]:
            col = QVBoxLayout(); col.setSpacing(4)
            col.addWidget(QLabel(label, objectName="Muted"))
            col.addWidget(field)
            parent_row.addLayout(col)

        save_btn = QPushButton("설정 저장"); save_btn.setObjectName("Primary")
        save_btn.setIcon(qta.icon("fa5s.save", color="white"))
        save_btn.clicked.connect(self.save_fee_config)
        self.fee_cfg_status = QLabel("")
        self.fee_cfg_status.setStyleSheet(f"color: {C_GREEN}; font-size: 19px;")

        cfg_lo.addLayout(row1); cfg_lo.addLayout(row2)
        save_row = QHBoxLayout()
        save_row.addWidget(save_btn); save_row.addWidget(self.fee_cfg_status); save_row.addStretch()
        cfg_lo.addLayout(save_row)
        lo.addWidget(cfg_card)

        # 요금 조회 및 계산 카드
        search_card = QFrame(); search_card.setObjectName("Card")
        sc_lo = QHBoxLayout(search_card); sc_lo.setContentsMargins(16, 12, 16, 12); sc_lo.setSpacing(8)
        self.f_plate = QLineEdit(); self.f_plate.setPlaceholderText("번호판 입력 (비워두면 전체)")
        self.f_plate.setFixedWidth(240)
        calc_btn = QPushButton("조회"); calc_btn.setObjectName("Success")
        calc_btn.setIcon(qta.icon("fa5s.calculator", color="white"))
        calc_btn.clicked.connect(self.calc_fee_ui)
        sc_lo.addWidget(self.f_plate); sc_lo.addWidget(calc_btn); sc_lo.addStretch()
        lo.addWidget(search_card)

        fee_scroll_lbl = QLabel("현재 주차 중인 차량 요금")
        fee_scroll_lbl.setStyleSheet(f"font-size: 19px; font-weight: bold; color: {C_TEXT};")
        lo.addWidget(fee_scroll_lbl)
        self.fee_scroll, self.fee_vbox = self._scroll_area()
        lo.addWidget(self.fee_scroll, 1)
        self.calc_fee_ui()

    def save_fee_config(self):
        try:
            FEE_CONFIG.update(
                free_min  = int(self.f_free.text()),
                base_fee  = int(self.f_base.text()),
                base_min  = int(self.f_basem.text()),
                unit_fee  = int(self.f_unit.text()),
                unit_min  = int(self.f_unitm.text()),
                daily_max = int(self.f_max.text()),
            )
            self.fee_cfg_status.setText("요금 설정 저장됨")
            self.show_toast("요금 설정 저장 완료")
        except ValueError:
            self.fee_cfg_status.setText("숫자를 입력하세요")

    def calc_fee_ui(self):
        plate  = self.f_plate.text().strip()
        parked = get_current_parked(self.conn)
        if plate:
            parked = [p for p in parked if plate in p["plate_text"]]
        clear_layout(self.fee_vbox)

        if not parked:
            empty = QFrame()
            empty.setStyleSheet(
                "QFrame { background-color: #FFFFFF; border: 1px solid transparent; border-radius: 10px; }"
            )
            el = QHBoxLayout(empty); el.setContentsMargins(20, 16, 20, 16)
            lbl = QLabel("현재 주차 중인 해당 차량이 없습니다.")
            lbl.setStyleSheet(f"color: {C_MUTED}; background: transparent; border: none;")
            el.addWidget(lbl)
            self.fee_vbox.addWidget(empty); return

        for p in parked:
            p = dict(p)
            plate_text   = p["plate_text"]
            is_wl        = is_whitelisted(self.conn, plate_text)

            card = QFrame(); card.setGraphicsEffect(create_shadow())
            card.setStyleSheet(
                "QFrame { background-color: #FFFFFF; border: 1px solid transparent; border-radius: 12px; } "
                "QFrame:hover { border: 1px solid #3B82F6; }"
            )
            cl = QHBoxLayout(card); cl.setContentsMargins(16, 14, 16, 14); cl.setSpacing(12)

            left = QVBoxLayout(); left.setSpacing(4)
            pl_lbl = QLabel(plate_text)
            pl_lbl.setStyleSheet(f"font-size: 19px; font-weight: bold; color: {C_TEXT}; background: transparent; border: none;")
            owner_lbl = QLabel(p.get("owner_name") or ("등록 차량" if is_wl else "미등록"))
            owner_lbl.setStyleSheet(f"color: {C_MUTED}; background: transparent; border: none;")
            left.addWidget(pl_lbl); left.addWidget(owner_lbl)

            right = QVBoxLayout(); right.setAlignment(Qt.AlignmentFlag.AlignRight); right.setSpacing(4)

            if is_wl:
                # 등록 차량 → 요금 면제
                info = calc_fee(p["entry_time"], config=FEE_CONFIG)
                dur_lbl = QLabel(info["duration_str"])
                dur_lbl.setStyleSheet(f"color: {C_MUTED}; background: transparent; border: none;")
                dur_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
                right.addWidget(dur_lbl)
                fee_lbl = QLabel("0원")
                fee_lbl.setStyleSheet(f"font-size: 23px; font-weight: bold; color: {C_GREEN}; background: transparent; border: none;")
                fee_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
                right.addWidget(fee_lbl)
                right.addWidget(make_tag("등록 차량 무료", C_GREEN),
                                alignment=Qt.AlignmentFlag.AlignRight)
            else:
                # 미등록 차량 일반 요금 계산
                info  = calc_fee(p["entry_time"], config=FEE_CONFIG)
                color = C_GREEN if info["is_free"] else C_ACCENT
                dur_lbl = QLabel(info["duration_str"])
                dur_lbl.setStyleSheet(f"color: {C_MUTED}; background: transparent; border: none;")
                dur_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
                fee_lbl = QLabel(info["fee_str"])
                fee_lbl.setStyleSheet(f"font-size: 23px; font-weight: bold; color: {color}; background: transparent; border: none;")
                fee_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
                right.addWidget(dur_lbl); right.addWidget(fee_lbl)
                if info["is_free"]:
                    right.addWidget(make_tag("10분 이내", C_GREEN),
                                    alignment=Qt.AlignmentFlag.AlignRight)
                else:
                    right.addWidget(make_tag("미등록 차량", C_YELLOW),
                                    alignment=Qt.AlignmentFlag.AlignRight)

            cl.addLayout(left); cl.addStretch(); cl.addLayout(right)
            self.fee_vbox.addWidget(card)

    # ════════════════════════════════════════════════════════════
    def closeEvent(self, event):
        for _, (_, se) in list(self.camera_threads.items()):
            try: se.set()
            except Exception: pass
        if self.worker and self.worker.isRunning():
            self.worker.stop(); self.worker.wait()
        try: self.conn.close()
        except Exception: pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    _font_dir = os.path.join(os.path.dirname(__file__), "assets", "fonts")
    _family = None
    # Regular/Bold만 사용 — Medium/SemiBold는 OTF 내부 family명이 "Pretendard
    # Medium" 등으로 따로 등록돼 있어 Qt가 "Pretendard" 패밀리로 못 찾는 문제가 있음
    for _weight_file in ("Pretendard-Regular.otf", "Pretendard-Bold.otf"):
        _fp = os.path.join(_font_dir, _weight_file)
        if os.path.exists(_fp):
            _fid = QFontDatabase.addApplicationFont(_fp)
            if _fid != -1 and _family is None:
                _family = QFontDatabase.applicationFontFamilies(_fid)[0]
    if _family:
        app.setFont(QFont(_family, 10))
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())

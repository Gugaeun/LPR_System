"""
generate_sample_data.py — 포트폴리오 데모용 합성 주차 데이터 생성

실제 카메라 배포 전이라 parking.db가 비어있어, 분석/대시보드 시연을 위한
  합성 데이터를 생성한다. 스키마는 lpr_system/database.py와 완전히 동일해서
  실제 데이터가 쌓이면 그대로 교체 가능하다.

시나리오: 오피스 빌딩 주차장 4주치 데이터
  - 정기 출퇴근 차량(화이트리스트) 40대: 평일 70% 확률로 출근/퇴근
  - 방문 차량 다수: 낮 시간대 산발적 방문, 체류시간 짧음
  - 주말엔 방문 차량만, 물량 대폭 감소
"""
import random
import sqlite3
import os
from datetime import datetime, timedelta

random.seed(42)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_parking.db")
DAYS = 28
END_DATE = datetime(2026, 9, 26)
START_DATE = END_DATE - timedelta(days=DAYS - 1)

# 번호판 용도 기호 40자 (비사업용 32자, 사업용·택배 5자, 렌터카 3자)
PLATE_HANGUL = "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주" "바사아자배" "하허호"
# 이전 글자 목록과 같은 56개로 맞춰 시드 42의 난수 순서를 유지한다
# (방문 시각·건수는 그대로, 번호의 한글만 실제 용도 기호로 바뀜)
KOR_CHARS = list(PLATE_HANGUL + "가나다라마거너더러머버서어저고노")


def rand_plate():
    front = random.choice([random.randint(10, 99), random.randint(100, 999)])
    return f"{front}{random.choice(KOR_CHARS)}{random.randint(1000, 9999)}"


def build_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS plate_logs (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_text   TEXT    NOT NULL,
        plate_color  TEXT    NOT NULL,
        plate_type   TEXT    NOT NULL,
        confidence   REAL    DEFAULT 0,
        timestamp    TEXT    NOT NULL,
        camera_id    TEXT    DEFAULT 'CAM_01',
        image_path   TEXT    DEFAULT '',
        frame_number INTEGER DEFAULT 0,
        source_file  TEXT    DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS entry_exit_log (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_text   TEXT NOT NULL,
        event_type   TEXT NOT NULL,
        camera_id    TEXT DEFAULT '',
        timestamp    TEXT NOT NULL,
        image_path   TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS whitelist (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_text    TEXT NOT NULL UNIQUE,
        owner_name    TEXT DEFAULT '',
        vehicle_desc  TEXT DEFAULT '',
        registered_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_pl_ts ON plate_logs (timestamp);
    CREATE INDEX IF NOT EXISTS idx_ee_ts ON entry_exit_log (timestamp);
    CREATE INDEX IF NOT EXISTS idx_ee_plate ON entry_exit_log (plate_text);
    """)
    conn.commit()


def minutes_normal(mean, sd, lo, hi):
    return max(lo, min(hi, int(random.gauss(mean, sd))))


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    build_schema(conn)

    regulars = [rand_plate() for _ in range(40)]
    for p in regulars:
        conn.execute(
            "INSERT OR IGNORE INTO whitelist (plate_text, owner_name, vehicle_desc, registered_at) VALUES (?,?,?,?)",
            (p, f"입주사_{regulars.index(p)+1:02d}", "정기권", START_DATE.strftime("%Y-%m-%d %H:%M:%S")),
        )

    events = []  # (plate, entry_dt, exit_dt)
    day = START_DATE
    for _ in range(DAYS):
        is_weekend = day.weekday() >= 5

        # 정기 출퇴근 차량
        if not is_weekend:
            for p in regulars:
                if random.random() < 0.72:
                    in_h, in_m = random.choice([7, 8, 8, 9]), random.randint(0, 59)
                    entry = day.replace(hour=in_h, minute=in_m, second=random.randint(0, 59))
                    stay = minutes_normal(560, 90, 240, 780)  # 약 9시간 근무 +- 편차
                    exit_ = entry + timedelta(minutes=stay)
                    events.append((p, entry, exit_))

        # 방문 차량 (평일 낮, 주말은 절반 이하)
        n_visitors = random.randint(15, 35) if not is_weekend else random.randint(4, 12)
        for _ in range(n_visitors):
            in_h = random.choices(
                population=list(range(8, 21)),
                weights=[2,3,5,6,7,8,7,6,7,8,6,5,3],  # 점심/오후 피크
            )[0]
            entry = day.replace(hour=in_h, minute=random.randint(0, 59), second=random.randint(0, 59))
            stay = minutes_normal(75, 60, 5, 300)
            exit_ = entry + timedelta(minutes=stay)
            events.append((rand_plate(), entry, exit_))

        day += timedelta(days=1)

    events.sort(key=lambda e: e[1])

    fee_cfg = dict(free_min=10, base_fee=1000, base_min=30, unit_fee=500, unit_min=10, daily_max=15000)

    def calc_fee_min(diff_min):
        if diff_min <= fee_cfg["free_min"]:
            return 0
        fee = fee_cfg["base_fee"]
        over = max(0, diff_min - fee_cfg["base_min"])
        units = -(-over // fee_cfg["unit_min"])
        fee += units * fee_cfg["unit_fee"]
        return min(fee, fee_cfg["daily_max"])

    fmt = "%Y-%m-%d %H:%M:%S"
    pl_rows, ee_rows = [], []
    for plate, entry, exit_ in events:
        color = random.choices(["WHITE", "GREENYELLOW", "YELLOW"], weights=[85, 10, 5])[0]
        ptype = "OLD" if color == "GREENYELLOW" else "NEW"
        conf_in = round(random.uniform(0.80, 0.99), 3)
        conf_out = round(random.uniform(0.78, 0.99), 3)

        pl_rows.append((plate, color, ptype, conf_in, entry.strftime(fmt), "CAM_01", "", 1, "demo"))
        pl_rows.append((plate, color, ptype, conf_out, exit_.strftime(fmt), "CAM_01", "", 1, "demo"))
        ee_rows.append((plate, "ENTRY", "CAM_01", entry.strftime(fmt), ""))
        ee_rows.append((plate, "EXIT",  "CAM_01", exit_.strftime(fmt), ""))

    conn.executemany(
        "INSERT INTO plate_logs (plate_text, plate_color, plate_type, confidence, timestamp, camera_id, image_path, frame_number, source_file) VALUES (?,?,?,?,?,?,?,?,?)",
        pl_rows,
    )
    conn.executemany(
        "INSERT INTO entry_exit_log (plate_text, event_type, camera_id, timestamp, image_path) VALUES (?,?,?,?,?)",
        ee_rows,
    )
    conn.commit()

    n_plate = conn.execute("SELECT COUNT(*) FROM plate_logs").fetchone()[0]
    n_ee = conn.execute("SELECT COUNT(*) FROM entry_exit_log").fetchone()[0]
    n_visits = len(events)
    total_revenue = sum(calc_fee_min(int((e[2]-e[1]).total_seconds()/60)) for e in events)
    print(f"완료: plate_logs {n_plate}건, entry_exit_log {n_ee}건, 방문 {n_visits}건")
    print(f"기간: {START_DATE.date()} ~ {END_DATE.date()} ({DAYS}일)")
    print(f"추정 총 매출: {total_revenue:,}원")
    print(f"DB 경로: {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()

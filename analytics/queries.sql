-- ============================================================
-- queries.sql — 주차장 입출차 데이터 SQL 분석
-- 대상: analytics/demo_parking.db (entry_exit_log, plate_logs)
-- ============================================================


-- ── 1. 입차·출차 페어링 (LAG 윈도우 함수) ────────────────────
--    entry_exit_log은 ENTRY/EXIT이 각각 별도 행이라, 같은 번호판의
--    직전 이벤트를 LAG로 가져와 한 행(방문 1건)으로 재구성한다.
WITH paired AS (
    SELECT plate_text, event_type, timestamp,
           LAG(timestamp)  OVER (PARTITION BY plate_text
                                 ORDER BY timestamp) AS prev_ts,
           LAG(event_type) OVER (PARTITION BY plate_text
                                 ORDER BY timestamp) AS prev_type
    FROM entry_exit_log
),
visits AS (
    SELECT plate_text,
           prev_ts   AS entry_time,
           timestamp AS exit_time,
           (JULIANDAY(timestamp) - JULIANDAY(prev_ts)) * 24 * 60
                     AS duration_min
    FROM paired
    WHERE event_type = 'EXIT' AND prev_type = 'ENTRY'
)
SELECT * FROM visits ORDER BY entry_time LIMIT 20;


-- ── 2. 일별 방문 대수 & 추정 매출 추이 ────────────────────────
--    요금 규칙(무료 10분 / 기본 1,000원(30분) / 10분당 500원 / 일 최대 15,000원)을
--    CASE 문으로 그대로 SQL에 이식해 매출을 계산한다.
WITH paired AS (
    SELECT plate_text, event_type, timestamp,
           LAG(timestamp)  OVER (PARTITION BY plate_text ORDER BY timestamp) AS prev_ts,
           LAG(event_type) OVER (PARTITION BY plate_text ORDER BY timestamp) AS prev_type
    FROM entry_exit_log
),
visits AS (
    SELECT plate_text, prev_ts AS entry_time, timestamp AS exit_time,
           CAST((JULIANDAY(timestamp) - JULIANDAY(prev_ts)) * 24 * 60 AS INTEGER) AS duration_min
    FROM paired
    WHERE event_type = 'EXIT' AND prev_type = 'ENTRY'
),
fee AS (
    SELECT *,
        CASE
            WHEN duration_min <= 10 THEN 0
            ELSE MIN(15000,
                1000 + CAST(((MAX(0, duration_min - 30) + 9) / 10) AS INTEGER) * 500)
        END AS fee
    FROM visits
)
SELECT
    DATE(entry_time)                       AS visit_date,
    STRFTIME('%w', entry_time)             AS weekday_num,   -- 0=일 .. 6=토
    COUNT(*)                               AS visit_count,
    SUM(fee)                               AS daily_revenue,
    ROUND(AVG(duration_min), 1)            AS avg_duration_min
FROM fee
GROUP BY visit_date
ORDER BY visit_date;


-- ── 3. 시간대별 입차 패턴 (피크타임 분석) ─────────────────────
SELECT
    STRFTIME('%H', timestamp) AS hour,
    COUNT(*)                  AS entry_count
FROM entry_exit_log
WHERE event_type = 'ENTRY'
GROUP BY hour
ORDER BY hour;


-- ── 4. 요일별 방문대수 / 평균 체류시간 (평일 vs 주말) ─────────
WITH paired AS (
    SELECT plate_text, event_type, timestamp,
           LAG(timestamp)  OVER (PARTITION BY plate_text ORDER BY timestamp) AS prev_ts,
           LAG(event_type) OVER (PARTITION BY plate_text ORDER BY timestamp) AS prev_type
    FROM entry_exit_log
),
visits AS (
    SELECT plate_text, prev_ts AS entry_time,
           CAST((JULIANDAY(timestamp) - JULIANDAY(prev_ts)) * 24 * 60 AS INTEGER) AS duration_min
    FROM paired
    WHERE event_type = 'EXIT' AND prev_type = 'ENTRY'
)
SELECT
    CASE STRFTIME('%w', entry_time)
        WHEN '0' THEN '일' WHEN '1' THEN '월' WHEN '2' THEN '화' WHEN '3' THEN '수'
        WHEN '4' THEN '목' WHEN '5' THEN '금' WHEN '6' THEN '토' END AS weekday,
    COUNT(*)                    AS visit_count,
    ROUND(AVG(duration_min), 1) AS avg_duration_min
FROM visits
GROUP BY STRFTIME('%w', entry_time)
ORDER BY STRFTIME('%w', entry_time);


-- ── 5. 재방문 차량 TOP 10 (단골 식별, RANK) ───────────────────
WITH visit_counts AS (
    SELECT plate_text, COUNT(*) AS visits
    FROM entry_exit_log
    WHERE event_type = 'ENTRY'
    GROUP BY plate_text
),
ranked AS (
    SELECT *, RANK() OVER (ORDER BY visits DESC) AS rnk
    FROM visit_counts
)
SELECT plate_text, visits, rnk FROM ranked WHERE rnk <= 10;


-- ── 6. 체류시간 구간별 분포 ───────────────────────────────────
WITH paired AS (
    SELECT plate_text, event_type, timestamp,
           LAG(timestamp)  OVER (PARTITION BY plate_text ORDER BY timestamp) AS prev_ts,
           LAG(event_type) OVER (PARTITION BY plate_text ORDER BY timestamp) AS prev_type
    FROM entry_exit_log
),
visits AS (
    SELECT CAST((JULIANDAY(timestamp) - JULIANDAY(prev_ts)) * 24 * 60 AS INTEGER) AS duration_min
    FROM paired
    WHERE event_type = 'EXIT' AND prev_type = 'ENTRY'
)
SELECT
    CASE
        WHEN duration_min <= 30  THEN '1) ~30분'
        WHEN duration_min <= 120 THEN '2) 30분~2시간'
        WHEN duration_min <= 360 THEN '3) 2~6시간'
        ELSE                          '4) 6시간~'
    END AS duration_bucket,
    COUNT(*) AS visit_count
FROM visits
GROUP BY duration_bucket
ORDER BY duration_bucket;


-- ── 7. 정기(화이트리스트) vs 일반 방문 비중 ────────────────────
SELECT
    CASE WHEN w.plate_text IS NOT NULL THEN '정기(화이트리스트)' ELSE '일반 방문' END AS visitor_type,
    COUNT(*) AS entry_count
FROM entry_exit_log e
LEFT JOIN whitelist w ON w.plate_text = e.plate_text
WHERE e.event_type = 'ENTRY'
GROUP BY visitor_type;

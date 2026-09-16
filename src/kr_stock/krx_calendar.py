"""KRX session calendar: weekends + holidays in SQLite `krx_holidays`."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Sequence

# Weekday KRX closures 2026 (weekends skipped separately).
KRX_HOLIDAYS_2026: Sequence[tuple[str, str]] = (
    ("2026-01-01", "신정"),
    ("2026-02-16", "설날"),
    ("2026-02-17", "설날"),
    ("2026-02-18", "설날"),
    ("2026-03-02", "삼일절 대체공휴일"),
    ("2026-05-01", "근로자의 날"),
    ("2026-05-05", "어린이날"),
    ("2026-05-25", "부처님오신날 대체공휴일"),
    ("2026-06-03", "전국동시지방선거"),
    ("2026-07-17", "제헌절"),
    ("2026-08-17", "광복절 대체공휴일"),
    ("2026-09-24", "추석"),
    ("2026-09-25", "추석"),
    ("2026-09-28", "추석 대체공휴일"),
    ("2026-10-05", "개천절 대체공휴일"),
    ("2026-10-09", "한글날"),
    ("2026-12-25", "성탄절"),
    ("2026-12-31", "연말 휴장"),
)


def ensure_holidays(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS krx_holidays (
            date TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL
        )"""
    )
    conn.executemany(
        "INSERT OR IGNORE INTO krx_holidays (date, name) VALUES (?, ?)",
        list(KRX_HOLIDAYS_2026),
    )
    conn.commit()
    conn.close()


def is_krx_holiday(db_path: str, date: str) -> bool:
    ensure_holidays(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT 1 FROM krx_holidays WHERE date = ?", (date,)).fetchone()
    conn.close()
    return row is not None


def is_session_day(db_path: str, date: str) -> bool:
    dt = datetime.strptime(date, "%Y-%m-%d")
    if dt.weekday() >= 5:
        return False
    return not is_krx_holiday(db_path, date)

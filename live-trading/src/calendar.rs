//! KRX session calendar: weekends + exchange holidays stored in SQLite.

use std::path::Path;

use chrono::{Datelike, NaiveDate, Weekday};
use rusqlite::Connection;
use tracing::info;

/// Weekday KRX closures in 2026 (weekends are handled separately).
/// Sources: KRX 월력 / 거래소 휴장 공지 (근로자의 날·연말 폐장·대체공휴일·지방선거 포함).
pub const KRX_HOLIDAYS_2026: &[(&str, &str)] = &[
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
];

pub fn ensure_holidays(db: &Path) -> rusqlite::Result<()> {
    let conn = Connection::open(db)?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS krx_holidays (
            date TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL
        )",
        [],
    )?;
    let mut inserted = 0i32;
    for (date, name) in KRX_HOLIDAYS_2026 {
        let n = conn.execute(
            "INSERT OR IGNORE INTO krx_holidays (date, name) VALUES (?1, ?2)",
            [*date, *name],
        )?;
        inserted += n as i32;
    }
    if inserted > 0 {
        info!("[calendar] seeded {inserted} KRX holiday row(s) into {}", db.display());
    }
    Ok(())
}

pub fn is_krx_holiday(db: &Path, date: &str) -> bool {
    let _ = ensure_holidays(db);
    Connection::open(db)
        .ok()
        .and_then(|c| {
            c.query_row(
                "SELECT 1 FROM krx_holidays WHERE date = ?1",
                [date],
                |_| Ok(()),
            )
            .ok()
        })
        .is_some()
}

pub fn is_session_day(db: &Path, date: &str) -> bool {
    let weekday = match NaiveDate::parse_from_str(date, "%Y-%m-%d") {
        Ok(d) => d.weekday(),
        Err(_) => return false,
    };
    matches!(
        weekday,
        Weekday::Mon | Weekday::Tue | Weekday::Wed | Weekday::Thu | Weekday::Fri
    ) && !is_krx_holiday(db, date)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp_db() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("krx_holidays_{nanos}.db"))
    }

    #[test]
    fn weekday_holiday_is_closed() {
        let p = tmp_db();
        ensure_holidays(&p).unwrap();
        assert!(is_krx_holiday(&p, "2026-05-01"));
        assert!(is_krx_holiday(&p, "2026-09-28"));
        assert!(is_krx_holiday(&p, "2026-12-31"));
        assert!(!is_session_day(&p, "2026-05-01"));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn weekend_is_closed_even_without_holiday_row() {
        let p = tmp_db();
        ensure_holidays(&p).unwrap();
        assert!(!is_session_day(&p, "2026-08-29")); // Sat
        assert!(!is_session_day(&p, "2026-08-30")); // Sun
        assert!(is_session_day(&p, "2026-08-28")); // Fri
        let _ = std::fs::remove_file(&p);
    }
}

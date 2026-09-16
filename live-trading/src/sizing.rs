//! Position sizing: hard cap per ticker, skip names that cannot buy 1 share.

use crate::scoring::Pick;

/// (sized pick, qty, buy_amount), skipped picks, alloc_per_stock
pub fn size_overnight_picks(
    picks: &[Pick],
    cash: f64,
    max_alloc: f64,
) -> (Vec<(Pick, i64, f64)>, Vec<Pick>, f64) {
    let cap = max_alloc.max(0.0);
    let mut skipped = Vec::new();
    let mut affordable = Vec::new();
    for p in picks {
        if p.close_price <= 0.0 {
            skipped.push(p.clone());
        } else if p.close_price > cap {
            skipped.push(p.clone());
        } else {
            affordable.push(p.clone());
        }
    }
    if affordable.is_empty() || cash <= 0.0 {
        return (Vec::new(), skipped, 0.0);
    }
    let alloc = (cash / affordable.len() as f64).min(cap);
    let mut sized = Vec::new();
    for p in affordable {
        let qty = (alloc / p.close_price).floor() as i64;
        if qty <= 0 {
            skipped.push(p);
            continue;
        }
        let buy_amount = qty as f64 * p.close_price;
        sized.push((p, qty, buy_amount));
    }
    (sized, skipped, alloc)
}

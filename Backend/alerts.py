"""Alert system: raises a warning when negative sentiment spikes."""
import os, datetime as dt

THRESHOLD = float(os.getenv("ALERT_NEGATIVE_THRESHOLD", 0.40))
WINDOW = int(os.getenv("ALERT_WINDOW_MINUTES", 10))
MIN_SAMPLE = 20


def evaluate(store):
    ratio, total = store.recent_negative_ratio(WINDOW)
    triggered = total >= MIN_SAMPLE and ratio >= THRESHOLD
    return {
        "alert": triggered,
        "level": "critical" if ratio >= THRESHOLD + 0.2 else ("warning" if triggered else "ok"),
        "negative_ratio": round(ratio, 3),
        "threshold": THRESHOLD,
        "window_minutes": WINDOW,
        "sample_size": total,
        "message": (f"Negative sentiment is {ratio:.0%} of the last {total} posts "
                    f"in {WINDOW} min (threshold {THRESHOLD:.0%})") if triggered
                   else "Sentiment within normal range",
        "checked_at": dt.datetime.utcnow().isoformat(),
    }

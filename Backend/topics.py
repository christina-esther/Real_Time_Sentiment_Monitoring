"""Topic / trending-keyword analysis over recent MongoDB documents."""
import re
from collections import Counter

STOP = set("""the and for that with this have from you your are was were will just about into over than
they them their our its his her not but all can has had how what when where which who why our""".split())


def trending_keywords(docs, limit=15):
    c = Counter()
    for d in docs:
        for w in re.findall(r"[a-z\u0900-\u097F]{4,}", (d.get("text") or "").lower()):
            if w not in STOP:
                c[w] += 1
    return [{"keyword": k, "count": n} for k, n in c.most_common(limit)]


def keyword_sentiment(docs, keywords):
    out = {}
    for kw in keywords:
        sub = [d for d in docs if kw in (d.get("text") or "").lower()]
        if not sub:
            continue
        pos = sum(1 for d in sub if d.get("sentiment") == "positive")
        neg = sum(1 for d in sub if d.get("sentiment") == "negative")
        out[kw] = {"mentions": len(sub),
                   "positive_pct": round(100 * pos / len(sub), 1),
                   "negative_pct": round(100 * neg / len(sub), 1)}
    return out

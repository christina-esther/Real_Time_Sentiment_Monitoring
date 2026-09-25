# MongoDB schema — `sentiment_monitor.sentiment_results`

```json
{
  "message_id": "9f2a1c7b4d...",        // md5(source + text), unique
  "text": "The product quality is amazing",
  "sentiment": "positive",              // positive | negative | neutral
  "confidence": 0.94,
  "timestamp": "2026-09-25T10:14:02",   // time of the original post
  "source": "reddit/technology",
  "language": "en",                     // en | hi | mr | ...
  "hashtags": ["camera", "review"],
  "ingested_at": ISODate("2026-09-25T10:14:05Z")
}
```

## Indexes

| Index | Type | Purpose |
|---|---|---|
| `message_id` | unique | idempotent writes, exactly-once effect |
| `ingested_at` | desc | live feed + timeline aggregations |
| `sentiment` | asc | analytics group-by |
| `hashtags` | multikey | trending topic queries |

## Useful queries

```js
db.sentiment_results.countDocuments()
db.sentiment_results.aggregate([{$group:{_id:"$sentiment",n:{$sum:1}}}])
db.sentiment_results.find().sort({ingested_at:-1}).limit(10)
```

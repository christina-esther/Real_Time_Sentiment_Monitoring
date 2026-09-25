"""
Kafka producer that streams REAL live social-media posts (Reddit API by default,
Twitter/X API optionally) into the Kafka topic. No random/fake message generation.

Usage:
    python Kafka/producer.py --source reddit --subs technology+india+movies
    python Kafka/producer.py --source twitter --query "iphone OR samsung lang:en"
"""
import argparse, hashlib, json, os, re, time, datetime as dt
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "social-posts")

HASHTAG_RE = re.compile(r"#(\w+)")


def detect_language(text):
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "unknown"


def envelope(text, source, author=None, created=None):
    return {
        "message_id": hashlib.md5((source + text[:120]).encode()).hexdigest(),
        "text": text,
        "timestamp": (created or dt.datetime.utcnow()).isoformat(),
        "source": source,
        "author": author,
        "language": detect_language(text),
        "hashtags": HASHTAG_RE.findall(text),
    }


def reddit_stream(subs):
    import praw
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "rt-sentiment/1.0"),
    )
    for comment in reddit.subreddit(subs).stream.comments(skip_existing=True):
        body = (comment.body or "").strip()
        if len(body) < 15 or body == "[deleted]":
            continue
        yield envelope(body, f"reddit/{comment.subreddit.display_name}",
                       str(comment.author),
                       dt.datetime.utcfromtimestamp(comment.created_utc))


def twitter_stream(query):
    import tweepy
    client = tweepy.Client(bearer_token=os.getenv("TWITTER_BEARER_TOKEN"))

    class P(tweepy.StreamingClient):
        def __init__(self, sink, **kw):
            super().__init__(**kw); self.sink = sink
        def on_tweet(self, tweet):
            self.sink.append(envelope(tweet.text, "twitter"))

    sink = []
    stream = P(sink, bearer_token=os.getenv("TWITTER_BEARER_TOKEN"))
    for r in stream.get_rules().data or []:
        stream.delete_rules(r.id)
    stream.add_rules(tweepy.StreamRule(query))
    stream.filter(threaded=True)
    while True:
        while sink:
            yield sink.pop(0)
        time.sleep(0.5)


def main(a):
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP.split(","),
        value_serializer=lambda v: json.dumps(v).encode(),
        key_serializer=lambda k: k.encode(),
        linger_ms=50, retries=5, acks="all")

    gen = reddit_stream(a.subs) if a.source == "reddit" else twitter_stream(a.query)
    sent = 0
    print(f"Streaming {a.source} -> kafka://{BOOTSTRAP}/{TOPIC}  (Ctrl-C to stop)")
    for msg in gen:
        producer.send(TOPIC, key=msg["message_id"], value=msg)
        sent += 1
        if sent % 10 == 0:
            producer.flush()
            print(f"[{sent}] {msg['source']} | {msg['language']} | {msg['text'][:70]!r}")
        if a.limit and sent >= a.limit:
            break
    producer.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["reddit", "twitter"], default="reddit")
    ap.add_argument("--subs", default=os.getenv("REDDIT_SUBREDDITS", "technology+india"))
    ap.add_argument("--query", default="(product OR service) lang:en -is:retweet")
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())

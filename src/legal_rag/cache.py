from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from legal_rag.config import (
    CACHE_ENABLED,
    CACHE_PREFIX,
    CACHE_VERSION,
    REDIS_SOCKET_TIMEOUT,
    REDIS_URL,
)

logger = logging.getLogger(__name__)

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None

_client: Any = None
_client_lock = threading.Lock()
_disabled_until = 0.0  # after a Redis failure, skip it for a short while
_RETRY_AFTER_SECONDS = 30

_stats = {"hits": 0, "misses": 0, "errors": 0, "writes": 0}
_stats_lock = threading.Lock()


def _count(name: str) -> None:
    with _stats_lock:
        _stats[name] += 1


# ============================================================
# Connection (thread-safe, fail-open)
# ============================================================

def _get_client() -> Any:
    global _client

    if not CACHE_ENABLED or redis is None:
        return None

    if time.monotonic() < _disabled_until:
        return None

    if _client is None:
        # FastAPI runs `ask` in many worker threads: create the client once.
        with _client_lock:
            if _client is None:
                _client = redis.Redis.from_url(
                    REDIS_URL,
                    socket_connect_timeout=REDIS_SOCKET_TIMEOUT,
                    socket_timeout=REDIS_SOCKET_TIMEOUT,
                    health_check_interval=30,
                    decode_responses=True,
                )

    return _client


def _fail(exc: Exception) -> None:
    global _disabled_until
    _disabled_until = time.monotonic() + _RETRY_AFTER_SECONDS
    _count("errors")
    logger.warning(
        "Redis unavailable, cache bypassed for %ss: %s",
        _RETRY_AFTER_SECONDS,
        exc,
    )


def ping() -> bool | None:
    """True = Redis answers, False = down, None = cache disabled."""
    global _disabled_until

    if not CACHE_ENABLED or redis is None:
        return None

    # An explicit health check ignores the back-off window.
    _disabled_until = 0.0
    client = _get_client()
    if client is None:
        return None

    try:
        return bool(client.ping())
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
        return False


def cache_stats() -> dict[str, Any]:
    with _stats_lock:
        return {"enabled": CACHE_ENABLED and redis is not None, **_stats}


# ============================================================
# Keys
# ============================================================

_AR_NORMALIZE = str.maketrans(
    {
        ord("أ"): "ا",
        ord("إ"): "ا",
        ord("آ"): "ا",
        ord("ى"): "ي",
    }
)


def normalize_question(question: str) -> str:
    """Make trivially different spellings of a question share one key."""
    text = unicodedata.normalize("NFKC", question)
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)  # tashkeel + tatweel
    text = text.translate(_AR_NORMALIZE)
    text = re.sub(r"\s+", " ", text).strip().rstrip("؟?!. ").lower()
    return text


def make_key(kind: str, question: str, **params: Any) -> str:
    raw = json.dumps(
        {"q": normalize_question(question), **params},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{CACHE_PREFIX}:{CACHE_VERSION}:{kind}:{digest}"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_fingerprint(path: Path) -> str:
    """Cheap fingerprint (size + mtime) so a data change invalidates the cache."""
    try:
        stat = path.stat()
    except OSError:
        return "missing"
    return f"{stat.st_size}-{int(stat.st_mtime)}"


# ============================================================
# Get / set
# ============================================================

def cache_get(key: str) -> Any | None:
    client = _get_client()
    if client is None:
        return None

    try:
        value = client.get(key)
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
        return None

    if value is None:
        _count("misses")
        return None

    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Corrupted cache entry dropped: %s", key)
        try:
            client.delete(key)
        except Exception:  # noqa: BLE001
            pass
        return None

    _count("hits")
    return result


def cache_set(key: str, value: Any, ttl: int) -> None:
    client = _get_client()
    if client is None or ttl <= 0:
        return

    # Serialization problems are our bug, not a Redis outage:
    # log them but do not switch the cache off.
    try:
        payload = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        logger.exception("Cache value is not serializable, skipped: %s", key)
        return

    try:
        client.set(key, payload, ex=ttl)
        _count("writes")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


def clear_cache() -> int:
    """Delete every key of this app (all versions). Returns deleted count."""
    global _disabled_until
    _disabled_until = 0.0

    client = _get_client()
    if client is None:
        return 0

    deleted = 0
    try:
        batch: list[str] = []
        for key in client.scan_iter(match=f"{CACHE_PREFIX}:*", count=500):
            batch.append(key)
            if len(batch) >= 500:
                deleted += client.unlink(*batch)
                batch = []
        if batch:
            deleted += client.unlink(*batch)
    except Exception as exc:  # noqa: BLE001
        _fail(exc)

    return deleted


# ============================================================
# Single flight: identical concurrent requests run the pipeline once
# ============================================================

_flight_lock = threading.Lock()
_flights: dict[str, list[Any]] = {}  # key -> [lock, waiters]


@contextmanager
def single_flight(key: str) -> Iterator[None]:
    """
    Serialize work for the same key inside this process. The second caller
    waits, then finds the answer in the cache instead of recomputing it.
    """
    with _flight_lock:
        entry = _flights.setdefault(key, [threading.Lock(), 0])
        entry[1] += 1

    try:
        with entry[0]:
            yield
    finally:
        with _flight_lock:
            entry[1] -= 1
            if entry[1] == 0:
                _flights.pop(key, None)


if __name__ == "__main__":  # python -m legal_rag.cache [ping|stats|clear]
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "ping"

    if command == "clear":
        print(f"deleted {clear_cache()} keys")
    elif command == "stats":
        print(cache_stats())
    else:
        print({"redis": ping()})
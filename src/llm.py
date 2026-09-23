"""Shared Groq client with self-throttling and retry.

The free tier caps tokens per minute, so a naive loop over the query set
trips a 429 partway through and leaves an experiment half-scored. This module
keeps a rolling one-minute token budget, sleeps before it would exceed it, and
retries on 429 with exponential backoff.

The gpt-oss models are reasoning models: they spend completion tokens on a
separate `reasoning` field before emitting `content`. MAX_TOKENS has to be
generous or `content` comes back empty.
"""

import os
import re
import time
from collections import deque

from dotenv import load_dotenv
from groq import Groq

from config import (
    DAILY_LIMIT_MAX_WAIT_SECONDS,
    DAILY_LIMIT_POLL_SECONDS,
    MAX_TOKENS,
    TEMPERATURE,
    TOKENS_PER_MINUTE,
)

load_dotenv()

# GROQ_NEW_API_KEY wins when it is set, so a run stopped by an exhausted daily
# quota can be finished on a fresh key without editing anything. The judge and
# generator models are unchanged by this -- only the account the calls are
# billed to -- so results stay comparable across the switch.
_api_key = os.getenv("GROQ_NEW_API_KEY") or os.getenv("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError(
        "Neither GROQ_NEW_API_KEY nor GROQ_API_KEY is set. "
        "Copy .env.example to .env and fill one in."
    )

client = Groq(api_key=_api_key)

# (timestamp, tokens) for calls made in the last 60 seconds.
_recent = deque()
_ESTIMATED_TOKENS_PER_CALL = 1200


def _spent_last_minute():
    now = time.time()
    while _recent and now - _recent[0][0] > 60:
        _recent.popleft()
    return sum(tokens for _, tokens in _recent)


def _throttle(estimate):
    """Block until `estimate` more tokens fit inside the per-minute budget."""
    while _recent and _spent_last_minute() + estimate > TOKENS_PER_MINUTE:
        wait = 60 - (time.time() - _recent[0][0]) + 0.5
        if wait <= 0:
            _recent.popleft()
            continue
        print(f"    [throttle] {_spent_last_minute()} tok used this minute, "
              f"sleeping {wait:.0f}s")
        time.sleep(wait)


_RETRY_HINT = re.compile(
    r"try again in\s*(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s", re.IGNORECASE)


def _retry_after(message):
    """Seconds the server asked us to wait, or None if it did not say."""
    match = _RETRY_HINT.search(message)
    if not match:
        return None
    minutes = float(match.group(1) or 0)
    return minutes * 60 + float(match.group(2))


def _is_daily_limit(message):
    lowered = message.lower()
    return "per day" in lowered or "tpd" in lowered or "tpd)" in lowered


def complete(prompt, model, max_retries=6):
    """Send a single-turn completion and return the message content.

    Two kinds of 429 are handled differently. A per-minute rate limit clears in
    seconds, so it is retried with exponential backoff and a small attempt
    budget. A per-day token limit clears over hours; backing off exponentially
    against it just exhausts the attempt budget and throws away the rest of the
    run, so it is instead waited out on the server's own schedule until
    DAILY_LIMIT_MAX_WAIT_SECONDS. Callers that cache per query -- as
    experiments.py does -- then resume rather than restart.
    """
    _throttle(_ESTIMATED_TOKENS_PER_CALL)

    delay = 5.0
    attempt = 0
    daily_waited = 0.0
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        except Exception as e:
            message = str(e)
            is_rate_limit = "429" in message or "rate_limit" in message.lower()
            if not is_rate_limit:
                raise

            if _is_daily_limit(message):
                # Honour the server's hint, but never spin: it reports the wait
                # for the next token to free, which early on is only seconds.
                hint = _retry_after(message) or DAILY_LIMIT_POLL_SECONDS
                wait = max(hint + 5.0, DAILY_LIMIT_POLL_SECONDS)
                if daily_waited + wait > DAILY_LIMIT_MAX_WAIT_SECONDS:
                    raise RuntimeError(
                        "daily token limit still in force after "
                        f"{daily_waited / 3600:.1f}h of waiting; giving up so "
                        "the cached results are not left in limbo. Re-run "
                        "experiments.py once the quota resets -- completed "
                        f"queries are cached and will be skipped.\n{message}"
                    ) from e
                daily_waited += wait
                print(f"    [daily quota] exhausted for {model}; sleeping "
                      f"{wait / 60:.1f}min "
                      f"({daily_waited / 3600:.1f}h waited so far)")
                time.sleep(wait)
                continue

            attempt += 1
            if attempt >= max_retries:
                raise
            print(f"    [retry {attempt}/{max_retries}] {message[:110]} "
                  f"-- waiting {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue

        usage = getattr(response, "usage", None)
        _recent.append((time.time(), getattr(usage, "total_tokens", _ESTIMATED_TOKENS_PER_CALL)))

        content = response.choices[0].message.content
        if content is None:
            content = ""
        return content.strip()

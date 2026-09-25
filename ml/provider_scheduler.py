"""Jikan pacing for the Python content-feature builder. No provider calls at import time."""

from __future__ import annotations

import email.utils
import json
import math
import random
import time
import urllib.error
import urllib.request
from typing import Callable, Dict


class RequestBudgetExceeded(RuntimeError):
    pass


class RequestCancelled(BaseException):
    # The feature loop catches ordinary Exception per item; cancellation stops the run.
    pass


def retry_after_seconds(value: str | None, wall_now: float) -> float | None:
    if not value:
        return None
    trimmed = value.strip()
    if trimmed.replace(".", "", 1).isdigit():
        seconds = float(trimmed)
        return seconds if math.isfinite(seconds) else None
    if trimmed.startswith(("-", "+")) and trimmed[1:].replace(".", "", 1).isdigit():
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(trimmed)
        return max(0.0, parsed.timestamp() - wall_now)
    except (TypeError, ValueError, OverflowError):
        return None


class JikanRequestScheduler:
    """One process-local Jikan quota for all feature requests and their retries."""

    def __init__(
        self,
        delay_ms: int = 350,
        timeout_s: float = 30.0,
        retry_budget_s: float = 120.0,
        opener: Callable = urllib.request.urlopen,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        if delay_ms < 0 or timeout_s <= 0 or retry_budget_s <= 0:
            raise ValueError("Invalid Jikan request limits")
        self.interval_s = delay_ms / 1000.0
        self.timeout_s = timeout_s
        self.retry_budget_s = retry_budget_s
        self.opener = opener
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.sleep = sleep
        self.random_value = random_value
        self.starts: list[float] = []
        self.cooldown_until = 0.0

    def get_json(
        self,
        url: str,
        max_retries: int,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Dict[str, object] | None:
        deadline = self.monotonic() + self.retry_budget_s
        for attempt in range(max(0, max_retries) + 1):
            self._wait_for_slot(deadline, cancelled)
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            try:
                with self.opener(request, timeout=self.timeout_s) as response:
                    self._check_cancel(cancelled)
                    if response.status == 404:
                        return None
                    if response.status != 200:
                        raise urllib.error.HTTPError(
                            url, response.status, "Unexpected status", response.headers, None
                        )
                    raw = response.read()
                    self._check_cancel(cancelled)
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                self._check_cancel(cancelled)
                if exc.code == 404:
                    return None
                retry_after = None
                if exc.code in (429, 503):
                    retry_after = retry_after_seconds(
                        exc.headers.get("Retry-After") if exc.headers else None,
                        self.wall_time(),
                    )
                    if retry_after is not None:
                        self.cooldown_until = max(
                            self.cooldown_until, self.monotonic() + retry_after
                        )
                if exc.code not in (408, 425, 429) and exc.code < 500:
                    raise
                if attempt >= max_retries:
                    raise
                delay = max(self._backoff(attempt), retry_after or 0.0)
                if self.monotonic() + delay >= deadline:
                    raise
            except (TimeoutError, urllib.error.URLError):
                self._check_cancel(cancelled)
                if attempt >= max_retries:
                    raise
                delay = self._backoff(attempt)
                if self.monotonic() + delay >= deadline:
                    raise
            self._sleep_checked(delay, cancelled)
        raise RequestBudgetExceeded("Jikan retry attempts exhausted")

    def _wait_for_slot(self, deadline: float, cancelled: Callable[[], bool]) -> None:
        while True:
            self._check_cancel(cancelled)
            now = self.monotonic()
            if now >= deadline:
                raise RequestBudgetExceeded("Jikan request waited past its budget")
            self.starts = [start for start in self.starts if start > now - 60.0]
            next_at = max(now, self.cooldown_until)
            for limit, period in ((3, 1.0), (60, 60.0), (1, self.interval_s)):
                if period <= 0:
                    continue
                recent = [start for start in self.starts if start > now - period]
                if len(recent) >= limit:
                    next_at = max(next_at, recent[-limit] + period)
            if next_at <= now:
                self.starts.append(now)
                return
            if next_at >= deadline:
                raise RequestBudgetExceeded("Jikan rate wait exceeds its request budget")
            self._sleep_checked(next_at - now, cancelled)

    def _backoff(self, attempt: int) -> float:
        jitter = min(1.0, max(0.0, self.random_value())) * 0.25
        return min(1.0 * (2 ** attempt), 8.0) + jitter

    def _sleep_checked(self, seconds: float, cancelled: Callable[[], bool]) -> None:
        end = self.monotonic() + seconds
        while self.monotonic() < end:
            self._check_cancel(cancelled)
            self.sleep(min(1.0, end - self.monotonic()))
        self._check_cancel(cancelled)

    @staticmethod
    def _check_cancel(cancelled: Callable[[], bool]) -> None:
        if cancelled():
            raise RequestCancelled("Jikan request canceled")

import sys
import unittest
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provider_scheduler import (  # noqa: E402
    JikanRequestScheduler,
    RequestBudgetExceeded,
    RequestCancelled,
    retry_after_seconds,
)


class FakeClock:
    def __init__(self):
        self.elapsed = 0.0
        self.epoch = datetime(2026, 9, 24, 12, tzinfo=timezone.utc).timestamp()
        self.sleeps = []

    def monotonic(self):
        return self.elapsed

    def wall_time(self):
        return self.epoch + self.elapsed

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.elapsed += seconds


class FakeResponse:
    status = 200
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, _kind, _value, _traceback):
        return False

    def read(self):
        return b'{"data": {}}'


class ProviderSchedulerTests(unittest.TestCase):
    def make_scheduler(self, clock, opener, delay_ms=0, retry_budget_s=120):
        return JikanRequestScheduler(
            delay_ms=delay_ms,
            retry_budget_s=retry_budget_s,
            opener=opener,
            monotonic=clock.monotonic,
            wall_time=clock.wall_time,
            sleep=clock.sleep,
            random_value=lambda: 0.5,
        )

    def test_second_and_minute_windows_cover_all_requests(self):
        clock = FakeClock()
        starts = []

        def opener(_request, timeout):
            self.assertEqual(timeout, 30)
            starts.append(clock.elapsed)
            return FakeResponse()

        scheduler = self.make_scheduler(clock, opener)
        for index in range(65):
            self.assertEqual(scheduler.get_json(f"https://api.jikan.moe/v4/anime/{index}", 0),
                             {"data": {}})
        self.assertEqual(len(starts), 65)
        for at in starts:
            self.assertLessEqual(sum(at <= time < at + 1 for time in starts), 3)
            self.assertLessEqual(sum(at <= time < at + 60 for time in starts), 60)
        self.assertGreaterEqual(starts[60], 60)

    def test_retry_after_seconds_date_and_budget(self):
        clock = FakeClock()
        self.assertEqual(retry_after_seconds("2", clock.wall_time()), 2)
        http_date = datetime.fromtimestamp(clock.epoch + 5, timezone.utc).strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        )
        self.assertEqual(retry_after_seconds(http_date, clock.wall_time()), 5)
        self.assertIsNone(retry_after_seconds("-1", clock.wall_time()))
        starts = []

        def opener(request, timeout):
            self.assertEqual(timeout, 30)
            starts.append(clock.elapsed)
            if len(starts) == 1:
                raise urllib.error.HTTPError(request.full_url, 429, "limited", {"Retry-After": "2"}, None)
            return FakeResponse()

        scheduler = self.make_scheduler(clock, opener)
        self.assertEqual(scheduler.get_json("https://api.jikan.moe/v4/anime/101", 2), {"data": {}})
        self.assertEqual(starts, [0, 2])

        blocked = self.make_scheduler(clock, lambda *_: (_ for _ in ()).throw(AssertionError("network")),
                                      retry_budget_s=30)
        blocked.cooldown_until = clock.elapsed + 120
        with self.assertRaises(RequestBudgetExceeded):
            blocked.get_json("https://api.jikan.moe/v4/anime/102", 1)

    def test_network_retry_has_bounded_backoff_and_timeout(self):
        clock = FakeClock()
        starts = []

        def opener(_request, timeout):
            self.assertEqual(timeout, 30)
            starts.append(clock.elapsed)
            if len(starts) == 1:
                raise urllib.error.URLError("synthetic timeout")
            return FakeResponse()

        scheduler = self.make_scheduler(clock, opener)
        self.assertEqual(scheduler.get_json("https://api.jikan.moe/v4/anime/101", 1), {"data": {}})
        self.assertGreaterEqual(starts[1] - starts[0], 1)
        self.assertLessEqual(starts[1] - starts[0], 1.25)

    def test_not_found_and_bad_request_do_not_retry(self):
        for status in (404, 400):
            clock = FakeClock()
            calls = []

            def opener(request, timeout):
                calls.append((request.full_url, timeout))
                raise urllib.error.HTTPError(request.full_url, status, "synthetic", {}, None)

            scheduler = self.make_scheduler(clock, opener)
            if status == 404:
                self.assertIsNone(scheduler.get_json("https://api.jikan.moe/v4/anime/999", 3))
            else:
                with self.assertRaises(urllib.error.HTTPError):
                    scheduler.get_json("https://api.jikan.moe/v4/anime/999", 3)
            self.assertEqual(len(calls), 1)
            self.assertEqual(clock.sleeps, [])

    def test_cancel_during_a_queued_rate_wait_never_enters_transport(self):
        clock = FakeClock()
        calls = []
        canceled = False

        def opener(_request, timeout):
            self.assertEqual(timeout, 30)
            calls.append(clock.elapsed)
            return FakeResponse()

        def sleep_and_cancel(seconds):
            nonlocal canceled
            clock.sleep(seconds)
            canceled = True

        scheduler = self.make_scheduler(clock, opener, delay_ms=5_000)
        scheduler.get_json("https://api.jikan.moe/v4/anime/101", 0)
        scheduler.sleep = sleep_and_cancel
        with self.assertRaises(RequestCancelled):
            scheduler.get_json("https://api.jikan.moe/v4/anime/102", 0,
                               cancelled=lambda: canceled)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()

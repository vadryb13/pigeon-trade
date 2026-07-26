"""Тесты retry + circuit breaker для MOEXAdapter."""
from __future__ import annotations

import time

import pytest
import requests

from aqr.data.moex import CircuitOpenError, MOEXAdapter

# ── Helpers ──────────────────────────────────────────────────────

class _FakeResponse:
    """Минимальный мок requests.Response."""

    def __init__(self, status_code=200, json_data=None, reason="OK"):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.reason = reason

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(
                f"{self.status_code} {self.reason}", response=self,
            )
            raise err

    def json(self):
        return self._json_data


def _make_adapter(call_log, response_seq=None):
    """Создать MOEXAdapter с подменённым session.request.

    Args:
        call_log: список (url, kwargs) каждого вызова
        response_seq: список Response|Exception — что вернуть на каждый вызов по индексу
    """
    if response_seq is None:
        response_seq = []

    adapter = MOEXAdapter(
        rate_limit_ms=0,  # убираем rate-limit для скорости тестов
        timeout=1,
        max_retries=3,
        cb_threshold=5,
        cb_recovery_seconds=60,
    )

    def fake_request(method, url, params=None, **kw):
        call_log.append((url, params))
        idx = len(call_log) - 1
        if idx < len(response_seq):
            v = response_seq[idx]
            if isinstance(v, Exception):
                raise v
            return v
        return _FakeResponse(200, {"candles": {"columns": [], "data": []}})

    adapter.session.request = fake_request  # type: ignore
    return adapter


# ── Timeout defaults ─────────────────────────────────────────────

class TestTimeoutDefaults:
    def test_default_timeout_is_10_seconds(self):
        """Per TASKS.md 7.1: HTTP timeout = 10 секунд."""
        adapter = MOEXAdapter(rate_limit_ms=0)
        assert adapter.timeout == 10

    def test_default_max_retries_is_3(self):
        adapter = MOEXAdapter(rate_limit_ms=0)
        assert adapter.max_retries == 3

    def test_default_cb_threshold_is_5(self):
        adapter = MOEXAdapter(rate_limit_ms=0)
        assert adapter.cb_threshold == 5


# ── Retry ────────────────────────────────────────────────────────

class TestRetry:
    def test_retry_on_5xx_eventually_succeeds(self):
        """500 → 500 → 200: должны получить ответ после 3-й попытки."""
        log = []
        adapter = _make_adapter(
            log,
            [
                _FakeResponse(500, reason="Internal Error"),
                _FakeResponse(500, reason="Internal Error"),
                _FakeResponse(200, {"candles": {"columns": [], "data": []}}),
            ],
        )
        resp = adapter._request_with_retry(
            "GET", "https://iss.moex.com/foo", ticker="SBER",
        )
        assert resp.status_code == 200
        # 3 попытки всего (1+1+1), хотя 5xx-ы ретраятся
        assert len(log) == 3

    def test_retry_on_connection_error(self):
        """ConnectionError → ConnectionError → 200: должны получить ответ."""
        log = []
        adapter = _make_adapter(log)
        # Подменяем session.request вместо session.get
        attempts = {"n": 0}

        def fake_request(method, url, params=None, **kw):
            log.append((url, params))
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise requests.ConnectionError("network down")
            return _FakeResponse(200, {"candles": {"columns": [], "data": []}})

        adapter.session.request = fake_request  # type: ignore
        resp = adapter._request_with_retry("GET", "https://iss.moex.com/foo", ticker="SBER")
        assert resp.status_code == 200
        assert len(log) == 3

    def test_retry_exhausted_raises_last_exception(self):
        """Все max_retries попыток падают → бросает оригинальное исключение."""
        log = []

        def fake_request(method, url, params=None, **kw):
            log.append(url)
            raise requests.ConnectionError("persistent failure")

        adapter = MOEXAdapter(rate_limit_ms=0, max_retries=3)
        adapter.session.request = fake_request  # type: ignore
        with pytest.raises(requests.ConnectionError, match="persistent failure"):
            adapter._request_with_retry("GET", "https://iss.moex.com/foo", ticker="SBER")
        # 3 попытки
        assert len(log) == 3

    def test_4xx_does_not_retry(self):
        """404 — НЕ ретраится, отдаём сразу."""
        log = []

        def fake_request(method, url, params=None, **kw):
            log.append(url)
            return _FakeResponse(404, reason="Not Found")

        adapter = MOEXAdapter(rate_limit_ms=0, max_retries=3)
        adapter.session.request = fake_request  # type: ignore
        resp = adapter._request_with_retry("GET", "https://iss.moex.com/foo", ticker="SBER")
        assert resp.status_code == 404
        # Только 1 вызов — без retry
        assert len(log) == 1


# ── Circuit Breaker ──────────────────────────────────────────────

class TestCircuitBreaker:
    def test_breaker_opens_after_n_failures(self):
        """После 5 ошибок подряд CB открыт, 6-й вызов сразу CircuitOpenError."""
        log = []

        def fake_request(method, url, params=None, **kw):
            log.append(url)
            raise requests.ConnectionError("boom")

        adapter = MOEXAdapter(
            rate_limit_ms=0, max_retries=1, cb_threshold=5, cb_recovery_seconds=60,
        )
        adapter.session.request = fake_request  # type: ignore

        # 5 ошибок
        for _ in range(5):
            with pytest.raises(requests.ConnectionError):
                adapter._request_with_retry("GET", "https://x", ticker="SBER")

        # CB должен быть открыт
        assert adapter.is_breaker_open("SBER") is True

        # 6-й вызов — мгновенный CircuitOpenError, БЕЗ сетевого вызова
        calls_before = len(log)
        with pytest.raises(CircuitOpenError, match="SBER"):
            adapter._request_with_retry("GET", "https://x", ticker="SBER")
        # Никаких новых сетевых вызовов
        assert len(log) == calls_before

    def test_breaker_isolated_per_ticker(self):
        """SBER падает → CB открыт для SBER. GAZP всё ещё работает."""
        log = []

        def fake_request(method, url, params=None, **kw):
            log.append((url, params))
            ticker = (params or {}).get("ticker", "")
            # Если тикер в URL содержит "sber" — падаем
            if "sber" in url.lower():
                raise requests.ConnectionError("sber down")
            return _FakeResponse(200, {"candles": {"columns": [], "data": []}})

        adapter = MOEXAdapter(
            rate_limit_ms=0, max_retries=1, cb_threshold=3, cb_recovery_seconds=60,
        )
        adapter.session.request = fake_request  # type: ignore

        # 3 ошибки для SBER — открываем CB
        for _ in range(3):
            with pytest.raises(requests.ConnectionError):
                adapter._request_with_retry(
                    "GET", "https://iss.moex.com/sber", ticker="SBER",
                )

        assert adapter.is_breaker_open("SBER") is True
        # GAZP не затронут
        assert adapter.is_breaker_open("GAZP") is False

        # SBER → CircuitOpenError мгновенно
        with pytest.raises(CircuitOpenError):
            adapter._request_with_retry("GET", "https://iss.moex.com/sber", ticker="SBER")

        # GAZP → успешный запрос (сетевой вызов происходит)
        resp = adapter._request_with_retry("GET", "https://iss.moex.com/gazp", ticker="GAZP")
        assert resp.status_code == 200

    def test_breaker_recovers_after_timeout(self):
        """CB автоматически закрывается через cb_recovery_seconds."""
        log = []

        def fake_request(method, url, params=None, **kw):
            log.append(url)
            raise requests.ConnectionError("boom")

        adapter = MOEXAdapter(
            rate_limit_ms=0, max_retries=1, cb_threshold=2, cb_recovery_seconds=1,
        )
        adapter.session.request = fake_request  # type: ignore

        # 2 ошибки → CB открыт на 1 секунду
        for _ in range(2):
            with pytest.raises(requests.ConnectionError):
                adapter._request_with_retry("GET", "https://x", ticker="SBER")
        assert adapter.is_breaker_open("SBER") is True

        # Ждём recovery
        time.sleep(1.1)
        # Теперь CB закрыт, но успеха нет — ошибка возвращается нормально
        with pytest.raises(requests.ConnectionError):
            adapter._request_with_retry("GET", "https://x", ticker="SBER")

    def test_success_resets_failure_counter(self):
        """Успех после 1-2 ошибок сбрасывает счётчик — CB не открывается."""
        log = []
        attempts = {"n": 0}

        def fake_request(method, url, params=None, **kw):
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise requests.ConnectionError("transient")
            return _FakeResponse(200, {"candles": {"columns": [], "data": []}})

        adapter = MOEXAdapter(
            rate_limit_ms=0, max_retries=1, cb_threshold=5,
        )
        adapter.session.request = fake_request  # type: ignore

        # 1-й: 1 попытка → ConnectionError (counter=1)
        with pytest.raises(requests.ConnectionError):
            adapter._request_with_retry("GET", "https://x", ticker="SBER")
        # 2-й: 1 попытка → ConnectionError (counter=2)
        with pytest.raises(requests.ConnectionError):
            adapter._request_with_retry("GET", "https://x", ticker="SBER")

        # Переключаем поведение на успех
        def fake_request_success(method, url, params=None, **kw):
            return _FakeResponse(200, {"candles": {"columns": [], "data": []}})

        adapter.session.request = fake_request_success  # type: ignore

        # Успех → counter сбрасывается на 0
        resp = adapter._request_with_retry("GET", "https://x", ticker="SBER")
        assert resp.status_code == 200
        assert adapter._breakers["SBER"].failures == 0

        # Теперь можно 4 раза упасть и CB всё ещё не открыт (counter начинает с 0)
        def fail_again(method, url, params=None, **kw):
            raise requests.ConnectionError("boom")

        adapter.session.request = fail_again  # type: ignore
        for _ in range(4):
            with pytest.raises(requests.ConnectionError):
                adapter._request_with_retry("GET", "https://x", ticker="SBER")
        # counter=4 < threshold=5 → CB не открыт
        assert adapter.is_breaker_open("SBER") is False

    def test_reset_breakers_clears_state(self):
        """reset_breakers() — админ-команда для сброса."""
        adapter = MOEXAdapter(rate_limit_ms=0, cb_threshold=1, cb_recovery_seconds=300)
        adapter._breakers["SBER"] = _FakeState(failures=10, open_until=time.time() + 300)
        assert adapter.is_breaker_open("SBER") is True
        adapter.reset_breakers()
        assert adapter.is_breaker_open("SBER") is False


class _FakeState:
    """Подмена _BreakerState для теста reset_breakers."""
    def __init__(self, failures, open_until):
        self.failures = failures
        self.open_until = open_until


# NOTE: `TestCorporateActions` и `TestSecuritiesList` удалены вместе с
# соответствующими методами MOEXAdapter (DEAD-2). Методы были заготовкой
# для PIT-пайплайна, но не подключены к `candles()`.

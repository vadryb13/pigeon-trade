# AGENTS.md

Контракт для LLM-агентов (Claude Code, Codex, Cursor), работающих с этим репозиторием.

## Что это за проект

AQR — тонкий пайплайн для проверки торговых гипотез на MOEX. Вход — цель на русском («проверь momentum на голубых фишках»), выход — топ-5 гипотез с Deflated Sharpe / CPCV / PBO + нарратив.

Трёхслойная архитектура:

- **Storage** — Postgres + pgvector + Alembic (`aqr/registry/`, `aqr/db.py`). Таблицы `sessions → runs → hypotheses` + `chat_messages` для истории WebSocket.
- **Tool Layer** — `aqr/tools/`: `ToolSpec` + `ToolRegistry` + **13 инструментов** (`plan_research`, `load_prices`, `generate_hypotheses`, `backtest_one`, `validate_portfolio`, `extract_insights`, `review_insights`, `narrate`, `get_run`, `list_runs`, `compare_runs`, `search_similar_hypotheses`, `find_duplicates`).
- **Agent Layer** — `aqr/agent/`: LangGraph-граф; `SessionContext` подмешивает историю и непроверенные комбинации в `session_context_prompt`.
- **Chat Layer** — `aqr/chat/`: WebSocket `/chat/{token}` + Web UI на vanilla JS (тёмная тема, markdown, slash-команды).

Один процесс. Фон — `aqr.tasks.schedule(...)` (с retention set против GC). Никакого Redis/Celery.

## Точки входа

| Что | Команда / URL |
|---|---|
| CLI | `python -m aqr "проверь momentum на Сбере"` (`--json`, `-q`) |
| HTTP API | `uvicorn aqr.main:app --port 8000` |
| Web UI | открыть `http://localhost:8000/chat` в браузере |
| Agent программно | `from aqr.agent.graph import run_agent; await run_agent(...)` |
| WebSocket | `WS /chat/{token}`, протокол в `aqr/chat/ws.py` (см. ниже) |
| Реестр | `from aqr.registry import RegistryStore` + `from aqr.db import get_db` |
| Токен для WS | `GET /chat/new?session_id=...` → `{token, session_id}` |

CLI и Web UI работают **без Postgres и без LLM-ключей** (fallback-планировщик + шаблонный нарратор). HTTP API и `run_agent` пишут в Postgres. LLM-режим: `AQR_LLM_MODEL` + один из `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GIGACHAT_CREDENTIALS`.

## Как запускать локально

```bash
# venv (Python 3.11+)
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev,llm,data,embeddings]"  # полный набор

# Postgres
docker run -d --name aqr-pg -e POSTGRES_PASSWORD=aqr -e POSTGRES_DB=aqr \
  -p 5432:5432 pgvector/pgvector:pg16
docker start aqr-pg   # после рестарта

# Миграции
DATABASE_URL=postgresql+asyncpg://postgres:aqr@localhost:5432/aqr \
  .venv/bin/alembic upgrade head

# Тесты + lint (212 passed, coverage ≥80%)
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m pytest tests/ --cov=aqr --cov-fail-under=80
.venv/bin/ruff check aqr/ tests/
```

`DATABASE_URL` — env-переменная для Postgres. Дефолт `postgresql+asyncpg://postgres:aqr@localhost:5432/aqr`. Читается в `aqr/db.py` и `alembic/env.py`.

## Структура проекта

```
aqr/
  pipeline/        # оркестратор: planner, executor, narrator, SSE-events, FastAPI роуты
  tools/           # Tool Layer (13 инструментов)
  agent/           # LangGraph граф + SessionContext + run_agent
  chat/            # WS /chat/{token} + Web UI
    ws.py          # WebSocket endpoint и контракт
    web.py         # GET /chat (HTML) + GET /chat/new (token)
    templates/     # chat.html — vanilla JS, dark theme
  registry/        # models, store, embeddings (OpenAI + hash)
  validation/      # Deflated Sharpe, PBO, CPCV, Reality Check — reference
  data/
    moex.py        # MOEXAdapter (timeout=10s, retry×3, CB per-ticker)
    ohlcv_cache.py # DuckDB-кэш OHLCV (lazy duckdb import)
    manifest.py    # DataManifest — не подключён к pipeline, опциональный
  logging_config.py # JsonFormatter + log_tool_call
  auth.py          # HMAC-подписанные session_id-токены (SEC-1)
  tasks.py         # Retention set для фоновых asyncio-задач (PERF-1)
  main.py          # FastAPI app: /health, /health/ready, /pipeline/*, /chat/*, /chat
  cli.py           # `python -m aqr <goal>`
alembic/versions/  # миграции: fcb396c4088d (initial) + a1b2c3d4e5f6 (chat_messages)
tests/             # 212 тестов в 19 файлах (см. ниже)
```

## WebSocket-протокол (контракт `aqr/chat/ws.py`)

Клиент → сервер:
```json
{"type": "message", "content": "проверь momentum на Сбере"}
{"type": "resume"}                              // запрос истории из БД
{"type": "ping"}                                // keepalive
```

Сервер → клиент:
```json
{"type": "connected",  "session_id": "..."}
{"type": "history",    "messages": [{"role", "content", "created_at"}]}
{"type": "user_echo",  "content": "..."}
{"type": "progress",   "node": "...", "data": {...}}
{"type": "tool_call",   "name": "...", "args": {...}}
{"type": "tool_result", "name": "...", "result": {...}}
{"type": "assistant",   "content": "..."}
{"type": "done",        "narrative": "...", "assistant": "...", "elapsed_seconds": N, "n_results": M}
{"type": "error",       "message": "..."}
{"type": "pong"}
```

Auth: `WS /chat/{token}` где `token` — это HMAC-подпись `exp:session_id`, выданная через `GET /chat/new?session_id=...`. Без токена — `close(1008)`. Legacy режим через `AQR_REQUIRE_WS_AUTH=0`.

## Инварианты (не нарушать)

1. **Никакого look-ahead.** В `aqr/tools/core.py:backtest_one` позиция сдвигается на 1 бар (`pos.shift(1).fillna(0.0)`).
2. **Fallback обязателен.** `ChatPlanner._fallback_plan`, `Narrator._fallback_narrate`, `InsightReviewer.review`, `Embedder.hash_embedding` — работают без LLM/API.
3. **Порядок событий SSE**: `planning → data → generating → backtesting × N → validating → insight × M → narrating → done`; при ошибке — `error`. Эмитятся через `await self._emit()` в `PipelineExecutor`.
4. **Валидация — источник истины.** Sharpe без DSR не показывать как «значимый». `DSR ≥ 0.95` → `significant`, `0.80–0.95` → `borderline`.
5. **Tool registry заполняется через `register_all()`** — идемпотентен (`if len(registry) >= 13: return` + `_safe_register`). Безопасно вызывать из `PipelineExecutor.run` и `agent/graph.py:run_agent`.
6. **UUID-консистентность для pipeline-run.** В `aqr/pipeline/api.py:start_run` генерируется `uuid.uuid4()` ОДИН раз и используется для BUS, БД и `_run_and_persist`. Если создавать UUID отдельно в BUS и store — будет FK-violation на `hypotheses.run_id`. Найдено и исправлено в REVIEW.md (BUG-FK-1).
7. **Lazy imports** для `[data]`/`[embeddings]` extras: `aqr/data/__init__.py` подгружает `DataManifest` через `__getattr__`; `ohlcv_cache.py` импортирует duckdb в методах; `embeddings.py` — openai только при наличии ключа.
8. **MOEX resilience:** `_request_with_retry` — 3 попытки с exponential backoff (0.5–4s), per-ticker CircuitBreaker на 60s после 5 ошибок подряд. `CircuitOpenError` в `load_prices` → fallback на synthetic GBM.
9. **Background-task retention.** `asyncio.create_task` без strong-ref теряется при GC. Использовать `aqr.tasks.schedule(coro)` вместо прямого вызова — retention set + done-callback. На FastAPI shutdown `tasks.drain(timeout=30)` дожидается завершения.
10. **WS auth.** Токен выпускается через `aqr.auth.sign_session(session_id)`, проверяется через `verify_token(token)`. `AQR_SESSION_SECRET` обязателен в проде (иначе ephemeral на процесс — клиенты теряют сессии при рестарте).

## Gotchas (то, что легко забыть)

### `MOEXAdapter.candles` принимает СТРОКУ interval, не число

`MOEXAdapter.INTERVAL_MAP = {"1min": 1, "10min": 10, "1H": 60, "D": 24, ...}` — строковые ключи.

- ❌ `adapter.candles("SBER", ..., interval=24)` — `INTERVAL_MAP.get(24, 24)` возвращает `24` (default), H1 молча становится D1.
- ✅ `adapter.candles("SBER", ..., interval="D")` или `interval="1H"`.

В `aqr/tools/core.py:load_prices` уже исправлено: `interval="D" if timeframe == "D1" else "1H"`.

### `backtest_one` принимает `cpcv_*` и `embargo_pct` параметры

RPC `plan.validation: {cpcv_splits, cpcv_test_splits, embargo_pct}` **должна** пробрасываться в `backtest_one(...)` явно. Дефолты в инструменте (`6/2/0.01`) — это fallback, не «волшебные числа».

Сейчас пробрасывается в:
- `aqr/agent/graph.py:backtest_node`
- `aqr/pipeline/executor.py:run`

При добавлении нового caller-а не забудь.

### `OhlcvCache` хранит NaN-safe: `close` is `NOT NULL`

`put_cache` пропускает строки с NaN/None close (использует хелпер `_safe_float`). Если добавить новое поле, тоже прогон через `_safe_float`.

### Путь к кэшу зависит от env

- Дефолт: `~/.aqr/ohlcv_cache.duckdb` (НЕ `data/` относительно CWD — раньше был, теперь нет).
- Override: `AQR_CACHE_DIR=/path/to/dir`.
- Тесты переопределяют через monkeypatch на `OhlcvCache.__init__`.

### Реестр инструментов

После `register_all()` в `aqr.tools.registry` ровно **13** инструментов:
- 8 pipeline: `plan_research, load_prices, generate_hypotheses, backtest_one, validate_portfolio, extract_insights, review_insights, narrate`
- 5 storage: `get_run, list_runs, compare_runs, search_similar_hypotheses, find_duplicates`

Тест: `tests/test_tools.py::TestToolRegistry::test_list_all_returns_13_tools`. При добавлении — обновить и `len(registry) >= N` в `register.py`.

### `AQR_REQUIRE_WS_AUTH` env

- `1` (default) — WS требует валидный HMAC-токен
- `0` — legacy/dev режим, WS принимает raw `session_id` без подписи

В тестах WS (`test_chat_ws.py`) и Web UI (`test_chat_web.py`) переопределяется через `monkeypatch.setenv` / `delenv` — иначе auth-mode перебивает друг друга при совместном прогоне.

### Двух-модульный import `_async_session_factory`

`aqr/tools/core.py:plan_research` импортирует `_async_session_factory` **лениво** внутри функции. Остальные модули (`tools/storage.py`, `agent/context.py`, `chat/ws.py`) — на top-level. В тестах при патче нужно мокать в обоих namespace (см. паттерн моков ниже).

## Что нельзя трогать без явной причины

- `aqr/validation/` — формулы Bailey & López de Prado. Покрыты `tests/test_validation.py`; правки — обязательный перепрогон.
- `aqr/data/moex.py` — адаптер к iss.moex.com. Менять по документации https://iss.moex.com/iss/reference/.
- `aqr/pipeline/events.py:Event` — поля `kind`, `stage`, `message`, `data`, `ts` завязаны на SSE UI.
- `aqr/registry/models.py` — менять схему только через новую миграцию Alembic.
- `aqr/chat/ws.py` — WebSocket-контракт. Web UI в `aqr/chat/templates/chat.html` зависит от точных имён полей. Менять — синхронно обновлять оба.
- `aqr/chat/templates/chat.html` — vanilla JS, без бандлера. Правки сразу в файле.

## Рецепты

### Добавить семейство гипотез
1. `_my_signal(p1, p2) -> Callable[[pd.Series], pd.Series]` в `aqr/pipeline/hypotheses.py`.
2. Ветка `if family == "my_family"` в `_make_one()` и `make_one_with_params()`.
3. В `aqr/pipeline/planner.py` ключевое слово → `"my_family"` в `_fallback_plan`, описание в `PLANNER_SYSTEM`.
4. Тест в `tests/test_pipeline_e2e.py`: ключевое слово → правильное family.

### Добавить MOEX-тикер
1. Тикер в `aqr/pipeline/planner.py::MOEX_TICKERS`.
2. Алиас или категория («голубые фишки», «банки») в `_extract_tickers`.

### Добавить новый инструмент
1. Async-функция в `aqr/tools/core.py`.
2. `ToolSpec` в `register.py::register_all()` через `_safe_register`.
3. Тест в `tests/test_tools.py`. Обновить `test_list_all_returns_13_tools` → `_N_tools` и порог в `register.py`.

### Добавить Web UI endpoint или страницу
1. FastAPI router в `aqr/chat/web.py` (или новый модуль).
2. Подключить в `aqr/main.py:app.include_router(...)`.
3. Если HTML — положить в `aqr/chat/templates/<name>.html` и загружать через `Path(__file__).parent / "templates" / "<name>.html"`.
4. Тест в `tests/test_chat_web.py` через `fastapi.testclient.TestClient`.

### Изменить WebSocket-протокол
1. Менять и сервер (`aqr/chat/ws.py:handle_message` / `_run_agent_for_session`), и клиент (`aqr/chat/templates/chat.html:handleServerMessage`).
2. Если меняются поля — обновить docstring в `aqr/chat/ws.py:1-18`.
3. Добавить тест в `tests/test_chat_ws.py`.

### Новая миграция Alembic
```bash
.venv/bin/alembic revision --autogenerate -m "add field x"
# отредактировать файл, перепрогнать:
.venv/bin/alembic upgrade head
.venv/bin/alembic check    # модели не расходятся с миграциями
```
`alembic/env.py` подхватывает `Base` из `aqr/registry/models.py` — autogenerate работает.

## Тесты

`pytest-asyncio` в режиме `auto` — async-тесты **без `@pytest.mark.asyncio`**, просто `async def test_...`.

**212 тестов проходят без Postgres и LLM, coverage ≥81%.** Полный список в `tests/` (19 файлов):

- `test_validation.py` — DSR/PBO/CPCV/Reality Check (reference)
- `test_pipeline_e2e.py` — fallback-планировщик + e2e на синтетике
- `test_smoke.py` — CLI subprocess + verbose/json режимы
- `test_tools.py` — `ToolRegistry` + 13 изолированных инструментов
- `test_agent.py` — граф, роутер, `SessionContext`, `run_agent()` e2e
- `test_chat_ws.py` — WebSocket через FastAPI TestClient (с моками агента и БД)
- `test_chat_web.py` — Web UI endpoints: `/chat`, `/chat/new`, integration с WS
- `test_ohlcv_cache.py` — DuckDB-кэш + NaN-handling + `AQR_CACHE_DIR`
- `test_cache_reuse.py` — кэш переиспользуется между прогонами
- `test_embeddings.py` — hash/OpenAI embeddings + дедуп
- `test_context.py` — `SessionContext` с моком БД
- `test_storage_tools.py` — storage-инструменты с моком БД
- `test_moex_retry.py` — retry + per-ticker circuit breaker
- `test_health.py` — `/health` + `/health/ready`
- `test_logging_config.py` — `JsonFormatter` + `log_tool_call`
- `test_cpcv_edge.py` — purge/embargo edge cases
- `test_api_routes.py` — FastAPI роуты + `_run_and_persist`
- `test_auth.py` — HMAC sign/verify round-trip + edge cases
- `test_smoke.py` — CLI subprocess

### Паттерн моков

- `_async_session_factory` → подменять в `aqr.db` И в импортирующем модуле (binding в namespace). Топ-уровневый импорт: `tools/storage.py`, `agent/context.py`, `chat/ws.py`. Lazy: `tools/core.py:plan_research`.
- `RegistryStore` → подменять в `aqr.registry.store` И в импортирующем модуле (`aqr.tools.storage`, `aqr.tools.core`, `aqr.pipeline.api`, `aqr.agent.context`).
- `MOEXAdapter` → `monkeypatch.setattr(aqr.data.moex, "MOEXAdapter", _FakeAdapter)` (через класс, не инстанс).
- `get_agent` в WS-тестах → патчить `aqr.agent.graph.get_agent` И `aqr.chat.ws.get_agent`.
- `AQR_REQUIRE_WS_AUTH` в WS-тестах → `test_chat_ws.py` ставит `0` на module level, `test_chat_web.py` через `monkeypatch.delenv(..., raising=False)` сбрасывает обратно к дефолту `1`. При совместном прогоне они интерферируют — см. fixture в `test_chat_web.py`.

### Известный баг pytest-cov

При `pytest --cov=aqr.chat.web` (или любой другой новый модуль с WebSocket-тестами) падают 12 тестов с `ImportError: cannot load module more than once per process` (numpy + coverage interaction). Без `--cov` всё работает (212 passed). Решение для CI — `coverage run -m pytest` вместо `pytest --cov`.

## Известные ограничения

- **ivfflat-индекс на `hypothesis.embedding`** — намеренно отложен (`__ivfflat_deferred__` в `models.py`). Нельзя на пустой таблице; добавить отдельной миграцией после >1000 строк.
- **PBO cross-ticker** — текущий `validate_portfolio` берёт общий суффикс `daily_returns` от разных тикеров; статистически некорректно. Per-ticker PBO — в TODO.
- **embeddings dedup через SQL search_similar** — нужен Postgres (pgvector). Без БД дедупликация в `plan_research` тихо отключается.
- **`DataManifest` (point-in-time lineage)** в `aqr/data/manifest.py` существует, но **не подключён к pipeline**. Файл-флаг — `import duckdb` на module-load; через `aqr.data.__init__` доступен лениво.
- **Web UI не покрыт browser-тестами.** Vanilla JS — рендеринг и WebSocket-клиент тестируются вручную. CI покрывает только серверные endpoints.
- **MOEX из sandbox/CI:** `/health/ready` отвечает 503 (MOEX недоступен). Это корректное поведение — в проде проверка падает, k8s не даёт трафик.

## Стиль

- `from __future__ import annotations` в каждом модуле.
- Type hints обязательны.
- Docstring — на русском или английском, **консистентно в модуле**.
- Без emoji в коде и коммитах.
- Модули до 400 строк (допустимо исключение для оркестраторов с >500).
- HTML-шаблон — без бандлера, vanilla JS, инлайнить CSS в `<style>` или отдельный файл в `templates/`.

## Как проверить проект руками

См. полную инструкцию в `REVIEW.md` секция «How to Verify the Fixes by Hand». Краткая версия:

```bash
# 1. Запуск
docker start aqr-pg
DATABASE_URL=postgresql+asyncpg://postgres:aqr@localhost:5432/aqr \
AQR_SESSION_SECRET=any-secret-32-bytes== \
  uvicorn aqr.main:app --port 8000

# 2. CLI smoke
.venv/bin/python -m aqr "проверь momentum на Сбере"

# 3. Web UI
open http://localhost:8000/chat   # ввести session_id, например "alice"

# 4. API
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/pipeline/runs \
  -H "Content-Type: application/json" \
  -d '{"goal":"проверь momentum на Сбере"}'

# 5. Остановить
pkill -f "uvicorn aqr.main"
```

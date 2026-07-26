# AGENTS.md

Контракт для LLM-агентов (Claude Code, Codex, Cursor), работающих с этим репозиторием.

## Что это за проект

AQR — тонкий пайплайн для проверки торговых гипотез на MOEX через T-Invest API. Вход — цель на русском («проверь momentum на голубых фишках»), выход — топ-5 гипотез с Deflated Sharpe / CPCV / PBO + нарратив.

**Строгий режим:** проект работает только при наличии всех зависимостей. Никаких fallback-путей. На старте валидируется наличие LLM, Postgres и Invest-токена; без них FastAPI не запускается.

Трёхслойная архитектура:

- **Storage** — Postgres + pgvector + Alembic (`aqr/registry/`, `aqr/db.py`). Таблицы `sessions → runs → hypotheses` + `chat_messages` для истории WebSocket.
- **Tool Layer** — `aqr/tools/`: `ToolSpec` + `ToolRegistry` + **13 инструментов** (`plan_research`, `load_prices`, `generate_hypotheses`, `backtest_one`, `validate_portfolio`, `extract_insights`, `review_insights`, `narrate`, `get_run`, `list_runs`, `compare_runs`, `search_similar_hypotheses`, `find_duplicates`).
- **Agent Layer** — `aqr/agent/`: LangGraph-граф; `SessionContext` подмешивает историю и непроверенные комбинации в `session_context_prompt`.
- **Chat Layer** — `aqr/chat/`: WebSocket `/chat/{token}` + Web UI на vanilla JS (тёмная тема, markdown, slash-команды).
- **Data Layer** — `aqr/data/`: `TInvestAdapter` (gRPC, sandbox-ready), `OhlcvCache` (DuckDB).

Один процесс. Фон — `aqr.tasks.schedule(...)` (с retention set против GC). Никакого Redis/Celery.

## Точки входа

| Что | Команда / URL |
|---|---|
| HTTP API | `uvicorn aqr.main:app --port 8000` |
| Web UI | открыть `http://localhost:8000/chat` в браузере |
| Agent программно | `from aqr.agent.graph import run_agent; await run_agent(...)` |
| WebSocket | `WS /chat/{token}`, протокол в `aqr/chat/ws.py` (см. ниже) |
| Реестр | `from aqr.registry import RegistryStore` + `from aqr.db import get_db` |
| Токен для WS | `GET /chat/new?session_id=...` → `{token, session_id}` |

**CLI удалён.** Все взаимодействие — через HTTP API / WebSocket / `run_agent`.

LLM-режим: `AQR_LLM_MODEL` + один из `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GIGACHAT_CREDENTIALS` (обязательно). Эмбеддинги — `OPENAI_API_KEY` (обязательно). Свечи — `INVEST_TOKEN` (обязательно).

## Как запускать локально

```bash
# venv (Python 3.11+)
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev,llm,embeddings]"

# t-tech-investments — из приватного index T-Bank
.venv/bin/pip install t-tech-investments \
  --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple

# Postgres
docker run -d --name aqr-pg -e POSTGRES_PASSWORD=aqr -e POSTGRES_DB=aqr \
  -p 5432:5432 pgvector/pgvector:pg16
docker start aqr-pg   # после рестарта

# Миграции
DATABASE_URL=postgresql+asyncpg://postgres:aqr@localhost:5432/aqr \
  .venv/bin/alembic upgrade head

# Обязательные env (минимальный набор для запуска):
export DATABASE_URL="postgresql+asyncpg://postgres:aqr@localhost:5432/aqr"
export AQR_LLM_MODEL="claude-3-5-sonnet-20241022"
export ANTHROPIC_API_KEY="sk-ant-..."     # или OPENAI_API_KEY, или GIGACHAT_CREDENTIALS
export OPENAI_API_KEY="sk-..."             # для эмбеддингов
export INVEST_TOKEN="t.INVEST_TOKEN..."    # t-Invest
export INVEST_SANDBOX=1                    # sandbox по умолчанию для dev/CI
export AQR_SESSION_SECRET="$(openssl rand -base64 32)"  # для WS HMAC

# Запуск
.venv/bin/uvicorn aqr.main:app --port 8000

# Без обязательной переменной — RuntimeError на старте, uvicorn падает
```

`DATABASE_URL` — env-переменная для Postgres. Дефолт `postgresql+asyncpg://postgres:aqr@localhost:5432/aqr`. Читается в `aqr/db.py` и `alembic/env.py`. На старте валидируется через `aqr.startup.validate_runtime()`.

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
  registry/        # models, store, embeddings (только OpenAI)
  validation/      # Deflated Sharpe, PBO, CPCV, Reality Check — reference
  data/
    tinvest.py     # TInvestAdapter (gRPC, sandbox, lazy FIGI cache)
    ohlcv_cache.py # DuckDB-кэш OHLCV (lazy duckdb import)
  logging_config.py # JsonFormatter + log_tool_call
  auth.py          # HMAC-подписанные session_id-токены (SEC-1)
  startup.py       # validate_runtime() — обязательные env + доступность зависимостей
  tasks.py         # Retention set для фоновых asyncio-задач (PERF-1)
  main.py          # FastAPI app: /health, /health/ready, /pipeline/*, /chat/*, /chat
alembic/versions/  # миграции: fcb396c4088d (initial) + a1b2c3d4e5f6 (chat_messages)
tests/             # тесты (см. ниже)
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

Auth: `WS /chat/{token}` где `token` — это HMAC-подпись `exp:session_id`, выданная через `GET /chat/new?session_id=...`. Без токена — `close(1008)`. **HMAC обязателен — legacy-режим удалён.**

## Инварианты (не нарушать)

1. **Никакого look-ahead.** В `aqr/tools/core.py:backtest_one` позиция сдвигается на 1 бар (`pos.shift(1).fillna(0.0)`).
2. **Fallback запрещён.** `ChatPlanner.plan`, `Narrator.narrate`, `InsightReviewer.review`, `Embedder.embed`, `TInvestAdapter.candles` — **всегда** требуют ключей / сети. Любая ошибка → raise, без возврата к шаблону или synthetic-данным.
3. **Порядок событий SSE**: `planning → data → generating → backtesting × N → validating → insight × M → narrating → done`; при ошибке — `error`. Эмитятся через `await self._emit()` в `PipelineExecutor`.
4. **Валидация — источник истины.** Sharpe без DSR не показывать как «значимый». `DSR ≥ 0.95` → `significant`, `0.80–0.95` → `borderline`.
5. **Tool registry заполняется через `register_all()`** — идемпотентен (`if len(registry) >= 13: return` + `_safe_register`). Безопасно вызывать из `PipelineExecutor.run` и `agent/graph.py:run_agent`.
6. **UUID-консистентность для pipeline-run.** В `aqr/pipeline/api.py:start_run` генерируется `uuid.uuid4()` ОДИН раз и используется для BUS, БД и `_run_and_persist`. Если создавать UUID отдельно в BUS и store — будет FK-violation на `hypotheses.run_id`. Найдено и исправлено в REVIEW.md (BUG-FK-1).
7. **Lazy imports** для duckdb (`ohlcv_cache.py`) и `t_tech.invest` (`tinvest.py`) — `import` только в методах, чтобы `aqr.main` стартовал без всех зависимостей поднятых сразу.
8. **T-Invest strict.** `TInvestAdapter.candles` — **одна попытка**, без retry/circuit-breaker. На любую ошибку (сеть, таймаут, неизвестный FIGI) — `raise`. Если нужно устойчивости — это проблема клиента, не адаптера.
9. **Background-task retention.** `asyncio.create_task` без strong-ref теряется при GC. Использовать `aqr.tasks.schedule(coro)` вместо прямого вызова — retention set + done-callback. На FastAPI shutdown `tasks.drain(timeout=30)` дожидается завершения.
10. **WS auth — только HMAC.** Токен выпускается через `aqr.auth.sign_session(session_id)`, проверяется через `verify_token(token)`. `AQR_SESSION_SECRET` обязателен в проде (иначе ephemeral на процесс — клиенты теряют сессии при рестарте).
11. **Startup validation.** `aqr.startup.validate_runtime()` вызывается в `lifespan` до `yield` и проверяет `DATABASE_URL`+SELECT 1, `AQR_LLM_MODEL`+один из ключей, `OPENAI_API_KEY`, `INVEST_TOKEN`, `AQR_SESSION_SECRET`. Без всех — `RuntimeError`, FastAPI не стартует.

## Gotchas (то, что легко забыть)

### `TInvestAdapter.candles` принимает интервалы из фиксированного набора

`TInvestAdapter.INTERVAL_MAP = {"1m": ..., "5m": ..., "15m": ..., "H1": ..., "D1": ..., "W": ..., "M": ...}` — 7 интервалов через T-Invest `CandleInterval`.

- ❌ `adapter.candles("SBER", ..., interval="2H")` — `KeyError`. Либо валидный ключ, либо отказ.
- ✅ `adapter.candles("SBER", ..., interval="H1")` или `interval="D1"`.

В `aqr/tools/core.py:load_prices` интервал пробрасывается из `plan.timeframe`. LLM-планировщик должен выбирать только из поддерживаемого набора.

### `TInvestAdapter._resolve_figi` — lazy с class-level cache

Первый вызов `candles("SBER", ...)` → gRPC `InstrumentsService.get_instrument_by_ticker`, кэш в `cls._figi_cache`. Неизвестный тикер → `raise ValueError`. Если нужно сбросить — `TInvestAdapter.clear_figi_cache()`.

Тесты патчат через `monkeypatch.setattr("t_tech.invest.Client", _FakeClient)` или оборачивают `TInvestAdapter.candles` напрямую.

### `backtest_one` принимает `cpcv_*` и `embargo_pct` параметры

RPC `plan.validation: {cpcv_splits, cpcv_test_splits, embargo_pct}` **должна** пробрасываться в `backtest_one(...)` явно. Дефолты в инструменте (`6/2/0.01`) — fallback только в смысле default-значения, не «магические числа».

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

### `t-tech-investments` — установка через кастомный index-url

PyPI-зеркало приватное: `https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple`. Без `--index-url` pip не найдёт пакет. CI должен использовать:

```bash
pip install t-tech-investments \
  --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple
```

`pyproject.toml` для uv:

```toml
[tool.uv.sources]
t-tech-investments = { index = "tbank" }

[[tool.uv.index]]
name = "tbank"
url = "https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple"
```

### `INVEST_SANDBOX` env

- `1` (default) — sandbox target (`INVEST_GRPC_API_SANDBOX`), заявки не уходят на биржу
- `0` — production target (`INVEST_GRPC_API`)

Для CI/тестов всегда sandbox. Для реальной торговли — production с реальным токеном.

### `SSL_TBANK_VERIFY` env

Если в РФ-окружении падает SSL-проверка, можно включить сертификат МинЦифры, поставляемый с `t-tech-investments`:

```bash
export SSL_TBANK_VERIFY=True
```

### Двух-модульный import `_async_session_factory`

`aqr/tools/core.py:plan_research` импортирует `_async_session_factory` **лениво** внутри функции. Остальные модули (`tools/storage.py`, `agent/context.py`, `chat/ws.py`) — на top-level. В тестах при патче нужно мокать в обоих namespace (см. паттерн моков ниже).

## Что нельзя трогать без явной причины

- `aqr/validation/` — формулы Bailey & López de Prado. Покрыты `tests/test_validation.py`; правки — обязательный перепрогон.
- `aqr/data/tinvest.py` — адаптер к T-Invest gRPC. Менять по документации https://developer.tbank.ru/invest/intro/intro и https://opensource.tbank.ru/invest/invest-python.
- `aqr/pipeline/events.py:Event` — поля `kind`, `stage`, `message`, `data`, `ts` завязаны на SSE UI.
- `aqr/registry/models.py` — менять схему только через новую миграцию Alembic.
- `aqr/chat/ws.py` — WebSocket-контракт. Web UI в `aqr/chat/templates/chat.html` зависит от точных имён полей. Менять — синхронно обновлять оба.
- `aqr/chat/templates/chat.html` — vanilla JS, без бандлера. Правки сразу в файле.
- `aqr/startup.py` — список обязательных env. Менять только при добавлении/удалении зависимости уровня infra.

## Рецепты

### Добавить семейство гипотез
1. `_my_signal(p1, p2) -> Callable[[pd.Series], pd.Series]` в `aqr/pipeline/hypotheses.py`.
2. Ветка `if family == "my_family"` в `_make_one()` и `make_one_with_params()`.
3. В `aqr/pipeline/planner.py` описание семейства в `PLANNER_SYSTEM` (LLM-режим; keyword-парсер удалён вместе с `_fallback_plan`).
4. Тест в `tests/test_pipeline_e2e.py`: ключевое слово → правильное family (через мок LLM).

### Добавить тикер T-Invest
Тикеры не нуждаются в регистрации — `TInvestAdapter._resolve_figi` лениво подтягивает FIGI через `InstrumentsService`. Если тикер не находится — `ValueError` в `load_prices`.

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

### Добавить новую обязательную зависимость на старте
1. Добавить проверку в `aqr/startup.py::validate_runtime()` — поднять `RuntimeError` с понятным сообщением, какая env-переменная отсутствует.
2. Обновить `.env.example` — пометить обязательную переменную.
3. Обновить раздел «Как запускать локально» в этом файле.

## Тесты

`pytest-asyncio` в режиме `auto` — async-тесты **без `@pytest.mark.asyncio``, просто `async def test_...`.

**Тесты запускаются только с обязательными env** (DATABASE_URL, AQR_LLM_MODEL, ANTHROPIC_API_KEY/OPENAI_API_KEY, OPENAI_API_KEY, INVEST_TOKEN, AQR_SESSION_SECRET). Без них часть тестов skip'ится или падает. Coverage ≥80%.

Полный список в `tests/`:

- `test_validation.py` — DSR/PBO/CPCV/Reality Check (reference)
- `test_pipeline_e2e.py` — LLM-планировщик + e2e с моком `litellm.completion`
- `test_tools.py` — `ToolRegistry` + 13 изолированных инструментов
- `test_agent.py` — граф, роутер, `SessionContext`, `run_agent()` e2e
- `test_chat_ws.py` — WebSocket через FastAPI TestClient (с моками агента и БД)
- `test_chat_web.py` — Web UI endpoints: `/chat`, `/chat/new`, integration с WS
- `test_ohlcv_cache.py` — DuckDB-кэш + NaN-handling + `AQR_CACHE_DIR`
- `test_cache_reuse.py` — кэш переиспользуется между прогонами
- `test_embeddings.py` — OpenAI embeddings + дедуп (с моком openai)
- `test_context.py` — `SessionContext` с моком БД
- `test_storage_tools.py` — storage-инструменты с моком БД
- `test_tinvest.py` — TInvestAdapter: FIGI cache, intervals, sandbox target, error propagation
- `test_health.py` — `/health` + `/health/ready` + startup validation
- `test_logging_config.py` — `JsonFormatter` + `log_tool_call`
- `test_cpcv_edge.py` — purge/embargo edge cases
- `test_api_routes.py` — FastAPI роуты + `_run_and_persist`
- `test_auth.py` — HMAC sign/verify round-trip + edge cases
- `test_startup.py` — validate_runtime() отказ при отсутствии env

### Паттерн моков

- `_async_session_factory` → подменять в `aqr.db` И в импортирующем модуле (binding в namespace). Топ-уровневый импорт: `tools/storage.py`, `agent/context.py`, `chat/ws.py`. Lazy: `tools/core.py:plan_research`.
- `RegistryStore` → подменять в `aqr.registry.store` И в импортирующем модуле (`aqr.tools.storage`, `aqr.tools.core`, `aqr.pipeline.api`, `aqr.agent.context`).
- `TInvestAdapter` → `monkeypatch.setattr(aqr.data.tinvest, "TInvestAdapter", _FakeAdapter)` (через класс, не инстанс). Для FIGI cache — `TInvestAdapter.clear_figi_cache()` в фикстуре.
- `t_tech.invest.Client` → `monkeypatch.setitem(sys.modules, "t_tech.invest", _FakeModule)` или `monkeypatch.setattr("t_tech.invest.Client", _FakeClient)`.
- `get_agent` в WS-тестах → патчить `aqr.agent.graph.get_agent` И `aqr.chat.ws.get_agent`.
- `litellm.completion` → `monkeypatch.setitem(sys.modules, "litellm", fake_module)` с фейковым `completion`, возвращающим нужный JSON.
- `openai.AsyncOpenAI` → `monkeypatch.setitem(sys.modules, "openai", fake_module)` (для Embedder).

### Известный баг pytest-cov

При `pytest --cov=aqr.chat.web` (или любой другой новый модуль с WebSocket-тестами) падают 12 тестов с `ImportError: cannot load module more than once per process` (numpy + coverage interaction). Без `--cov` всё работает. Решение для CI — `coverage run -m pytest` вместо `pytest --cov`.

## Известные ограничения

- **ivfflat-индекс на `hypothesis.embedding`** — намеренно отложен (`__ivfflat_deferred__` в `models.py`). Нельзя на пустой таблице; добавить отдельной миграцией после >1000 строк.
- **PBO cross-ticker** — текущий `validate_portfolio` берёт общий суффикс `daily_returns` от разных тикеров; статистически некорректно. Per-ticker PBO — в TODO.
- **FIGI cache in-memory** — `TInvestAdapter._figi_cache` живёт до перезапуска процесса. Несколько воркеров uvicorn получают разные кэши — это OK, т.к. lookup идёт через gRPC idempotent.
- **SSL в РФ-окружении** — если корневые CA не проходят проверку, нужен `SSL_TBANK_VERIFY=True`. Дефолт выключен.
- **T-Invest rate limits** — 300 req/min на user-units. `load_prices` сейчас без backpressure; при массовых прогонах может упереться. В TODO — semaphore per session.
- **Web UI не покрыт browser-тестами.** Vanilla JS — рендеринг и WebSocket-клиент тестируются вручную. CI покрывает только серверные endpoints.
- **Sandbox vs prod** — `INVEST_SANDBOX=1` по умолчанию. Sandbox имеет ограниченный universe инструментов и тестовые счета; реальные FIGI/свечи только в production-токене.

## Что уже сделано (v0.3)

Работает end-to-end в strict-режиме (без fallback):

- ✅ **T-Invest gRPC**: `aqr/data/tinvest.py` — `TInvestAdapter` с lazy FIGI cache, 7 интервалов (1m/5m/15m/H1/D1/W/M), sandbox по дефолту.
- ✅ **Validation**: DSR (Bailey-López de Prado), CPCV, PBO в `aqr/validation/`.
- ✅ **Tool Layer**: 13 инструментов в `aqr.tools.registry` (8 pipeline + 5 storage).
- ✅ **Agent Layer**: линейный LangGraph-граф в `aqr/agent/graph.py` (plan → load → generate → backtest → validate → narrate → respond).
- ✅ **Storage**: Postgres + pgvector, `session_settings` с Fernet-encrypted credentials (HKDF от `AQR_SESSION_SECRET`), `chat_messages` для WS-истории.
- ✅ **Chat**: WS `/chat/{token}` с HMAC auth, per-session credentials через ContextVar (`/chat/{token}/settings` форма), vanilla-JS UI.
- ✅ **Startup validation**: `aqr/startup.py::validate_runtime()` — auto-provision Postgres-контейнера + обязательные env.
- ✅ **Tests**: 239 passed, coverage 83% (≥80% gate).

Слабые места v0.3 (обоснование для v0.4):
- Агент — линейный, без параллелизма и специализации.
- Скрининг гипотез — медленный (один backtest за раз через executor).
- Код стратегий — параметризованные шаблоны в `hypotheses.py`, не выразительный Python для сложных сигналов.
- Execution model — упрощённый (без реалистичного моделирования комиссий/slippage/частичного исполнения).

## План переделывания (v0.4 → v0.5)

| Компонент | Рекомендация | Обоснование |
|---|---|---|
| Быстрый скрининг идей | **VectorBT** (open-source, с осознанием что развитие остановлено) | Скорость итераций для проверки «есть ли edge». Numba-ускоренные бэктесты на 1000+ параметров за минуты, а не часы. Подходит для фазы discovery; для production execution — следующий уровень. |
| Валидация отобранных стратегий | **NautilusTrader** | Реалистичное моделирование исполнения перед paper trading: комиссии, slippage, частичное исполнение, latency, order book. Event-driven backtester на Rust-ядре. Переход от «сигнал даёт Sharpe» → «сигнал даёт Sharpe после реалистичных издержек». |
| Оркестрация LLM-агентов | **LangGraph с ролями** Researcher/Coder/Reviewer/Writer (расширение существующего `aqr/agent/`) | Полный контроль над данными (T-Invest), параллельный research гипотез через `asyncio.gather`. Заменяет линейный v0.3-граф на иерархию: Editor → {Browser, Analyst} → Reviewer → Writer. |
| Доступ к брокерским данным | **T-Invest MCP Server + T-Bank Invest API** | Готовая интеграция без написания коннекторов с нуля. MCP-протокол даёт стандартизованный интерфейс для LLM-агентов (tools/JSON-RPC). Не зависит от внутреннего gRPC-SDK. |

### Стратегия миграции

v0.3 продолжает работать параллельно (backward compat). Новые компоненты добавляются в `aqr/v04/`:

```
aqr/v04/
  screener/        # VectorBT-обёртки (Phase 1: discovery)
  executor/        # NautilusTrader-бэктесты (Phase 2: validation)
  agents/          # 5 ролей через LangGraph (Phase 3: orchestration)
  mcp/             # T-Invest MCP client (Phase 4: data layer)
```

WebSocket получает slash-команды `/run` (старый pipeline) и `/team` (новый orchestrator с 5 ролями). Settings-форма расширяется чекбоксом «use team agents» (default off — для backward compat).

### Порядок реализации

1. **VectorBT-screener** — отдельный endpoint `POST /screener/vectorbt`, возвращает топ-N параметров по Sharpe. Без замены существующего flow.
2. **NautilusTrader-executor** — `aqr/v04/executor/` для paper-trading валидации топ-стратегий из v0.3 pipeline. Опциональный шаг перед реальной торговлей.
3. **5-agent team** — `aqr/v04/agents/` с LangGraph orchestrator, параллельный research/analysis, доступен через `/team`.
4. **T-Invest MCP integration** — последним, после стабилизации agent API.

## Стиль

- `from __future__ import annotations` в каждом модуле.
- Type hints обязательны.
- Docstring — на русском или английском, **консистентно в модуле**.
- Без emoji в коде и коммитах.
- Модули до 400 строк (допустимо исключение для оркестраторов с >500).
- HTML-шаблон — без бандлера, vanilla JS, инлайнить CSS в `<style>` или отдельный файл в `templates/`.

## Как проверить проект руками

```bash
# 1. Запуск с обязательными env
docker start aqr-pg
export DATABASE_URL="postgresql+asyncpg://postgres:aqr@localhost:5432/aqr"
export AQR_LLM_MODEL="claude-3-5-sonnet-20241022"
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export INVEST_TOKEN="t.INVEST_TOKEN..."
export INVEST_SANDBOX=1
export AQR_SESSION_SECRET="$(openssl rand -base64 32)"

.venv/bin/alembic upgrade head
.venv/bin/uvicorn aqr.main:app --port 8000

# Без обязательной переменной — упадёт с RuntimeError на старте
unset INVEST_TOKEN
.venv/bin/uvicorn aqr.main:app --port 8000
# RuntimeError: validate_runtime() failed: INVEST_TOKEN is required

# 2. Web UI
open http://localhost:8000/chat   # ввести session_id, например "alice"

# 3. API
curl http://127.0.0.1:8000/health            # 200
curl http://127.0.0.1:8000/health/ready      # 200 если все зависимости OK
curl -X POST http://127.0.0.1:8000/pipeline/runs \
  -H "Content-Type: application/json" \
  -d '{"goal":"проверь momentum на Сбере"}'

# 4. Остановить
pkill -f "uvicorn aqr.main"
```
# AGENTS.md

Контракт для LLM-агентов, работающих с этим репозиторием.

## Проект

AQR — пайплайн для проверки торговых гипотез на MOEX через T-Invest API.
Вход — цель на русском, выход — топ-5 гипотез с Deflated Sharpe / CPCV / PBO + нарратив.

Строгий режим: любая ошибка → raise, без fallback к шаблону или синтетическим данным.

## Quick start (Docker)

```bash
# .env с минимум DATABASE_URL и AQR_SESSION_SECRET + LLM/Invest ключами
# DATABASE_URL=postgresql+asyncpg://postgres:aqr@postgres:5432/aqr
docker compose -f aqr-compose.yml up -d --build
.venv/bin/alembic upgrade head  # или из контейнера: docker compose exec app alembic upgrade head
open http://localhost:8000/chat
```

## Quick start (pip, без Docker)

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev,llm,embeddings,data,screener]"
.venv/bin/pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple

# Postgres
docker compose -f aqr-compose.yml up -d postgres
.venv/bin/alembic upgrade head

# .env (DATABASE_URL=...@localhost:5432/...)
.venv/bin/uvicorn aqr.main:app --port 8000
```

.env (автоподгружается через python-dotenv в startup.py, override=False):

```bash
DATABASE_URL=postgresql+asyncpg://postgres:aqr@localhost:5432/aqr  # pip
# DATABASE_URL=postgresql+asyncpg://postgres:aqr@postgres:5432/aqr  # Docker
AQR_SESSION_SECRET=$(openssl rand -base64 32)
AQR_LLM_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
INVEST_TOKEN=t.INVEST_TOKEN...
SSL_TBANK_VERIFY=true
```

Startup (`validate_runtime`) требует `DATABASE_URL` + `AQR_SESSION_SECRET`.
LLM/Invest/embeddings-ключи проверяются при первом использовании.

## Архитектура

```
aqr/
  graph/      LangGraph граф (plan→load→generate→backtest→validate→narrate→respond)
  tools/      ToolSpec + ToolRegistry, 13 инструментов (8 pipeline + 5 storage)
  agents/     5-role team: Editor/Browser/Analyst/Reviewer/Writer + orchestrator
  chat/       WS /chat/{token} (HMAC auth) + Web UI (vanilla JS, dark theme)
  pipeline/   PipelineExecutor, EventBus (SSE), hypothesis families, planner/narrator
  registry/   Postgres + pgvector (Alembic), Embedder, RegistryStore
  data/       TInvestAdapter (AsyncClient), OhlcvCache (DuckDB)
  validation/ DSR, CPCV, PBO, Reality Check (Bailey & López de Prado)
  screener/   VectorBT — screen_momentum (SMA-crossover grid search)
  executor/   NautilusTrader — с комиссиями и slippage (native fallback)
  api/        POST /team/run, /executor/nautilus, /mcp/rpc
  mcp/        JSON-RPC 2.0 (get_candles, resolve_figi, search_similar, find_duplicates)
```

## Ключевые точки входа

| Что | Команда |
|---|---|
| Docker (всё) | `docker compose -f aqr-compose.yml up -d --build` |
| HTTP | `uvicorn aqr.main:app --port 8000` |
| Web UI | `http://localhost:8000/chat` |
| Migrations | `.venv/bin/alembic upgrade head` или `docker compose exec app alembic upgrade head` |
| Agent вызов | `from aqr.graph import run_agent; await run_agent(...)` |
| WS | `WS /chat/{token}`, токен через `GET /chat/new?session_id=...` |
| CI | `ruff check aqr/ tests/` → `PYTHONPATH=. pytest tests/ --cov=aqr --cov-fail-under=80` |

## Инварианты (не нарушать)

1. **Look-ahead запрещён.** `backtest_one` сдвигает позицию на 1 бар (`pos.shift(1).fillna(0.0)`).
2. **Fallback запрещён.** Planner, narrator, reviewer, embedder, TInvestAdapter — всегда требуют ключи/сеть. Любая ошибка → raise.
3. **Tool registry** заполняется через `register_all()` — идемпотентен через `_registration_done`.
4. **Lazy imports.** `duckdb`, `vectorbt`, `t_tech.invest` — import только в методах.
5. **Background-task retention.** `aqr.background.schedule(coro)` вместо `asyncio.create_task`.
6. **WS auth — только HMAC.** `AQR_SESSION_SECRET` обязателен в проде.
7. **Per-session credentials** через `ContextVar` (`set_credentials`/`current_credentials`/`reset_credentials`).
8. **Контракт ошибок инструментов:** recoverable → `{"error": "..."}`, критические → raise. Проверять `"error" in result`.

## Gotchas

### TInvestAdapter

Async client, sandbox не отдаёт свечи. Все market-data запросы — production target.
FIGI resolution: предпочитает Bloomberg (`BBG*`).

```python
adapter = TInvestAdapter()
df = await adapter.candles("SBER", "2023-01-01", "2024-12-31", interval="D1")
```

### LLM-провайдеры

litellm требует provider prefix: `deepseek/deepseek-chat`, `anthropic/claude-3-5-sonnet-20241022`.
Без префикса — `BadRequestError`.

Поддерживаемые env: `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GIGACHAT_CREDENTIALS`.
`_has_llm_key()` проверяет их все + `current_credentials()` из ContextVar.

`_llm_route()`: при `step == "done"` возвращает `END` без вызова LLM (предотвращает бесконечный цикл с моками).

### Эмбеддинги

`Embedder` по умолчанию — `nomic-embed-text` (768d). `OPENAI_BASE_URL` для кастомного endpoint.
`EMBEDDING_DIM = 768` в `aqr/types.py`, `embeddings.py`, `models.py` — менять синхронно.

### `load_dotenv(override=False)`

`aqr/startup.py` вызывает `load_dotenv(_env_file, override=False)` при импорте.
Это значит:
- Если `.env` существует и переменная не задана — она устанавливается.
- `monkeypatch.delenv` в тестах после импорта `aqr.*` сработает,
  но если вызвать delenv ДО импорта — `load_dotenv` восстановит значение из `.env`.
- **Правило:** импортировать `aqr.*` модули до `monkeypatch.delenv` в тестах,
  или делать `monkeypatch.delenv` для всех трёх LLM-ключей + `INVEST_TOKEN`.

### `screen_momentum` возвращает `list[dict]`, не `list[VariantResult]`

```python
result = screen_momentum("SBER", candles=df)
# result — list[dict] с ключами: fast, slow, sharpe, sortino, max_drawdown,
# total_return, n_trades, ticker
```

Без `candles` пытается загрузить из T-Invest через `asyncio.run()`.
В тестах всегда передавать цены напрямую.

### CPCV-параметры

`plan.validation: {cpcv_splits, cpcv_test_splits, embargo_pct}` пробрасывается в `backtest_one(...)` из `graph/graph.py:backtest_node` и `pipeline/executor.py:run`.

### MCP dispatch — async, возвращает dict

```python
result = await dispatch("get_candles", {"ticker": "SBER", "from_date": "2023-01-01", ...})
# result — dict, не MCPResponse. Ключи: "jsonrpc", "result"/"error", "id".
```

`MCPError.to_dict()` не принимает `id` (id на уровне response-обёртки).

### ToolSpec — обязательный `input_schema`

```python
ToolSpec(name="x", description="x", input_schema={"type": "object", "properties": {}}, fn=my_fn)
```

## Тесты

31 файл, ~280 тестов. `pytest-asyncio` в режиме `auto` — `async def` без декоратора.
Нужен `.env` с `DATABASE_URL` + `AQR_SESSION_SECRET`.
Local coverage: `coverage run --source=aqr -m pytest tests/` (быстрее `--cov`).

### Паттерн моков

- **TInvestAdapter** → `monkeypatch.setattr(aqr.data.tinvest, "TInvestAdapter", _FakeAdapter)`
- **async_session_factory** → патчить в `aqr.session`
- **litellm** → `monkeypatch.setitem(sys.modules, "litellm", fake_module)` с `AsyncMock`
- **openai (embeddings)** → `monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)`
- **Tool registry (executor)** → `monkeypatch.setattr("aqr.tools.register.register_all", lambda: None)`,
  затем `registry._tools = {name: ToolSpec(...), ...}`.
  **Важно:** сохранять и восстанавливать `registry._tools` в autouse fixture, иначе ломаются тесты в других файлах.

### Тесты на отсутствие credentials

Если `.env` содержит ключи (`DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `INVEST_TOKEN`),
`load_dotenv(override=False)` восстанавливает их после `monkeypatch.delenv`.
Тесты на `RuntimeError` без credentials должны делать `monkeypatch.delenv` **после** импорта
или в том же порядке, что и `load_dotenv`.

## Известные gaps (не исправлены)

| Модуль | Проблема |
|---|---|
| `graph/context.py` | 4 метода: DB упала → `[]/None/""`. Вызывающий не отличает «нет данных» от «БД недоступна». |
| `agents/analyst.py` | Сбой скринера/бэктеста → `[]/None`, indistinguishable от «не нашёл идей». |
| `agents/writer.py` | Fallback-строки на русском идут в финальный отчёт; оркестратор считает прогон успешным. |
| Системная | 7 разных паттернов ошибок (raise, `{"error": ...}`, `None`, falsy defaults, catch-reraise, fallback, RuntimeError). |

## Стиль

## Docker

Docker-образ собирается из `Dockerfile` (Python 3.11-slim, все extras + t-tech-investments).
Compose-файл (`aqr-compose.yml`) поднимает два сервиса: `app` (uvicorn) и `postgres` (pgvector:pg16).

```bash
# Билд + запуск
docker compose -f aqr-compose.yml up -d --build

# Миграции
docker compose exec app alembic upgrade head

# Логи
docker compose logs -f app

# Пересборка после изменений в aqr/
docker compose build app && docker compose up -d
```

**Важно для `.env`:** в Docker DATABASE_URL должен использовать `postgres` как хост
(`postgresql+asyncpg://postgres:aqr@postgres:5432/aqr`), а не `localhost`.

**Кэш DuckDB** хранится в Docker volume `aqr-cache` → `/root/.aqr` внутри контейнера.

## Стиль

- `from __future__ import annotations` в каждом модуле
- Type hints обязательны
- Без emoji в коде и коммитах
- HTML-шаблоны — vanilla JS, без бандлера
- Комментарии — только когда неочевидно; не комментировать тривиальный код

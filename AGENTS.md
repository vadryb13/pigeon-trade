# AGENTS.md

Контракт для LLM-агентов, работающих с этим репозиторием.

## Проект

AQR — пайплайн для проверки торговых гипотез на MOEX через T-Invest API. Вход — цель на русском, выход — топ-5 гипотез с Deflated Sharpe / CPCV / PBO + нарратив.

Строгий режим: любая ошибка → raise, без fallback к шаблону или синтетическим данным.

## Quick start

```bash
# venv + deps
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev,llm,embeddings]"
.venv/bin/pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple

# Postgres (через docker compose)
docker compose -f aqr-compose.yml up -d
.venv/bin/alembic upgrade head

# .env — автоподгружается через python-dotenv в startup.py
cat > .env << EOF
DATABASE_URL=postgresql+asyncpg://postgres:aqr@localhost:5432/aqr
AQR_SESSION_SECRET=\$(openssl rand -base64 32)
AQR_LLM_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
INVEST_TOKEN=t.INVEST_TOKEN...
SSL_TBANK_VERIFY=true
EOF

.venv/bin/uvicorn aqr.main:app --port 8000
```

Startup (`validate_runtime`) требует только `DATABASE_URL` + `AQR_SESSION_SECRET`. LLM/Invest/embeddings-ключи проверяются при первом использовании. Без одного из них — RuntimeError на старте.

## Архитектура

```
aqr/
  agent/      LangGraph граф (plan→load→generate→backtest→validate→narrate→respond)
  tools/      ToolSpec + ToolRegistry, 13 инструментов (8 pipeline + 5 storage)
  agents/     5-role team: Editor/Browser/Analyst/Reviewer/Writer + orchestrator
  chat/       WS /chat/{token} (HMAC auth) + Web UI (vanilla JS, dark theme)
  pipeline/   PipelineExecutor, EventBus (SSE), hypothesis families, planner/narrator
  registry/   Postgres + pgvector (Alembic), Embedder, RegistryStore
  data/       TInvestAdapter (AsyncClient), OhlcvCache (DuckDB)
  validation/ DSR, CPCV, PBO, Reality Check (Bailey & López de Prado)
  screener/   VectorBT — screen_momentum (SMA-crossover grid search)
  executor/   NautilusTrader — с комиссиями и slippage (native fallback если не установлен)
  api/        POST /team/run, /executor/nautilus, /mcp/rpc
  mcp/        JSON-RPC 2.0 (get_candles, resolve_figi, search_similar, find_duplicates)
```

## Ключевые точки входа

| Что | Команда |
|---|---|
| HTTP | `uvicorn aqr.main:app --port 8000` |
| Web UI | открыть `http://localhost:8000/chat` |
| Agent программно | `from aqr.agent import run_agent; await run_agent(...)` |
| WS | `WS /chat/{token}`, токен через `GET /chat/new?session_id=...` |
| Реестр | `from aqr.registry import RegistryStore` + `from aqr.db import _async_session_factory` |

## WebSocket-протокол

Клиент → сервер: `{"type": "message"|"resume"|"ping"}`
Сервер → клиент: `{"type": "connected"|"history"|"user_echo"|"progress"|"tool_call"|"tool_result"|"assistant"|"done"|"error"|"pong"}`

Auth: HMAC-подпись через `aqr.auth.sign_session(session_id)`. Без токена — `close(1008)`.

## Инварианты (не нарушать)

1. **Look-ahead запрещён.** `backtest_one` сдвигает позицию на 1 бар (`pos.shift(1).fillna(0.0)`).
2. **Fallback запрещён.** Planner, narrator, reviewer, embedder, TInvestAdapter — всегда требуют ключи/сеть. Любая ошибка → raise.
3. **Tool registry** заполняется через `register_all()` — идемпотентен через `_registration_done` флаг.
4. **Lazy imports.** `duckdb` (`ohlcv_cache.py`), `vectorbt` (`screener/vectorbt.py`), `t_tech.invest` (`tinvest.py`) — import только в методах.
5. **Background-task retention.** Использовать `aqr.tasks.schedule(coro)` вместо `asyncio.create_task` (GC съедает без strong-ref).
6. **WS auth — только HMAC.** `AQR_SESSION_SECRET` обязателен в проде (иначе сессии теряются при рестарте).
7. **Per-session credentials** пробрасываются через `ContextVar` (`set_credentials`/`current_credentials`/`reset_credentials`). Этот же механизм использует и `_api_key_from_context()` в Embedder, и `TInvestAdapter.__init__`.

## Gotchas

### TInvestAdapter — новый SDK (1.0.0+)

Переписан под `AsyncClient` (async context manager). Методы `candles()` и `_resolve_figi()` — **async**:

```python
adapter = TInvestAdapter()
figi = await adapter._resolve_figi("SBER")
df = await adapter.candles("SBER", "2023-01-01", "2024-12-31", interval="D1")
```

Sandbox endpoint не отдаёт свечи. Все запросы market data идут на production target (`INVEST_GRPC_API`), независимо от `INVEST_SANDBOX`.

FIGI resolution: при нескольких FIGI предпочитает Bloomberg (начинается с `BBG`). Для правильного поиска шер передаёт `instrument_kind=InstrumentType.INSTRUMENT_TYPE_SHARE`.

### CandleInterval — расширенный набор

Дополнительно к 7 базовым (`1m/5m/15m/H1/D1/W/M`) поддерживаются `2m/3m/10m/30m/2H/4H`.

### SSL в РФ-окружении

```bash
export SSL_TBANK_VERIFY=true   # сертификат МинЦифры поставляется с t-tech-investments
```

### LLM-провайдеры

litellm требует provider prefix: `deepseek/deepseek-chat`, `anthropic/claude-3-5-sonnet-20241022`. Без префикса — `BadRequestError`.

Поддерживаемые env: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GIGACHAT_CREDENTIALS`. `_has_llm_key()` в `graph.py` проверяет их все + `current_credentials()` из ContextVar.

### Эмбеддинги — Ollama (дефолт) или OpenAI

`Embedder` по умолчанию использует `nomic-embed-text` (768d). `OPENAI_BASE_URL` задаёт кастомный endpoint; `OPENAI_API_KEY=ollama` для локального Ollama (игнорируется).

Размерность вектора: 768 (задана в `EMBEDDING_DIM` в `embeddings.py` и в `Vector(768)` в `models.py`). Менять — синхронно в обоих файлах + `ALTER TABLE hypotheses ALTER COLUMN embedding TYPE vector(N)`.

### `screen_momentum` принимает опциональные цены

```python
screen_momentum("SBER", candles=preloaded_df)
```

Без `candles` пытается загрузить из T-Invest через `asyncio.run()`. Из тестов передавать цены напрямую.

### CPCV-параметры

`plan.validation: {cpcv_splits, cpcv_test_splits, embargo_pct}` должна явно пробрасываться в `backtest_one(...)`. Сейчас пробрасывается из `agent/graph.py:backtest_node` и `pipeline/executor.py:run`.

## Тесты

- `pytest-asyncio` в режиме `auto` — `async def` без `@pytest.mark.asyncio`
- Нужен `.env` с переменными (DATABASE_URL, AQR_SESSION_SECRET) — без них часть тестов падает
- `pytest --cov` вызывает `ImportError` на некоторых модулях. В CI: `coverage run -m pytest`

### Паттерн моков

- `TInvestAdapter` → `monkeypatch.setattr(aqr.data.tinvest, "TInvestAdapter", _FakeAdapter)` (класс целиком)
- `_async_session_factory` → патчить в `aqr.db` И в импортирующем модуле
- `litellm` → `monkeypatch.setitem(sys.modules, "litellm", fake_module)` с `AsyncMock(fake_resp)`
- `get_agent` → патчить `aqr.agent.graph.get_agent` И `aqr.chat.ws.get_agent`
- `REGISTRY` → `aqr.tools.register._reset_registration_done()` + `reset_for_testing()` между тестами

## Стиль

- `from __future__ import annotations` в каждом модуле
- Type hints обязательны
- Без emoji в коде и коммитах
- HTML-шаблоны — vanilla JS, без бандлера

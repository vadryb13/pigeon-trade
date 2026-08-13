# AQR

Strict-mode quant pipeline: вход — цель на естественном языке, выход — топ-5 гипотез с **Deflated Sharpe / CPCV / PBO** и нарративом на русском. Данные — MOEX через T-Invest gRPC. Бэктест — vectorized (native) + опционально NautilusTrader (комиссии / slippage).

Строгий режим: любая ошибка → raise. Без fallback к шаблону или синтетическим данным.

## Быстрый старт

### Docker (рекомендуется)

```bash
# .env с минимум DATABASE_URL, AQR_SESSION_SECRET + LLM/Invest ключами
cp .env.example .env
# отредактировать .env: задать ключи

docker compose -f aqr-compose.yml up -d --build
docker compose exec app alembic upgrade head
open http://localhost:8000/chat
```

### Pip (ручная установка)

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev,llm,embeddings,data,screener]"
.venv/bin/pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple

# Postgres
docker compose -f aqr-compose.yml up -d postgres
.venv/bin/alembic upgrade head

# .env с DATABASE_URL=...@localhost:5432/...
.venv/bin/uvicorn aqr.main:app --port 8000
```

### Обязательные env-переменные

```bash
DATABASE_URL=postgresql+asyncpg://postgres:aqr@localhost:5432/aqr  # pip
# DATABASE_URL=postgresql+asyncpg://postgres:aqr@postgres:5432/aqr  # Docker
AQR_SESSION_SECRET=$(openssl rand -base64 32)
AQR_LLM_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=sk-...
INVEST_TOKEN=t.INVEST_TOKEN...
```

`STARTUP` (`validate_runtime`) требует `DATABASE_URL` + `AQR_SESSION_SECRET`. LLM/Invest/embeddings-ключи проверяются при первом использовании.

## HTTP API

```bash
uvicorn aqr.main:app --port 8000
```

| Метод | Эндпоинт | Описание |
|---|---|---|
| `GET` | `/health` | Liveness (всегда 200) |
| `GET` | `/health/ready` | Readiness — Postgres, 503 если degraded |
| `POST` | `/pipeline/runs` | Стартовать прогон: `{"goal": "..."}` |
| `GET` | `/pipeline/runs/{run_id}` | Снимок событий и статус |
| `GET` | `/pipeline/runs/{run_id}/stream` | SSE-лента событий |
| `GET` | `/chat` | Web UI чата (HTML) |
| `POST` | `/chat/new` | Новая серверная сессия и HMAC-токен для WS |
| `WS` | `/chat/{token}` | Двусторонний диалог с агентом |
| `GET` | `/chat/{token}/settings` | Форма настроек сессии (LLM/Invest keys) |
| `POST` | `/chat/{token}/settings` | Сохранение credentials |
| `GET` | `/explore` | Web UI explore (HTML) |
| `GET` | `/activity` | Web UI активности (HTML) |
| `GET` | `/api/explore/hypotheses` | Список гипотез (JSON) |
| `GET` | `/api/explore/stats` | Агрегатные метрики (JSON) |
| `POST` | `/team/run` | 5-agent оркестратор: `{"goal": "..."}` |
| `POST` | `/executor/nautilus` | NautilusTrader бэктест |
| `POST` | `/mcp/rpc` | JSON-RPC 2.0: `get_candles`, `resolve_figi`, `search_similar`, `find_duplicates` |

## Web UI

Две HTML-страницы (vanilla JS, без бандлера):

**Чат** (`/chat`) — тёмная тема, моноширинный шрифт:
- Markdown-rendering (headings, lists, code blocks, bold/italic)
- Slash-команды: `/help`, `/run <goal>`, `/history`, `/clear`, `/exit`
- Auto-reconnect при обрыве WebSocket
- localStorage сохраняет токен между перезагрузками
- Auth-флоу: сервер создаёт непрозрачную сессию → HMAC-токен → WS. При истечении (30 дней) — баннер.

**Explore** (`/explore`) — светлая тема, таблица гипотез:
- Сортировка по sharpe / DSR / PBO
- График equity curve на детальной странице
- Presence tracking (кто онлайн / что смотрит)
- Pessimistic lock на редактирование

## WebSocket протокол

Клиент → сервер:
```json
{"type": "message", "content": "проверь momentum на Сбере"}
{"type": "resume"}
{"type": "ping"}
```

Сервер → клиент:
```json
{"type": "history",    "messages": [...]}
{"type": "user_echo",  "content": "..."}
{"type": "progress",   "node": "backtest", "data": {...}}
{"type": "done",       "narrative": "...", "assistant": "..."}
{"type": "error",      "message": "..."}
{"type": "pong"}
```

Полный контракт в `aqr/chat/ws.py`.

## Архитектура

```
aqr/
  main.py       FastAPI app + lifespan (validate_runtime + drain)
  graph/        LangGraph граф (plan→load→generate→backtest→validate→narrate→respond)
  agents/       5-role team: Editor/Browser/Analyst/Reviewer/Writer + orchestrator
  tools/        ToolSpec + ToolRegistry, 13 инструментов (8 pipeline + 5 storage)
  chat/         WS /chat/{token} (HMAC auth) + Web UI
  pipeline/     PipelineExecutor, EventBus (SSE), hypothesis families, planner/narrator/reviewer
  registry/     Postgres + pgvector (Alembic), Embedder, RegistryStore
  data/         TInvestAdapter (AsyncClient), OhlcvCache (DuckDB)
  validation/   DSR, CPCV, PBO, Reality Check (Bailey & Lopez de Prado)
  screener/     VectorBT — screen_momentum (SMA-crossover grid search)
  executor/     NautilusTrader — комиссии + slippage (native fallback)
  mcp/          JSON-RPC 2.0 (get_candles, resolve_figi, search_similar, find_duplicates)
  explore/      REST /api/explore + SSE + presence tracking
```

### Что делает пайплайн

1. **Plan** — LLM превращает цель в `ResearchPlan` (тикеры, семейства, таймфрейм, N).
2. **Load** — DuckDB-кэш → T-Invest gRPC (retry + circuit breaker).
3. **Generate** — параметризованные семейства: `momentum`, `mean_reversion`, `breakout`, `volatility`.
4. **Backtest** — vectorized с `shift(1)` (look-ahead запрещён), Sharpe, drawdown, CPCV OOS.
5. **Validate** — PBO по всему портфелю.
6. **Insights** — детерминистичные + LLM-review (0-3 дополнительных наблюдения).
7. **Narrate** — LLM пишет 3-6 абзацев по-русски.

Каждый шаг публикует Event в EventBus. UI подписывается через SSE или WebSocket.

## Эмбеддинги (семантический поиск)

`nomic-embed-text` (768d). `OPENAI_BASE_URL` для кастомного endpoint. Без ключа — детерминистический hash-вектор.

```python
from aqr.registry.embeddings import Embedder
emb = await Embedder().embed_hypothesis("momentum", "SBER", {"fast": 5, "slow": 50})
```

Используется для дедупликации (`cosine >= 0.92`) и семантического поиска (`threshold=0.7`).

## Тесты

```bash
pytest tests/ -x -q --tb=short               # 317 тестов в 31 файле
ruff check aqr/ tests/                        # линтер
coverage run --source=aqr -m pytest tests/   # coverage (быстрее --cov)
```

## Структура проекта

```
aqr/
  graph/         LangGraph граф + SessionContext
  agents/        5-role team + orchestrator
  tools/         ToolSpec + ToolRegistry (13 инструментов)
  chat/          WS /chat/{token} + HTML-шаблоны
  pipeline/      PipelineExecutor, EventBus, planner/narrator/reviewer
  registry/      SQLAlchemy модели, pgvector, RegistryStore, Embedder
  data/          TInvestAdapter, OhlcvCache (DuckDB)
  validation/    DSR, CPCV, PBO, Reality Check
  screener/      VectorBT screen_momentum
  executor/      NautilusTrader (native fallback)
  mcp/           JSON-RPC 2.0 диспетчер
  explore/       REST + SSE + presence tracking
  api/           POST /team/run, /executor/nautilus, /mcp/rpc
alembic/         Миграции (Alembic)
tests/           31 файл, 317 тестов
```

Полное описание для LLM-агентов: [AGENTS.md](AGENTS.md).

## Ограничения

- T-Invest sandbox не отдаёт свечи — все market-data запросы в production target.
- DuckDB: конкурентные подключения к одному файлу падают (см. AGENTS.md).
- Backtest не учитывает комиссии и проскальзывание в native-режиме (NautilusTrader — учитывает).

## Лицензия

Apache 2.0.

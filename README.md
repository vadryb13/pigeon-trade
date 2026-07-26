# AQR

Thin pipeline для проверки торговых гипотез на MOEX: вход — цель на естественном языке («проверь momentum на Сбере»), выход — топ-5 гипотез с **Deflated Sharpe / CPCV / PBO** и нарративом на русском.

## Быстрый старт

```bash
pip install -e ".[dev]"
python -m aqr "проверь momentum на Сбере"
```

CLI работает **без LLM-ключей** (fallback-планировщик и fallback-нарратор) и **без Postgres** (синтетические данные при недоступности MOEX). Для полного режима:

```bash
docker run -d --name aqr-pg -e POSTGRES_PASSWORD=aqr -p 5432:5432 pgvector/pgvector:pg16
alembic upgrade head
```

С LLM (Claude / GPT / GigaChat):

```bash
export ANTHROPIC_API_KEY=...
export AQR_LLM_MODEL=claude-3-5-sonnet-20241022
python -m aqr "что работает у металлургов?"
```

## HTTP API

```bash
uvicorn aqr.main:app --port 8000
```

| Метод | Эндпоинт | Описание |
|---|---|---|
| `POST` | `/pipeline/runs` | Стартовать прогон: `{"goal": "..."}` → `{run_id, plan}` |
| `GET` | `/pipeline/runs/{run_id}` | Снимок событий и статус |
| `GET` | `/pipeline/runs/{run_id}/stream` | SSE-лента событий в реальном времени |
| `GET` | `/chat` | Web-UI (HTML-страница чата в браузере) |
| `GET` | `/chat/new?session_id=...` | Выпустить HMAC-токен для WS-подключения |
| `WS` | `/chat/{token}` | Двусторонний диалог с агентом (требует валидный токен) |
| `GET` | `/health` | Liveness (всегда 200) |
| `GET` | `/health/ready` | Readiness — проверяет Postgres + MOEX, 503 если degraded |

## Web UI

Откройте `http://localhost:8000/chat` в браузере после запуска сервера.

- **Тёмная тема** с моноширинным шрифтом в стиле терминала
- **Markdown-rendering** в ответах агента (headings, lists, code blocks, bold/italic)
- **Звуковое уведомление** на `done` (subtle beep через Web Audio API)
- **Slash-команды**:
  - `/help` — список команд
  - `/run <goal>` — стартовать новый run (алиас для обычного сообщения)
  - `/history` — загрузить историю чата из БД
  - `/clear` — очистить окно (история в БД сохраняется)
  - `/exit` — закрыть сессию
- **Auto-reconnect** при обрыве WebSocket
- **localStorage** сохраняет токен между перезагрузками страницы

Auth-флоу:
1. Пользователь вводит `session_id` → `GET /chat/new?session_id=alice`
2. Сервер возвращает `{token, session_id}` (подписан через `AQR_SESSION_SECRET`)
3. Браузер открывает `ws://localhost:8000/chat/{token}`
4. При истечении токена (30 дней) клиент показывает баннер и предлагает войти заново

Legacy режим для dev: `AQR_REQUIRE_WS_AUTH=0` отключает проверку токена (WS принимает любой session_id напрямую).

## WebSocket протокол

Клиент → сервер:
```json
{"type": "message", "content": "проверь momentum на Сбере"}
{"type": "resume"}
{"type": "ping"}
```

Сервер → клиент:
```json
{"type": "history",    "messages": [{"role": "user", "content": "..."}]}
{"type": "user_echo",  "content": "..."}
{"type": "progress",   "node": "backtest", "data": {...}}
{"type": "done",       "narrative": "...", "assistant": "..."}
{"type": "error",      "message": "..."}
{"type": "pong"}
```

Полный контракт в `aqr/chat/ws.py`. Клиент: `websocat ws://localhost:8000/chat/sess-1`.

## Архитектура

```
┌────────────────────────────────────────────────────────────────┐
│ Storage layer (Postgres + pgvector + Alembic)                  │
│   sessions → runs → hypotheses                                 │
│   + chat_messages (история WebSocket-диалогов)                │
│   + Hypothesis.embedding (Vector 1536)                         │
└────────────────────────────────────────────────────────────────┘
                          ▲
┌────────────────────────────────────────────────────────────────┐
│ Tool layer (aqr/tools/)                                        │
│   13 инструментов в registry: plan_research, load_prices,     │
│   backtest_one, validate_portfolio, narrate, search_similar,  │
│   find_duplicates, ...                                         │
│   Каждый = ToolSpec(name, description, input_schema, fn)        │
└────────────────────────────────────────────────────────────────┘
                          ▲
┌────────────────────────────────────────────────────────────────┐
│ Agent layer (LangGraph)                                        │
│   route → plan → load_data → generate → backtest →            │
│     validate → narrate → respond                              │
│   + SessionContext (история, лучшая стратегия, белые пятна)   │
└────────────────────────────────────────────────────────────────┘
```

## Что делает пайплайн

1. **ChatPlanner** — превращает цель в `ResearchPlan` (тикеры, семейства, таймфрейм, N).
2. **load_prices** — DuckDB-кэш → MOEX ISS → синтетический GBM fallback.
3. **generate_hypotheses** — параметризованные семейства: `momentum`, `mean_reversion`, `breakout`, `volatility`.
4. **backtest_one** — vectorized backtest с `shift(1)`, Sharpe, drawdown, число сделок, CPCV OOS.
5. **validate_portfolio** — PBO по всему портфелю.
6. **extract_insights** + **review_insights** — детерминистичные + LLM-наблюдения.
7. **narrate** — LLM или fallback пишет 3-6 абзацев по-русски.

Каждый шаг публикует Event в EventBus. UI/CLI подписываются через SSE или WebSocket.

## Resilience (этап 7)

- **HTTP timeout 10s** на каждый запрос к MOEX (было 30s).
- **3 retry** с exponential backoff (0.5s → 1s → 2s) на 5xx и ConnectionError.
- **Per-ticker circuit breaker**: 5 ошибок подряд → 60s все запросы к этому тикеру идут в synthetic fallback без сетевых вызовов.
- **DuckDB-кэш OHLCV**: повторные прогоны на тех же тикерах не ходят в MOEX.
- **Structured logging**: JSON-формат при `AQR_LOG_JSON=1` (поля `run_id, tool, duration_ms, status, error`).
- **Readiness probe**: `/health/ready` → 503 если Postgres или MOEX недоступны.

## Эмбеддинги (семантический поиск)

OpenAI `text-embedding-3-small` (1536d, ~$0.02/1M токенов ≈ десятки центов в месяц).
Без API-ключа — детерминистический hash-вектор.

```python
from aqr.registry.embeddings import Embedder
emb = await Embedder().embed_hypothesis("momentum", "SBER", {"fast": 5, "slow": 50})
```

Используется для:
- **Дедупликации**: `plan_research` предупреждает если похожая гипотеза уже проверялась (`cosine ≥ 0.92`).
- **Семантического поиска**: `search_similar_hypotheses` (threshold=0.7) и `find_duplicates` (threshold=0.92).

## Тесты

```bash
pytest tests/ -v                          # 212 тестов, без Postgres и LLM
pytest tests/ --cov=aqr --cov-report=term # coverage > 80%
ruff check aqr/ tests/                    # линтер
```

19 файлов тестов:
- `test_validation.py` — DSR / PBO / CPCV / Reality Check (reference)
- `test_pipeline_e2e.py` — fallback-планировщик + полный e2e на синтетике
- `test_reviewer.py` — InsightReviewer
- `test_tools.py` — `ToolRegistry` + изолированные вызовы 13 инструментов
- `test_agent.py` — граф, роутер, `SessionContext`, `run_agent()` e2e
- `test_chat_ws.py` — WebSocket через FastAPI TestClient
- `test_chat_web.py` — Web UI endpoints (`/chat`, `/chat/new`)
- `test_ohlcv_cache.py` — DuckDB-кэш + NaN-handling + `AQR_CACHE_DIR`
- `test_cache_reuse.py` — кэш переиспользуется между прогонами
- `test_embeddings.py` — hash/OpenAI embeddings + дедуп
- `test_context.py` — `SessionContext` с моком БД
- `test_storage_tools.py` — storage-инструменты с моком БД
- `test_moex_retry.py` — retry + circuit breaker
- `test_health.py` — `/health` + `/health/ready`
- `test_logging_config.py` — JsonFormatter + `log_tool_call`
- `test_cpcv_edge.py` — purge/embargo edge cases
- `test_api_routes.py` — FastAPI роуты + `_run_and_persist`
- `test_auth.py` — HMAC sign/verify round-trip
- `test_smoke.py` — CLI subprocess

## Структура проекта

```
aqr/
├── pipeline/         # оркестратор: planner, executor, narrator, SSE-events, api
├── tools/            # Tool layer (13 инструментов)
├── agent/            # LangGraph граф + SessionContext
├── chat/             # WebSocket-диалог + Web UI (HTML+JS чат)
│   ├── ws.py        # WebSocket endpoint /chat/{token}
│   ├── web.py       # GET /chat (HTML) + GET /chat/new (token)
│   └── templates/   # chat.html (vanilla JS, dark theme)
├── registry/         # SQLAlchemy модели + RegistryStore + Embedder
├── validation/       # Deflated Sharpe, CPCV, PBO, Reality Check
├── data/             # MOEXAdapter (с retry+CB), OhlcvCache (DuckDB)
├── logging_config.py # JsonFormatter + log_tool_call
├── main.py           # FastAPI app
├── cli.py            # `python -m aqr <goal>`
alembic/              # миграции
tests/                # 212 тестов в 19 файлах
```

Полное описание для LLM-агентов: [AGENTS.md](AGENTS.md).

## Ограничения

- MOEX ISS исторически не даёт H1 глубже нескольких месяцев — для H1-стратегий покрытие короткое.
- Backtest не учитывает комиссии и проскальзывание (planned).
- Fallback-планировщик — по ключевым словам, не понимает сложные формулировки.
- ivfflat-индекс на `hypothesis.embedding` намеренно отложен (нельзя на пустой таблице).

## Out of scope

Транзакционные издержки / slippage, многопользовательский режим, аутентификация, RL/auto-ML, live-trading, источники кроме MOEX.

## Лицензия

Apache 2.0.
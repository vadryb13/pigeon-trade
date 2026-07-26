# TASKS.md — поэтапный план реализации

## Этап 1: Registry & Storage (Postgres + pgvector + Alembic) ✅

**Статус:** завершён.

- [x] SQLAlchemy-модели `Run`, `Hypothesis`, `Session` с pgvector (Vector 1536)
- [x] Alembic-конфигурация (async engine, autogenerate-ready)
- [x] Миграция `fcb396c4088d`: sessions → runs → hypotheses (FK-каскад, индексы)
- [x] `RegistryStore`: 15 методов CRUD + batch-сохранение `save_pipeline_result`
- [x] `aqr/db.py`: фабрика async-сессий + FastAPI-зависимость `get_db()`
- [x] API-интеграция: `POST /pipeline/runs` сохраняет run в БД, фон завершает запись

**Результат:** пайплайн пишет все прогоны и топ-5 гипотез в Postgres.

---

## Этап 2: Извлечение инструментов из пайплайна (3–5 часов) ✅

**Цель:** разобрать `PipelineExecutor.run()` на независимые инструменты с общим контрактом `ToolSpec`.

### 2.1 Контракт инструмента ✅

- [x] Создать `aqr/tools/__init__.py`
- [x] Определить `ToolSpec` — датакласс: `name`, `description` (на русском), `input_schema` (JSON Schema), `fn` (Callable), `category`
- [x] Реализовать `ToolRegistry` — реестр: `register(tool)`, `get(name)`, `list_all()`, `list_for_llm()` (имена + описания для промпта)

### 2.2 Инструменты ядра ✅

- [x] `plan_research` — обёртка над `ChatPlanner.plan()`: `goal → ResearchPlan`
- [x] `load_prices` — извлечь `executor._load_data` в `aqr/tools/core.py`: `(tickers, start, end, timeframe) → dict[str, pd.Series]` (+ synthetic fallback)
- [x] `generate_hypotheses` — обёртка над существующей `generate_hypotheses()`: `(tickers, families, n) → list[HypothesisSpec]`
- [x] `backtest_one` — извлечь `executor._backtest_one` в `aqr/tools/core.py`: `(HypothesisSpec, prices) → BacktestResult`
- [x] `validate_portfolio` — извлечь `executor._portfolio_pbo` в `aqr/tools/core.py`: `(list[BacktestResult]) → PBO dict`
- [x] `extract_insights` — извлечь `executor._extract_insights` в `aqr/tools/core.py`: `(PipelineResult) → list[str]`
- [x] `review_insights` — обёртка над существующим `InsightReviewer.review()`: `(result, det_insights) → list[str]`
- [x] `narrate` — обёртка над существующим `Narrator.narrate()`: `(PipelineResult) → str`

> Реализовано в `aqr/tools/core.py` (8 функций). Все зарегистрированы в `aqr/tools/register.py` (`register_all()`).

### 2.3 Инструменты хранилища ✅

- [x] `get_run` — `(run_id) → Run | None` (из RegistryStore)
- [x] `compare_runs` — `(run_id_a, run_id_b) → dict` (сравнение метрик двух прогонов)
- [x] `list_runs` — `(session_id, limit) → list[Run]`

> Реализовано в `aqr/tools/storage.py`. Итого 11 инструментов в реестре.

### 2.4 Сохранить совместимость ✅

- [x] Переписать `PipelineExecutor.run()` чтобы он использовал ToolRegistry вместо приватных методов
- [x] Убедиться что существующий CLI и `/pipeline/*` endpoint'ы продолжают работать
- [x] `register_all()` сделан идемпотентным — повторные вызовы из `executor.run()` и `agent/graph` не падают
- [x] `aqr/data/__init__.py` использует ленивый импорт `DataManifest` через `__getattr__` (PEP 562) — `import aqr.data` больше не требует duckdb
- [x] Прогнать `pytest tests/ -v` — все тесты зелёные (76 passed)


### 2.5 Тесты на инструменты ✅

- [x] `tests/test_tools.py`: каждый инструмент вызывается изолированно с минимальными входными данными
- [x] `tests/test_tools.py`: `ToolRegistry.list_all()` возвращает все 11 инструментов


**Результат:** каждый шаг пайплайна можно вызвать отдельно. Пайплайн становится композицией инструментов.

---

## Этап 3: Агентный слой на LangGraph (5–8 часов) ✅

**Статус:** завершён.

**Цель:** заменить фиксированную последовательность шагов на агента, который сам решает какие инструменты вызывать.

### 3.1 LangGraph-граф ✅

- [x] Добавить `langgraph` в зависимости (добавлен в pyproject.toml)
- [x] Создать `aqr/agent/__init__.py`
- [x] Определить `AgentState` — TypedDict: `messages`, `goal`, `plan`, `prices`, `hypotheses`, `results`, `pbo`, `insights`, `narrative`, `session_id`, `step`, `error`, `metadata`
- [x] Реализовать узлы графа в `aqr/agent/graph.py`:
  - `plan` — вызывает `plan_research` (через ToolRegistry)
  - `load_data` — вызывает `load_prices`
  - `generate` — вызывает `generate_hypotheses`
  - `backtest` — вызывает `backtest_one` × N (цикл)
  - `validate` — вызывает `validate_portfolio`
  - `narrate` — вызывает `extract_insights` + `review_insights` + `narrate`
  - `respond` — формирует финальный ответ
  - `route` — входной узел: различает новую цель от уточнения
- [x] Собрать граф: `route → plan → load_data → generate → backtest → validate → narrate → respond → END`
- [x] Добавить conditional edge: после `respond` → `route` (ждёт уточнения или завершает)
- Два роутера:
  - `_deterministic_route` — линейная последовательность для пайплайна
  - `_llm_route` — через litellm для follow-up-роутинга

> Реализовано в `aqr/agent/graph.py` (~550 строк). API: `run_agent(message, session_id)`.

### 3.2 Контекст сессии ✅

- [x] Создать `aqr/agent/context.py` — `SessionContext`:
  - `get_recent_runs(session_id, limit=5)` — последние прогоны
  - `get_best_strategy(session_id)` — лучшая гипотеза по DSR
  - `get_untested_combos(session_id)` — «белые пятна» (семейства × тикеры без прогонов)
- [x] `SessionContext.build_context_prompt()` — собирает всё в промпт-строку
- [x] Контекст подмешивается в `AgentState.session_context_prompt` и используется `_llm_route` для augmenting LLM-промпта
- [x] Устойчив к отсутствию БД — все методы возвращают пустые значения (тесты и CLI работают без Postgres)

> Реализовано в `aqr/agent/context.py`, интегрирован в `aqr/agent/graph.py:run_agent` и `_llm_route`.

### 3.3 Тесты на граф ✅

- [x] `tests/test_agent.py`: граф проходит от `plan` до `narrate` без ошибок
- [x] `tests/test_agent.py`: роутер завершает диалог после `narrate`
- [x] `tests/test_agent.py`: роутер принимает уточняющее сообщение и перезапускает пайплайн (`route_node` сбрасывает state при новом user message после `done`)
- [x] `tests/test_agent.py`: `SessionContext` устойчив к отсутствию БД


**Результат:** агент может провести полное исследование и ответить на уточняющие вопросы без перезапуска всего пайплайна.

---

## Этап 4: WebSocket-интерфейс чата (2–3 часа) ✅

**Статус:** завершён.

**Цель:** заменить SSE-ленту на двусторонний WebSocket-диалог с агентом.

### 4.1 WebSocket endpoint ✅

- [x] Добавить `WebSocket` endpoint в `aqr/main.py`: `WS /chat/{session_id}`
- [x] Каждое входящее сообщение пользователя → запуск графа агента
- [x] Промежуточные результаты стримятся как JSON-сообщения (`{"type": "progress", "node": "...", "data": {...}}`)
- [x] Финальный ответ — `{"type": "done", "narrative": "...", "assistant": "..."}`

> Реализовано в `aqr/chat/ws.py` (~210 строк). Использует `astream` LangGraph для прогресса.

### 4.2 Сохранение истории ✅

- [x] Сохранять историю диалога в отдельной таблице `chat_messages` (`role`, `content`, `meta`, `created_at`) с FK на `sessions(id) ON DELETE CASCADE`
- [x] При reconnect'е — `{"type": "resume"}` → сервер шлёт `{"type": "history", "messages": [...]}`
- [x] Alembic-миграция `a1b2c3d4e5f6_add_chat_messages.py`
- [x] `RegistryStore.save_chat_message` / `list_chat_history`

### 4.3 Совместимость ✅

- [x] `GET /pipeline/runs/{id}/stream` (SSE) — продолжает работать
- [x] `POST /pipeline/runs` — продолжает работать fire-and-forget

**Результат:** пользователь может общаться с агентом в реальном времени через WebSocket.

---

## Этап 5: Эмбеддинги и семантический поиск (2–3 часа) ✅

**Статус:** завершён.

**Цель:** векторизовать гипотезы для поиска похожих и дедупликации.

### 5.1 Генерация эмбеддингов ✅

- [x] Создать `aqr/registry/embeddings.py`:
  - `Embedder.embed(text) → list[float]` — OpenAI API или hash-fallback (1536d, L2-норма=1)
  - `Embedder.embed_hypothesis(family, ticker, params)` — канонический текст `"{family} on {ticker}: {sorted_params}"` → эмбеддинг
  - Модель: `text-embedding-3-small` (OpenAI, 1536d, $0.02/1M токенов ≈ <$1/месяц при typical usage)
  - Fallback: SHA256-based детерминистический вектор (L2-норма=1), работает без ключа

### 5.2 Поиск похожих ✅

- [x] `RegistryStore.search_similar(embedding, threshold=0.0, limit=10)` — cosine distance через pgvector, возвращает `[(Hypothesis, similarity)]`
- [x] `RegistryStore.search_by_text(query, embedder, limit=10)` — текст → embedding → search_similar

### 5.3 Дедупликация ✅

- [x] `plan_research` после разбора цели проверяет `search_similar(threshold=0.92)`, добавляет `dedup_warning` и `similar_runs` в план
- [x] В `_run_and_persist` (`aqr/pipeline/api.py`): `embedding` заполняется при сохранении топ-5 гипотез каждого прогона
- [x] Два storage-инструмента в реестре: `search_similar_hypotheses`, `find_duplicates` (всего 13 tools)

### 5.4 Тесты ✅

- [x] `tests/test_embeddings.py`: hash-fallback детерминистичный и нормализованный
- [x] `tests/test_embeddings.py`: cosine similarity семантика для похожих/разных текстов
- [x] `tests/test_embeddings.py`: `plan_research` без БД не падает; с моком `search_similar` — добавляет `dedup_warning`

**Результат:** платформа помнит что уже проверяла и предупреждает о дубликатах.

---

## Этап 6: DuckDB-кэш OHLCV (2–3 часа) ✅

**Статус:** завершён.

**Цель:** ускорить загрузку данных и снизить нагрузку на MOEX ISS API.

### 6.1 Кэш ✅

- [x] Создать `aqr/data/ohlcv_cache.py`:
  - `OhlcvCache.get_cached(ticker, start, end, timeframe) → pd.DataFrame | None` — read-through кэш
  - `OhlcvCache.put_cache(ticker, df, timeframe)` — upsert (ON CONFLICT DO UPDATE)
  - `OhlcvCache.invalidate(ticker=None)` — очистка по тикеру или целиком
  - `OhlcvCache.stats()` — `{tickers, rows}` для отладки
  - Файл: `data/ohlcv_cache.duckdb`, создаётся автоматически при первом обращении
  - PRAGMA: `PRIMARY KEY (ticker, timeframe, begin)`, индекс `(ticker, timeframe, begin)`
  - Lazy import `duckdb` (требует `[data]` extra)

### 6.2 Интеграция с `load_prices` ✅

- [x] `load_prices` сначала проверяет DuckDB-кэш для всех тикеров
- [x] Для промахов — идёт в MOEX ISS API и сохраняет в кэш (полный DataFrame с OHLCV)
- [x] Fallback на synthetic GBM сохраняется (если MOEX недоступен)
- [x] Повторные прогоны на тех же тикерах не ходят в MOEX (verified в тесте `test_load_prices_uses_cache_on_second_call`)

### 6.3 Тесты ✅

- [x] `tests/test_ohlcv_cache.py`: put 500 баров → get возвращает те же данные (`test_put_get_roundtrip`)
- [x] `tests/test_ohlcv_cache.py`: промах возвращает None (`test_get_miss_returns_none`)
- [x] `tests/test_ohlcv_cache.py`: частичное окно, разные таймфреймы, upsert, invalidate, stats
- [x] `tests/test_ohlcv_cache.py`: интеграция с `load_prices` (монkeypatch на MOEXAdapter)

**Результат:** повторные прогоны на тех же тикерах не ходят в MOEX API.

---

## Этап 7: Полировка и production hardening (3–5 часов) ✅

**Статус:** завершён.

### 7.1 Обработка ошибок ✅

- [x] Таймауты на MOEX ISS (10 секунд) — `MOEXAdapter(timeout=10)` по умолчанию
- [x] Retry с exponential backoff (3 попытки) — tenacity `@retry(stop=stop_after_attempt(3), wait=wait_exponential(0.5-4s))`
- [x] Circuit breaker per-ticker: 5 ошибок подряд → 60s CB открыт, синтетический fallback без сетевых вызовов
- [x] Reset CB через `MOEXAdapter.reset_breakers()` (admin-команда)

> Реализовано в `aqr/data/moex.py` (`_request_with_retry`, `_check_breaker`, `_record_success/failure`). Тесты: `tests/test_moex_retry.py` (15 тестов).

### 7.2 Мониторинг ✅

- [x] Структурированное логирование — `aqr/logging_config.py` (JsonFormatter, log_tool_call)
- [x] Поля `{run_id, tool, duration_ms, status, error}` фиксированы в JsonFormatter
- [x] `AQR_LOG_JSON=1` → JSON, иначе human-readable
- [x] `log_tool_call(logger, run_id, tool, duration_ms, status, error)` — единый хелпер
- [x] Инструментирован `aqr/pipeline/executor.py` (`_emit` пишет structured-лог)
- [x] Health-check: `/health` (liveness) + `/health/ready` (readiness: Postgres + MOEX, 503 если degraded)

> Тесты: `tests/test_logging_config.py` (9), `tests/test_health.py` (5).

### 7.3 Документация ✅

- [x] README.md переписан (~180 строк): архитектура, WebSocket протокол, resilience, эмбеддинги, тесты
- [x] Docstrings в `aqr/main.py`, `aqr/chat/ws.py`, `aqr/logging_config.py`, `aqr/data/moex.py` — публичные методы покрыты

### 7.4 Финальное тестирование ✅

- [x] `pytest tests/ --cov=aqr` → **coverage 81%** (>80%) ✅
- [x] `ruff check aqr/ tests/` → 28 ошибок остались (136 auto-fixed, оставшиеся — pre-existing UP035/F401)
- [x] Ручной прогон автоматизирован — `tests/test_cache_reuse.py` (5 тестов: `test_followup_question_reuses_cache`)

> **Итог:** 187 тестов проходят, coverage 81%, все resilience-фичи работают.

---

## Сводка по этапам

| Этап | Название | Часы | Статус |
|---|---|---|---|
| 1 | Registry & Storage | 4 | ✅ Завершён |
| 2 | Извлечение инструментов | 3–5 | ✅ Завершён |
| 3 | Агентный слой (LangGraph) | 5–8 | ✅ Завершён |
| 4 | WebSocket чат | 2–3 | ✅ Завершён |
| 5 | Эмбеддинги и поиск | 2–3 | ✅ Завершён |
| 6 | DuckDB-кэш OHLCV | 2–3 | ✅ Завершён |
| 7 | Полировка | 3–5 | ✅ Завершён |
| **Итого** | | **25–36** | |

**Все 7 этапов завершены.** 187 тестов проходят, coverage 81%. Проект готов к проду.

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

## Этап 2: Извлечение инструментов из пайплайна (3–5 часов)

**Цель:** разобрать `PipelineExecutor.run()` на независимые инструменты с общим контрактом `ToolSpec`.

### 2.1 Контракт инструмента

- [ ] Создать `aqr/tools/__init__.py`
- [ ] Определить `ToolSpec` — датакласс: `name`, `description` (на русском), `input_schema` (JSON Schema), `fn` (Callable)
- [ ] Реализовать `ToolRegistry` — реестр: `register(tool)`, `get(name)`, `list_all()`, `list_for_llm()` (имена + описания для промпта)

### 2.2 Инструменты ядра

- [ ] `plan_research` — обёртка над `ChatPlanner.plan()`: `goal → ResearchPlan`
- [ ] `load_prices` — извлечь `executor._load_data` в `aqr/tools/data.py`: `(tickers, start, end, timeframe) → dict[str, pd.Series]` (+ synthetic fallback)
- [ ] `generate_hypotheses` — обёртка над существующей `generate_hypotheses()`: `(tickers, families, n) → list[HypothesisSpec]`
- [ ] `backtest_one` — извлечь `executor._backtest_one` в `aqr/tools/backtest.py`: `(HypothesisSpec, prices) → BacktestResult`
- [ ] `validate_portfolio` — извлечь `executor._portfolio_pbo` в `aqr/tools/validate.py`: `(list[BacktestResult]) → PBO dict`
- [ ] `extract_insights` — извлечь `executor._extract_insights` в `aqr/tools/insights.py`: `(PipelineResult) → list[str]`
- [ ] `review_insights` — обёртка над существующим `InsightReviewer.review()`: `(result, det_insights) → list[str]`
- [ ] `narrate` — обёртка над существующим `Narrator.narrate()`: `(PipelineResult) → str`

### 2.3 Инструменты хранилища

- [ ] `get_run` — `(run_id) → Run | None` (из RegistryStore)
- [ ] `compare_runs` — `(run_id_a, run_id_b) → dict` (сравнение метрик двух прогонов)
- [ ] `list_runs` — `(session_id, limit) → list[Run]`

### 2.4 Сохранить совместимость

- [ ] Переписать `PipelineExecutor.run()` чтобы он использовал ToolRegistry вместо приватных методов
- [ ] Убедиться что существующий CLI и `/pipeline/*` endpoint'ы продолжают работать
- [ ] Прогнать `pytest tests/ -v` — все 22 теста должны остаться зелёными

### 2.5 Тесты на инструменты

- [ ] `tests/test_tools.py`: каждый инструмент вызывается изолированно с минимальными входными данными
- [ ] `tests/test_tools.py`: `ToolRegistry.list_all()` возвращает все 11 инструментов

**Результат:** каждый шаг пайплайна можно вызвать отдельно. Пайплайн становится композицией инструментов.

---

## Этап 3: Агентный слой на LangGraph (5–8 часов)

**Цель:** заменить фиксированную последовательность шагов на агента, который сам решает какие инструменты вызывать.

### 3.1 LangGraph-граф

- [ ] Добавить `langgraph` в зависимости
- [ ] Создать `aqr/agent/__init__.py`
- [ ] Определить `AgentState` — TypedDict: `messages`, `plan`, `prices`, `hypotheses`, `results`, `narrative`, `session_id`
- [ ] Реализовать узлы графа в `aqr/agent/graph.py`:
  - `plan_node` — вызывает `plan_research`
  - `load_node` — вызывает `load_prices`
  - `generate_node` — вызывает `generate_hypotheses`
  - `backtest_node` — вызывает `backtest_one` × N (цикл)
  - `validate_node` — вызывает `validate_portfolio`
  - `narrate_node` — вызывает `extract_insights` + `review_insights` + `narrate`
  - `router_node` — LLM решает какой узел вызывать следующим (на основе `messages` + состояния)
- [ ] Собрать граф: `plan → load → generate → backtest → validate → narrate → END`
- [ ] Добавить conditional edge: после `narrate` → `router` (ждёт уточнения или завершает)

### 3.2 Контекст сессии

- [ ] Создать `aqr/agent/context.py` — `SessionContext`:
  - `get_recent_runs(session_id, limit=5)` — последние прогоны
  - `get_best_strategy(session_id)` — лучшая гипотеза по DSR
  - `get_untested_combos(session_id)` — «белые пятна» (семейства × тикеры без прогонов)
- [ ] `SessionContext` подмешивается в системный промпт роутера

### 3.3 Тесты на граф

- [ ] `tests/test_agent.py`: граф проходит от `plan` до `narrate` без ошибок
- [ ] `tests/test_agent.py`: роутер завершает диалог после `narrate`
- [ ] `tests/test_agent.py`: роутер принимает уточняющее сообщение и перезапускает backtest

**Результат:** агент может провести полное исследование и ответить на уточняющие вопросы без перезапуска всего пайплайна.

---

## Этап 4: WebSocket-интерфейс чата (2–3 часа)

**Цель:** заменить SSE-ленту на двусторонний WebSocket-диалог с агентом.

### 4.1 WebSocket endpoint

- [ ] Добавить `WebSocket` endpoint в `aqr/main.py`: `WS /chat/{session_id}`
- [ ] Каждое входящее сообщение пользователя → запуск графа агента
- [ ] Промежуточные результаты стримятся как JSON-сообщения (`{"type": "progress", "node": "backtest", "data": {...}}`)
- [ ] Финальный ответ — `{"type": "done", "narrative": "..."}`

### 4.2 Сохранение истории

- [ ] Сохранять историю диалога в `Session` (поле `messages` JSONB или отдельная таблица)
- [ ] При reconnect'е — отдавать историю

### 4.3 Совместимость

- [ ] Старый `GET /pipeline/runs/{id}/stream` (SSE) продолжает работать для обратной совместимости
- [ ] `POST /pipeline/runs` продолжает работать для fire-and-forget прогонов

**Результат:** пользователь может общаться с агентом в реальном времени через WebSocket.

---

## Этап 5: Эмбеддинги и семантический поиск (2–3 часа)

**Цель:** векторизовать гипотезы для поиска похожих и дедупликации.

### 5.1 Генерация эмбеддингов

- [ ] Создать `aqr/registry/embeddings.py`:
  - `embed_hypothesis(hypothesis) → list[float]` — текст из `family + ticker + config_json` → эмбеддинг
  - Модель: `text-embedding-3-small` (OpenAI) или `intfloat/multilingual-e5-large` (локально)
  - Fallback: детерминистический хеш-вектор если нет API-ключа

### 5.2 Поиск похожих

- [ ] `RegistryStore.search_similar(embedding, threshold=0.7, limit=10)` — cosine distance через pgvector
- [ ] `RegistryStore.search_by_text(query, limit=10)` — текст → эмбеддинг → поиск

### 5.3 Дедупликация

- [ ] Перед запуском прогона: `search_similar(embedding, threshold=0.92)` → если найдено, агент предупреждает
- [ ] В `_run_and_persist`: заполнять `hypothesis.embedding` при сохранении

### 5.4 Тесты

- [ ] `tests/test_embeddings.py`: эмбеддинг генерируется без ошибок
- [ ] `tests/test_embeddings.py`: `search_similar` находит гипотезу с одинаковым family+ticker
- [ ] `tests/test_embeddings.py`: `search_similar` не находит гипотезу с разным family+ticker

**Результат:** платформа помнит что уже проверяла и предупреждает о дубликатах.

---

## Этап 6: DuckDB-кэш OHLCV (2–3 часа)

**Цель:** ускорить загрузку данных и снизить нагрузку на MOEX ISS API.

### 6.1 Кэш

- [ ] Создать `aqr/data/ohlcv_cache.py`:
  - `get_cached(ticker, start, end, timeframe) → pd.DataFrame | None`
  - `put_cache(ticker, df)` — upsert в DuckDB
  - Файл: `data/ohlcv_cache.duckdb`, создаётся автоматически при первом использовании

### 6.2 Интеграция с `load_prices`

- [ ] `load_prices` сначала проверяет DuckDB-кэш
- [ ] При промахе — идёт в MOEX ISS API и сохраняет в кэш
- [ ] Fallback на synthetic GBM сохраняется

### 6.3 Тесты

- [ ] `tests/test_ohlcv_cache.py`: put → get возвращает те же данные
- [ ] `tests/test_ohlcv_cache.py`: промах возвращает None

**Результат:** повторные прогоны на тех же тикерах не ходят в MOEX API.

---

## Этап 7: Полировка и production hardening (3–5 часов)

### 7.1 Обработка ошибок

- [ ] Таймауты на MOEX ISS (10 секунд)
- [ ] Retry с exponential backoff (3 попытки)
- [ ] Circuit breaker: после 5 ошибок подряд — только synthetic до рестарта

### 7.2 Мониторинг

- [ ] Структурированное логирование (структура: `{run_id, tool, duration_ms, status, error}`)
- [ ] Health-check эндпоинт проверяет Postgres + MOEX доступность

### 7.3 Документация

- [ ] README.md: архитектура, быстрый старт, примеры запросов к агенту
- [ ] Docstring'и на всех публичных методах инструментов

### 7.4 Финальное тестирование

- [ ] `pytest tests/ -v --cov=aqr` — coverage > 80%
- [ ] `ruff check aqr/ tests/` — чисто
- [ ] Ручной прогон: «проверь momentum на Сбере» → «а что если mean reversion?» → агент переиспользует кэш и не ходит в MOEX повторно

---

## Сводка по этапам

| Этап | Название | Часы | Статус |
|---|---|---|---|
| 1 | Registry & Storage | 4 | ✅ Завершён |
| 2 | Извлечение инструментов | 3–5 | 🔲 Запланирован |
| 3 | Агентный слой (LangGraph) | 5–8 | 🔲 Запланирован |
| 4 | WebSocket чат | 2–3 | 🔲 Запланирован |
| 5 | Эмбеддинги и поиск | 2–3 | 🔲 Запланирован |
| 6 | DuckDB-кэш OHLCV | 2–3 | 🔲 Запланирован |
| 7 | Полировка | 3–5 | 🔲 Запланирован |
| **Итого** | | **21–31** | |

Этапы 2 и 3 — критический путь: без них нет агента. Этапы 4, 5, 6 можно делать параллельно после завершения этапа 3.

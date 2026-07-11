# AGENTS.md

Контракт для LLM-агентов (Claude Code, Codex, Cursor), работающих с этим репозиторием.

## Что это за проект

Трёхслойная платформа автоматизированного квант-исследования (в процессе перехода от монолитного пайплайна):

- **Chat Layer** (запланирован) — верхний слой: пользователь общается с агентом в чате, агент вызывает инструменты, стримит ответ. Никакой прямой связи с пайплайном — только через Tool Registry.
- **Tool Layer** (запланирован) — средний слой: реестр независимых инструментов. Каждый инструмент = одна функция. Агент не знает о пайплайне напрямую.
- **Storage Layer** (частично реализован) — нижний слой: Postgres + pgvector для прогонов и гипотез, Alembic для миграций. DuckDB для OHLCV-кэша запланирован.

Никакого Redis, никакого Kafka, никакой очереди — всё в одном процессе.

### Почему уходим от монолитного пайплайна к инструментам

Плюсы:
- Пользователь сможет итеративно уточнять гипотезы (а не гонять полный пайплайн заново)
- Инструменты можно переиспользовать в разных сценариях
- Каждый инструмент тестируется изолированно
- Агент может сам решать какие инструменты вызвать — не жёсткая последовательность шагов

Риски (приемлемые):
- Дополнительная абстракция Tool Registry
- Качество агента зависит от системного промпта

## Что уже работает

### Pipeline (монолит, текущая точка входа)

- **CLI**: `python -m aqr "цель на русском"` — прогон пайплайна с живым логом
- **HTTP API**: FastAPI + SSE, три endpoint'а под `/pipeline/*`
- **Планировщик**: разбор цели на русском → план (тикеры, семейства, параметры). LLM + regex-ный fallback.
- **Загрузка данных**: MOEX ISS API, fallback на синтетический GBM
- **Гипотезы**: momentum (SMA crossover, z-score), mean reversion, breakout, volatility filter
- **Бэктестинг**: позиция сдвигается на 1 бар (`shift(1)`) — без look-ahead
- **Валидация**: Deflated Sharpe (DSR), Combinatorial Purged CV (CPCV), Probability of Backtest Overfitting (PBO)
- **Инсайты**: детерминистичные наблюдения + LLM-review топ-5
- **Нарратор**: генерация русского отчёта (LLM + шаблонный fallback)
- **LLM-точки**: `ChatPlanner`, `InsightReviewer`, `Narrator`. Все через `litellm`, все с fallback

### Registry (хранилище гипотез)

- **Таблицы** (`sessions` → `runs` → `hypotheses`): PostgreSQL с pgvector (Vector 1536), Alembic-миграция `fcb396c4088d`
- **`RegistryStore`**: асинхронный CRUD — `get_or_create_session`, `create_run`, `update_run_status`, `create_hypothesis`, `list_hypotheses_by_run`, `get_top_valid_hypotheses`, `save_pipeline_result` (batch)
- **`aqr/db.py`**: фабрика async-сессий + FastAPI-зависимость `get_db()`
- **API-интеграция**: `POST /pipeline/runs` сохраняет run в БД; по завершении пишет summary-метрики и топ-5 гипотез

### Тесты

- **22 теста проходят**: 12 валидация, 3 e2e, 7 reviewer
- **2 теста требуют duckdb** (предустановленная проблема в `aqr/data/manifest.py`)

## Что в процессе (ближайшие шаги)

### Шаг 2: извлечь независимые инструменты из пайплайна (3–5 часов)

Сейчас ядро пайплайна — это 250-строчный `PipelineExecutor.run()`. Его нужно разобрать на инструменты с общим контрактом:

```python
@dataclass
class ToolSpec:
    name: str           # "load_prices", "backtest", "validate_portfolio"
    description: str    # русское описание для LLM-агента
    input_schema: dict  # JSON Schema параметров
    fn: Callable        # асинхронная функция
```

Инструменты на извлечение:

| # | Инструмент | Откуда взять | Вход → Выход |
|---|---|---|---|
| 1 | `plan_research` | уже в `planner.py` | `goal: str → ResearchPlan` |
| 2 | `load_prices` | `executor._load_data` | `(tickers, start, end, timeframe) → dict[str, Series]` |
| 3 | `generate_hypotheses` | уже в `hypotheses.py` | `(tickers, families, n) → list[HypothesisSpec]` |
| 4 | `backtest_one` | `executor._backtest_one` | `(HypothesisSpec, prices) → BacktestResult` |
| 5 | `validate_portfolio` | `executor._portfolio_pbo` | `(list[BacktestResult]) → PBO dict` |
| 6 | `extract_insights` | `executor._extract_insights` | `(PipelineResult) → list[str]` |
| 7 | `review_insights` | уже в `reviewer.py` | `(result, det_insights) → list[str]` |
| 8 | `narrate` | уже в `narrator.py` | `(PipelineResult) → str` |

Файлы: новый пакет `aqr/tools/` — `registry.py` (ToolSpec + реестр), отдельные модули под каждый инструмент.

### Шаг 3: агентный слой вместо фиксированного пайплайна (5–8 часов)

Заменить `PipelineExecutor.run()` на агента, который в цикле вызывает инструменты. Подход: LangGraph.

```
пользователь: "проверь momentum на Сбере"
    → агент: plan_research → load_prices → generate_hypotheses → [backtest × N] → validate → narrate

пользователь: "а что если убрать волатильность?"
    → агент: filter(семейство != volatility) → перезапуск backtest → новый отчёт
```

Ключевые компоненты:
- `aqr/tools/registry.py` — ToolRegistry
- `aqr/agent/graph.py` — LangGraph-граф (planner → loader → backtester → validator → narrator + router)
- WebSocket вместо SSE для двустороннего диалога
- `aqr/agent/context.py` — контекст сессии (история диалога, последние прогоны, лучшая стратегия)

## Структура проекта (целевая)

```
aqr/
  agent/           # Chat Layer (запланирован)
    graph.py       # LangGraph-граф
    context.py     # Контекст сессии
  tools/           # Tool Layer (в процессе)
    registry.py    # ToolSpec + ToolRegistry
    plan.py        # plan_research
    data.py        # load_prices
    backtest.py    # backtest_one
    validate.py    # validate_portfolio
    insights.py    # extract_insights
  registry/        # Hypothesis Registry
    store.py       # RegistryStore — CRUD к Postgres + pgvector
    models.py      # SQLAlchemy-модели (Run, Hypothesis, Session)
  pipeline/        # Pipeline (существующий монолит — будет разобран)
    executor.py    # PipelineExecutor.run() — подлежит разбору на tools/
    planner.py     # ChatPlanner — будет tool "plan_research"
    hypotheses.py  # generate_hypotheses — будет tool
    narrator.py    # Narrator — будет tool
    reviewer.py    # InsightReviewer — будет tool
    events.py      # EventBus — сохраняется для SSE-стрима
    api.py         # FastAPI роутер — замена на WebSocket
  validation/      # Валидация (не трогать)
    ...
  data/            # Данные
    moex.py        # MOEX ISS адаптер (не трогать)
  db.py            # Фабрика async-сессий + get_db()
alembic/           # Миграции схемы Postgres
```

## Что НЕ надо трогать без явной причины

- `aqr/validation/` — reference-имплементации из книг López de Prado / Bailey. Формулы уже проверены тестами. Если меняешь — обязательно перепрогони `tests/test_validation.py`.
- `aqr/data/moex.py` — MOEX ISS адаптер. API MOEX — внешнее, менять эндпоинты только по документации https://iss.moex.com.
- `aqr/pipeline/events.py` — контракт `Event`, на нём завязан SSE UI. Ломать поля осторожно.

## Разрешено активно менять

- `aqr/tools/` — создание и регистрация инструментов
- `aqr/agent/` — LangGraph-граф, контекст сессии, WebSocket endpoint
- `aqr/registry/store.py` — методы RegistryStore
- `aqr/registry/models.py` — SQLAlchemy-модели
- `aqr/pipeline/planner.py` — правила разбора русского запроса. Добавить тикеры/семейства/категории — здесь.
- `aqr/pipeline/hypotheses.py` — новые семейства гипотез. Формат: функция `(prices: pd.Series) -> pd.Series` возвращает позицию -1/0/+1.
- `aqr/pipeline/narrator.py` — стиль отчёта.
- `aqr/pipeline/reviewer.py` — system-prompt для LLM-review топ-5 (что искать: concentration risk, слабые данные, несоответствие цели).
- `aqr/pipeline/executor.py` — оркестрация шагов. Если добавляешь новый шаг, эмить события через `_emit()`.

## Инварианты, которые нельзя нарушать

1. **Никакого look-ahead**. В `_backtest_one` позиция сдвигается на 1 бар (`shift(1)`). Не убирать.
2. **Fallback обязателен**. Планировщик и нарратор ДОЛЖНЫ работать без LLM-ключей. Не удаляй `_fallback_plan` / `_fallback_narrate`.
3. **События идут в порядке**: `planning → data → generating → backtesting × N → validating → insight × M → narrating → done`. При ошибке — `error`.
4. **Один процесс, никаких брокеров**. Если нужен фон — `asyncio.create_task`. Не тащить Redis / Celery / RQ.
5. **Валидация — источник истины**. Sharpe без DSR не показывать пользователю как «значимый».

## Как запускать локально

```bash
pip install -e ".[dev]"

# 0. Поднять Postgres + pgvector
docker run -d --name aqr-pg -e POSTGRES_PASSWORD=aqr -p 5432:5432 pgvector/pgvector:pg16
alembic upgrade head

# 1. Прогон CLI
python -m aqr "проверь momentum на голубых фишках"

# 2. HTTP-сервер
uvicorn aqr.main:app --reload --port 8000
curl -s -X POST http://localhost:8000/pipeline/runs \
  -H "Content-Type: application/json" \
  -d '{"goal":"проверь mean reversion на Газпроме"}'

# 3. Тесты
pytest tests/ -v
```

## Как добавить новое семейство гипотез

1. В `aqr/pipeline/hypotheses.py` добавить функцию-сигнал `_my_signal(param1, param2)` возвращающую `(prices) -> positions`.
2. В `_make_one()` добавить ветку `if family == "my_family": ...` с генерацией параметров.
3. В `aqr/pipeline/planner.py`:
   - добавить ключевые слова в `_fallback_plan()` (например, `"мой_паттерн" → "my_family"`)
   - добавить в `PLANNER_SYSTEM` описание для LLM
4. В `tests/test_pipeline_e2e.py` добавить проверку что план с этим ключевым словом даёт правильный family.
5. Прогнать `pytest tests/`.

## Как добавить новый MOEX-инструмент

1. В `aqr/pipeline/planner.py::MOEX_TICKERS` добавить тикер.
2. В `_extract_tickers()` добавить русское название в `aliases` или в категорию (голубые фишки / банки / металлурги).
3. Проверить `python -m aqr "проверь <название>"` находит тикер.

## Проверка перед PR

```bash
pytest tests/ -v                              # все тесты должны быть зелёные
python -m aqr "проверь momentum на Сбере"     # end-to-end проходит
ruff check aqr/ tests/                        # линтер
alembic check                                 # миграции не расходятся с моделями
```

## Что явно вне scope сейчас

- Транзакционные издержки и slippage в бэктесте
- Многопользовательский режим, аутентификация
- Reinforcement learning / auto-ML на гипотезах
- Live-trading, брокерская интеграция
- Источники данных кроме MOEX

Если пользователь просит что-то из этого списка — обсудить план прежде чем реализовывать: это большой шаг.

## Файлы, которые ты почти всегда трогаешь

| Задача | Файлы |
|---|---|
| Новый инструмент | `tools/<tool_name>.py`, `tools/registry.py` |
| LangGraph-граф / агент | `agent/graph.py`, `agent/context.py` |
| Новый метод RegistryStore | `registry/store.py`, `registry/models.py` |
| Новое семейство гипотез | `pipeline/hypotheses.py`, `pipeline/planner.py`, `tests/test_pipeline_e2e.py` |
| Улучшение отчёта | `pipeline/narrator.py` |
| Новый MOEX-тикер / алиас | `pipeline/planner.py` |
| Новый шаг пайплайна | `pipeline/executor.py`, `pipeline/events.py` |
| Новый HTTP/WS endpoint | `pipeline/api.py` или `agent/`, `main.py` |
| Новая валидационная метрика | `validation/` + `tests/test_validation.py` |
| Миграция схемы | `alembic/versions/` |

## Стиль

- Type hints с `from __future__ import annotations`
- Комментарии по-русски там, где помогают понять контекст; docstring по-английски или по-русски, консистентно в модуле
- Никаких emoji в коде
- Модуль не длиннее 400 строк — иначе разбивать

# Tasks: Explore UI — closing gaps with mockups

Проект соответствует макетам на ~70%. Ниже — план по 6 оставшимся gap'ам.

## Gap 1: Full graphs page (`/explore/{id}/graphs`)

**Макет:** `04_graphs_full.html` (559 строк)

**Что нужно:**
- Новый маршрут `GET /explore/{hyp_id}/graphs` → HTML-страница
- 7 графических карточек (каждая — SVG-заглушка, в будущем — live Chart.js/Plotly):
  1. TL;DR summary (3 stat-блока: Sharpe, DSR, PBO с цветовым verdict)
  2. Equity curve (strategy vs buy-and-hold, заполненная область)
  3. Drawdown (underwater curve)
  4. Rolling Sharpe (12M window)
  5. Monthly returns heatmap (2 года × 12 месяцев, цвет от зелёного до красного)
  6. 4-window cross-val mini-charts
  7. Risk/return scatter (все гипотезы)
  8. Distribution of daily returns (с sign-раскраской)
  9. Trade analysis (P/L distribution + hold time)

**Оценка:** ~400 строк HTML+CSS+SVG

---

## Gap 2: Activity page (`/activity`)

**Макет:** `05_activity.html` (212 строк)

**Что нужно:**
- Новый маршрут `GET /activity` → HTML-страница
- Топбар с табами (те же, что в explore: Spreadsheet · Activity · By ticker)
- Фильтр-чипы по типу события (7 типов: created/edit/rerun/approve/reject/comment/AI/status)
- Timeline за последние 7 дней (разбивка по дням)
- Каждое событие: иконка, кто, что, когда
- Авто-статус-изменения помечены `actor=system`
- Pagination / «Show more»

**Оценка:** ~250 строк HTML+CSS+JS

---

## Gap 3: By ticker tabview

**Макет:** таб «By ticker» в топбаре (02 и 05)

**Что нужно:**
- Добавить таб «By ticker» в explore.html + activity.html
- Группировка гипотез по тикеру с коллапсируемыми секциями
- Для каждого тикера: count hypothesis, avg Sharpe, best DSR

**Оценка:** +50 строк JS + 100 строк HTML/CSS в explore.html

---

## Gap 4: Presence indicators

**Макет:** зелёные/жёлтые dots в топбаре + в notebook

**Что нужно:**
- WebSocket-подписка на присутствие (может reuse `/chat` WS или новый endpoint)
- Показывать кто онлайн, кто смотрит какую гипотезу
- В spreadsheet — колонка «presence» или строка в statsbar
- В notebook — индикатор «bob online · viewing this»

**Оценка:** 3-4 дня (требует backend: SSE или WS для push-уведомлений о присутствии)

---

## Gap 5: Pessimistic lock

**Макет:** locked row (#51) в spreadsheet + lock-banner в notebook

**Что нужно:**
- При начале редактирования — блокировка гипотезы на N минут
- Другие пользователи видят locked row (opacity + disabled checkbox)
- В notebook — жёлтый lock-banner с таймером авто-освобождения
- Кнопка «Notify when free»

**Оценка:** 2-3 дня (требует backend: lock в Postgres или Redis + SSE)

---

## Статус на 2026-07-30

| Gap | Статус |
|---|---|
| P0. Full graphs page (`/explore/{id}/graphs`) | ✅ |
| P0. Activity page (`/activity`) | ✅ |
| P1. By ticker tab | ✅ |
| P1. Dynamic data (RegistryStore API + frontend fetch) | ✅ |

---

## Остаётся (P2)

- **Presence indicators** — WS/SSE + кто смотрит какую гипотезу
- **Pessimistic lock** — блокировка строк при редактировании, таймер авто-освобождения

Оба требуют backend SSE-инфраструктуры. Решать отдельно.

## Gap 6: Dynamic data from backend (✅ completed)

4 новых метода в `RegistryStore`, 4 API-эндпоинта, фронтенд-рендер вместо статики.

**Что нужно:**
- Заменить статические sample-data на реальные данные из реестра
- Explore spreadsheet: список гипотез из `RegistryStore`
- Activity: события из `hypothesis_events`
- Stats bar: реальные метрики (win rate, count)
- Notebook: реальная история событий, метрики, статусы

**Оценка:** 3-5 дней (зависит от полноты бэкенд-моделей)

---

## Приоритеты

| Приоритет | Gap | Статус |
|---|---|---|
| P0 | **Full graphs page** | ✅ |
| P0 | **Activity page** | ✅ |
| P1 | **By ticker tab** | ✅ |
| P1 | **Dynamic data** | ✅ — RegistryStore API + endpoints + frontend fetch |
| P2 | **Presence indicators** | ❌ — требуется backend SSE |
| P2 | **Pessimistic lock** | ❌ — требуется backend SSE + Postgres |

# Explore UI — design mockups (v0.4)

Static HTML-черновики будущего интерфейса `/explore` (master-detail
для quant-гипотез). Не живой код — для обсуждения UX, ревью,
onboarding новых членов команды.

## Файлы

| # | Файл | Что показывает |
|---|---|---|
| 01 | [`01_explore_empty.html`](01_explore_empty.html) | Empty state — пустой лист после первого деплоя. Один CTA: `+ New hypothesis`. Без templates и без AI (по решению команды). |
| 02 | [`02_explore_spreadsheet.html`](02_explore_spreadsheet.html) | Leaderboard из 6 гипотез с разными статусами. Bulk-select для CSS-overlay compare. Locked-row (`#51`) показывает pessimistic-lock presence. Auto-screening (`#38`) — системное событие, не пользователь. |
| 03 | [`03_notebook.html`](03_notebook.html) | Drill-down на `#42`. 5 вкладок (Code / Backtest / Equity Curve / History / Discussion). Equity-кривая inline SVG, stability 4/4, audit trail на 5 событий. Lazy AI-кнопки (`🤖 explain / suggest / report`) появляются только на релевантном шаге status-flow. |
| 04 | [`04_activity.html`](04_activity.html) | Глобальный activity feed за 7 дней. 7 типов событий (created/edit/rerun/approve/reject/comment/AI/status). Авто-status-изменения помечены actor=system. Filter-chips по типу события. |
| 05 | [`05_explore_new.html`](05_explore_new.html) | Следующий шаг после клика `+ New hypothesis`. Форма `/explore/new`: identity → params → period → metadata → footer с `Save as draft` и `Save & Run backtest →`. Без AI на пустом листе — вручную. |

## Как открыть

```bash
# Локальный preview через http.server
python3 -m http.server -d docs/mockups/explore 8080
# затем http://localhost:8080/02_explore_spreadsheet.html
```

Или напрямую:
```bash
xdg-open docs/mockups/explore/02_explore_spreadsheet.html
```

## Связь с кодовой базой

Эти макеты — **reference design**, а не шаблоны. Кодовая база этих
экранов появится в ветке `v0.4-explore` (Phase 1 roadmap в AGENTS.md).

Используйте:
- при планировании CSS layout для `/explore`
- как референс для safe-area / spacing
- при обсуждении UX с коллегами

Не используйте:
- для извлечения компонентов в React/whatever
- как единственный источник правды для v0.4 — правила и flows в
  AGENTS.md → раздел "UX-flow (Phase 1 v0.4)"

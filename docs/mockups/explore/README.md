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
| 05 | [`05_explore_new.html`](05_explore_new.html) | Следующий шаг после клика `Generate 3 hypotheses in parallel`. **Batch notebook:** каждая гипотеза в изолированной ячейке — AI делает research (Browser → Analyst → Editor) и формулирует экономическую гипотезу. Ячейки в 3 состояниях: `● researching` (лог шагов с pulse-анимацией) / `✓ ready pending review` (editable поля + rationale) / `○ queued`. Каждая подтверждается отдельно → попадает в spreadsheet как `idea`. |

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

## UX-flow создания гипотез (v0.4 Phase 1)

```
   /explore (empty)
       │
       │ [+ Generate 3 hypotheses in parallel] ← batch-kомпонент: 1-5 ячеек
       ▼
   /explore/new (batch notebook)
       │
       ├─ Cell 1: ● researching (Browser → Analyst → Editor, лог шагов)
       ├─ Cell 2: ✓ ready pending review (editable поля + rationale)
       ├─ Cell 3: ○ queued
       └─ [+ Add another hypothesis]
       │
       │ каждая подтверждается отдельно:
       │   [✓ Confirm & add] → spreadsheet row со статусом `idea`
       │   [🔄 Regenerate]  → перезапуск AI-агентов в этой ячейке
       │   [✗ Reject]       → удаляется без записи в spreadsheet
       ▼
   /explore (spreadsheet · N новых rows)
```

**Изоляция ячеек:** каждый AI-агент (Browser/Analyst/Editor) работает в
отдельной песочнице. Падение в одной ячейке не ломает другие.
Ячейка идёт в статус `failed` и не блокирует batch.

**Параллелизм:** количество параллельных AI-задач ограничено
concurrency-лимитом (default 3). Если выбрать 5 — две встанут в очередь
(visual: queued state с estimated wait).

**После confirm гипотезы** в spreadsheet она появляется со специальной
подсветкой «new» и авто-переходит в `screening` после первого
successful backtest (см. lazy status-flow в AGENTS.md).


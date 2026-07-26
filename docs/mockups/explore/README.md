# Explore UI — design mockups (v0.4)

Static HTML-черновики будущего интерфейса `/explore` (master-detail
для quant-гипотез). Не живой код — для обсуждения UX, ревью,
onboarding новых членов команды.

## Файлы

| # | Файл | Что показывает |
|---|---|---|
| 01 | [`01_explore_empty.html`](01_explore_empty.html) | Empty state — пустой лист после первого деплоя. Number-input для batch count (max 20, default 3). Кнопка «Generate & research in parallel». |
| 02 | [`02_explore_spreadsheet.html`](02_explore_spreadsheet.html) | Leaderboard из 6 гипотез с разными статусами. Каждая строка кликабельна → открывает `03_notebook.html`. Bulk-select для CSS-overlay compare. Locked-row (`#51`) показывает pessimistic-lock presence. Auto-screening (`#38`) — системное событие, не пользователь. |
| 03 | [`03_notebook.html`](03_notebook.html) | **Drill-down на одну стратегию.** Combined view с 3 tabs: 💬 Workspace (status panel для decision support: editable params, metrics, stability, quick actions; live chat-thread где работа + adjustments; sticky composer с /run /edit /approve /help) · 📜 History (timeline) · 📈 Graphs (equity/drawdown/stability). Status: `screening` с активным диалогом. Lock-banner сверху при presence-edit. |
| 04 | [`04_activity.html`](04_activity.html) | Глобальный activity feed за 7 дней. 7 типов событий (created/edit/rerun/approve/reject/comment/AI/status). Авто-status-изменения помечены actor=system. Filter-chips по типу события. |

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

## UX-flow создания и drill-down (v0.4 Phase 1)

```
   /explore (empty, count input + 'Generate & research')
       │
       │  AI создаёт N изолированных ячеек в worktree-like песочницах:
       │  Browser-agent (поиск новостей/фундаментала)
       │  → Analyst-agent (похожие гипотезы через find_duplicates)
       │  → Editor-agent (формулирует economic hypothesis + params)
       ▼
   Research log live-streaming в Notebook tab ячейки
       │
       │  ✓ Confirm в каждой ячейке → spreadsheet row со статусом `idea`
       │  🔄 Regenerate         → перезапуск AI-agent'ов в ячейке
       │  ✗ Reject              → удаляется без записи
       ▼
   /explore (spreadsheet)
       │
       │  клик по любой строке → 03_notebook.html (combined view)
       ▼
   /explore/{id} (3 tabs)
       │
       ├─ 📓 Notebook: research log + editable params + AI rationale
       │           + metrics panel + lazy actions (status-зависимые)
       ├─ 📜 History: timeline всех событий гипотезы
       └─ 📈 Graphs: equity curve SVG + drawdown + 4-window stability
```

**Кликабельная строка:** `02_explore_spreadsheet.html` → клик по `<tr>`
→ `onclick="location='03_notebook.html'"`. Checkbox использует
`onclick="event.stopPropagation()"` чтобы bulk-select не открывал
notebook.

**Изоляция ячеек при batch-создании:** каждый AI-агент работает в
отдельной песочнице. Падение в одной ячейке не ломает другие.
Concurrency-limit default 3, остальные встают в `queued` state.

**После confirm гипотезы** в spreadsheet она появляется со специальной
подсветкой «new» и авто-переходит в `screening` после первого
successful backtest (см. lazy status-flow в AGENTS.md).


# Code Review Report — AQR (pigeon-trade)

**Date:** 2026-07-28
**Scope:** Full codebase (49 source files, ~7,600 LOC; 28 test files, ~6,300 LOC)
**Reviewer:** Automated code review via opencode

---

## Executive Summary

The AQR codebase is well-structured with clean three-layer architecture, comprehensive test coverage (~83%), and thoughtful engineering practices (lazy imports, per-session credentials via ContextVar, background task retention, strict startup validation). The code reads like it was written by experienced engineers.

However, the review found **5 critical bugs**, **12 high-severity issues**, **14 medium-severity issues**, and **8 security concerns** that should be addressed before production deployment.

---

## Critical Bugs

### 1. `aqr/executor/nautilus.py:78` — NautilusTrader metrics computed on index data

```python
spec.fn(pd.Series(strat_ret.index, name=spec.ticker))
```

`_compute_metrics` feeds the **integer index** (position numbers: 0, 1, 2, ...) of the strategy return series into the hypothesis signal function, which expects **price data**. Every metric produced by this path — trade count, turnover, Sharpe — is garbage. This affects all backtest results that flow through `execute_with_slippage`.

**Severity:** Critical — wrong results silently returned as valid.

**Fix:** Replace `pd.Series(strat_ret.index, name=spec.ticker)` with the actual price DataFrame passed to `execute_with_slippage`. The function needs access to the original OHLCV data.

### 2. `aqr/agent/graph.py:341` — LLM routing unreachable in WebSocket mode

```python
if not _has_llm_key():  # checks os.environ only
```

`_has_llm_key()` inspects `os.environ` for LLM API keys. In WebSocket mode, credentials are set per-session via ContextVar (`set_credentials()` in `chat/ws.py`), never written to `os.environ`. This means the LLM-based follow-up routing (`_llm_route`) is **always skipped** for WebSocket users, falling through to deterministic routing. The feature is dead for the primary UI.

**Severity:** Critical — WS users cannot use conversational follow-up features.

**Fix:** Also check `current_credentials()` from ContextVar:
```python
if not _has_llm_key() and not (creds := current_credentials()) or not creds.llm_api_key:
    return "respond_fast"
```

### 3. `aqr/agent/graph.py:423` — State mutation has no effect in LangGraph

```python
state["step"] = ""  # line 423
```

LangGraph nodes return a dict that gets merged into state. Direct mutation of the state dict inside a node function has **no effect** on the graph state — only returned values are applied. After the first pipeline run completes, `step` remains `"done"` forever. Subsequent messages will never restart the pipeline because `_deterministic_route` (line 308) reads `state.get("step", "")` and sees `"done"`.

**Severity:** Critical — pipeline silently stops working after first run.

**Fix:** `route_node` must return `{"step": ""}` as part of its return dict.

### 4. `aqr/chat/ws.py:280-296` — `last_state` undefined when `astream` yields zero events

```python
async for event in agent.astream(initial_state, config=config):
    # process events
    last_state = event  # only assigned inside loop
# line 296: last_state is undefined if loop has 0 iterations
```

If `agent.astream()` yields zero events (which can happen if the graph immediately returns), `last_state` is never assigned, causing `NameError`. This crashes the WebSocket handler.

**Severity:** Critical — unhandled crash in WebSocket handler.

**Fix:** Initialize `last_state = initial_state` before the `async for` loop.

### 5. `aqr/chat/ws.py:270` — WebSocket users never get session context

```python
"session_context_prompt": ""  # hardcoded empty string
```

In `graph.py:run_agent` (line 519-527), `session_context_prompt` is populated by `SessionContext.build_context_prompt()`, which includes recent runs, best strategy, and untested combinations. The WebSocket path hardcodes it to `""`, making the entire context subsystem dead for the primary user interface. Users never see historical context in their LLM prompts.

**Severity:** Critical — key feature entirely absent from primary UI path.

**Fix:** Call `SessionContext.build_context_prompt()` in the WS handler before constructing `initial_state`.

---

## High Severity Issues

### 6. `aqr/pipeline/executor.py:208` — `pd.date_range(freq="B")` misaligns index with T-Invest calendar-day data

T-Invest D1 candles use calendar-day dates. `pd.date_range(freq="B")` generates business-day dates. The resulting Series index is misaligned with actual trading dates, causing incorrect return calculations in `backtest_one` (which uses `pct_change()`).

**Fix:** Use the actual timestamp index from the OHLCV data, or use `freq="D"` and filter to business days via the data, not the index generator.

### 7. `aqr/agents/orchestrator.py:90` — `tasks.schedule()` RuntimeError crashes orchestrator

`schedule()` raises `RuntimeError` when the background task limit (64) is exceeded. The orchestrator calls `tasks.schedule()` without try/except, so this crashes the entire team run.

**Fix:** Wrap in try/except with graceful degradation (log warning, continue without background persistence).

### 8. `aqr/agents/editor.py:26-34` — RuntimeError instead of AgentResult

The Editor agent raises `RuntimeError` on failure instead of returning `AgentResult(ok=False)`. The orchestrator at `orchestrator.py:71` checks `plan_result.ok` but is NOT wrapped in try/except, so this RuntimeError crashes the orchestrator, skipping Browser and all Analyst steps.

**Fix:** Return `AgentResult(ok=False, error=str(exc))` consistently with all other agents.

### 9. `aqr/agents/reviewer.py:86,104,122` / `aqr/agents/writer.py:62,73` — Key mismatch: `verdict` vs `pbo_verdict`

`_compute_pbo` returns a dict with key `"verdict"`, but WriterAgent reads `"pbo_verdict"`. The Writer will always get an empty string for PBO verdict. This propagates to the narrative — users never see PBO results.

**Fix:** Align the key names. Either rename the return key in Reviewer or update Writer to read `"verdict"`.

### 10. `aqr/agents/browser.py:108-119` — `num_families_tested` returns wrong metric

`_count_tested_families` returns `len(recent)` (the number of recent runs), but the variable name and output key claim it counts distinct (ticker, family) pairs. The function never iterates over or deduplicates by pairs.

**Fix:** Iterate over recent hypotheses and count unique `(h.ticker, h.family)` tuples.

### 11. `aqr/agents/writer.py:119-120` — Exception details leaked into LLM/user-visible output

```python
return [f"Ошибка при анализе: {exc}"]
```

Exception messages (potentially containing API keys, connection strings) are directly embedded in insights/LLM prompts. Same pattern at line 152.

**Fix:** Sanitize the exception before including it. Use the same credential-redaction pattern from `reviewer.py` or log the full error and return a generic message.

### 12. `aqr/llm_env.py:36-37` — Entire LLM call serialized under lock

The pattern `async with await acquire_llm_env_lock() as make_env: with make_env(creds): resp = await litellm.acompletion(...)` holds the asyncio.Lock through the entire HTTP request. The docstring claims "блокирует только другие LLM-вызовы" (only blocks other LLM calls), but this means **all parallel LLM calls are fully serialized**.

**Fix:** Use a context manager that sets env vars and releases the lock before the actual API call. Or use litellm's built-in per-call credential mechanism.

### 13. `aqr/chat/web.py:134` — Redirect to WebSocket URL (404 for browsers)

```python
return RedirectResponse(url=f"/chat/{token}", ...)
```

After successful settings submission, the user is redirected to `/chat/{token}` which is a **WebSocket endpoint** (`ws://`), not an HTTP GET endpoint. Browsers receive a 404.

**Fix:** Redirect to `/chat?token={token}` or `/chat` with a query parameter.

### 14. `aqr/chat/web.py:43-59` — Rate-limit dict never expires (memory leak)

`_rate_buckets` stores per-IP entries but never removes old entries. On a long-running server, this grows unboundedly per unique IP.

**Fix:** Periodically prune stale entries (e.g., remove entries with `last_refill` older than the bucket window). Use a background task or prune on each call.

### 15. `aqr/registry/store.py:67` — `list_chat_history` returns oldest messages, not latest

```python
select(ChatMessage).order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).limit(limit)
```

ASC order + `limit(N)` returns the N **oldest** messages. The caller (`list_chat_history`) presumably wants the most recent messages for display.

**Fix:** Use `order_by(desc())` and reverse after fetch, or use a subquery.

### 16. `aqr/registry/models.py:109` — Dual cascade (ORM + DB-level) on FK relationships

`Run.hypotheses` uses `cascade="all, delete-orphan"` (ORM-level) while the FK has `ondelete="CASCADE"` (DB-level, documented at file header line 4-7). When Postgres cascades a delete, the ORM may attempt a second delete on already-removed rows, producing StaleDataError warnings.

**Fix:** Remove ORM cascade and rely on DB-level cascade only (matching the file header rationale).

### 17. `aqr/pipeline/executor.py:154` / `aqr/agents/analyst.py:211` — Backtest results from analyst path bypass validation

In the Analyst agent, `_deep_backtest` calls `backtest_one` but the results are returned to the orchestrator without running `validate_portfolio`. The orchestrator then passes them to Reviewer, which computes PBO over potentially unvalidated data.

---

## Medium Severity Issues

### 18. `aqr/agent/context.py:91` — `h.dsr and ...` treats DSR=0.0 as missing

```python
if h.dsr and h.dsr > best_dsr:
```

DSR can legitimately be `0.0`, which is falsy in Python. A strategy with DSR=0.0 is silently skipped.

**Fix:** `if h.dsr is not None and h.dsr > best_dsr`

### 19. `aqr/agent/graph.py:386-397` — Silent exception swallowing in `_llm_route`

The entire LLM routing is wrapped in `except Exception: pass` with no logging. Misconfigured LLM keys, bad responses, or rate limits are completely invisible.

**Fix:** Add `logger.exception("LLM routing failed, falling back to deterministic")`

### 20. `aqr/executor/nautilus.py:154` — Double computation of metrics in native path

`execute_with_slippage` computes `strat_ret` (line 133), then calls `_compute_metrics` (line 154), which REAPPLIES `spec.fn` on the wrong data. This results in double-computation with inconsistent input data.

**Fix:** Either pass pre-computed returns to `_compute_metrics` or restructure so metrics are computed once from correct data.

### 21. `aqr/executor/nautilus.py:57,88-89` — CPCV metrics always zero

`BacktestResult` dataclass fields `cpcv_mean_sharpe` and `cpcv_std_sharpe` are always initialized to `0.0` in the nautilus executor. Downstream code uses these for ranking but always gets zeros.

### 22. `aqr/validation/deflated_sharpe.py:130` — Float precision produces infinite E[max SR]

```python
stats.norm.ppf(1.0 - 1.0 / (n * np.e))
```

For moderately large `n` (`n_trials > ~3`), `1.0 / (n * np.e)` underflows float precision to `0.0`, making `ppf(1.0)` → `inf`. This produces an infinite expected maximum Sharpe ratio and zero DSR for all results with more than a few trials.

**Fix:** Use `ppf(1.0 - epsilon)` with a small epsilon (e.g., `1e-12`) or use the asymptotic approximation from Bailey & Lopez de Prado.

### 23. `aqr/validation/cpcv.py:24-49` — `_get_test_ranges` is dead code

The function builds combinatorial test paths but `CombinatorialPurgedCV.split` recomputes the fold logic independently. Never called in the codebase.

**Fix:** Remove dead code.

### 24. `aqr/tools/storage.py:21,54-55` — Unhandled `ValueError` on invalid UUID parse

`uuid.UUID(run_id)` has no try/except. Malformed UUIDs from user input raise unhandled exceptions.

**Fix:** Catch `ValueError` and return a descriptive error.

### 25. `aqr/pipeline/planner.py:93` — `json.loads` on potentially empty/garbled LLM response

```python
json.loads(resp.choices[0].message.content)
```

No check for empty `resp.choices`, no handling of markdown code fences (`` ```json ... ``` ``), and no try/except around `json.loads`. Non-JSON LLM responses crash the pipeline.

**Fix:** Add `resp.choices` check, strip markdown fences, and handle `JSONDecodeError` with a retry or clear error.

### 26. `aqr/pipeline/narrator.py:67` / `aqr/pipeline/reviewer.py:105-106` — Crash on `None` content

LLM responses can have `content=None` (some models use `tool_calls` instead, or return empty). Both files access `.content.strip()` without None check.

**Fix:** Check `if content is None: raise RuntimeError("LLM returned empty response")`

### 27. `aqr/agents/analyst.py:74` — `top_n=0` interpreted as "all"

```python
specs[:top_n] if top_n else specs
```

User passing `top_n=0` expects 0 results but gets all.

**Fix:** `specs[:top_n] if top_n is not None else specs`

### 28. `aqr/agents/analyst.py:127-128` — Silent fallback violates strict-mode invariant 2

`_build_specs` silently catches tool errors and falls back to `_default_specs`. Per AGENTS.md invariant 2, errors should raise, not fall back silently.

### 29. `aqr/startup.py:86-94` — `UnboundLocalError` in retry loop

If the engine creation itself fails (e.g., bad URL), `engine` is undefined in the `finally` block.

**Fix:** Initialize `engine = None` before the try block.

### 30. `aqr/data/tinvest.py:48-51` — `sys.modules` monkey-patch leaks from tests to production

`_get_tinvest` checks `sys.modules.get("t_tech.invest")`. If a test sets a mock in `sys.modules` and doesn't remove it, the production path uses the fake module.

**Fix:** Use a dedicated test-only flag or import the module directly instead of checking `sys.modules`.

### 31. `aqr/registry/embeddings.py:45` — Obscure error when credentials missing

`_api_key_from_context` returns `creds.openai_api_key` without checking if the attribute exists. If credential structure changes, the error is `AttributeError` instead of a clear `RuntimeError("OpenAI API key not configured")`.

---

## Security Issues

### 32. `aqr/chat/web.py:64-67` — Trusts `X-Forwarded-For` without validation

If no reverse proxy is configured, an attacker can spoof the header to bypass rate limiting.

**Fix:** Validate that request originates from a trusted proxy IP before trusting `X-Forwarded-For`.

### 33. `aqr/main.py:62-63` — Wildcard CORS as default

```python
allow_origins=["*"], allow_headers=["*"]
```

When `AQR_ALLOWED_ORIGINS` is not set (easy to miss in production), CORS is wide open.

**Fix:** Default to localhost-only, require explicit configuration for wildcard in production.

### 34. `aqr/auth.py:135-138` — `verify_token_async` fail-open on DB error

When the database is down, the function falls back to HMAC-only validation. A session that was deleted after the DB went down would still be authorized.

**Fix:** Fail closed — return `(None, None)` when DB is unavailable in strict mode.

### 35. `aqr/mcp/server.py:72` — Exception messages exposed without credential redaction

```python
error_response(MCPError(code=-32603, message=str(exc)), req_id)
```

Any handler exception's string representation is directly exposed in the JSON-RPC error. TInvestAdapter or OpenAI errors could leak API keys/tokens.

**Fix:** Redact sensitive patterns from exception messages before exposing.

### 36. `aqr/pipeline/api.py:37-44` — No control-character filter on user input

`_strip_and_check` only strips whitespace. Control characters (`\x00`, `\r\n`) in `goal` input can cause log-injection or JSON-serialization issues.

**Fix:** Add control character filtering or use a regex validator.

### 37. `aqr/logging_config.py:34-40` — Incomplete sensitive-key list

Missing: `authorization`, `x-api-key`, `private_key`, `secret_key`, `connection_string`, `DATABASE_URL`.

**Fix:** Expand `_SENSITIVE_KEYS` to cover all common credential patterns.

### 38. `aqr/agents/reviewer.py:116-121` — Credential redaction regex is incomplete

The regex won't match JSON with escaped quotes, URL query parameters, or values containing spaces. False sense of security.

**Fix:** Use a more robust regex or a dedicated sensitive-value redaction library.

### 39. `aqr/crypto.py:38-45,48-61` — Fernet instance not cached

HKDF derivation + `Fernet()` constructor called on every encrypt/decrypt. Should be cached at module level.

---

## Performance Issues

### 40. `aqr/agent/graph.py:193-207` — Sequential backtests

`backtest_node` runs hypotheses one-by-one with only `asyncio.sleep(0)`. For 50 hypotheses, this is ~50x slower than `asyncio.gather` with a semaphore.

### 41. `aqr/agent/context.py:94` — O(n^2) goal lookup

```python
next((r.goal for r in runs if r.id == run), "")
```

Build a `{r.id: r.goal}` dict once instead of scanning the list per hypothesis.

### 42. `aqr/data/ohlcv_cache.py:165` — `df.iterrows()` for cache insertion

Row-by-row iteration is slow for intraday data (100K+ minute bars). Use vectorized `executemany` from numpy.

### 43. `aqr/validation/pbo.py:75` — Unbounded combination explosion

```python
list(combinations(all_indices, S // 2))
```

`C(n, n/2)` grows combinatorially. For `n_partitions=20`, `C(20,10)=184,756`. For 30 it's 155M. No guard.

**Fix:** Cap at a reasonable maximum with a warning.

### 44. `aqr/screener/vectorbt.py:77` — TInvestAdapter re-instantiated per call

Creates a new gRPC channel for each `screen_momentum` call. Cache the adapter instance.

### 45. `aqr/main.py:89-101` — `/health/ready` re-runs full validation on every probe

Each kubelet probe triggers Docker commands, Postgres health checks, and env validation. Use a cached flag set once in lifespan.

---

## Code Quality & Maintainability Issues

### 46. `aqr/tools/register.py:35` — Hardcoded tool count `>= 13`

Breaks when new tools are added. Use a boolean `_initialized` flag or compute expected count dynamically.

### 47. `aqr/registry/models.py:131` — `Vector(1536)` hardcoded

Tied to `text-embedding-3-small`. Changing models breaks all inserts with dimension mismatch.

**Fix:** Make dimension configurable or derive from the embedder.

### 48. `aqr/tools/core.py:128` — `contextlib.suppress(Exception)` swallows cache failures

Disk-full and permission errors are silently dropped. Violates strict-mode invariant.

### 49. `aqr/tools/core.py:258-265` — DSR thresholds managed in two places

`deflated_sharpe.py` defines `0.95` for significant, `tools/core.py` defines `0.80` for borderline. No shared constants — threshold divergence over time is likely.

### 50. `aqr/tools/storage.py:135-148` — Threshold filtering in Python instead of SQL

`search_similar_hypotheses` fetches `limit*2` rows then filters in Python. If all are below threshold, result is empty even though more matching rows exist.

### 51. `aqr/agents/base.py:44-53` — Dead `credentials` property

Never used by any agent subclass.

### 52. `aqr/pipeline/narrator.py:30` / `aqr/pipeline/reviewer.py:42` — Dead `self.model` constructor parameter

`creds.llm_model` is used instead. The constructor parameter is dead code.

### 53. `aqr/pipeline/planner.py` / `aqr/agent/context.py` — Hardcoded ticker lists

`PLANNER_SYSTEM` prompt and `SessionContext.all_tickers` have hardcoded Russian ticker lists. Same data in two places — will diverge.

### 54. `aqr/agents/analyst.py:179-211` — Missing CPCV parameter threading

`_deep_backtest` doesn't pass `cpcv_splits`, `cpcv_test_splits`, or `embargo_pct` to `backtest_one`. Per AGENTS.md gotcha, these must be threaded explicitly.

### 55. `aqr/executor/nautilus.py:19` / `aqr/agents/analyst.py:19` — Duplicate `MIN_PRICES = 126`

Same constant defined in two places independently.

### 56. `aqr/agents/browser.py:102` — Redundant exception catch

`except (ValueError, Exception)` — ValueError is a subclass of Exception.

### 57. `aqr/agents/browser.py:102-119` — Catch-all exception swallowing

Multiple `except Exception: return []` or `return {}` silently hide DB, network, and credential errors.

### 58. `aqr/api/routes.py:87` — JSON-RPC `id` not echoed

`post_mcp_rpc` doesn't accept/pass `req_id`, so all responses have `id: null`, breaking JSON-RPC 2.0 client correlation.

### 59. `aqr/pipeline/executor.py:194-213` — Per-ticker SSE events fire pre-load

"Загружаю" (loading) events are emitted for all tickers BEFORE `load_prices` is called, then all "OK" events appear at once. Misleading progress UX.

### 60. `aqr/executor/nautilus.py:137-138, 157-176` — `_require_nautilus()` return value unused

The module reference is checked but the placeholder doesn't use nautilus_trader at all.

### 61. `aqr/types.py:18-22` — `TYPE_CHECKING`-only imports used in runtime dataclass

`HypothesisSpec` and `ResearchPlan` are imported only for type checking but `BacktestResult` includes them as field types. Runtime `isinstance()` checks would fail.

### 62. `aqr/chat/ws.py:308-314` — Chat history save failure is silent

`_save_history` silently catches exceptions (line 62-63). If the DB write fails, the user sees "done" but messages are lost permanently with no indication.

### 63. `README.md` — Out of sync with AGENTS.md

README references deleted files (`cli.py`) and removed features (fallback planner, synthetic data, CLI mode). AGENTS.md is the source of truth but the README misleads new readers.

### 64. `build/` and `UNKNOWN.egg-info/` — Stale build artifacts

The `build/lib/aqr/` directory contains old code versions (including deleted modules like `cli.py`, `__main__.py`, `data/moex.py`). These should be cleaned up.

---

## Summary by File

| File | Critical | High | Medium | Low/Quality |
|---|---|---|---|---|
| `aqr/agent/graph.py` | 2 (LLM routing, state mutation) | 0 | 0 | 3 |
| `aqr/chat/ws.py` | 2 (last_state, session context) | 0 | 0 | 2 |
| `aqr/executor/nautilus.py` | 1 (index data) | 0 | 0 | 2 |
| `aqr/chat/web.py` | 0 | 2 (redirect, memory leak) | 0 | 1 |
| `aqr/agents/editor.py` | 0 | 1 | 0 | 0 |
| `aqr/agents/reviewer.py` | 0 | 1 | 0 | 0 |
| `aqr/agents/browser.py` | 0 | 1 | 0 | 0 |
| `aqr/agents/writer.py` | 0 | 1 | 0 | 0 |
| `aqr/agents/orchestrator.py` | 0 | 1 | 0 | 0 |
| `aqr/llm_env.py` | 0 | 1 | 0 | 0 |
| `aqr/registry/store.py` | 0 | 1 | 0 | 0 |
| `aqr/registry/models.py` | 0 | 1 | 0 | 0 |
| `aqr/pipeline/executor.py` | 0 | 0 | 1 | 0 |
| Others | 0 | 2 | 13 | 18 |
| **Total** | **5** | **12** | **14** | **26** |

---

## Recommendations

### Immediate (before production)

1. Fix all 5 critical bugs (nautilus metrics, LLM routing, state mutation, last_state, session context).
2. Fix high-severity data correctness issues (#7-9) in the agents layer.
3. Fix the WebSocket URL redirect (#13) and rate-limit leak (#14).

### Short-term (within a sprint)

4. Address data alignment (#6) in the executor.
5. Fix `verdict`/`pbo_verdict` key mismatch (#9).
6. Fix the wrong `num_families_tested` metric (#10).
7. Fix the exception leak in WriterAgent (#11).
8. Fix LLM serialization bottleneck (#12).
9. Fix chat history ordering (#15).
10. Fix the dual cascade issue (#16).
11. Fix the float precision issue in DSR (#22).

### Medium-term (technical debt)

12. Clean up stale `build/` and `UNKNOWN.egg-info/` directories.
13. Remove dead code (`_get_test_ranges` in cpcv.py, `credentials` property in base.py, dead constructor params).
14. Add missing timeout handling for LLM and embedding API calls.
15. Fix `sys.modules` monkey-patch leak in tinvest.py (#30).
16. Implement WebSocket slash-commands for v0.4 features.
17. Fix CORS default to localhost-only.
18. Implement the deferred `ivfflat` index on `Hypothesis.embedding`.
19. Update README to match current architecture.

### Long-term

20. Implement per-ticker PBO (fix the cross-ticker limitation).
21. Add browser tests for Web UI.
22. Implement NautilusTrader `BacktestEngine` integration (replace the placeholder).
23. Add proper backpressure/rate-limiting for T-Invest API calls.

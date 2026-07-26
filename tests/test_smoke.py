"""Smoke-тесты: CLI, entry points, edge cases.

Эти тесты поднимают coverage для модулей, которые сложно протестировать
через прямой import (CLI использует sys.exit, asyncio.run и т.д.).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ── CLI smoke (subprocess) ───────────────────────────────────────

class TestCLI:
    def test_cli_help(self):
        """`python -m aqr --help` → exit 0, help в stdout."""
        result = subprocess.run(
            [sys.executable, "-m", "aqr", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30,
        )
        assert result.returncode == 0
        assert "goal" in result.stdout.lower()

    def test_cli_runs_simple_goal(self):
        """`python -m aqr "проверь momentum на Сбере"` → exit 0, есть нарратив."""
        result = subprocess.run(
            [sys.executable, "-m", "aqr", "-q", "проверь momentum на Сбере"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
            timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # В stdout должно быть "Топ-5" и нарратив
        assert "Топ-5" in result.stdout or "Sharpe" in result.stdout or "проверено" in result.stdout.lower()

    def test_cli_json_output(self):
        """`python -m aqr --json ...` → валидный JSON в stdout."""
        result = subprocess.run(
            [sys.executable, "-m", "aqr", "--json", "проверь momentum на Сбере"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
            timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Должен быть парсимый JSON
        import json as json_mod
        try:
            data = json_mod.loads(result.stdout)
            assert "plan" in data or "narrative" in data or "top" in data
        except json_mod.JSONDecodeError:
            pytest.fail(f"Not valid JSON: {result.stdout[:200]}")

    def test_cli_quiet_suppresses_progress(self):
        """`-q` → в stdout нет progress-строк [PLAN] [DATA]."""
        result = subprocess.run(
            [sys.executable, "-m", "aqr", "-q", "проверь momentum на Сбере"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
            timeout=60,
        )
        assert "[PLAN]" not in result.stdout
        assert "[DATA]" not in result.stdout


# ── main module entry point ─────────────────────────────────────

class TestMainModule:
    def test_python_dash_m_aqr_invokes_cli(self):
        """`python -m aqr --help` работает (через __main__.py)."""
        result = subprocess.run(
            [sys.executable, "-m", "aqr", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30,
        )
        assert result.returncode == 0
        assert "goal" in result.stdout.lower()


# ── Event format helper ──────────────────────────────────────────

class TestEventFormat:
    def test_fmt_event_includes_icon_and_stage(self):
        from aqr.cli import _fmt_event
        from aqr.pipeline.events import Event

        ev = Event(run_id="r", kind="planning", stage="plan", message="started")
        formatted = _fmt_event(ev)
        assert "[PLAN]" in formatted
        assert "plan" in formatted
        assert "started" in formatted

    def test_fmt_event_unknown_kind(self):
        from aqr.cli import _fmt_event
        from aqr.pipeline.events import Event

        ev = Event(run_id="r", kind="custom_kind", stage="custom", message="x")
        formatted = _fmt_event(ev)
        # 4 пробела как иконка по умолчанию — "[    ] custom — x"
        assert "[    ]" in formatted
        assert "custom" in formatted


# ── _run с моками (покрытие веток asyncio.wait_for) ─────────────

class TestRunAsync:
    @pytest.mark.asyncio
    async def test_run_with_quiet_no_print_progress(self, capsys):
        """quiet=True → print_events() печатает nothing."""
        from aqr.cli import _run

        # Мокаем PipelineExecutor.run чтобы не делать реальный прогон
        from aqr.pipeline import executor as exec_mod

        async def fake_run(self, run_id, plan):
            from aqr.pipeline.executor import PipelineResult
            return PipelineResult(
                run_id=run_id,
                plan=plan,
                n_hypotheses_tested=0,
                n_survived_dsr=0,
                portfolio_pbo=0.0,
                portfolio_pbo_verdict="insufficient",
                top=[],
                elapsed_seconds=0.1,
                narrative="test narrative",
            )

        original = exec_mod.PipelineExecutor.run
        exec_mod.PipelineExecutor.run = fake_run
        try:
            exit_code = await _run("проверь momentum на Сбере", as_json=False, quiet=True)
            assert exit_code == 0
            captured = capsys.readouterr()
            # При quiet=True не должно быть [PLAN] и пр.
            assert "[PLAN]" not in captured.out
            # Должны быть РЕЗУЛЬТАТ и НАРРАТИВ
            assert "РЕЗУЛЬТАТ" in captured.out
            assert "НАРРАТИВ" in captured.out
            assert "test narrative" in captured.out
        finally:
            exec_mod.PipelineExecutor.run = original

    @pytest.mark.asyncio
    async def test_run_with_json_output(self, capsys):
        """as_json=True → валидный JSON в stdout."""
        from aqr.cli import _run
        from aqr.pipeline import executor as exec_mod

        async def fake_run(self, run_id, plan):
            from aqr.pipeline.executor import PipelineResult
            return PipelineResult(
                run_id=run_id, plan=plan,
                n_hypotheses_tested=5, n_survived_dsr=2,
                portfolio_pbo=0.3, portfolio_pbo_verdict="ok",
                top=[], elapsed_seconds=1.0, narrative="narr",
            )

        original = exec_mod.PipelineExecutor.run
        exec_mod.PipelineExecutor.run = fake_run
        try:
            exit_code = await _run("проверь momentum", as_json=True, quiet=True)
            assert exit_code == 0
            captured = capsys.readouterr()
            import json as json_mod
            data = json_mod.loads(captured.out)
            assert data["n_hypotheses_tested"] == 5
            assert data["narrative"] == "narr"
        finally:
            exec_mod.PipelineExecutor.run = original

    @pytest.mark.asyncio
    async def test_run_verbose_prints_progress_and_top(self, capsys):
        """verbose режим (quiet=False, json=False) → прогресс и топ-5 печатаются."""
        from aqr.cli import _run
        from aqr.pipeline import executor as exec_mod
        from aqr.pipeline.executor import BacktestResult, PipelineResult
        from aqr.pipeline.hypotheses import HypothesisSpec

        spec = HypothesisSpec(
            name="SMA10/50", family="momentum", ticker="SBER",
            params={"fast": 10, "slow": 50}, fn=lambda x: x,
        )
        top_result = BacktestResult(
            hypothesis=spec, sharpe=1.2, dsr=0.95, dsr_verdict="significant",
            cpcv_mean_sharpe=0.7, cpcv_std_sharpe=0.1, max_drawdown=-0.1,
            n_trades=10, daily_returns=[],
        )

        async def fake_run(self, run_id, plan):
            return PipelineResult(
                run_id=run_id, plan=plan,
                n_hypotheses_tested=5, n_survived_dsr=2,
                portfolio_pbo=0.3, portfolio_pbo_verdict="ok",
                top=[top_result], elapsed_seconds=1.0, narrative="narr",
            )

        original = exec_mod.PipelineExecutor.run
        exec_mod.PipelineExecutor.run = fake_run
        try:
            exit_code = await _run("проверь momentum", as_json=False, quiet=False)
            assert exit_code == 0
            captured = capsys.readouterr()
            # Verbose: печатает "Цель:" и "План:" в начале
            assert "Цель:" in captured.out
            assert "План:" in captured.out
            # Топ-5 секция
            assert "Топ-5 по DSR" in captured.out
            assert "SMA10/50" in captured.out
        finally:
            exec_mod.PipelineExecutor.run = original


class TestMainEntry:
    def test_main_function_uses_argparse(self, monkeypatch):
        """main() парсит argv и запускает _run."""
        from aqr.cli import main

        calls = {"goal": None, "json": None, "quiet": None}

        async def fake_run(goal, as_json, quiet):
            calls["goal"] = goal
            calls["json"] = as_json
            calls["quiet"] = quiet
            return 0

        # Подменяем sys.argv
        monkeypatch.setattr("sys.argv", ["aqr", "test goal", "--json", "-q"])

        # Подменяем _run
        monkeypatch.setattr("aqr.cli._run", fake_run)
        # Подменяем sys.exit чтобы не выходить из теста
        monkeypatch.setattr("aqr.cli.sys.exit", lambda code: None)

        main()
        assert calls["goal"] == "test goal"
        assert calls["json"] is True
        assert calls["quiet"] is True

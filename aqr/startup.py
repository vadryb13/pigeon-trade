"""Runtime validation at FastAPI startup.

Вызывается в `lifespan` ДО `yield`. Бросает RuntimeError со списком
недостающих env или упавших проверок — FastAPI не стартует пока всё не OK.

.env файл загружается автоматически если `python-dotenv` установлен.

Обязательные переменные окружения:
- DATABASE_URL: postgres+asyncpg://...
- AQR_SESSION_SECRET: >=32 символов (мастер-ключ для HMAC WS-токенов
  и Fernet-шифрования per-session credentials)

Опционально (auto-provision):
- Если Docker доступен — `docker compose -f aqr-compose.yml up -d postgres`
  поднимает Postgres-контейнер идемпотентно. Если Docker недоступен —
  БД должна быть уже запущена (DATABASE_URL ведёт на существующий инстанс).

Concurrency: docker-команды запускаются через `asyncio.to_thread` —
`subprocess.run` блокирует event loop на 60 секунд (B7).
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine

# Автозагрузка .env
try:
    from dotenv import load_dotenv

    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=True)
except ImportError:
    pass

_COMPOSE_PATH = Path(__file__).parent.parent / "aqr-compose.yml"
_PG_READY_TIMEOUT = 30.0
_PG_READY_INTERVAL = 1.0
_DOCKER_TIMEOUT = 5.0
_COMPOSE_TIMEOUT = 60.0


def _check_env(errors: list[str]) -> None:
    """Проверяет обязательные env-переменные."""
    if not os.getenv("DATABASE_URL"):
        errors.append("DATABASE_URL is required")

    secret = os.getenv("AQR_SESSION_SECRET")
    if not secret:
        errors.append("AQR_SESSION_SECRET is required")
    elif len(secret) < 32:
        errors.append(f"AQR_SESSION_SECRET must be ≥32 chars (got {len(secret)})")


def _docker_available() -> bool:
    """True если Docker daemon отвечает. Тихо проглатывает ошибки."""
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=_DOCKER_TIMEOUT,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _compose_up_postgres() -> tuple[bool, str]:
    """Поднимает postgres-сервис. Возвращает (ok, error_msg)."""
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(_COMPOSE_PATH), "up", "-d", "postgres"],
            check=True,
            capture_output=True,
            timeout=_COMPOSE_TIMEOUT,
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        return False, f"docker compose up failed: {stderr[:200]}"
    except subprocess.TimeoutExpired:
        return False, f"docker compose up timed out after {_COMPOSE_TIMEOUT}s"
    except FileNotFoundError:
        return False, "docker compose CLI not found"


async def _wait_for_pg(db_url: str) -> tuple[bool, str]:
    """SELECT 1 с retry до PG_READY_TIMEOUT."""
    last_err: Exception | None = None
    deadline = asyncio.get_event_loop().time() + _PG_READY_TIMEOUT
    while asyncio.get_event_loop().time() < deadline:
        engine = None
        try:
            engine = create_async_engine(db_url)
            async with engine.connect() as conn:
                await conn.execute(sqlalchemy.text("SELECT 1"))
            return True, ""
        except Exception as e:
            last_err = e
        finally:
            if engine is not None:
                await engine.dispose()
        await asyncio.sleep(_PG_READY_INTERVAL)
    return False, f"Postgres not ready after {_PG_READY_TIMEOUT}s: {type(last_err).__name__}: {last_err}"


async def validate_runtime() -> dict[str, str]:
    """Проверяет обязательные env и доступность Postgres.

    Returns:
        {"status": "ready", "postgres": "ok|auto-provisioned|already-up"}

    Raises:
        RuntimeError: со списком всех проблем (env, docker, pg).
    """
    errors: list[str] = []
    _check_env(errors)

    if not errors:
        db_url = os.environ["DATABASE_URL"]
        # Пробуем подключиться к БД сразу. Если не получается — docker compose.
        pg_ok, pg_err = await _wait_for_pg(db_url)
        if not pg_ok:
            if await asyncio.to_thread(_docker_available):
                ok, err = await asyncio.to_thread(_compose_up_postgres)
                if not ok:
                    errors.append(
                        f"Postgres unreachable and docker compose failed: {err}"
                    )
                else:
                    pg_ok, pg_err = await _wait_for_pg(db_url)
                    if not pg_ok:
                        errors.append(pg_err)
            else:
                errors.append(pg_err)

    if errors:
        raise RuntimeError(
            "validate_runtime() failed:\n - " + "\n - ".join(errors)
        )

    return {"status": "ready", "postgres": "ok"}

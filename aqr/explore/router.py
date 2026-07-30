"""Explore UI router — serves explore, notebook, graphs, and activity HTML pages."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

_templates = Path(__file__).resolve().parent / "templates"

router = APIRouter()


def _read(name: str) -> str:
    return (_templates / name).read_text(encoding="utf-8")


@router.get("/explore", response_class=HTMLResponse)
async def explore_page() -> str:
    return _read("explore.html")


@router.get("/explore/{hyp_id}/graphs", response_class=HTMLResponse)
async def graphs_page(hyp_id: str) -> str:
    html = _read("graphs_full.html")
    return html.replace("{{HYP_ID}}", hyp_id)


@router.get("/explore/{hyp_id}", response_class=HTMLResponse)
async def notebook_page(hyp_id: str) -> str:
    html = _read("notebook.html")
    return html.replace("{{HYP_ID}}", hyp_id)


@router.get("/activity", response_class=HTMLResponse)
async def activity_page() -> str:
    return _read("activity.html")

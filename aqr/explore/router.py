"""Explore UI router — serves the explore and notebook HTML pages."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

_EXPLORE_TEMPLATE = Path(__file__).resolve().parent / "templates" / "explore.html"
_NOTEBOOK_TEMPLATE = Path(__file__).resolve().parent / "templates" / "notebook.html"

router = APIRouter()


@router.get("/explore", response_class=HTMLResponse)
async def explore_page() -> str:
    return _EXPLORE_TEMPLATE.read_text(encoding="utf-8")


@router.get("/explore/{hyp_id}", response_class=HTMLResponse)
async def notebook_page(hyp_id: str) -> str:
    html = _NOTEBOOK_TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("{{HYP_ID}}", hyp_id)
    return html

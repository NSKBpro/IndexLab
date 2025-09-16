# app/ui/routes_eval_pages.py
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

ui_router = APIRouter(tags=["ui"])

# This template directory must match your project layout.
templates = Jinja2Templates(directory="app/ui/templates")


@ui_router.get("/eval", response_class=HTMLResponse)
def eval_page(request: Request):
    return templates.TemplateResponse("eval.html", {"request": request})


@ui_router.get("/eval-compare", response_class=HTMLResponse)
def eval_compare_page(request: Request):
    return templates.TemplateResponse("eval-compare.html", {"request": request})

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .core.logging import setup_logging
from .models.db import init_db
from .api import eval_api as eval_api, files, answerless_search as search, health, sse, config, analytics
from .api import sources as sources_api
from .api import chunk_preview as chunk_preview_api
# main.py
from .api import indexes as indexes_api
from .api import versions as versions_api
from .api import download as download_api

setup_logging()
init_db()

app = FastAPI(title="VectorDash")

app.include_router(health.router,   prefix="/api")
app.include_router(files.router,    prefix="/api")
app.include_router(search.router,   prefix="/api")
app.include_router(sse.router,      prefix="/api")
app.include_router(config.router,   prefix="/api")
app.include_router(eval_api.router, prefix="/api")
app.include_router(analytics.router,prefix="/api")
app.include_router(sources_api.router, prefix="/api")
app.include_router(chunk_preview_api.router, prefix="/api")
app.include_router(indexes_api.router, prefix="/api")
app.include_router(versions_api.router, prefix="/api")
app.include_router(download_api.router, prefix="/api")

templates = Jinja2Templates(directory="app/ui/templates")
app.mount("/static", StaticFiles(directory="app/ui/templates"), name="static")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/sources", response_class=HTMLResponse)
def sources_page(request: Request):
    return templates.TemplateResponse("sources.html", {"request": request})


@app.get("/preview", response_class=HTMLResponse)
def chunk_preview_page(request: Request):
    return templates.TemplateResponse("preview.html", {"request": request})

templates = Jinja2Templates(directory="app/ui/templates")  # already present above

@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request})

from app.api.eval_api import router as eval_api_router
from app.routes_eval_pages import ui_router as eval_ui_router

# --- add where you wire routers into the app ---

@app.get("/eval", response_class=HTMLResponse)
def eval_page(request: Request):
    return templates.TemplateResponse("eval.html", {"request": request})


@app.get("/eval-compare", response_class=HTMLResponse)
def eval_compare_page(request: Request):
    return templates.TemplateResponse("eval-compare.html", {"request": request})
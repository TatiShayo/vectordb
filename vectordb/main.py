"""VectorDB v2 — FastAPI application entry point."""
from __future__ import annotations
import logging, os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

@asynccontextmanager
async def lifespan(app: FastAPI):
    from utils.logging_config import setup
    from config import LOG_LEVEL, LOG_FORMAT
    setup(LOG_LEVEL)
    logger = logging.getLogger("vectordb")
    logger.info("VectorDB v2 starting up …")

    from config import DATA_DIR, COLLECTIONS_DIR, SNAPSHOT_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    from core.engine import get_engine
    engine = get_engine()
    logger.info(f"Ready — {len(engine.list())} collection(s), "
                f"{engine.total_vectors()} total vectors")
    yield
    logger.info("Shutting down — saving state …")
    engine.save_all()
    engine.shutdown()
    logger.info("Done.")


app = FastAPI(
    title="VectorDB", description="Lightweight production-ready vector database.",
    version="2.0.0", lifespan=lifespan, docs_url="/docs", redoc_url="/redoc",
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

from api.middleware import RateLimitMiddleware, RequestLogMiddleware
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLogMiddleware)

# ── Routers ────────────────────────────────────────────────────────────────────
from api.collections import router as collections_router
from api.vectors     import router as vectors_router
from api.search      import router as search_router
from api.management  import router as admin_router

app.include_router(collections_router)
app.include_router(vectors_router)
app.include_router(search_router)
app.include_router(admin_router)

# ── Static UI ──────────────────────────────────────────────────────────────────
_web = Path(__file__).parent / "web"
if _web.exists():
    app.mount("/ui", StaticFiles(directory=str(_web), html=True), name="ui")

# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", tags=["root"])
async def root():
    return {"service": "VectorDB", "version": "2.0.0",
            "docs": "/docs", "ui": "/ui",
            "health": "/admin/health",
            "metrics": "/admin/metrics/prometheus"}

@app.exception_handler(Exception)
async def global_exc(request: Request, exc: Exception):
    logging.getLogger("vectordb").exception(
        f"Unhandled error: {request.method} {request.url.path}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",
                host=os.environ.get("VDB_HOST", "0.0.0.0"),
                port=int(os.environ.get("VDB_PORT", 8000)),
                reload=os.environ.get("VDB_RELOAD", "false").lower() == "true")

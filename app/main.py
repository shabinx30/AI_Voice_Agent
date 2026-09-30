"""Main FastAPI application entry point.

Initializes the FastAPI application, mounts REST and WebSocket routers,
serves static web assets, and handles application lifespan events.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import pipeline, router as api_router
from app.api.websocket import ws_router
from app.config import settings

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event context manager for application startup and shutdown.

    Warms up the OpenVINO STT and Qwen-TTS models and validates LM Studio
    availability on server launch.
    """
    logger.info("Initializing Personal Assistant Server...")
    logger.info("OpenVINO STT Model: %s (Device: %s)", settings.stt_model_id, settings.stt_device)
    logger.info("LM Studio Endpoint: %s (Model: %s)", settings.lm_studio_base_url, settings.lm_studio_model)
    logger.info("Kokoro TTS Model: %s (Device: %s)", settings.tts_model_id, settings.tts_device)

    # Perform non-blocking background warmup
    import asyncio

    async def _safe_warmup() -> None:
        try:
            await asyncio.to_thread(pipeline.warmup)
        except Exception as exc:
            logger.debug("Warmup finished with notice: %s", exc)

    asyncio.create_task(_safe_warmup())

    yield

    logger.info("Shutting down Personal Assistant Server...")
    await pipeline.llm.close()


def create_app() -> FastAPI:
    """Factory function to build and configure the FastAPI application instance.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Personal Voice Assistant",
        description=(
            "Voice assistant server integrating OpenVINO Whisper Base INT8 STT, "
            "LM Studio LLM inference, and Kokoro-82M TTS speech generation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API and WebSocket routers
    app.include_router(api_router)
    app.include_router(ws_router)

    # Static assets directory
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index() -> FileResponse:
            """Serves the frontend single-page application dashboard."""
            return FileResponse(static_dir / "index.html")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug,
    )

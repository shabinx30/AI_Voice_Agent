"""Server bootstrap and launcher script.

Ensures LM Studio local server and default model are initialized before
launching the FastAPI assistant server on the specified port.
"""

import logging
import subprocess
import sys
import time
import httpx
import uvicorn

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("launcher")


def verify_lm_studio() -> None:
    """Verifies that LM Studio server is running and a model is loaded."""
    logger.info("Checking LM Studio status at %s...", settings.lm_studio_base_url)

    is_online = False
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{settings.lm_studio_base_url}/models")
            if resp.status_code == 200:
                is_online = True
                models = [m["id"] for m in resp.json().get("data", [])]
                logger.info("LM Studio is online. Loaded models: %s", models)
    except Exception:
        pass

    if not is_online:
        logger.info("Starting LM Studio server via CLI...")
        try:
            res = subprocess.run(
                ["lms", "server", "start"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
            logger.info("LMS Server output: %s", res.stdout.strip())
            time.sleep(2)
        except Exception as exc:
            logger.warning("Could not auto-start LM Studio: %s", exc)

    # Load model if needed
    try:
        subprocess.run(
            [
                "lms",
                "load",
                settings.lm_studio_model,
                "--gpu",
                "max",
                "-y",
                "--identifier",
                settings.lm_studio_model,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except Exception as exc:
        logger.debug("Model load check note: %s", exc)


def main() -> None:
    """Entry point for the server launcher."""
    print("=" * 65)
    print("      NexusVoice Personal Assistant Server Starting...")
    print("=" * 65)
    print(f"  • Host:         http://localhost:{settings.server_port}")
    print(f"  • STT Model:    {settings.stt_model_id} (OpenVINO {settings.stt_device})")
    print(f"  • LM Studio:    {settings.lm_studio_base_url} ({settings.lm_studio_model})")
    print(f"  • TTS Model:    {settings.tts_model_id} (Speaker: {settings.tts_speaker})")
    print("=" * 65)

    verify_lm_studio()

    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()

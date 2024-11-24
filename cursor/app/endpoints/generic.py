from typing import Any
from fastapi.routing import APIRouter
from cursor.app.constants import llm_models

async def models() -> list[dict[str, Any]]:
    return models()


router = APIRouter()
router.add_api_route("/v1/models", models, methods=["GET"], tags=["Text"], response_model=None)

from fastapi import FastAPI, Depends, HTTPException, Request
from cursor.app.core.config import config
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
# Your predefined valid API keys
VALID_API_KEYS = {config.api_key}


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Get the API key from the `Authorization` header
        if request.method == "OPTIONS":
            return await call_next(request)

        if not request.headers.get("Authorization"):
            return JSONResponse(
                {"detail": "Invalid or missing API Key"}, status_code=401
            )

        api_key = request.headers.get("Authorization").split(" ")[1]

        # Validate the API key
        if not api_key or api_key not in VALID_API_KEYS:
            return JSONResponse(
                {"detail": "Invalid or missing API Key"}, status_code=401
            )

        # Proceed to the next middleware or route handler
        return await call_next(request)

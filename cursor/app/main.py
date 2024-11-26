from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from .endpoints.text import router as chat_router
from .endpoints.generic import router as generic_router
from cursor.app.core.config import config
from starlette.middleware.base import BaseHTTPMiddleware

# Your predefined valid API keys
VALID_API_KEYS =  {config.api_key}


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Get the API key from the `Authorization` header
        api_key = request.headers.get("Authorization")

        # Validate the API key
        if not api_key or api_key not in VALID_API_KEYS:
            return JSONResponse(
                {"detail": "Invalid or missing API Key"}, status_code=401
            )

        # Proceed to the next middleware or route handler
        return await call_next(request)


app = FastAPI()
# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(chat_router)
app.include_router(generic_router)


from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from .endpoints.text import router as chat_router



app = FastAPI()
app.include_router(chat_router)

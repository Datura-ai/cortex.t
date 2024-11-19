from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from . import curd, models, schemas
from .database import create_table, conn, cur
from typing import List
from .endpoints.text import router as chat_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    await create_table(None)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
)
app.include_router(chat_router)

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import asyncio

app = FastAPI()
organic_server_url = "https://0.0.0.0:8000"


@app.post("/apis.datura.ai/s18_sigma")
async def stream_prompt():
    try:
        # Create an asynchronous HTTP client session
        async with httpx.AsyncClient() as client:
            # Make a streaming GET request to the external server
            response = await client.stream("GET", organic_server_url)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

            # Define an async generator to read chunks of data from the external server
            async def stream_generator():
                async for chunk in response.aiter_bytes():
                    yield chunk
                    await asyncio.sleep(0)  # Allow other tasks to run

            # Return a StreamingResponse that forwards the data from the external server
            return StreamingResponse(stream_generator(), media_type="application/json")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail="Failed to fetch data from external server")

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import openai
import httpx

router = APIRouter()


class MessageRequest(BaseModel):
    session_id: str | None = None
    content: str


@router.post("/message")
def agent_message(req: MessageRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    client = openai.OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are Trailblip Agent. Be concise and helpful."},
                {"role": "user", "content": req.content},
            ],
        )
        return {
            "success": True,
            "data": {
                "session_id": req.session_id,
                "message": completion.choices[0].message.content,
                "tokens": completion.usage.total_tokens if hasattr(completion, "usage") else 0,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/stream")
def agent_message_stream(req: MessageRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    client = openai.OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def event_generator():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are Trailblip Agent. Be concise and helpful."},
                    {"role": "user", "content": req.content},
                ],
                stream=True,
            )
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"\n[stream-error] {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/plain")


@router.get("/tools/external/adventure-data")
async def proxy_adventure_data(lat: float, lng: float, radius: int = 5000, interests: str | None = None):
    base = os.getenv("NODE_BACKEND_URL", "http://localhost:5000")
    url = f"{base}/api/external/adventure-data"
    params = {"lat": lat, "lng": lng, "radius": radius}
    if interests:
        params["interests"] = interests
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Proxy error: {str(e)}")


@router.get("/tools/external/directions")
async def proxy_directions(fromLat: float, fromLng: float, toLat: float, toLng: float, mode: str = "walking"):
    base = os.getenv("NODE_BACKEND_URL", "http://localhost:5000")
    url = f"{base}/api/external/directions"
    params = {"fromLat": fromLat, "fromLng": fromLng, "toLat": toLat, "toLng": toLng, "mode": mode}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Proxy error: {str(e)}")


@router.get("/tools/external/weather/current")
async def proxy_weather_current(lat: float, lng: float):
    base = os.getenv("NODE_BACKEND_URL", "http://localhost:5000")
    url = f"{base}/api/external/weather/current"
    params = {"lat": lat, "lng": lng}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Proxy error: {str(e)}")


@router.get("/tools/external/weather/forecast")
async def proxy_weather_forecast(lat: float, lng: float, days: int = 1):
    base = os.getenv("NODE_BACKEND_URL", "http://localhost:5000")
    url = f"{base}/api/external/weather/forecast"
    params = {"lat": lat, "lng": lng, "days": days}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Proxy error: {str(e)}")



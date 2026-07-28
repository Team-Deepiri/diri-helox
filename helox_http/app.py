"""Mount training routes on a FastAPI app (cyrex/helox HTTP surface)."""

from fastapi import FastAPI

from helox_http.training_api import router as training_router

app = FastAPI(title="Diri-Helox Training API", version="1.0.0")
app.include_router(training_router)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "diri-helox-training"}

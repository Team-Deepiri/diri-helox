from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .routes.agent import router as agent_router
from .settings import settings

app = FastAPI(title="Trailblip Python Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "ai": "ready" if settings.OPENAI_API_KEY else "disabled"
    }

app.include_router(agent_router, prefix="/agent", tags=["agent"]) 



from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .routes.agent import router as agent_router
from .settings import settings
import time
import uuid
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Trailblip Python Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("pyagent")
logging.basicConfig(level=logging.INFO)

# Prometheus metrics
REQ_COUNTER = Counter("pyagent_requests_total", "Total requests", ["path", "method", "status"])
REQ_LATENCY = Histogram("pyagent_request_duration_seconds", "Request latency", ["path", "method"])

@app.middleware("http")
async def add_request_id_and_metrics(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.time()
    path = request.url.path
    method = request.method
    response = None
    try:
        # API key guard for non-health/metrics
        if not path.startswith("/health") and not path.startswith("/metrics"):
            api_key = request.headers.get("x-api-key")
            if settings.PYAGENT_API_KEY and api_key != settings.PYAGENT_API_KEY:
                raise HTTPException(status_code=401, detail="Invalid API key")
        response = await call_next(request)
        return response
    finally:
        status = response.status_code if response else 500
        duration = time.time() - start
        REQ_COUNTER.labels(path=path, method=method, status=str(status)).inc()
        REQ_LATENCY.labels(path=path, method=method).observe(duration)
        logger.info(f"request_id=%s method=%s path=%s status=%s duration_ms=%d", request_id, method, path, status, int(duration*1000))

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "ai": "ready" if settings.OPENAI_API_KEY else "disabled"
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

app.include_router(agent_router, prefix="/agent", tags=["agent"]) 



import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.services.ai_inference_service import AIInferenceService
from src.services.genai_service import GenAIService
from src.controllers.prediction_controller import PredictionController
from src.routes.prediction_routes import router as prediction_router

# ─── APPLICATION INITIALIZATION & LIFESPAN ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    inference_service = AIInferenceService(version=os.environ.get("MODEL_VERSION", "v1"))
    genai_service = GenAIService()
    app.state.inference_service = inference_service
    app.state.genai_service = genai_service
    app.state.prediction_controller = PredictionController(inference_service, genai_service)
    yield

app = FastAPI(title="SmartDigital Adaptive Twin API", version="1.6.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register prediction router
app.include_router(prediction_router)

# ─── ROUTING ENDPOINTS ───────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ONLINE",
        "model_version": os.environ.get("MODEL_VERSION", "v1"),
        "message": "SmartDigital Adaptive Twin API is running."
    }

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)
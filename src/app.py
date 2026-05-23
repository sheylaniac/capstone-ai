import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.services.ai_inference_service import AIInferenceService
from src.services.genai_service import GenAIService
from src.controllers.prediction_controller import PredictionController
from src.routes.prediction_routes import router as prediction_router

def load_env():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    workspace = os.path.dirname(current_dir)
    
    env_path = os.path.join(workspace, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, val = line.split('=', 1)
                        os.environ[key.strip()] = val.strip()
        print(f"Successfully loaded .env variables from: {env_path}")
    else:
        print(f"WARNING: .env file not found at {env_path}")

load_env()

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print(f"Starting SmartDigital API in lifespan context (Version: {MODEL_VERSION})...")
        
        inference_service = AIInferenceService(version=MODEL_VERSION)
        genai_service = GenAIService()
        
        app.state.prediction_controller = PredictionController(
            inference_service=inference_service,
            genai_service=genai_service
        )
        print("Initialization completed successfully. Services are bound and active.")
    except Exception as e:
        print(f"CRITICAL: Failed to load services on startup: {e}")
        
    yield  
    
    print("Shutting down SmartDigital API...")

app = FastAPI(
    title="SmartDigital Production API (dtwin-ai)",
    description="REST API untuk memprediksi produktivitas harian dan kategori status kelelahan.",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "message": "SmartDigital Multi-Output LSTM Prediction API is fully operational."
    }

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)

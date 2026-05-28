from fastapi import APIRouter, Request, Depends, HTTPException
from src.controllers.prediction_controller import (
    LogSequenceRequest,
    UserPredictionRequest,
    PredictFromLogsRequest,
    PredictionController
)

router = APIRouter(prefix="/api/v1", tags=["Prediction"])

def get_prediction_controller(request: Request) -> PredictionController:
    controller = getattr(request.app.state, "prediction_controller", None)
    if controller is None:
        raise HTTPException(status_code=503, detail="Prediction service is currently unavailable.")
    return controller

@router.post("/predict-from-logs")
async def predict_from_logs(
    payload: PredictFromLogsRequest,
    controller: PredictionController = Depends(get_prediction_controller)
):
    return await controller.predict_from_logs(payload)

@router.post("/predict")
async def predict_full(
    payload: LogSequenceRequest,
    controller: PredictionController = Depends(get_prediction_controller)
):
    return await controller.predict_full(payload)

@router.post("/predict-flexible")
async def predict_flexible(
    payload: UserPredictionRequest,
    controller: PredictionController = Depends(get_prediction_controller)
):
    return await controller.predict_flexible(payload)
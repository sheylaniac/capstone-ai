from fastapi import APIRouter, Request, Depends, HTTPException
from src.controllers.prediction_controller import LogSequenceRequest, PredictionResponse, PredictionController

router = APIRouter(prefix="/api/v1", tags=["Prediction"])

def get_prediction_controller(request: Request) -> PredictionController:
    controller = getattr(request.app.state, "prediction_controller", None)
    if controller is None:
        raise HTTPException(status_code=503, detail="Prediction service is currently unavailable.")
    return controller

@router.post("/predict", response_model=PredictionResponse)
async def predict_productivity(
    request_data: LogSequenceRequest,
    controller: PredictionController = Depends(get_prediction_controller)
):
    try:
        response = await controller.get_prediction(request_data)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

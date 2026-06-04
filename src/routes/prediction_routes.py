from fastapi import APIRouter, Request, Depends, HTTPException, Header
from src.schemas.prediction_schema import AIPredictionPayload
from src.controllers.prediction_controller import PredictionController

router = APIRouter(prefix="/api/v1", tags=["Prediction"])

def get_prediction_controller(request: Request) -> PredictionController:
    controller = getattr(request.app.state, "prediction_controller", None)
    if controller is None:
        raise HTTPException(status_code=503, detail="Prediction service unavailable.")
    return controller

@router.post("/predict")
async def generate_prediction(
    payload: AIPredictionPayload,
    authorization: str = Header(None),
    controller: PredictionController = Depends(get_prediction_controller)
):
    return await controller.process_prediction(payload, authorization)
from pydantic import BaseModel, Field
from typing import Annotated, List, Dict
from src.services.ai_inference_service import AIInferenceService
from src.services.genai_service import GenAIService

class DailyLog(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sleep_duration": 7.5,
                    "sleep_quality": 7.0,
                    "study_work_duration": 6.5,
                    "break_duration": 0.75,
                    "physical_activity_duration": 30.0,
                    "screen_time_duration": 3.0,
                    "stress_level": 4.0,
                    "mood_score": 7.0,
                    "focus_score": 8.0,
                    "task_planned": 5.0,
                    "task_completed": 4.0,
                    "task_completion_rate": 0.8,
                    "day_of_week": 0,
                    "month": 5,
                    "is_weekend": 0
                }
            ]
        }
    }

    sleep_duration: float = Field(..., description="Durasi tidur dalam jam")
    sleep_quality: float = Field(..., description="Kualitas tidur (skala 1-10)")
    study_work_duration: float = Field(..., description="Durasi belajar/bekerja dalam jam")
    break_duration: float = Field(..., description="Durasi istirahat dalam jam")
    physical_activity_duration: float = Field(..., description="Durasi aktivitas fisik dalam menit")
    screen_time_duration: float = Field(..., description="Screen time dalam jam")
    stress_level: float = Field(..., description="Tingkat stres (skala 1-10)")
    mood_score: float = Field(..., description="Mood score (skala 1-10)")
    focus_score: float = Field(..., description="Focus score (skala 1-10)")
    task_planned: float = Field(..., description="Jumlah tugas yang direncanakan")
    task_completed: float = Field(..., description="Jumlah tugas yang diselesaikan")
    task_completion_rate: float = Field(..., description="Rasio penyelesaian tugas (0.0 - 1.0)")
    day_of_week: int = Field(..., description="Hari dalam seminggu (0=Senin, 6=Minggu)")
    month: int = Field(..., description="Bulan (1-12)")
    is_weekend: int = Field(..., description="Apakah hari libur/akhir pekan? (0 atau 1)")

class LogSequenceRequest(BaseModel):
    logs: Annotated[List[DailyLog], Field(min_length=7, max_length=7)] = Field(
        ...,
        description="Sekuens log aktivitas harian tepat sepanjang 7 hari."
    )

class PredictionResponse(BaseModel):
    predicted_productivity_score: float
    predicted_category: str
    probabilities: Dict[str, float]
    recommendation: str

class PredictionController:
    def __init__(self, inference_service: AIInferenceService, genai_service: GenAIService):
        self.inference_service = inference_service
        self.genai_service = genai_service

    async def get_prediction(self, request: LogSequenceRequest) -> PredictionResponse:
        raw_logs = [log.model_dump() for log in request.logs]
        
        score, category, probabilities, metrics = self.inference_service.predict(raw_logs)
        
        recommendation = self.genai_service.get_recommendation(category, score, metrics)
        
        return PredictionResponse(
            predicted_productivity_score=score,
            predicted_category=category,
            probabilities=probabilities,
            recommendation=recommendation
        )

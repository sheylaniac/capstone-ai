from pydantic import BaseModel
from typing import List

class DailyLogSchema(BaseModel):    
    id: str    
    user_id: str    
    log_date: str     
    is_weekend: bool    
    sleep_duration: float    
    study_work_duration: float    
    break_duration: float    
    exercise_duration: float    
    downtime_duration: float    
    stress_level: int    
    mood_score: int    
    focus_score: int    
    task_planned: int    
    task_completed: int    
    completion_ratio: float    
    fatigue_index: float    
    cumulative_fatigue: float

class AIPredictionPayload(BaseModel):
    last_7_logs: List[DailyLogSchema]
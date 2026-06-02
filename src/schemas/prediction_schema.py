from pydantic import BaseModel, Field
from typing import List, Optional

class UserGoals(BaseModel):
    focus_sleep: bool
    focus_productivity: bool
    focus_fitness: bool
    focus_screen_time: bool

class DailyActivity(BaseModel):
    # Node.js akan mengirim data ini sesuai dengan skema database
    log_date: str 
    is_weekend: bool
    sleep_duration: float
    sleep_quality: int
    study_work_duration: float
    break_duration: float
    physical_activity_duration: float
    screen_time_duration: float
    stress_level: int
    mood_score: int
    focus_score: int
    task_planned: int
    task_completed: int
    completion_ratio: float

class AIPredictionPayload(BaseModel):
    user_id: str
    user_goals: UserGoals
    last_7_logs: List[DailyActivity] = Field(..., description="Wajib berisi 7 data harian")
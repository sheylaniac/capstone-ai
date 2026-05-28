import os
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, List, Dict, Optional
from datetime import datetime, timedelta
from fastapi import HTTPException
from src.services.ai_inference_service import AIInferenceService
from src.services.genai_service import GenAIService

# ─── DATABASE CONFIGURATION & CONSTANTS ──────────────────────────────────────

USER_LOGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "user_activity_logs.csv"
)

GLOBAL_DEFAULTS = {
    "sleep_duration": 7.22,
    "sleep_quality": 6.18,
    "study_work_duration": 6.51,
    "break_duration": 2.47,
    "physical_activity_duration": 27.60,
    "screen_time_duration": 6.49,
    "stress_level": 4.22,
    "mood_score": 6.13,
    "focus_score": 5.66,
    "task_planned": 5.96,
    "task_completed": 3.47
}

ACTIVITY_FIELDS = [
    "sleep_duration", "sleep_quality", "study_work_duration", "break_duration",
    "physical_activity_duration", "screen_time_duration", "stress_level",
    "mood_score", "focus_score", "task_planned", "task_completed"
]

def load_user_logs() -> pd.DataFrame:
    if os.path.exists(USER_LOGS_PATH):
        try:
            return pd.read_csv(USER_LOGS_PATH)
        except Exception:
            pass
    columns = [
        "log_id", "user_id", "log_date", "is_weekend", "sleep_duration", "sleep_quality",
        "study_work_duration", "break_duration", "physical_activity_duration",
        "screen_time_duration", "stress_level", "mood_score", "focus_score",
        "task_completed", "task_planned", "completion_ratio", "fatigue_accumulation",
        "productivity_score", "created_at"
    ]
    return pd.DataFrame(columns=columns)

def save_user_logs(df: pd.DataFrame):
    os.makedirs(os.path.dirname(USER_LOGS_PATH), exist_ok=True)
    df.to_csv(USER_LOGS_PATH, index=False)


# ─── PYDANTIC MODELS ─────────────────────────────────────────────────────────

# ─── PYDANTIC MODELS ─────────────────────────────────────────────────────────
# ─── PYDANTIC MODELS (SOLUSI FIX 422 & URUTAN SWAGGER KEMBAR) ────────────────

class DailyLog(BaseModel):
    """Skema untuk Predict Full: Semua field WAJIB DIISI."""
    model_config = {"populate_by_name": True}

    sleep_duration: float = Field(description="Durasi tidur (jam)", example=7.2)
    sleep_quality: float = Field(description="Kualitas tidur harian (1-10)", example=6.5)
    study_work_duration: float = Field(description="Durasi belajar/bekerja (jam)", example=6.0)
    break_duration: float = Field(description="Durasi istirahat (jam)", example=2.0)
    physical_activity_duration: float = Field(description="Durasi olahraga (menit)", example=15.0)
    screen_time_duration: float = Field(description="Durasi layar (jam)", example=6.0)
    stress_level: float = Field(description="Skala tingkat stres (1-10)", example=4.0)
    mood_score: float = Field(description="Skor suasana hati harian (1-10)", example=6.0)
    focus_score: float = Field(description="Skala tingkat fokus (1-10)", example=7.0)
    task_planned: int = Field(description="Jumlah tugas direncanakan", example=5)
    task_completed: int = Field(description="Jumlah tugas diselesaikan", example=3)
    is_weekend: int = Field(description="Akhir pekan (0 atau 1)", example=0)
    log_date: str = Field(description="Tanggal log (YYYY-MM-DD)", example="2026-05-28")


class LogSequenceRequest(BaseModel):
    user_id: int = Field(..., description="User ID untuk pencarian riwayat data", example=12)
    current_log: DailyLog = Field(..., description="Log harian lengkap (wajib diisi semua fitur).")


class DailyActivityLogOptional(BaseModel):
    """Skema untuk Predict Flexible: Semua field OPSIONAL (Boleh Kosong)."""
    model_config = {"populate_by_name": True}

    # Diberikan default=None agar Pydantic tahu field ini boleh tidak dikirim/kosong
    sleep_duration: Optional[float] = Field(None, description="[OPSIONAL] Durasi tidur (jam)", example=7.2)
    sleep_quality: Optional[float] = Field(None, description="[OPSIONAL] Kualitas tidur harian (1-10)", example=6.5)
    study_work_duration: Optional[float] = Field(None, description="[OPSIONAL] Durasi belajar/bekerja (jam)", example=6.0)
    break_duration: Optional[float] = Field(None, description="[OPSIONAL] Durasi istirahat (jam)", example=2.0)
    physical_activity_duration: Optional[float] = Field(None, description="[OPSIONAL] Durasi olahraga (menit)", example=15.0)
    screen_time_duration: Optional[float] = Field(None, description="[OPSIONAL] Durasi layar (jam)", example=6.0)
    stress_level: Optional[float] = Field(None, description="[OPSIONAL] Skala tingkat stres (1-10)", example=4.0)
    mood_score: Optional[float] = Field(None, description="[OPSIONAL] Skor suasana hati harian (1-10)", example=6.0)
    focus_score: Optional[float] = Field(None, description="[OPSIONAL] Skala tingkat fokus (1-10)", example=7.0)
    task_planned: Optional[int] = Field(None, description="[OPSIONAL] Jumlah tugas direncanakan", example=5)
    task_completed: Optional[int] = Field(None, description="[OPSIONAL] Jumlah tugas diselesaikan", example=3)
    is_weekend: Optional[int] = Field(None, description="[OPSIONAL] Akhir pekan (0 atau 1)", example=0)
    log_date: Optional[str] = Field(None, description="[OPSIONAL] Tanggal log (YYYY-MM-DD)", example="2026-05-28")


class UserPredictionRequest(BaseModel):
    user_id: int = Field(..., description="User ID untuk pencarian riwayat data", example=12)
    current_log: DailyActivityLogOptional = Field(..., description="Form fleksibel. Isi yang diinginkan saja, sisanya otomatis diisi AI.")


class PredictFromLogsRequest(BaseModel):
    user_id: int = Field(..., description="User ID untuk memuat 7 log terakhir dari database", example=12)
# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

class FlexibleDayLog(BaseModel):
    """Skema log harian di mana semua field aktivitas bersifat opsional."""
    model_config = {"populate_by_name": True}

    log_date: str = Field(description="Tanggal log (YYYY-MM-DD), wajib diisi untuk urutan sekuens", example="2026-05-28")
    is_weekend: Optional[int] = Field(None, description="Akhir pekan (0 atau 1)", example=0)
    
    sleep_duration: Optional[float] = Field(None, description="Durasi tidur (jam)")
    sleep_quality: Optional[float] = Field(None, description="Kualitas tidur harian (1-10)")
    study_work_duration: Optional[float] = Field(None, description="Durasi belajar/bekerja (jam)")
    break_duration: Optional[float] = Field(None, description="Durasi istirahat (jam)")
    physical_activity_duration: Optional[float] = Field(None, description="Durasi olahraga (menit)")
    screen_time_duration: Optional[float] = Field(None, description="Durasi layar (jam)")
    stress_level: Optional[float] = Field(None, description="Skala tingkat stres (1-10)")
    mood_score: Optional[float] = Field(None, description="Skor suasana hati harian (1-10)")
    focus_score: Optional[float] = Field(None, description="Skala tingkat fokus (1-10)")
    task_planned: Optional[int] = Field(None, description="Jumlah tugas direncanakan")
    task_completed: Optional[int] = Field(None, description="Jumlah tugas diselesaikan")

class UserFlexibleSequenceRequest(BaseModel):
    user_id: int = Field(..., description="User ID pencarian riwayat data", example=12)
    logs_sequence: List[FlexibleDayLog] = Field(..., description="List data sekuens wajib berisi 7 hari")

    @field_validator("logs_sequence")
    @classmethod
    def validate_must_be_7_days(cls, value: List[FlexibleDayLog]) -> List[FlexibleDayLog]:
        if len(value) != 7:
            raise ValueError("logs_sequence harus berisi tepat 7 hari data berturut-turut!")
        return value
def format_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str.split(" ")[0], "%Y-%m-%d")
        return dt.strftime("%b %d")
    except Exception:
        return date_str

def safe_weekday(d) -> int:
    try:
        return datetime.strptime(str(d).strip(), "%Y-%m-%d").weekday()
    except Exception:
        return 0

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def build_model_input(row: pd.Series) -> dict:
    try:
        dt = datetime.strptime(str(row["log_date"]), "%Y-%m-%d")
        day_of_week = dt.weekday()
        month = dt.month
    except Exception:
        day_of_week = 0
        month = 1
    return {
        "sleep_duration": float(row["sleep_duration"]),
        "sleep_quality": float(row["sleep_quality"]),
        "study_work_duration": float(row["study_work_duration"]),
        "break_duration": float(row["break_duration"]),
        "physical_activity_duration": float(row["physical_activity_duration"]),
        "screen_time_duration": float(row["screen_time_duration"]),
        "stress_level": float(row["stress_level"]),
        "mood_score": float(row["mood_score"]),
        "focus_score": float(row["focus_score"]),
        "task_planned": float(row["task_planned"]),
        "task_completed": float(row["task_completed"]),
        "task_completion_rate": float(row["completion_ratio"]),
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": int(row["is_weekend"])
    }

def ensure_user_has_7_days(df_logs: pd.DataFrame, user_id: int) -> pd.DataFrame:
    user_logs = df_logs[df_logs["user_id"] == user_id].sort_values("log_date")
    num_logs = len(user_logs)
    if num_logs >= 7:
        return df_logs
    
    needed = 7 - num_logs
    if num_logs > 0:
        try:
            start_dt = datetime.strptime(str(user_logs.iloc[0]["log_date"]).split(" ")[0], "%Y-%m-%d")
        except Exception:
            start_dt = datetime.now()
    else:
        start_dt = datetime.now()
        
    for i in range(1, needed + 1):
        target_dt = start_dt - timedelta(days=i)
        target_date_str = target_dt.strftime("%Y-%m-%d")
        
        stress = GLOBAL_DEFAULTS["stress_level"]
        study = GLOBAL_DEFAULTS["study_work_duration"]
        sleep_dur = GLOBAL_DEFAULTS["sleep_duration"]
        sleep_qual = GLOBAL_DEFAULTS["sleep_quality"]
        planned = GLOBAL_DEFAULTS["task_planned"]
        completed = GLOBAL_DEFAULTS["task_completed"]
        completion = float(completed / planned) if planned > 0 else 0.0
        focus = GLOBAL_DEFAULTS["focus_score"]
        mood = GLOBAL_DEFAULTS["mood_score"]
        
        fatigue_calc = (stress / 10.0) * 0.4 + (study / (sleep_dur + 1e-5)) * 0.3 + (1.0 - sleep_qual / 10.0) * 0.3
        fatigue_accumulation = round(float(np.clip(fatigue_calc, 0.0, 5.0)), 2)
        
        prod_calc = (completion * 0.40 + (focus / 10.0) * 0.25 + (mood / 10.0) * 0.15 + ((10.0 - stress) / 10.0) * 0.10 + (sleep_qual / 10.0) * 0.10) * 100.0
        productivity_score = round(float(np.clip(prod_calc, 15.0, 98.0)), 2)
        
        log_id = int(df_logs["log_id"].max() + 1) if not df_logs.empty and pd.notna(df_logs["log_id"].max()) else 1
        new_row = {
            "log_id": log_id,
            "user_id": user_id,
            "log_date": target_date_str,
            "is_weekend": 1 if target_dt.weekday() >= 5 else 0,
            "sleep_duration": float(GLOBAL_DEFAULTS["sleep_duration"]),
            "sleep_quality": float(GLOBAL_DEFAULTS["sleep_quality"]),
            "study_work_duration": float(GLOBAL_DEFAULTS["study_work_duration"]),
            "break_duration": float(GLOBAL_DEFAULTS["break_duration"]),
            "physical_activity_duration": float(GLOBAL_DEFAULTS["physical_activity_duration"]),
            "screen_time_duration": float(GLOBAL_DEFAULTS["screen_time_duration"]),
            "stress_level": float(GLOBAL_DEFAULTS["stress_level"]),
            "mood_score": float(GLOBAL_DEFAULTS["mood_score"]),
            "focus_score": float(GLOBAL_DEFAULTS["focus_score"]),
            "task_planned": int(GLOBAL_DEFAULTS["task_planned"]),
            "task_completed": int(GLOBAL_DEFAULTS["task_completed"]),
            "completion_ratio": completion,
            "fatigue_accumulation": fatigue_accumulation,
            "productivity_score": productivity_score,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        df_logs = pd.concat([df_logs, pd.DataFrame([new_row])], ignore_index=True)
        
    return df_logs

def build_response_components(
    score: float, category: str, probabilities: dict, metrics: dict,
    reconciled_log: dict, log_date: str, user_logs_updated: pd.DataFrame,
    last_7_logs_updated: pd.DataFrame, genai_service: GenAIService
) -> dict:
    # Component 1: Main Dashboard
    main_dashboard = {
        "productivity_status": category,
        "productivity_score": round(score, 2),
        "prediction_confidence": round(probabilities.get(category, 0.0) * 100, 2),
        "probabilities": {k: float(v) for k, v in probabilities.items()},
        "fatigue_level": round(metrics.get("last_fatigue", 0.0), 2),
        "completion_rate": round(reconciled_log["completion_ratio"] * 100, 2),
        "risk_signal": "HIGH" if category == "At Risk" else "NORMAL"
    }

    # Component 2: Analytics Dashboard
    daily_chart = []
    for _, row in last_7_logs_updated.iterrows():
        score_val = row.get("productivity_score")
        score_val = float(score_val) if pd.notna(score_val) else 0.0
        daily_chart.append({
            "date": format_date(str(row["log_date"])),
            "score": round(score_val, 2)
        })

    if len(last_7_logs_updated) >= 3:
        first_3_avg = last_7_logs_updated.head(3)["productivity_score"].fillna(0).mean()
        last_3_avg = last_7_logs_updated.tail(3)["productivity_score"].fillna(0).mean()
        trend = "Increasing" if last_3_avg > first_3_avg + 2.0 else "Decreasing" if last_3_avg < first_3_avg - 2.0 else "Stable"
    else:
        trend = "Stable"

    durations = {
        "Study/Work": last_7_logs_updated["study_work_duration"].fillna(0).mean(),
        "Sleep": last_7_logs_updated["sleep_duration"].fillna(0).mean(),
        "Screen Time": last_7_logs_updated["screen_time_duration"].fillna(0).mean(),
        "Break": last_7_logs_updated["break_duration"].fillna(0).mean(),
        "Physical Activity": last_7_logs_updated["physical_activity_duration"].fillna(0).mean() / 60.0
    }
    dominant_activity = max(durations, key=durations.get)
    mean_focus = last_7_logs_updated["focus_score"].fillna(0).mean()
    peak_hours = "08:30–11:30" if mean_focus > 7.5 else "09:00–11:00"

    day_mapping = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    user_logs_with_day = user_logs_updated.copy()
    user_logs_with_day["day_of_week_num"] = user_logs_with_day["log_date"].apply(safe_weekday)
    heatmap_grouped = user_logs_with_day.groupby("day_of_week_num")["productivity_score"].mean().to_dict()
    activity_heatmap = {}
    for i, name in day_mapping.items():
        val = heatmap_grouped.get(i, 0.0)
        if pd.isna(val) or val is None:
            val = 0.0
        activity_heatmap[name] = round(float(val), 2)

    analytics_dashboard = {
        "daily_productivity_chart": daily_chart,
        "weekly_productivity_trend": trend,
        "most_dominant_activity": dominant_activity,
        "peak_productive_hours": peak_hours,
        "activity_heatmap": activity_heatmap
    }

    # Component 3: AI Insight
    recommendation = genai_service.get_recommendation(category, score, metrics)
    paragraphs = [p.strip() for p in recommendation.split("\n\n") if p.strip()]
    while len(paragraphs) < 4:
        paragraphs.append("Insufficient data for this analysis section.")

    ai_insight = {
        "condition_insight": paragraphs[0],
        "performance_cause": paragraphs[1],
        "activity_recommendation": paragraphs[2],
        "burnout_warning": paragraphs[3],
        "recommendation_text": recommendation
    }

    # Component 4: Similar Productivity History
    past_logs = user_logs_updated[user_logs_updated["log_date"] != log_date]
    feature_cols = [
        "sleep_duration", "sleep_quality", "study_work_duration", "break_duration",
        "physical_activity_duration", "screen_time_duration", "stress_level",
        "mood_score", "focus_score", "task_planned", "task_completed"
    ]
    current_vec = np.array([reconciled_log[f] for f in feature_cols], dtype=float)

    similarities = []
    for _, row in past_logs.iterrows():
        row_vec = row[feature_cols].fillna(0).values.astype(float)
        sim_score = calculate_cosine_similarity(current_vec, row_vec)
        score_hist = float(row.get("productivity_score")) if pd.notna(row.get("productivity_score")) else 0.0
        cat_hist = "Thriving" if score_hist >= 67.0 else "Steady" if score_hist >= 55.0 else "At Risk"
        similarities.append({
            "date": format_date(str(row["log_date"])),
            "similarity_score": f"{round(sim_score * 100)}%",
            "historical_productivity_score": round(score_hist, 2),
            "historical_category": cat_hist,
            "raw_score": score_hist
        })

    similarities.sort(key=lambda x: float(x["similarity_score"].replace("%", "")), reverse=True)
    top_3 = similarities[:3]
    avg_prod = sum(d["raw_score"] for d in top_3) / len(top_3) if top_3 else 0.0
    for d in top_3:
        d.pop("raw_score", None)

    similar_history = {
        "top_3_similar_days": top_3,
        "average_productivity_from_similar_days": round(avg_prod, 2)
    }

    # Component 5: Reconciliation
    reconciled_log["fatigue_accumulation"] = round(metrics.get("last_fatigue", 0.0), 2)
    reconciled_log["productivity_score"] = round(score, 2)

    form_reconciliation = {
        "imputation_status": "Historical profile average applied to empty fields." if len(past_logs) > 0 else "Global default average applied to empty fields.",
        "reconciled_log_today": reconciled_log
    }

    return {
        "1_main_dashboard": main_dashboard,
        "2_productivity_analytics_dashboard": analytics_dashboard,
        "3_ai_insight_and_recommendation": ai_insight,
        "4_similar_productivity_history": similar_history,
        "5_activity_input_form_reconciliation": form_reconciliation
    }


# ─── CONTROLLER CLASS ────────────────────────────────────────────────────────

class PredictionController:
    def __init__(self, inference_service: AIInferenceService, genai_service: GenAIService):
        self.inference_service = inference_service
        self.genai_service = genai_service

    async def predict_from_logs(self, payload: PredictFromLogsRequest) -> dict:
        user_id = payload.user_id
        df_logs = load_user_logs()
        
        # Ensure user has 7 days of logs (creates synthetic logs if needed)
        df_logs = ensure_user_has_7_days(df_logs, user_id)
        save_user_logs(df_logs)
        
        # Reload updated user logs
        df_logs = load_user_logs()
        user_logs = df_logs[df_logs["user_id"] == user_id].sort_values("log_date")
        
        num_logs = len(user_logs)
        last_7_logs = user_logs.tail(7)
        model_input_logs = [build_model_input(row) for _, row in last_7_logs.iterrows()]
        
        try:
            score, category, probabilities, metrics = self.inference_service.predict(model_input_logs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
            
        log_date = str(last_7_logs.iloc[-1]["log_date"])
        
        # Update today's log in CSV with predicted score and fatigue
        today_idx = df_logs[(df_logs["user_id"] == user_id) & (df_logs["log_date"] == log_date)].index
        if not today_idx.empty:
            idx = today_idx[0]
            df_logs.loc[idx, "productivity_score"] = float(score)
            df_logs.loc[idx, "fatigue_accumulation"] = float(metrics.get("last_fatigue", 0.0))
            save_user_logs(df_logs)
            
        df_logs = load_user_logs()
        user_logs_updated = df_logs[df_logs["user_id"] == user_id].sort_values("log_date")
        last_7_logs_updated = user_logs_updated.tail(7)
        
        reconciled_log = {field: float(last_7_logs.iloc[-1][field]) for field in ACTIVITY_FIELDS}
        reconciled_log["completion_ratio"] = float(last_7_logs.iloc[-1]["completion_ratio"])
        
        components = build_response_components(
            score, category, probabilities, metrics,
            reconciled_log, log_date, user_logs_updated, last_7_logs_updated,
            self.genai_service
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "days_logged": num_logs,
            "message": "Prediction generated from saved activity logs.",
            **components
        }

    async def predict_full(self, payload: LogSequenceRequest) -> dict:
        user_id = payload.user_id
        current_input = payload.current_log.model_dump()
        log_date = current_input.get("log_date")
        
        if not log_date:
            raise HTTPException(status_code=400, detail="Format log_date tidak valid. Gunakan format YYYY-MM-DD.")

        # 1. Ambil data logs saat ini dari database CSV
        df_logs = load_user_logs()
        
        # 2. Susun record data baru dari input (Validasi Pydantic menjamin data ini terisi penuh)
        reconciled_log = {
            "user_id": user_id,
            "log_date": log_date,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for field in ACTIVITY_FIELDS:
            reconciled_log[field] = float(current_input[field])
            
        planned = reconciled_log["task_planned"]
        completed = reconciled_log["task_completed"]
        reconciled_log["completion_ratio"] = float(completed / planned) if planned > 0 else 0.0
        reconciled_log["is_weekend"] = int(current_input["is_weekend"])
        
        # Kalkulasi nilai awal fatigue & productivity harian sebelum prediksi model AI
        stress = reconciled_log["stress_level"]
        study = reconciled_log["study_work_duration"]
        sleep_dur = reconciled_log["sleep_duration"]
        sleep_qual = reconciled_log["sleep_quality"]
        focus = reconciled_log["focus_score"]
        mood = reconciled_log["mood_score"]
        completion = reconciled_log["completion_ratio"]
        
        fatigue_calc = (stress / 10.0) * 0.4 + (study / (sleep_dur + 1e-5)) * 0.3 + (1.0 - sleep_qual / 10.0) * 0.3
        reconciled_log["fatigue_accumulation"] = round(float(np.clip(fatigue_calc, 0.0, 5.0)), 2)
        
        prod_calc = (completion * 0.40 + (focus / 10.0) * 0.25 + (mood / 10.0) * 0.15 + ((10.0 - stress) / 10.0) * 0.10 + (sleep_qual / 10.0) * 0.10) * 100.0
        reconciled_log["productivity_score"] = round(float(np.clip(prod_calc, 15.0, 98.0)), 2)
        
        # 3. Masukkan atau timpa record log tanggal ini di CSV
        existing_idx = df_logs[(df_logs["user_id"] == user_id) & (df_logs["log_date"] == log_date)].index
        if not existing_idx.empty:
            idx = existing_idx[0]
            reconciled_log["log_id"] = int(df_logs.loc[idx, "log_id"])
            for col, val in reconciled_log.items():
                df_logs.loc[idx, col] = val
        else:
            log_id = int(df_logs["log_id"].max() + 1) if not df_logs.empty and pd.notna(df_logs["log_id"].max()) else 1
            reconciled_log["log_id"] = log_id
            df_logs = pd.concat([df_logs, pd.DataFrame([reconciled_log])], ignore_index=True)
            
        save_user_logs(df_logs)
        
        # 4. Ambil history user yang sudah diperbarui & Cek jumlah harinya
        df_logs = load_user_logs()
        user_logs = df_logs[df_logs["user_id"] == user_id].sort_values("log_date")
        num_logs = len(user_logs)
        
        # ─── PERINGATAN STRIP JIKA SEKUENS KURANG DARI 7 HARI ───
        if num_logs < 7:
            raise HTTPException(
                status_code=400, 
                detail=f"Gagal memproses prediksi. Data aktivitas user baru terisi {num_logs} hari di database (termasuk hari ini). Minimal dibutuhkan riwayat 7 hari aktivitas!"
            )
            
        # 5. Jika lolos sekuens 7 hari, potong 7 baris terakhir untuk input LSTM model
        last_7_logs = user_logs.tail(7)
        model_input_logs = [build_model_input(row) for _, row in last_7_logs.iterrows()]
        
        try:
            score, category, probabilities, metrics = self.inference_service.predict(model_input_logs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
            
        # 6. Update skor final hasil prediksi model AI ke database
        df_logs.loc[df_logs["log_id"] == reconciled_log["log_id"], "productivity_score"] = float(score)
        df_logs.loc[df_logs["log_id"] == reconciled_log["log_id"], "fatigue_accumulation"] = float(metrics.get("last_fatigue", 0.0))
        save_user_logs(df_logs)
        
        # Reload database final untuk penyusunan komponen visual response dashboard
        df_logs = load_user_logs()
        user_logs_final = df_logs[df_logs["user_id"] == user_id].sort_values("log_date")
        last_7_logs_final = user_logs_final.tail(7)
        
        reconciled_log_final = {field: float(last_7_logs_final.iloc[-1][field]) for field in ACTIVITY_FIELDS}
        reconciled_log_final["completion_ratio"] = float(last_7_logs_final.iloc[-1]["completion_ratio"])
        
        components = build_response_components(
            score, category, probabilities, metrics,
            reconciled_log_final, log_date, user_logs_final, last_7_logs_final,
            self.genai_service
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "days_logged": num_logs,
            "message": "Productivity prediction successfully calculated.",
            **components
        }

    async def predict_flexible(self, payload: UserPredictionRequest) -> dict:
        user_id = payload.user_id
        current_input = payload.current_log.model_dump()
        log_date = current_input.get("log_date")
        
        if not log_date:
            log_date = datetime.now().strftime("%Y-%m-%d")
            
        # 1. Load data masa lalu dari database untuk keperluan imputasi nilai rata-rata
        df_logs = load_user_logs()
        past_user_logs = df_logs[df_logs["user_id"] == user_id]
        
        # 2. Susun record data baru dari input (imputasi field opsional)
        reconciled_log = {
            "user_id": user_id,
            "log_date": log_date,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # --- PROSES IMPUTASI FITUR OPSIONAL ---
        for field in ACTIVITY_FIELDS:
            val = current_input.get(field)
            if val is None:
                # Jika user mengosongkan fitur ini, ambil dari rata-rata histori atau global default
                if len(past_user_logs) > 0 and field in past_user_logs.columns:
                    mean_val = past_user_logs[field].mean()
                    reconciled_log[field] = float(mean_val) if not np.isnan(mean_val) else GLOBAL_DEFAULTS[field]
                else:
                    reconciled_log[field] = GLOBAL_DEFAULTS[field]
            else:
                reconciled_log[field] = float(val)
                
        # --- HITUNG METRIK TURUNAN MANDIRI ---
        planned = reconciled_log["task_planned"]
        completed = reconciled_log["task_completed"]
        reconciled_log["completion_ratio"] = float(completed / planned) if planned > 0 else 0.0
        
        # Deteksi otomatis weekend jika tidak diisi user
        is_weekend = current_input.get("is_weekend")
        if is_weekend is None:
            try:
                dt = datetime.strptime(log_date, "%Y-%m-%d")
                reconciled_log["is_weekend"] = 1 if dt.weekday() >= 5 else 0
            except Exception:
                reconciled_log["is_weekend"] = 0
        else:
            reconciled_log["is_weekend"] = int(is_weekend)
            
        # Hitung kalkulasi awal fatigue & productivity harian statis
        stress = reconciled_log["stress_level"]
        study = reconciled_log["study_work_duration"]
        sleep_dur = reconciled_log["sleep_duration"]
        sleep_qual = reconciled_log["sleep_quality"]
        focus = reconciled_log["focus_score"]
        mood = reconciled_log["mood_score"]
        completion = reconciled_log["completion_ratio"]
        
        fatigue_calc = (stress / 10.0) * 0.4 + (study / (sleep_dur + 1e-5)) * 0.3 + (1.0 - sleep_qual / 10.0) * 0.3
        reconciled_log["fatigue_accumulation"] = round(float(np.clip(fatigue_calc, 0.0, 5.0)), 2)
        
        prod_calc = (completion * 0.40 + (focus / 10.0) * 0.25 + (mood / 10.0) * 0.15 + ((10.0 - stress) / 10.0) * 0.10 + (sleep_qual / 10.0) * 0.10) * 100.0
        reconciled_log["productivity_score"] = round(float(np.clip(prod_calc, 15.0, 98.0)), 2)
        
        # 3. Simpan atau perbarui log hari ini ke dalam DataFrame master
        existing_idx = df_logs[(df_logs["user_id"] == user_id) & (df_logs["log_date"] == log_date)].index
        if not existing_idx.empty:
            idx = existing_idx[0]
            reconciled_log["log_id"] = int(df_logs.loc[idx, "log_id"])
            for col, val in reconciled_log.items():
                df_logs.loc[idx, col] = val
        else:
            log_id = int(df_logs["log_id"].max() + 1) if not df_logs.empty and pd.notna(df_logs["log_id"].max()) else 1
            reconciled_log["log_id"] = log_id
            df_logs = pd.concat([df_logs, pd.DataFrame([reconciled_log])], ignore_index=True)
            
        save_user_logs(df_logs)
        
        # 4. Ambil history user yang sudah diperbarui & Cek jumlah harinya
        df_logs = load_user_logs()
        user_logs = df_logs[df_logs["user_id"] == user_id].sort_values("log_date")
        num_logs = len(user_logs)
        
        if num_logs < 7:
            raise HTTPException(
                status_code=400, 
                detail=f"Gagal memproses prediksi. Data aktivitas user baru terisi {num_logs} hari di database (termasuk hari ini). Minimal dibutuhkan riwayat 7 hari aktivitas!"
            )
            
        # 5. Jika lolos sekuens 7 hari, potong 7 baris terakhir untuk input LSTM model
        last_7_logs = user_logs.tail(7)
        model_input_logs = [build_model_input(row) for _, row in last_7_logs.iterrows()]
        
        try:
            score, category, probabilities, metrics = self.inference_service.predict(model_input_logs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
            
        # 6. Update skor prediksi final hasil AI pada hari terakhir
        df_logs.loc[df_logs["log_id"] == reconciled_log["log_id"], "productivity_score"] = float(score)
        df_logs.loc[df_logs["log_id"] == reconciled_log["log_id"], "fatigue_accumulation"] = float(metrics.get("last_fatigue", 0.0))
        save_user_logs(df_logs)
        
        # Reload database final untuk penyusunan komponen grafik dashboard
        df_logs = load_user_logs()
        user_logs_final = df_logs[df_logs["user_id"] == user_id].sort_values("log_date")
        last_7_logs_final = user_logs_final.tail(7)
        
        reconciled_log_final = {field: float(last_7_logs_final.iloc[-1][field]) for field in ACTIVITY_FIELDS}
        reconciled_log_final["completion_ratio"] = float(last_7_logs_final.iloc[-1]["completion_ratio"])
        
        components = build_response_components(
            score, category, probabilities, metrics,
            reconciled_log_final, log_date, user_logs_final, last_7_logs_final,
            self.genai_service
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "days_logged": num_logs,
            "message": "Productivity analysis successfully processed over 7-day flexible input.",
            **components
        }
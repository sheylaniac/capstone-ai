import os
from datetime import datetime
from fastapi import HTTPException, Header
from src.schemas.prediction_schema import AIPredictionPayload
from src.services.ai_inference_service import AIInferenceService
from src.services.genai_service import GenAIService

# Ambil token dari .env (Samakan dengan yang ada di Node.js)
SECRET_TOKEN = os.getenv("SECRET_TOKEN_AI", "fallback_token_jika_kosong")
print("ini token: ", SECRET_TOKEN)

class PredictionController:
    def __init__(self, inference_service: AIInferenceService, genai_service: GenAIService):
        self.inference_service = inference_service
        self.genai_service = genai_service

    async def process_prediction(self, payload: AIPredictionPayload, authorization: str = Header(None)):
        # 1. Gatekeeper: Cek Token
        if authorization != f"Bearer {SECRET_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized - Akses AI Ditolak")

        # 2. Validasi Jumlah Data
        if len(payload.last_7_logs) != 7:
            raise HTTPException(status_code=400, detail="AI membutuhkan tepat 7 data historis.")

        try:
            # 3. Format Data untuk Mesin AI
            raw_logs = []
            for log in payload.last_7_logs:
                log_dict = log.model_dump()
                
                # Ekstrak Bulan & Hari untuk kebutuhan fitur LSTM
                dt = datetime.strptime(str(log_dict["log_date"]).split("T")[0], "%Y-%m-%d")
                log_dict["day_of_week"] = dt.weekday()
                log_dict["month"] = dt.month
                log_dict["is_weekend"] = int(log_dict["is_weekend"])
                
                # Mapping nama kolom dari Node.js (completion_ratio) ke model AI (task_completion_rate)
                log_dict["task_completion_rate"] = log_dict.pop("completion_ratio")
                
                raw_logs.append(log_dict)

            # 4. Eksekusi Prediksi LSTM
            score, category, probabilities, metrics = self.inference_service.predict(raw_logs)

            # 5. Eksekusi Rekomendasi Gemini (Dengan Target User Goals)
            user_goals_dict = payload.user_goals.model_dump()
            recommendation = self.genai_service.get_recommendation(category, score, metrics, user_goals_dict)

            # Pecah rekomendasi jadi 4 paragraf rapi
            paragraphs = [p.strip() for p in recommendation.split("\n\n") if p.strip()]
            while len(paragraphs) < 4:
                paragraphs.append("Insufficient data for this analysis section.")

            # 6. Susun Respons JSON
            return {
                "status": "success",
                "data": {
                    "user_id": payload.user_id,
                    "prediction": {
                        "productivity_status": category,
                        "productivity_score": round(score, 2),
                        "probabilities": probabilities,
                        "fatigue_level": round(metrics.get("last_fatigue", 0.0), 2),
                    },
                    "ai_insight": {
                        "condition_insight": paragraphs[0],
                        "performance_cause": paragraphs[1],
                        "activity_recommendation": paragraphs[2],
                        "burnout_warning": paragraphs[3]
                    }
                }
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Gagal memproses prediksi: {str(e)}")

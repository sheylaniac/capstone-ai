import os
from datetime import datetime
from fastapi import HTTPException, Header
from src.schemas.prediction_schema import AIPredictionPayload
from src.services.ai_inference_service import AIInferenceService
from src.services.genai_service import GenAIService

SECRET_TOKEN = os.getenv("SECRET_TOKEN_AI", "fallback_token_jika_kosong")

class PredictionController:
    def __init__(self, inference_service: AIInferenceService, genai_service: GenAIService):
        self.inference_service = inference_service
        self.genai_service = genai_service

    async def process_prediction(self, payload: AIPredictionPayload, authorization: str = Header(None)):
        if authorization != f"Bearer {SECRET_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized - AI Access Denied")

        if len(payload.last_7_logs) != 7:
            raise HTTPException(status_code=400, detail="AI requires exactly 7 historical data logs.")

        try:
            raw_logs = []
            for log in payload.last_7_logs:
                log_dict = log.model_dump()
                log_dict["is_weekend"] = int(log_dict["is_weekend"])
                raw_logs.append(log_dict)

            score, category, probabilities, metrics = self.inference_service.predict(raw_logs)

            def get_category_from_score(s: float) -> str:
                if s < 55.0:
                    return "At Risk"
                elif s < 70.0:
                    return "Steady"
                else:
                    return "Thriving"

            category = get_category_from_score(score)

            recommendation = self.genai_service.get_recommendation(category, score, metrics)

            paragraphs = [p.strip() for p in recommendation.split("\n\n") if p.strip()]
            while len(paragraphs) < 5:
                paragraphs.append("Insufficient data for this analysis section.")

            grafik_data = []
            heatmap_data = {}
            days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            historical_days = []
            
            for idx, log in enumerate(payload.last_7_logs):
                try:
                    dt_str = log.log_date.split("T")[0]
                    dt_obj = datetime.strptime(dt_str, "%Y-%m-%d")
                    eng_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    date_label = f"{eng_months[dt_obj.month - 1]} {dt_obj.day}"
                    day_name = days_of_week[dt_obj.weekday()]
                except Exception:
                    date_label = f"Day {idx+1}"
                    day_name = days_of_week[idx % 7]
                                
                completion_ratio = log.completion_ratio if log.completion_ratio is not None else (log.task_completed / max(1, log.task_planned))
                day_score = float(completion_ratio * 40 + log.focus_score * 4 + (10 - log.stress_level) * 2)
                day_score = round(max(0.0, min(100.0, day_score)), 2)
                                
                grafik_data.append({"date": date_label, "score": day_score})
                heatmap_data[day_name] = day_score
                                
                diff = abs(day_score - score)
                similarity_pct = max(0.0, 100.0 - diff)
                                
                historical_days.append({
                    "date": date_label,
                    "similarity_score": f"{round(similarity_pct, 1)}%",
                    "similarity_val": similarity_pct,
                    "historical_productivity_score": day_score,
                    "historical_category": get_category_from_score(day_score)
                })

            if grafik_data:
                grafik_data[-1]["score"] = round(score, 2)
            heatmap_data[day_name] = round(score, 2)
                        
            avg_focus = sum(l.focus_score for l in payload.last_7_logs) / 7
            avg_sleep = sum(l.sleep_duration for l in payload.last_7_logs) / 7

            if avg_focus >= 7.5:
                jam_produktif = "09:00-11:00" if avg_sleep >= 7.0 else "10:00-12:00"
            elif avg_focus >= 6.0:
                jam_produktif = "09:00-12:00" if avg_sleep >= 6.0 else "13:00-15:00"
            else:
                jam_produktif = "19:00-21:00" if avg_sleep < 5.5 else "14:00-16:00"
                        
            avg_work = sum(l.study_work_duration for l in payload.last_7_logs) / 7
            avg_break = sum(l.break_duration for l in payload.last_7_logs) / 7
            avg_exercise = sum(l.exercise_duration for l in payload.last_7_logs) / 7
            avg_downtime = sum(l.downtime_duration for l in payload.last_7_logs) / 7
                        
            activities = {
                "Sleep": avg_sleep,
                "Study/Work": avg_work,
                "Break": avg_break,
                "Exercise": avg_exercise,
                "Downtime": avg_downtime
            }
            most_dominant_activity = max(activities, key=activities.get)
                        
            historical_days_sorted = sorted(historical_days, key=lambda x: x["similarity_val"], reverse=True)
            top_3 = historical_days_sorted[:3]
                        
            top_3_formatted = [
                {
                    "date": item["date"],
                    "similarity_score": item["similarity_score"] if item["similarity_val"] < 100.0 else "100%",
                    "historical_productivity_score": item["historical_productivity_score"],
                    "historical_category": item["historical_category"]
                }
                for item in top_3
            ]
            similar_history = {
                "top_3_similar_days": top_3_formatted,
                "average_productivity_from_similar_days": round(sum(item["historical_productivity_score"] for item in top_3) / len(top_3), 2)
            }

            return {
                "success": True,
                "user_id": payload.last_7_logs[0].user_id,
                "days_logged": len(payload.last_7_logs),
                "message": "Productivity analysis successfully processed over 7-day flexible input.",
                "1_main_dashboard": {
                    "productivity_status": category,
                    "productivity_score": round(score, 2),
                    "prediction_confidence": round(probabilities.get(category, 0.0) * 100, 2),
                    "probabilities": probabilities,
                    "fatigue_level": round(metrics.get("last_fatigue", 0.0), 2),
                    "completion_rate": round(sum(log.completion_ratio if log.completion_ratio is not None else (log.task_completed / max(1, log.task_planned)) for log in payload.last_7_logs) / 7 * 100, 2),
                    "risk_signal": "HIGH RISK" if category == "At Risk" else ("WARNING" if category == "Steady" else "NORMAL")
                },
                "2_productivity_analytics_dashboard": {
                    "daily_productivity_chart": grafik_data,
                    "weekly_productivity_trend": "Increasing" if category == "Thriving" else ("Decreasing" if category == "At Risk" else "Stable"),
                    "most_dominant_activity": most_dominant_activity,
                    "peak_productive_hours": jam_produktif,
                    "activity_heatmap": heatmap_data
                },
                "3_ai_insight_and_recommendation": {
                    "condition_insight": paragraphs[0],
                    "performance_cause": paragraphs[1],
                    "activity_recommendation": paragraphs[2],
                    "tomorrow_prediction": paragraphs[3],
                    "burnout_warning": paragraphs[4],
                    "recommendation_text": recommendation
                },
                "4_similar_productivity_history": similar_history
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to process prediction: {str(e)}")
import os
import google.generativeai as genai

class GenAIService:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.gemini_available = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                self.gemini_available = True
                print("Gemini API configured successfully.")
            except Exception as e:
                print(f"Failed to initialize Gemini API: {e}")
        else:
            print("GEMINI_API_KEY not found. Using fallback.")

    def get_recommendation(self, category: str, score: float, metrics: dict, user_goals: dict) -> str:
        
        # Menerjemahkan Boolean Goals ke Teks
        active_goals = []
        if user_goals.get('focus_sleep'): active_goals.append("Sleep Quality & Duration")
        if user_goals.get('focus_productivity'): active_goals.append("Work/Study Productivity")
        if user_goals.get('focus_fitness'): active_goals.append("Physical Activity")
        if user_goals.get('focus_screen_time'): active_goals.append("Screen Time Management")
        
        focus_str = ", ".join(active_goals) if active_goals else "General Well-being"

        if self.gemini_available:
            try:
                prompt = (
                    f"Act as an analytical and professional AI Personal Health and Productivity Assistant.\n"
                    f"USER TARGET GOALS: The user explicitly wants to improve: {focus_str}.\n\n"
                    
                    f"Prediction for tomorrow:\n"
                    f"- Productivity Score: {score:.2f}% (scale 0-100)\n"
                    f"- Condition Category: {category}\n\n"
                    
                    f"Last 7 days average:\n"
                    f"- Sleep: {metrics.get('avg_sleep', 7.0):.1f} hours (quality: {metrics.get('avg_sleep_quality', 7.0):.1f}/10)\n"
                    f"- Work/study: {metrics.get('avg_work', 6.0):.1f} hours\n"
                    f"- Stress level: {metrics.get('avg_stress', 5.0):.1f}/10\n"
                    f"- Screen time: {metrics.get('avg_screen_time', 4.0):.1f} hours\n"
                    f"- Fatigue accumulation: {metrics.get('last_fatigue', 2.0):.2f}\n\n"
                    
                    f"Write a 4-paragraph recommendation report (use \\n\\n between paragraphs). DO NOT use markdown/bullet points:\n"
                    f"1. Condition Insight: State the category and score objectively.\n"
                    f"2. Root Cause: Explain the primary metric driving this score.\n"
                    f"3. Activity Recommendation: Provide concrete advice. CRITICAL: Tailor this advice specifically toward the user's target goals ({focus_str}).\n"
                    f"4. Burnout Warning: Provide a measured warning if the current pattern continues."
                )
                
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Gemini error: {e}. Falling back.")
                
        return self._get_fallback_recommendation(category, score, metrics)

    def _get_fallback_recommendation(self, category: str, score: float, metrics: dict) -> str:
        # (Isi fallback logic kamu tetap sama seperti sebelumnya)
        return (
            f"Prediction: {category} ({score:.2f}%).\n\n"
            f"Metrics indicate a shift in performance based on your recent 7-day pattern.\n\n"
            f"Please adjust your daily routine to balance work duration and recovery periods.\n\n"
            f"Monitor your fatigue levels to prevent burnout."
        )
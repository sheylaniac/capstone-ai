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
                    "Act as an analytical, professional, and objective AI Personal Health and Productivity Assistant.\n"
                    f"USER TARGET GOALS: The user explicitly wants to improve: {focus_str}.\n\n"
                    "You are provided with the user's activity data from the last 7 days as the basis for your analysis:\n\n"
                    "Activity Data for the Last 7 Days:\n"
                    f"- Weekly Productivity Score: {score:.2f}% (scale 0-100)\n"
                    f"- Current Condition Category: {category}\n"
                    f"- Average sleep: {metrics.get('avg_sleep', 7.0):.1f} hours (quality: {metrics.get('avg_sleep_quality', 7.0):.1f}/10)\n"
                    f"- Average work/study hours: {metrics.get('avg_work', 6.0):.1f} hours\n"
                    f"- Average stress level: {metrics.get('avg_stress', 5.0):.1f}/10\n"
                    f"- Average screen time/downtime: {metrics.get('avg_downtime', 4.0):.1f} hours\n"
                    f"- Fatigue accumulation: {metrics.get('last_fatigue', 2.0):.2f}\n\n"
                    "Write a recommendation report based on the following requirements:\n"
                    "- Use formal, professional, and analytical English.\n"
                    "- DO NOT use casual greetings, numbering, bullet points, markdown formatting, or any symbols.\n"
                    "- MUST consist of exactly 5 paragraphs, separated by double newlines (\\n\\n).\n\n"
                    "- Keep each paragraph between 40 and 60 words.\n"
                    "Paragraph 1: Explain the user's current condition based on the weekly productivity score.\n"
                    "Paragraph 2: Analyze the root causes based on the provided 7-day data summary.\n"
                    f"Paragraph 3: Provide concrete, measurable activity recommendations tailored to user goals ({focus_str}) to improve performance.\n"
                    "Paragraph 4: Provide an estimated productivity score prediction for tomorrow.\n"
                    "Paragraph 5: Provide a warning regarding burnout or fatigue risks if the current activity pattern persists."
                )
                
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Gemini error: {e}. Falling back.")
                
        return self._get_fallback_recommendation(category, score, metrics)

    def _get_fallback_recommendation(self, category: str, score: float, metrics: dict) -> str:
        pred = min(100, score * 1.05) if category == 'Thriving' else (score * 0.95 if category == 'At Risk' else score)

        if category == 'Thriving':
            return (f"Current condition is prime (Thriving) with a weekly productivity score of {score:.2f}%. This performance reflects excellent stability over the past week.\n\n"
                    "This optimization is driven by ideal sleep duration and highly controlled stress management.\n\n"
                    "It is recommended to maintain the current operational pacing while consistently integrating active micro-breaks to preserve energy.\n\n"
                    f"The projected productivity score for tomorrow is {pred:.2f}%.\n\n"
                    "Continued adherence to this routine is crucial to stabilize energy levels and mitigate the risk of latent fatigue accumulation.")

        elif category == 'Steady':
            return (f"Current condition is stable (Steady) with a weekly productivity score of {score:.2f}%. Performance levels are currently well-maintained.\n\n"
                    "Prolonged screen exposure remains the primary factor potentially compromising physiological recovery.\n\n"
                    "Implementing a strict screen restriction for one hour before nocturnal rest is advised to improve sleep efficiency.\n\n"
                    f"The projected productivity score for tomorrow is {pred:.2f}%.\n\n"
                    "This preventive measure is essential to forestall stamina depletion and prevent a decline into higher-risk fatigue zones.")

        else:  # At Risk
            return (f"Current condition is high-risk (At Risk) with a weekly productivity score of {score:.2f}%. A decline in performance requires immediate intervention.\n\n"
                    "This indicator drop stems from significant fatigue accumulation, correlated with insufficient sleep and elevated stress indexes.\n\n"
                    "The immediate priority entails reducing computational workloads tonight and dedicating remaining hours to physiological recovery.\n\n"
                    f"The projected productivity score for tomorrow is {pred:.2f}%.\n\n"
                    "Disregarding these recovery signals will accelerate acute burnout and cause a substantial degradation in overall cognitive capabilities.")
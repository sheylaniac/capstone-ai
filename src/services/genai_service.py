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
                print("Gemini API configured successfully in GenAIService.")
            except Exception as e:
                print(f"Failed to initialize Gemini API in GenAIService: {e}")
        else:
            print("GEMINI_API_KEY not found in environment. Using rule-based fallback recommendations.")

    def get_recommendation(self, category: str, score: float, metrics: dict) -> str:

        if self.gemini_available:
            try:
                prompt = (
                    f"Act as an analytical and professional AI Personal Health and Productivity Assistant.\n"
                    f"Given the user's prediction data for tomorrow based on the last 7 days of activity logs:\n\n"
                    f"Prediction for tomorrow:\n"
                    f"- Productivity Score: {score:.2f}% (scale 0-100)\n"
                    f"- Condition Category: {category}\n\n"
                    f"Summary of the last 7 days of activity:\n"
                    f"- Average sleep: {metrics.get('avg_sleep', 7.0):.1f} hours (quality: {metrics.get('avg_sleep_quality', 7.0):.1f}/10)\n"
                    f"- Average work/study hours: {metrics.get('avg_work', 6.0):.1f} hours\n"
                    f"- Average stress level: {metrics.get('avg_stress', 5.0):.1f}/10\n"
                    f"- Average screen time: {metrics.get('avg_screen_time', 4.0):.1f} hours\n"
                    f"- Fatigue accumulation: {metrics.get('last_fatigue', 2.0):.2f}\n\n"
                    f"Write a recommendation report with the following strict constraints:\n"
                    f"- Use formal, professional, objective, and analytical English.\n"
                    f"- DO NOT use casual greetings (e.g., Hi, Hello) or informal pronouns.\n"
                    f"- DO NOT use numbering, bullet points, markdown formatting (*, #), or any symbols.\n"
                    f"- CRITICAL: Separate each core point into its own paragraph using double newlines (\\n\\n), resulting in exactly 4 distinct lines/paragraphs as follows:\n"
                    f"  1. Paragraph 1 (User Condition Insight): Objectively state the predicted condition category (At Risk / Steady / Thriving) and the productivity score ({score:.2f}%).\n"
                    f"  2. Paragraph 2 (Root Cause Analysis): Explain the primary factor driving the performance trend based on the provided metrics (such as sleep duration, screen time, or stress levels).\n"
                    f"  3. Paragraph 3 (Activity Recommendation): Provide specific, actionable, and concrete behavioral recommendations to address the underlying data metrics.\n"
                    f"  4. Paragraph 4 (Burnout or Fatigue Warning): Deliver a measured warning regarding the risks of physical fatigue or mental burnout if the current activity pattern persists."
                )
                
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Gemini API generation error: {e}. Falling back to rules.")
                err_msg = str(e).lower()
                if any(x in err_msg for x in ["quota", "exhausted", "429", "limit", "api key", "invalid", "blocked"]):
                    print("Temporarily disabling Gemini API service due to quota/rate-limiting/invalid key/blocked content.")
                    self.gemini_available = False
        
        return self._get_fallback_recommendation(category, score, metrics)

    def _get_fallback_recommendation(self, category: str, score: float, metrics: dict) -> str:
        if category == 'Thriving':
            return (
                f"Activity log evaluation predicts a prime condition (Thriving) with a projected productivity score of {score:.2f}% for tomorrow.\n\n"
                "This performance optimization is driven by an ideal daily sleep duration and highly controlled stress management over the past week.\n\n"
                "It is recommended to maintain the current operational pacing while consistently integrating active micro-breaks throughout daily tasks.\n\n"
                "This sustained effort is crucial to stabilize energy levels and mitigate the risk of latent fatigue accumulation moving forward."
            )
        elif category == 'Steady':
            return (
                f"Data analysis indicates that tomorrow's condition is projected to remain stable (Steady) with an estimated productivity score of {score:.2f}%.\n\n"
                "While current performance levels are well-maintained, prolonged exposure to screen time requires careful monitoring as it may compromise physiological recovery.\n\n"
                "A tactical restriction of screen usage for at least one hour prior to nocturnal rest is highly advised to improve sleep efficiency.\n\n"
                "This preventive measure is essential to forestall macro-stamina depletion and prevent a decline into higher-risk fatigue zones."
            )
        else:  # At Risk
            return (
                f"Projections detect a high-risk condition (At Risk) with a decline in tomorrow's predicted productivity score to {score:.2f}%.\n\n"
                "This indicator drop stems from a significant accumulation of fatigue, directly correlating with deficient sleep duration and elevated daily stress indexes.\n\n"
                "The immediate priority entails reducing computational workloads tonight and dedicating the remaining hours entirely to physiological recovery.\n\n"
                "Disregarding these recovery signals will accelerate acute burnout and cause a substantial degradation in overall cognitive capabilities."
            )
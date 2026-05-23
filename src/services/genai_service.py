import os
import google.generativeai as genai


class GenAIService:

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.gemini_available = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
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
                    f"Bertindaklah sebagai Asisten Kesehatan dan Produktivitas Personal berbasis AI.\n"
                    f"Diberikan data prediksi pengguna untuk hari esok berdasarkan log aktivitas 7 hari terakhir:\n"
                    f"- Prediksi Skor Produktivitas: {score:.2f}% (skala 0-100)\n"
                    f"- Prediksi Kategori Kondisi: {category} (At Risk / Steady / Thriving)\n\n"
                    f"Rangkuman aktivitas 7 hari terakhir:\n"
                    f"- Rata-rata Tidur: {metrics.get('avg_sleep', 7.0):.1f} jam dengan Kualitas Tidur: {metrics.get('avg_sleep_quality', 7.0):.1f}/10\n"
                    f"- Rata-rata Jam Kerja/Belajar: {metrics.get('avg_work', 6.0):.1f} jam\n"
                    f"- Rata-rata Tingkat Stres: {metrics.get('avg_stress', 5.0):.1f}/10\n"
                    f"- Rata-rata Screen Time: {metrics.get('avg_screen_time', 4.0):.1f} jam\n"
                    f"- Akumulasi Fatigue (Kelelahan): {metrics.get('last_fatigue', 2.0):.2f}\n\n"
                    f"Berikan rekomendasi kesehatan dan produktivitas yang personal, praktis, dan memotivasi dalam 3-4 kalimat singkat dalam bahasa Indonesia. Berikan tips konkret sesuai dengan kategori status mereka."
                )
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Gemini API generation error: {e}. Falling back to rules.")
        
        return self._get_fallback_recommendation(category, score, metrics)

    def _get_fallback_recommendation(self, category: str, score: float, metrics: dict) -> str:
        avg_sleep = metrics.get('avg_sleep', 7.0)
        avg_stress = metrics.get('avg_stress', 5.0)
        
        if category == 'Thriving':
            return (
                f"Berdasarkan analisis log 7 hari terakhir, Anda berada dalam kondisi prima (Thriving) dengan "
                f"skor produktivitas besok yang diprediksi sebesar {score:.2f}%. Pola tidur Anda sangat baik "
                f"(rata-rata {avg_sleep:.1f} jam) dengan tingkat stres yang rendah (rata-rata {avg_stress:.1f}/10). "
                f"Rekomendasi: Pertahankan ritme kerja saat ini, lakukan istirahat aktif secara konsisten, dan "
                f"hindari peningkatan beban kerja secara mendadak agar terhindar dari kelelahan di masa mendatang."
            )
        elif category == 'Steady':
            return (
                f"Berdasarkan analisis log 7 hari terakhir, Anda berada dalam kondisi stabil (Steady) dengan "
                f"skor produktivitas besok sebesar {score:.2f}%. Pola tidur Anda cukup (rata-rata {avg_sleep:.1f} jam) "
                f"dengan tingkat stres sedang (rata-rata {avg_stress:.1f}/10). Rekomendasi: Anda berkinerja dengan baik, "
                f"namun pastikan untuk menjaga keseimbangan antara bekerja dan beristirahat. Sisipkan jeda singkat (break) "
                f"5-10 menit setiap 90 menit bekerja untuk menyegarkan pikiran."
            )
        else:  # At Risk
            return (
                f"PERINGATAN KRITIS! Berdasarkan log 7 hari terakhir, Anda berada dalam kondisi berisiko tinggi (At Risk) "
                f"burnout dengan skor produktivitas besok yang diprediksi menurun ke {score:.2f}%. Pola tidur Anda kurang "
                f"(rata-rata {avg_sleep:.1f} jam) dengan tingkat stres yang tinggi (rata-rata {avg_stress:.1f}/10). "
                f"Rekomendasi: Segera kurangi beban kerja Anda hari ini. Prioritaskan tidur minimal 7-8 jam malam ini, "
                f"batasi screen time di luar jam kerja, dan luangkan waktu untuk relaksasi total guna memulihkan kondisi fisik Anda."
            )

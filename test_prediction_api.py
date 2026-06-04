import time
import requests
import subprocess
import sys
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

def run_integration_tests():
    print("=" * 70)
    print("   [TEST] STARTING SMART DIGITAL TWIN INTEGRATION TEST")
    print("=" * 70)
    
    # 1. Start uvicorn server in background
    python_exe = "D:\\anaconda\\envs\\tf-env\\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable
    print(f"Using Python: {python_exe}")
    
    server_port = "8005"
    proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "src.app:app", "--port", server_port],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 2. Wait for server to become online
    print("Waiting for uvicorn server to start...")
    started = False
    for _ in range(15):
        if proc.poll() is not None:
            print("Error: Server failed to start immediately.")
            break
        try:
            res = requests.get(f"http://127.0.0.1:{server_port}/")
            if res.status_code == 200:
                started = True
                print("Server is ONLINE.")
                print(f"Server response: {res.json()}")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        
    if not started:
        print("Error: Server failed to start.")
        proc.terminate()
        return

    # 3. Construct a valid payload
    user_id = "test_user_123"
    
    import random
    
    # Pilih skenario secara acak: 0=At Risk, 1=Steady, 2=Thriving
    scenario = random.choice([0, 1, 2])
    scenario_names = {0: "At Risk (Kondisi Buruk)", 1: "Steady (Kondisi Normal)", 2: "Thriving (Kondisi Prima)"}
    print(f"\n[TEST INFO] Menjalankan pengujian dengan Skenario Acak: {scenario_names[scenario]}")
    
    from datetime import datetime, timedelta
    last_7_logs = []
    for i in range(7):
        target_date = datetime.now() - timedelta(days=6 - i)
        log_date_str = target_date.strftime("%Y-%m-%dT00:00:00.000Z")
        is_weekend_val = target_date.weekday() >= 5

        if scenario == 0:  # At Risk
            sleep = 4.5
            work = 10.0
            breaks = 0.2
            exercise = 0.0
            downtime = 6.0
            stress = 9
            mood = 3
            focus = 3
            planned = 10
            completed = 3
            ratio = 0.3
            fatigue = 85.0
            cum_fatigue = 80.0
        elif scenario == 1:  # Steady
            sleep = 6.5
            work = 7.0
            breaks = 1.0
            exercise = 15.0
            downtime = 3.5
            stress = 5
            mood = 6
            focus = 6
            planned = 6
            completed = 4
            ratio = 0.67
            fatigue = 45.0
            cum_fatigue = 45.0
        else:  # Thriving
            sleep = 8.0
            work = 5.0
            breaks = 2.0
            exercise = 45.0
            downtime = 1.5
            stress = 2
            mood = 9
            focus = 9
            planned = 5
            completed = 5
            ratio = 1.0
            fatigue = 15.0
            cum_fatigue = 15.0

        last_7_logs.append({
            "log_date": log_date_str,
            "is_weekend": is_weekend_val,
            "sleep_duration": sleep,
            "study_work_duration": work,
            "break_duration": breaks,
            "exercise_duration": exercise,
            "downtime_duration": downtime,
            "stress_level": stress,
            "mood_score": mood,
            "focus_score": focus,
            "task_planned": planned,
            "task_completed": completed,
            "completion_ratio": ratio,
            "fatigue_index": fatigue,
            "cumulative_fatigue": cum_fatigue
        })
        
    payload = {
        "user_id": user_id,
        "last_7_logs": last_7_logs
    }
    
    # Auth headers
    token = os.getenv("SECRET_TOKEN_AI", "fallback_token_jika_kosong")
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        # Test Case 1: Unauthorized access
        print("\n--- TEST CASE 1: Unauthorized request (no token) ---")
        res = requests.post(f"http://127.0.0.1:{server_port}/api/v1/predict", json=payload)
        print(f"Status Code: {res.status_code} (Expected: 401)")
        print(f"Response: {res.json()}")
        
        # Test Case 2: Authorized prediction with correct schema
        print("\n--- TEST CASE 2: Authorized prediction with 14-feature schema ---")
        res = requests.post(f"http://127.0.0.1:{server_port}/api/v1/predict", json=payload, headers=headers)
        print(f"Status Code: {res.status_code} (Expected: 200)")
        if res.status_code == 200:
            data = res.json()
            print("Response successfully received:")
            print(f"   Success Status: {data.get('success')}")
            print(f"   User ID: {data.get('user_id')}")
            print(f"   Days Logged: {data.get('days_logged')}")
            print(f"   Message: {data.get('message')}")
            
            main_dash = data.get('1_main_dashboard', {})
            print(f"   Main Dashboard:")
            print(f"      * Productivity Status: {main_dash.get('productivity_status')}")
            print(f"      * Productivity Score: {main_dash.get('productivity_score')}%")
            print(f"      * Prediction Confidence: {main_dash.get('prediction_confidence')}%")
            print(f"      * Probabilities: {main_dash.get('probabilities')}")
            print(f"      * Fatigue Level: {main_dash.get('fatigue_level')}")
            print(f"      * Completion Rate: {main_dash.get('completion_rate')}%")
            print(f"      * Risk Signal: {main_dash.get('risk_signal')}")
 
            analytics_dash = data.get('2_productivity_analytics_dashboard', {})
            print("   Productivity Analytics Dashboard:")
            print(f"      * Daily graph (last day score): {analytics_dash.get('daily_productivity_chart')[-1] if analytics_dash.get('daily_productivity_chart') else 'N/A'}")
            print(f"      * Weekly trend: {analytics_dash.get('weekly_productivity_trend')}")
            print(f"      * Dominant activity: {analytics_dash.get('most_dominant_activity')}")
            print(f"      * Most productive hours: {analytics_dash.get('peak_productive_hours')}")
            
            ai_insight = data.get('3_ai_insight_and_recommendation', {})
            print("   Gemini recommendation (5 paragraphs):")
            print(f"      [1] Insight: {ai_insight.get('condition_insight')[:70]}...")
            print(f"      [2] Root Cause: {ai_insight.get('performance_cause')[:70]}...")
            print(f"      [3] Recs: {ai_insight.get('activity_recommendation')[:70]}...")
            print(f"      [4] Tomorrow: {ai_insight.get('tomorrow_prediction')[:70]}...")
            print(f"      [5] Burnout: {ai_insight.get('burnout_warning')[:70]}...")
            
            # Print new fields
            similar_hist = data.get('4_similar_productivity_history', {})
            print("   Similar Productivity History:")
            print(f"      * Top 3 similar days: {similar_hist.get('top_3_similar_days')}")
            print(f"      * Average productivity: {similar_hist.get('average_productivity_from_similar_days')}%")
        else:
            print(f"Test failed with: {res.text}")
            
    except Exception as e:
        print(f"Exception during testing: {e}")
    finally:
        print("\nStopping FastAPI server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("FastAPI server stopped.")
        print("=" * 70)

if __name__ == "__main__":
    run_integration_tests()

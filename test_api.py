import time
import requests
import subprocess
import sys
import random
import os

def run_tests():
    print("Starting FastAPI server...")
    python_exe = "D:\\anaconda\\envs\\tf-env\\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable
    print(f"Using python executable: {python_exe}")
    proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "src.app:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print("Waiting for server to start (polling endpoint)...")
    started = False
    for attempt in range(35):
        if proc.poll() is not None:
            print("Error: FastAPI server failed to start (process exited).")
            break
        try:
            res = requests.get("http://127.0.0.1:8000/", timeout=1)
            if res.status_code == 200:
                started = True
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        
    if not started:
        print("Error: FastAPI server did not start in time or crashed.")
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
        except Exception:
            pass
        return
        
    print("Server started. Sending test requests...")
    
    # Define scenarios for POST /api/v1/predict (full manual sequence)
    logs_thriving = []
    for i in range(7):
        logs_thriving.append({
            'sleep_duration': 8.0,
            'sleep_quality': 9.0,
            'study_work_duration': 6.0,
            'break_duration': 1.0,
            'physical_activity_duration': 45.0,
            'screen_time_duration': 2.0,
            'stress_level': 2.0,
            'mood_score': 9.0,
            'focus_score': 9.0,
            'task_planned': 5,
            'task_completed': 5,
            'is_weekend': 1 if (i % 7 in [5, 6]) else 0,
            'log_date': f"2026-05-{10+i}"
        })
        
    logs_steady = []
    for i in range(7):
        logs_steady.append({
            'sleep_duration': 6.5,
            'sleep_quality': 6.0,
            'study_work_duration': 7.5,
            'break_duration': 0.5,
            'physical_activity_duration': 20.0,
            'screen_time_duration': 4.0,
            'stress_level': 5.0,
            'mood_score': 6.0,
            'focus_score': 7.0,
            'task_planned': 6,
            'task_completed': 4,
            'is_weekend': 1 if (i % 7 in [5, 6]) else 0,
            'log_date': f"2026-05-{10+i}"
        })
        
    logs_at_risk = []
    for i in range(7):
        logs_at_risk.append({
            'sleep_duration': 4.5,
            'sleep_quality': 3.0,
            'study_work_duration': 10.0,
            'break_duration': 0.167,
            'physical_activity_duration': 0.0,
            'screen_time_duration': 8.0,
            'stress_level': 9.0,
            'mood_score': 2.0,
            'focus_score': 3.0,
            'task_planned': 8,
            'task_completed': 2,
            'is_weekend': 1 if (i % 7 in [5, 6]) else 0,
            'log_date': f"2026-05-{10+i}"
        })
        
    scenarios = {
        "Thriving Scenario": logs_thriving,
        "Steady Scenario": logs_steady,
        "At Risk Scenario": logs_at_risk
    }
    
    try:
        root_res = requests.get("http://127.0.0.1:8000/")
        print("Root Endpoint Response:", root_res.json())
        
        test_user_id = random.randint(100000, 999999)
        print(f"\n--- 1. Testing Endpoint: /predict (Full) with NO history for user_id={test_user_id} ---")
        url_predict = "http://127.0.0.1:8000/api/v1/predict"
        
        single_log = {
            "sleep_duration": 8.0,
            "sleep_quality": 8.0,
            "study_work_duration": 7.0,
            "break_duration": 1.0,
            "physical_activity_duration": 30.0,
            "screen_time_duration": 3.0,
            "stress_level": 3.0,
            "mood_score": 8.0,
            "focus_score": 8.0,
            "task_planned": 6,
            "task_completed": 5,
            "is_weekend": 0,
            "log_date": "2026-05-28"
        }
        
        payload_predict = {
            "user_id": test_user_id,
            "current_log": single_log
        }
        
        res_predict = requests.post(url_predict, json=payload_predict)
        print(f"Status Code (Expected 400): {res_predict.status_code}")
        print(f"Response (Expected Error): {res_predict.text}")
        
        print(f"\n--- 2. Testing Endpoint: /predict-from-logs for user_id={test_user_id} to populate history ---")
        url_from_logs = "http://127.0.0.1:8000/api/v1/predict-from-logs"
        payload_from_logs = {"user_id": test_user_id}
        res_from_logs = requests.post(url_from_logs, json=payload_from_logs)
        if res_from_logs.status_code == 200:
            data = res_from_logs.json()
            print("Status Code: 200 OK")
            print(f"Success Status: {data.get('success')}")
            print(f"Days Logged: {data.get('days_logged')}")
            print(f"Message: {data.get('message')}")
        else:
            print(f"Error Code {res_from_logs.status_code}: {res_from_logs.text}")

        print(f"\n--- 3. Testing Endpoint: /predict (Full) again with populated history for user_id={test_user_id} ---")
        payload_predict["current_log"]["log_date"] = "2026-05-29" # Next day log
        res_predict_2 = requests.post(url_predict, json=payload_predict)
        if res_predict_2.status_code == 200:
            data = res_predict_2.json()
            print("Status Code: 200 OK")
            print(f"Success Status: {data.get('success')}")
            print(f"Days Logged (Expected 8): {data.get('days_logged')}")
            main_db = data.get('1_main_dashboard', {})
            print(f"Prediction Score: {main_db.get('productivity_score')}")
            print(f"Category: {main_db.get('productivity_status')}")
            print(f"Confidence: {main_db.get('prediction_confidence')}%")
        else:
            print(f"Error Code {res_predict_2.status_code}: {res_predict_2.text}")

        print(f"\n--- 4. Testing Endpoint: /predict-flexible for user_id={test_user_id} ---")
        url_flexible = "http://127.0.0.1:8000/api/v1/predict-flexible"
        payload_flexible = {
            "user_id": test_user_id,
            "current_log": {
                "sleep_duration": 8.0,
                "sleep_quality": 8.0,
                "study_work_duration": 6.0,
                "log_date": "2026-05-30"
            }
        }
        res_flexible = requests.post(url_flexible, json=payload_flexible)
        if res_flexible.status_code == 200:
            data = res_flexible.json()
            print("Status Code: 200 OK")
            print(f"Success Status: {data.get('success')}")
            print(f"Days Logged (Expected 9): {data.get('days_logged')}")
            main_db = data.get('1_main_dashboard', {})
            reconciliation = data.get('5_activity_input_form_reconciliation', {})
            print(f"Prediction Score: {main_db.get('productivity_score')}")
            print(f"Imputation Status: {reconciliation.get('imputation_status')}")
        else:
            print(f"Error Code {res_flexible.status_code}: {res_flexible.text}")
                 
    except Exception as e:
        print("Exception occurred during requests:", e)
    finally:
        print("\nStopping FastAPI server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server stopped.")

if __name__ == '__main__':
    run_tests()

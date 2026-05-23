import time
import requests
import subprocess
import sys

def run_tests():
    # 1. Start uvicorn server in a subprocess
    print("Starting FastAPI server...")
    # Use python -m uvicorn main:app
    proc = subprocess.Popen(
        ["D:\\anaconda\\envs\\tf-env\\python.exe", "-m", "uvicorn", "main:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start by polling
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
        # Terminate and show output
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
        except Exception:
            pass
        return
        
    print("Server started. Sending test requests...")
    
    # Define scenarios
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
            'task_planned': 5.0,
            'task_completed': 5.0,
            'task_completion_rate': 1.0,
            'day_of_week': i % 7,
            'month': 5,
            'is_weekend': 1 if (i % 7 in [5, 6]) else 0
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
            'task_planned': 6.0,
            'task_completed': 4.0,
            'task_completion_rate': 0.667,
            'day_of_week': i % 7,
            'month': 5,
            'is_weekend': 1 if (i % 7 in [5, 6]) else 0
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
            'task_planned': 8.0,
            'task_completed': 2.0,
            'task_completion_rate': 0.25,
            'day_of_week': i % 7,
            'month': 5,
            'is_weekend': 1 if (i % 7 in [5, 6]) else 0
        })
        
    scenarios = {
        "Thriving Scenario": logs_thriving,
        "Steady Scenario": logs_steady,
        "At Risk Scenario": logs_at_risk
    }
    
    url = "http://127.0.0.1:8000/predict"
    
    try:
        # Check root endpoint
        root_res = requests.get("http://127.0.0.1:8000/")
        print("Root Endpoint Response:", root_res.json())
        
        for name, logs in scenarios.items():
            print(f"\n--- Testing {name} ---")
            payload = {"logs": logs}
            res = requests.post(url, json=payload)
            if res.status_code == 200:
                data = res.json()
                print("Status Code: 200 OK")
                print(f"Productivity Score: {data['predicted_productivity_score']:.2f}")
                print(f"Category: {data['predicted_category']}")
                print(f"Probabilities: {data['probabilities']}")
                print(f"Recommendation: {data['recommendation']}")
            else:
                print(f"Error Code {res.status_code}: {res.text}")
                
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

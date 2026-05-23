import os
import random
import numpy as np
import pandas as pd
import requests
import json

def main():
    print("=" * 60)
    print("[TEST] DTWIN-AI: INTEGRATION & SYSTEM TEST CLIENT")
    print("=" * 60)

    url = "http://127.0.0.1:8000/api/v1/predict"
    workspace = r"c:\Users\HP\OneDrive - unesa.ac.id\kuliah\SEMESTER 6\STUDI INDEPENDENT\CAPSTON EPROJECT\dtwin"
    
    # Paths to processed test arrays and raw test logs
    x_test_path = os.path.join(workspace, "data", "processed", "x_test.npy")
    y_test_reg_path = os.path.join(workspace, "data", "processed", "y_test_reg.npy")
    y_test_clf_path = os.path.join(workspace, "data", "processed", "y_test_clf.npy")
    raw_test_df_path = os.path.join(workspace, "data", "processed", "test_raw_dataframe.csv")

    # 1. Check if server is running
    try:
        health = requests.get("http://127.0.0.1:8000/")
        health.raise_for_status()
        print("Server status: ONLINE")
        print(f"   Model Version: {health.json().get('model_version', 'N/A')}")
    except requests.exceptions.ConnectionError:
        print("Server status: OFFLINE")
        print("\n   [PENTING] Mohon jalankan server FastAPI Anda terlebih dahulu di terminal terpisah:")
        print("   Perintah: uvicorn src.app:app --reload")
        print("=" * 60)
        return

    # 2. RUN TEST CASE 1: Data dari Data Splitting (Reconstructed Raw sequence)
    print("\n" + "-" * 50)
    print("TEST CASE 1: Menguji Menggunakan Data Hasil Splitting")
    print("-" * 50)

    if os.path.exists(raw_test_df_path):
        test_df = pd.read_csv(raw_test_df_path)
        # Select a random user and extract 7 consecutive days
        users = test_df['user_id'].unique()
        random_user = random.choice(users)
        user_logs = test_df[test_df['user_id'] == random_user].sort_values('day_of_week')
        
        if len(user_logs) >= 7:
            sample_logs = user_logs.head(7)
            # Reconstruct raw logs to JSON
            log_list = []
            for _, row in sample_logs.iterrows():
                log_list.append({
                    "sleep_duration": float(row['sleep_duration']),
                    "sleep_quality": float(row['sleep_quality']),
                    "study_work_duration": float(row['study_work_duration']),
                    "break_duration": float(row['break_duration']),
                    "physical_activity_duration": float(row['physical_activity_duration']),
                    "screen_time_duration": float(row['screen_time_duration']),
                    "stress_level": float(row['stress_level']),
                    "mood_score": float(row['mood_score']),
                    "focus_score": float(row['focus_score']),
                    "task_planned": float(row['task_planned']),
                    "task_completed": float(row['task_completed']),
                    "task_completion_rate": float(row['task_completion_rate']),
                    "day_of_week": int(row['day_of_week']),
                    "month": int(row['month']),
                    "is_weekend": int(row['is_weekend'])
                })
            
            # Ground truth actual tomorrow productivity score
            actual_score = float(sample_logs.iloc[-1]['target_reg'])
            actual_class_idx = int(sample_logs.iloc[-1]['target_clf'])
            classes_map = {0: 'At Risk', 1: 'Steady', 2: 'Thriving'}
            actual_class_name = classes_map[actual_class_idx]
            
            payload = {"logs": log_list}
            
            print(f"Mengirimkan 7 hari sekuens log mentah untuk User ID: {random_user}")
            print(f"Nilai Aktual (Ground Truth) Esok Hari: Score={actual_score:.2f}% | Category={actual_class_name}")
            
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                res_data = response.json()
                print("Prediksi Berhasil Diterima:")
                print(f"   - Prediksi Skor Produktivitas: {res_data['predicted_productivity_score']:.2f}%")
                print(f"   - Prediksi Kategori Kondisi: {res_data['predicted_category']}")
                print(f"   - Probabilitas Kelas:")
                for k, v in res_data['probabilities'].items():
                    print(f"     * {k}: {v*100:.2f}%")
                print(f"   - Rekomendasi AI:")
                print(f"     \"{res_data['recommendation']}\"")
            else:
                print(f"Error API ({response.status_code}): {response.text}")
        else:
            print("Data sekuens user kurang dari 7 hari.")
    else:
        print("Berkas raw_test_dataframe.csv tidak ditemukan. Harap jalankan data_splitter.py terlebih dahulu.")

    # 3. RUN TEST CASE 2: Menggunakan Input Manual User (Simulated Profile Skenario)
    print("\n" + "-" * 50)
    print("TEST CASE 2: Menguji Menggunakan Skenario Input Manual User")
    print("-" * 50)
    
    # 7-day logs simulating high exhaustion & low sleep (AT RISK scenario)
    manual_logs_at_risk = [
        {"sleep_duration": 4.5, "sleep_quality": 3.0, "study_work_duration": 9.0, "break_duration": 0.2, "physical_activity_duration": 0.0, "screen_time_duration": 8.0, "stress_level": 8.0, "mood_score": 3.0, "focus_score": 4.0, "task_planned": 10.0, "task_completed": 3.0, "task_completion_rate": 0.3, "day_of_week": i, "month": 5, "is_weekend": 1 if i >= 5 else 0}
        for i in range(7)
    ]
    
    # 7-day logs simulating high wellness & productivity (THRIVING scenario)
    manual_logs_thriving = [
        {"sleep_duration": 8.0, "sleep_quality": 9.0, "study_work_duration": 6.5, "break_duration": 1.0, "physical_activity_duration": 45.0, "screen_time_duration": 2.5, "stress_level": 2.0, "mood_score": 9.0, "focus_score": 9.0, "task_planned": 5.0, "task_completed": 5.0, "task_completion_rate": 1.0, "day_of_week": i, "month": 5, "is_weekend": 1 if i >= 5 else 0}
        for i in range(7)
    ]
    
    scenarios = [
        ("[SKENARIO AT RISK] (Kurang tidur, stres tinggi, burnout)", manual_logs_at_risk),
        ("[SKENARIO THRIVING] (Cukup tidur, stres rendah, fokus tinggi)", manual_logs_thriving)
    ]
    
    for sc_title, logs in scenarios:
        print(f"\nMenjalankan: {sc_title}")
        payload = {"logs": logs}
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            res_data = response.json()
            print("Prediksi Berhasil Diterima:")
            print(f"   - Prediksi Skor Produktivitas: {res_data['predicted_productivity_score']:.2f}%")
            print(f"   - Prediksi Kategori Kondisi: {res_data['predicted_category']}")
            print(f"   - Probabilitas Kelas:")
            for k, v in res_data['probabilities'].items():
                print(f"     * {k}: {v*100:.2f}%")
            print(f"   - Rekomendasi AI:")
            print(f"     \"{res_data['recommendation']}\"")
        else:
            print(f"Error API ({response.status_code}): {response.text}")
            
    print("\n" + "=" * 60)
    print("PENGUJIAN INTEGRASI SELESAI")
    print("=" * 60)

if __name__ == "__main__":
    main()

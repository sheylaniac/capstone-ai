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
    workspace = os.path.dirname(os.path.abspath(__file__))
    
    x_test_path = os.path.join(workspace, "data", "processed", "x_test.npy")
    y_test_reg_path = os.path.join(workspace, "data", "processed", "y_test_reg.npy")
    y_test_clf_path = os.path.join(workspace, "data", "processed", "y_test_clf.npy")
    raw_test_df_path = os.path.join(workspace, "data", "processed", "test_raw_dataframe.csv")

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

    print("\n" + "-" * 50)
    print("TEST CASE 1: Menguji Menggunakan Data Hasil Splitting")
    print("-" * 50)

    if os.path.exists(raw_test_df_path):
        test_df = pd.read_csv(raw_test_df_path)
        users = test_df['user_id'].unique()
        random_user = random.choice(users)
        user_logs = test_df[test_df['user_id'] == random_user].sort_values('day_of_week')
        
        if len(user_logs) >= 7:
            sample_logs = user_logs.head(7)
            log_list = []
            for idx, row in sample_logs.reset_index(drop=True).iterrows():
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
                    "task_planned": int(row['task_planned']),
                    "task_completed": int(row['task_completed']),
                    "is_weekend": int(row['is_weekend']),
                    "log_date": f"2026-{int(row.get('month', 5)):02d}-{10+idx:02d}"
                })
            
            actual_score = float(sample_logs.iloc[-1]['target_reg'])
            actual_class_idx = int(sample_logs.iloc[-1]['target_clf'])
            classes_map = {0: 'At Risk', 1: 'Steady', 2: 'Thriving'}
            actual_class_name = classes_map[actual_class_idx]
            
            payload = {
                "user_id": int(random_user),
                "current_log": log_list[-1]
            }
            
            print(f"Mengirimkan log harian hari terakhir untuk User ID: {random_user} (riwayat sudah ada di database)")
            print(f"Nilai Aktual (Ground Truth) Esok Hari: Score={actual_score:.2f}% | Category={actual_class_name}")
            
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                res_data = response.json()
                main_db = res_data.get('1_main_dashboard', {})
                ai_insight = res_data.get('3_ai_insight_and_recommendation', {})
                print("Prediksi Berhasil Diterima:")
                print(f"   - Prediksi Skor Produktivitas: {main_db.get('productivity_score', 0.0):.2f}%")
                print(f"   - Prediksi Kategori Kondisi: {main_db.get('productivity_status', 'N/A')}")
                print(f"   - Probabilitas Kelas:")
                for k, v in main_db.get('probabilities', {}).items():
                    print(f"     * {k}: {v*100:.2f}%")
                print(f"   - Rekomendasi AI:")
                print(f"     \"{ai_insight.get('recommendation_text', 'N/A')}\"")
            else:
                print(f"Error API ({response.status_code}): {response.text}")
        else:
            print("Data sekuens user kurang dari 7 hari.")
    else:
        print("Berkas raw_test_dataframe.csv tidak ditemukan. Harap jalankan data_splitter.py terlebih dahulu.")

    print("\n" + "-" * 50)
    print("TEST CASE 2: Menguji Menggunakan Skenario Input Manual User")
    print("-" * 50)

    manual_logs_thriving = [
        {"sleep_duration": 8.0, "sleep_quality": 9.0, "study_work_duration": 6.0,
         "break_duration": 1.0, "physical_activity_duration": 45.0, "screen_time_duration": 2.0,
         "stress_level": 2.0, "mood_score": 9.0, "focus_score": 9.0,
         "task_planned": 5, "task_completed": 5,
         "is_weekend": 1 if (i % 7 in [5, 6]) else 0,
         "log_date": f"2026-05-{10+i:02d}"}
        for i in range(7)
    ]

    manual_logs_steady = [
        {"sleep_duration": 6.0, "sleep_quality": 5.0, "study_work_duration": 8.0,
         "break_duration": 0.4, "physical_activity_duration": 15.0, "screen_time_duration": 4.5,
         "stress_level": 5.0, "mood_score": 5.0, "focus_score": 6.0,
         "task_planned": 6, "task_completed": 4,
         "is_weekend": 1 if (i % 7 in [5, 6]) else 0,
         "log_date": f"2026-05-{10+i:02d}"}
        for i in range(7)
    ]

    manual_logs_at_risk = [
        {"sleep_duration": 4.5, "sleep_quality": 3.0, "study_work_duration": 10.0,
         "break_duration": 0.167, "physical_activity_duration": 0.0, "screen_time_duration": 8.0,
         "stress_level": 9.0, "mood_score": 2.0, "focus_score": 3.0,
         "task_planned": 8, "task_completed": 2,
         "is_weekend": 1 if (i % 7 in [5, 6]) else 0,
         "log_date": f"2026-05-{10+i:02d}"}
        for i in range(7)
    ]

    scenarios = [
        ("[SKENARIO THRIVING] (Prima & sehat, fokus tinggi)", manual_logs_thriving),
        ("[SKENARIO STEADY] (Stabil & seimbang)", manual_logs_steady),
        ("[SKENARIO AT RISK] (Lelah & burnout, stres tinggi)", manual_logs_at_risk),
    ]
    
    for sc_title, logs in scenarios:
        print(f"\nMenjalankan: {sc_title}")
        sim_user_id = random.randint(100000, 999999)
        
        for idx, log in enumerate(logs):
            payload = {
                "user_id": sim_user_id,
                "current_log": log
            }
            response = requests.post(url, json=payload)
            if idx < 6:
                if response.status_code != 400:
                    print(f"   [WARN] Hari ke-{idx+1} harusnya error 400, tetapi mendapat {response.status_code}.")
            else:
                if response.status_code == 200:
                    res_data = response.json()
                    main_db = res_data.get('1_main_dashboard', {})
                    ai_insight = res_data.get('3_ai_insight_and_recommendation', {})
                    print("Prediksi Berhasil Diterima (Hari ke-7):")
                    print(f"   - Prediksi Skor Produktivitas: {main_db.get('productivity_score', 0.0):.2f}%")
                    print(f"   - Prediksi Kategori Kondisi: {main_db.get('productivity_status', 'N/A')}")
                    print(f"   - Probabilitas Kelas:")
                    for k, v in main_db.get('probabilities', {}).items():
                        print(f"     * {k}: {v*100:.2f}%")
                    print(f"   - Rekomendasi AI:")
                    print(f"     \"{ai_insight.get('recommendation_text', 'N/A')}\"")
                else:
                    print(f"   [ERROR] Hari ke-7 gagal dengan status {response.status_code}: {response.text}")
        print(f"Skenario selesai untuk User ID: {sim_user_id}")
            
    print("\n" + "=" * 60)
    print("PENGUJIAN INTEGRASI SELESAI")
    print("=" * 60)

if __name__ == "__main__":
    main()

import os
import argparse
import pickle
import io
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from src.ai_pipeline.components.layers import CustomAttention

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it as a tensor compatible with tf.summary.image."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def main():
    parser = argparse.ArgumentParser(description="Explain and analyze trained LSTM Model (Temporal Ablation & Attention Weights)")
    parser.add_argument("--version", type=str, default="v1", help="Version of the model to explain")
    parser.add_argument("--tensorboard", action="store_true", default=True, help="Enable logging results to TensorBoard")
    args = parser.parse_args()

    workspace = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(workspace, "saved_models", args.version)
    model_path = os.path.join(model_dir, "lstmmultioutput.keras")
    feat_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    if not os.path.exists(feat_scaler_path):
        feat_scaler_path = os.path.join(model_dir, "artifacts", "feature_scaler.pkl")

    targ_scaler_path = os.path.join(model_dir, "target_scaler.pkl")
    if not os.path.exists(targ_scaler_path):
        targ_scaler_path = os.path.join(model_dir, "artifacts", "target_scaler.pkl")

    x_test_path = os.path.join(workspace, "data", "processed", "x_test.npy")
    y_test_reg_path = os.path.join(workspace, "data", "processed", "y_test_reg.npy")
    y_test_clf_path = os.path.join(workspace, "data", "processed", "y_test_clf.npy")

    # Validasi file
    if not all(os.path.exists(p) for p in [model_path, feat_scaler_path, targ_scaler_path]):
        print(f"[ERROR] Berkas model atau scaler tidak ditemukan di {model_dir}")
        print("Pastikan Anda sudah menjalankan pelatihan model: python -m src.ai_pipeline.train")
        return

    print("=" * 70)
    print(f"DIAGNOSTIK DIAGRAM ATENSI & ABLATION MODEL (VERSI: {args.version})")
    print("=" * 70)

    # 1. Load Scalers and Model
    print("Memuat scalers dan model Keras...")
    with open(feat_scaler_path, 'rb') as f:
        feat_scaler = pickle.load(f)
    with open(targ_scaler_path, 'rb') as f:
        targ_scaler = pickle.load(f)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'CustomAttention': CustomAttention}
    )

    # 2. RUN TEMPORAL ABLATION TEST (SKENARIO A vs B)
    print("\n[ 1 ] MENJALANKAN TEMPORAL ABLATION TEST...")
    
    # Hari ke-7 normal (identik untuk kedua skenario)
    day7 = {
        'sleep_duration': 7.0, 'study_work_duration': 6.0,
        'break_duration': 1.0, 'exercise_duration': 30.0, 'downtime_duration': 1.5,
        'stress_level': 4, 'mood_score': 7, 'focus_score': 7,
        'task_planned': 5, 'task_completed': 4,
        'is_weekend': 0
    }

    # Skenario A: 6 hari sebelumnya BURUK, hari ke-7 normal
    logs_A = []
    for _ in range(6):
        logs_A.append({
            'sleep_duration': 4.0, 'study_work_duration': 11.0,
            'break_duration': 0.1, 'exercise_duration': 0.0, 'downtime_duration': 0.3,
            'stress_level': 9, 'mood_score': 2, 'focus_score': 3,
            'task_planned': 8, 'task_completed': 1,
            'is_weekend': 0
        })
    logs_A.append(day7)

    # Skenario B: 6 hari sebelumnya PRIMA, hari ke-7 normal
    logs_B = []
    for _ in range(6):
        logs_B.append({
            'sleep_duration': 8.5, 'study_work_duration': 4.0,
            'break_duration': 2.0, 'exercise_duration': 60.0, 'downtime_duration': 3.0,
            'stress_level': 1, 'mood_score': 10, 'focus_score': 10,
            'task_planned': 5, 'task_completed': 5,
            'is_weekend': 0
        })
    logs_B.append(day7)

    def process_scenario(raw_logs):
        df = pd.DataFrame(raw_logs)
        # Re-calculate fatigue index & cumulative fatigue
        df['fatigue_index'] = (
            (df['stress_level'] / 8.0) * 40 +
            (df['downtime_duration'] / 11.21) * 30 +
            (df['study_work_duration'] / 17.405) * 30
        ).clip(0, 100).round(2)
        df['cumulative_fatigue'] = (
            df['fatigue_index'].ewm(alpha=0.1, adjust=False).mean()
        ).clip(0, 100).round(2)
        df['completion_ratio'] = (df['task_completed'] / df['task_planned'].clip(lower=1)).round(3)
        
        feature_cols = [
            'is_weekend', 'sleep_duration', 'study_work_duration', 'break_duration',
            'exercise_duration', 'downtime_duration', 'stress_level', 'mood_score',
            'focus_score', 'task_planned', 'task_completed', 'completion_ratio',
            'fatigue_index', 'cumulative_fatigue'
        ]
        feats_scaled = feat_scaler.transform(df[feature_cols])
        return np.expand_dims(feats_scaled, axis=0) # shape (1, 7, 14)

    X_A = process_scenario(logs_A)
    X_B = process_scenario(logs_B)

    pred_reg_A, pred_clf_A = model.predict(X_A, verbose=0)
    pred_reg_B, pred_clf_B = model.predict(X_B, verbose=0)

    score_A = targ_scaler.inverse_transform(pred_reg_A)[0][0]
    score_B = targ_scaler.inverse_transform(pred_reg_B)[0][0]

    classes = ['At Risk', 'Steady', 'Thriving']
    cat_A = classes[np.argmax(pred_clf_A[0])]
    cat_B = classes[np.argmax(pred_clf_B[0])]

    print(f"  Skenario A (6 Hari Buruk + Hari 7 Normal):")
    print(f"     -> Skor Produktivitas : {score_A:.2f}%")
    print(f"     -> Kategori Kelelahan : {cat_A}")
    print(f"     -> Distribusi Kelas   : At Risk: {pred_clf_A[0][0]*100:.2f}%, Steady: {pred_clf_A[0][1]*100:.2f}%, Thriving: {pred_clf_A[0][2]*100:.2f}%")
    
    print(f"  Skenario B (6 Hari Prima + Hari 7 Normal):")
    print(f"     -> Skor Produktivitas : {score_B:.2f}%")
    print(f"     -> Kategori Kelelahan : {cat_B}")
    print(f"     -> Distribusi Kelas   : At Risk: {pred_clf_B[0][0]*100:.2f}%, Steady: {pred_clf_B[0][1]*100:.2f}%, Thriving: {pred_clf_B[0][2]*100:.2f}%")

    diff = abs(score_A - score_B)
    print("-" * 70)
    if diff > 1.0:
        print(f"  [SUCCESS] Model TERBUKTI peka terhadap waktu (Temporal Awareness)! Selisih skor: {diff:.2f} poin.")
    else:
        print("  [WARNING] Model kurang peka terhadap data historis (selisih skor sangat kecil).")
    print("=" * 70)


    # 3. EXTRACT CUSTOM ATTENTION WEIGHTS (NUMPY RECONSTRUCTION)
    print("\n[ 2 ] MENGANALISIS BOBOT CUSTOM ATTENTION...")
    if not os.path.exists(x_test_path):
        print("  [ERROR] File x_test.npy tidak ditemukan. Lewati ekstraksi bobot atensi batch.")
        mean_alpha = None
    else:
        X_test = np.load(x_test_path)
        print(f"  Mengekstrak LSTM outputs pada {len(X_test)} data uji...")

        # Bangun sub-model untuk mendapatkan output LSTM
        lstm_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('lstm_layer').output)
        lstm_outputs = lstm_model.predict(X_test, verbose=0) # shape (N, 7, 64)

        # Ambil bobot (weights) dari attention layer
        att_layer = model.get_layer('attention_layer')
        W = att_layer.weights[0].numpy() # shape (64, 1)
        b = att_layer.weights[1].numpy() # shape (7, 1)

        # Hitung e = tanh(lstm_outputs @ W + b) secara manual dalam numpy
        e = np.tanh(np.dot(lstm_outputs, W) + b) # shape (N, 7, 1)

        # Hitung softmax di axis 1 (time steps)
        exp_e = np.exp(e)
        alpha = exp_e / np.sum(exp_e, axis=1, keepdims=True) # shape (N, 7, 1)

        # Hitung rata-rata kontribusi per hari
        mean_alpha = np.mean(alpha, axis=0).flatten() # shape (7,)

        print("  Rata-rata pengaruh/atensi model per hari:")
        for i, val in enumerate(mean_alpha):
            marker = "<- Terbesar" if val == max(mean_alpha) else ""
            print(f"     Hari {i+1} (Day {i+1}): {val*100:.2f}% {marker}")

        # Render grafik batang atensi
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = ['steelblue' if v < max(mean_alpha) else 'crimson' for v in mean_alpha]
        ax.bar(range(1, 8), mean_alpha * 100, color=colors, edgecolor='black', alpha=0.85)
        ax.set_xlabel('Hari ke- (Jendela Aktivitas 7 Hari)', fontsize=11)
        ax.set_ylabel('Rata-rata Bobot Perhatian (Atensi %)', fontsize=11)
        ax.set_title('Kontribusi Pengaruh Tiap Hari terhadap Estimasi Model', fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, 8))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for i, val in enumerate(mean_alpha):
            ax.text(i + 1, val * 100 + 0.5, f"{val*100:.1f}%", ha='center', fontweight='bold')
        plt.tight_layout()

        # Simpan grafik
        chart_path = os.path.join(model_dir, "attention_weights.png")
        plt.savefig(chart_path)
        print(f"  [SUCCESS] Grafik bobot atensi disimpan ke '{chart_path}'")


    # 4. LOG RESULTS TO TENSORBOARD
    if args.tensorboard:
        print("\n[ 3 ] MENCATAT HASIL DIAGNOSTIK KE TENSORBOARD...")
        log_dir = os.path.join(workspace, "logs", "explainability", datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = tf.summary.create_file_writer(log_dir)

        with writer.as_default():
            # A. Catat scalar hasil Ablation
            tf.summary.scalar('Ablation/Scenario_A_Score', score_A, step=0)
            tf.summary.scalar('Ablation/Scenario_B_Score', score_B, step=0)
            tf.summary.scalar('Ablation/Score_Difference', diff, step=0)

            # B. Catat laporan markdown ke tab Text
            markdown_text = (
                f"## Laporan Temporal Ablation Test\n\n"
                f"Pengujian ini membuktikan apakah model memiliki *Temporal Awareness* (peka terhadap urutan waktu) "
                f"dengan membandingkan dua skenario input yang memiliki data hari terakhir (hari ke-7) yang identik, "
                f"tetapi riwayat 6 hari sebelumnya berbeda.\n\n"
                f"### Hasil Simulasi:\n"
                f"1. **Skenario A** (6 Hari Aktivitas Buruk + Hari 7 Normal):\n"
                f"   * Estimasi Skor Produktivitas: **{score_A:.2f}%**\n"
                f"   * Kategori Kelelahan: **{cat_A}**\n"
                f"2. **Skenario B** (6 Hari Aktivitas Prima + Hari 7 Normal):\n"
                f"   * Estimasi Skor Produktivitas: **{score_B:.2f}%**\n"
                f"   * Kategori Kelelahan: **{cat_B}**\n\n"
                f"### Kesimpulan:\n"
                f"* **Selisih Skor**: **{diff:.2f}** poin\n"
                f"* Status: **{'Lolos (Temporal Aware)' if diff > 1.0 else 'Gagal (Kurang Peka)'}**"
            )
            tf.summary.text('Ablation/Summary_Report', markdown_text, step=0)

            # C. Catat grafik batang atensi ke tab Images
            if mean_alpha is not None:
                # Buat ulang gambar untuk summary
                fig_tb, ax_tb = plt.subplots(figsize=(6, 4))
                ax_tb.bar(range(1, 8), mean_alpha * 100, color='darkorange', edgecolor='black')
                ax_tb.set_xlabel('Day Sequence')
                ax_tb.set_ylabel('Attention Weight (%)')
                ax_tb.set_title('Custom Attention Time Weights')
                ax_tb.set_xticks(range(1, 8))
                plt.tight_layout()
                
                img_tensor = plot_to_image(fig_tb)
                tf.summary.image('Attention/Day_Contribution_Chart', img_tensor, step=0)
                print("  [SUCCESS] Gambar grafik bobot atensi berhasil dicatat ke TensorBoard (tab Images).")

        print(f"  [SUCCESS] Log TensorBoard disimpan di: {log_dir}")
        print("  Silakan jalankan perintah berikut untuk melihat hasil visualisasi:")
        print("  tensorboard --logdir logs/")
    
    print("\n" + "=" * 70)
    print("PROSES ANALISIS SELESAI")
    print("=" * 70)

if __name__ == "__main__":
    main()

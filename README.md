# Smart Digital Twin: LSTM Multi-Output System

Sistem Digital Twin Cerdas berbasis kecerdasan buatan (*AI*) untuk memprediksi **Skor Produktivitas** (regresi) dan **Status Kelelahan Pengguna** (klasifikasi 3-kelas: *At Risk*, *Steady*, *Thriving*) secara simultan (*multi-output*) berdasarkan sekuens data log aktivitas 7 hari ke belakang.

Sistem ini dilengkapi dengan layanan rekomendasi kesehatan adaptif berbasis **Google Gemini API (`gemini-2.5-flash`)** dengan mekanisme pertahanan **Auto Fallback & Rate Limit Bypass** otomatis ke *Rule-Based Fallback* jika API Key tidak disediakan, kuota habis (Error 429), atau diblokir.

---

## Pemetaan Kriteria Penilaian Capstone (AI Checklist)

Proyek ini telah dirancang untuk memenuhi **100% kriteria penilaian AI Capstone Project** (kriteria wajib & opsional/nilai tambah):

### 1. Main Quest (Checklist Wajib / MVP)
* **Model Deep Learning Multi-Output**: Dibangun dengan **TensorFlow Functional API** di `src/ai_pipeline/model.py` untuk secara simultan menghasilkan skor produktivitas (regresi) dan status kelelahan (klasifikasi 3-kelas).
* **Komponen Kustom Lanjutan**: Mengimplementasikan 3 komponen kustom:
  - *Custom Layer*: `CustomAttention` di `src/ai_pipeline/components/layers.py` untuk pembobotan bobot sekuensial dinamis.
  - *Custom Loss Function*: `calculate_multi_objective_loss` di `src/ai_pipeline/components/losses.py` untuk menyeimbangkan MAE & Categorical Crossentropy.
  - *Custom Callback*: `ModelCheckpointCallback` di `src/ai_pipeline/components/callbacks.py` untuk penyimpanan checkpoint cerdas.
* **Format Ekspor Siap Produksi**: Model penuh disimpan dalam format `.keras` di `saved_models/v1/lstmmultioutput.keras`.
* **Kode Inferensi Model**: Terstruktur dengan rapi pada berkas modular `src/services/ai_inference_service.py` untuk memproses scaling data dan prediksi.

### 2. Side Quest (Checklist Opsional / Nilai Tambah)
* **REST API Mandiri**: Dikembangkan menggunakan **FastAPI** asinkron di folder `src/` untuk melayani model inferensi secara efisien.
* **Custom Training Loop (`tf.GradientTape`)**: Pelatihan model tidak menggunakan `.fit()` biasa, melainkan loop kustom penuh di `src/ai_pipeline/train.py` dengan kalkulasi gradien manual via `tf.GradientTape()`.
* **Integrasi Generative AI**: Menggunakan **Google Gemini API** (`gemini-2.5-flash`) dengan mekanisme ketahanan otomatis (*auto-fallback*) jika mengalami limit kuota.
* **Integrasi TensorBoard**: Menyimpan log pelatihan kustom secara dinamis di folder `logs/gradient_tape/` untuk pemantauan kurva metrik.
* **Performa Model Optimal**: Hasil pelatihan menghasilkan akurasi klasifikasi tingkat kelelahan >95% (minimal target 85%) dan MAE <0.02.

---

## Fitur Utama & Pembaruan Sistem

1. **Standardisasi Input Tunggal (Single-Day Input)**:
   API tidak lagi menerima array 7 hari langsung dari klien, melainkan menerima data log harian tunggal (`current_log`) dan memvalidasi riwayat 7 hari ke belakang dari database CSV (`user_activity_logs.csv`) secara otomatis. Hal ini menjaga integritas data dan keamanan sekuens.
2. **Imputasi Cerdas (Flexible Input Imputation)**:
   Melalui endpoint `/predict-flexible`, pengguna dapat mengirimkan data sebagian (opsional). Sistem secara cerdas akan mengisi nilai kosong menggunakan rata-rata riwayat aktivitas pengguna tersebut, atau menggunakan nilai default global jika pengguna baru pertama kali masuk.
3. **Mekanisme Resilience Gemini API**:
   Jika Gemini API mencapai kuota limit harian (Error 429 / Resource Exhausted) atau kunci API tidak valid, sistem akan mendeteksinya secara dinamis, menghentikan sementara pemanggilan API Gemini (untuk menghindari overhead penundaan koneksi), dan langsung mengalihkan pembuatan asisten ke logika *Rule-Based Fallback* dalam bahasa Inggris profesional.
4. **Struktur Dashboard Visual 5 Komponen**:
   Respons API dikemas dalam format yang siap digunakan oleh UI Dashboard, yang terdiri dari:
   - **Main Dashboard**: Status produktivitas, skor, tingkat keyakinan (*confidence*), dan tingkat kelelahan harian.
   - **Analytics Dashboard**: Grafik tren mingguan, aktivitas paling dominan, jam puncak produktivitas, dan heatmap aktivitas mingguan.
   - **AI Insight**: Laporan rekomendasi kesehatan & produktivitas 4 paragraf (Insight, Root Cause, Rekomendasi Aktivitas, dan Peringatan Burnout).
   - **Similar History**: Membandingkan profil hari ini dengan 3 hari terbaik di masa lalu menggunakan *Cosine Similarity*.
   - **Form Reconciliation**: Menampilkan status imputasi dan log final setelah data diisi lengkap.

---

## Desain Arsitektur Folder Modular

Struktur repositori ini disusun secara berlapis (*layered architecture*) berstandar industri:

```text
dtwin-ai/
│
├── .env                        # Konfigurasi Environment (API Key, Model Version)
├── .gitignore                  # Mengabaikan berkas virtual env, keras model, logs, dll.
├── README.md                   # Dokumentasi panduan instalasi & API
├── requirements.txt            # Dependensi Python ter-update
├── data_splitter.py            # Skrip pembagi data & pengekspor file numpy test
├── test_api.py                 # Skrip pengujian otomatis API (FastAPI)
├── test_dtwin_system.py        # Skrip pengujian otomatis komprehensif sistem
│
├── data/                       # Tempat Penyimpanan Dataset
│   ├── raw/                    # Dataset mentah (datasetdailylogs.csv)
│   ├── processed/              # Hasil data pembagian splitting untuk pengujian
│   └── user_activity_logs.csv  # Database log aktivitas lokal (CSV)
│
├── saved_models/               # Tempat Penyimpanan Model (Mendukung Versioning)
│   └── v1/                     # Folder versi model (v1, v2, dst.)
│       ├── lstmmultioutput.keras # File ekspor model penuh TensorFlow versi 1
│       └── artifacts/          # Objek scaling scaler versi 1
│           ├── feature_scaler.pkl  
│           └── target_scaler.pkl   
│
└── src/                        # KODE UTAMA BACKEND API
    ├── app.py                  # Entrypoint server FastAPI & lifespan startup loader
    │
    ├── routes/                 # Layer 1: API Routing
    │   ├── __init__.py
    │   └── prediction_routes.py# Routing endpoint API
    │
    ├── controllers/            # Layer 2: API Controller & DTO
    │   ├── __init__.py
    │   └── prediction_controller.py # Validasi Pydantic, koordinasi service, & visual components
    │
    └── services/               # Layer 3: Logika Bisnis & Inferensi AI
        ├── __init__.py
        ├── ai_inference_service.py # Feature engineering dinamis & model.predict()
        └── genai_service.py    # Integrasi API Google Gemini & Smart Fallback
```

---

## Panduan Cara Mencoba & Menguji

### 1. Prasyarat & Instalasi Dependensi
Pastikan Anda menggunakan Python dengan TensorFlow terpasang (disarankan menggunakan Anaconda environment).
Klon repositori dan jalankan instalasi dependensi berikut:
```bash
pip install -r requirements.txt
```

### 2. Konfigurasi Kunci API Gemini (Opsional)
Buat file `.env` pada folder root proyek:
```env
GEMINI_API_KEY=AIzaSy... (Masukkan kunci API Gemini Anda di sini)
MODEL_VERSION=v1
```
> [!NOTE]
> Jika `GEMINI_API_KEY` tidak diisi atau kuota limit Anda habis, sistem akan secara otomatis beralih ke logika *rule-based fallback* berkualitas tinggi tanpa merusak alur aplikasi.

### 3. Menjalankan Server API FastAPI
Untuk menyalakan server lokal secara interaktif dengan auto-reload:
```bash
uvicorn src.app:app --reload
```
Server akan aktif di `http://127.0.0.1:8000/`. Anda bisa membuka dokumentasi interaktif Swagger UI di `http://127.0.0.1:8000/docs` untuk mencoba langsung lewat web.

### 4. Menjalankan Skrip Pengujian Otomatis (`test_api.py`)
Dalam terminal terpisah, Anda dapat menjalankan skrip pengujian API otomatis:
```bash
python test_api.py
```
Skrip ini akan menguji seluruh skenario secara berurutan:
- **Skenario 1**: Menguji `/predict` dengan user ID baru yang tidak memiliki riwayat data. *Ekspektasi: Menghasilkan error 400 karena kurang dari 7 hari.*
- **Skenario 2**: Menguji `/predict-from-logs` untuk user ID tersebut guna mengisi data riwayat 7 hari awal secara otomatis (Synthetic Cold Start). *Ekspektasi: Berhasil 200 OK.*
- **Skenario 3**: Menguji kembali `/predict` dengan data hari ke-8. *Ekspektasi: Berhasil 200 OK dan menghasilkan prediksi model LSTM.*
- **Skenario 4**: Menguji `/predict-flexible` dengan data sebagian (hanya durasi tidur dan kualitas tidur). *Ekspektasi: Berhasil 200 OK dengan melakukan imputasi otomatis rata-rata.*

---

## Panduan Format JSON Request & Response API

### 1. POST `/api/v1/predict-from-logs`
Memproses prediksi berdasarkan data riwayat aktivitas yang sudah tersimpan di database CSV lokal. Jika riwayat kurang dari 7 hari, ia akan membuat data sintetis (Cold Start) secara otomatis.

* **Payload Request:**
  ```json
  {
    "user_id": 428489
  }
  ```

* **Contoh Respons Sukses (200 OK):**
  ```json
  {
    "success": true,
    "user_id": 428489,
    "days_logged": 7,
    "message": "Prediction generated from saved activity logs.",
    "1_main_dashboard": { ... },
    "2_productivity_analytics_dashboard": { ... },
    "3_ai_insight_and_recommendation": { ... },
    "4_similar_productivity_history": { ... },
    "5_activity_input_form_reconciliation": { ... }
  }
  ```

---

### 2. POST `/api/v1/predict`
Mengirimkan data aktivitas harian lengkap. Endpoint ini akan mencatat log hari ini ke database, dan mengharuskan user memiliki minimal 7 hari riwayat aktivitas di database (termasuk hari ini) agar model LSTM dapat beroperasi.

* **Payload Request:**
  ```json
  {
    "user_id": 428489,
    "current_log": {
      "sleep_duration": 7.2,
      "sleep_quality": 6.5,
      "study_work_duration": 6.0,
      "break_duration": 2.0,
      "physical_activity_duration": 15.0,
      "screen_time_duration": 6.0,
      "stress_level": 4.0,
      "mood_score": 6.0,
      "focus_score": 7.0,
      "task_planned": 5,
      "task_completed": 3,
      "is_weekend": 0,
      "log_date": "2026-05-28"
    }
  }
  ```

* **Ekspektasi Respons Gagal (400 Bad Request) jika riwayat data kurang dari 7 hari:**
  ```json
  {
    "detail": "Gagal memproses prediksi. Data aktivitas user baru terisi 1 hari di database (termasuk hari ini). Minimal dibutuhkan riwayat 7 hari aktivitas!"
  }
  ```

---

### 3. POST `/api/v1/predict-flexible`
Mengirimkan data aktivitas harian dengan field yang fleksibel (opsional). Kolom yang dikosongkan secara otomatis akan diisi menggunakan nilai rata-rata profil historis user, atau default global sistem.

* **Payload Request (Hanya mengisi sebagian field):**
  ```json
  {
    "user_id": 428489,
    "current_log": {
      "sleep_duration": 8.0,
      "sleep_quality": 8.0,
      "study_work_duration": 6.0,
      "log_date": "2026-05-30"
    }
  }
  ```

* **Contoh Respons Sukses (200 OK) beserta rincian 5 Komponen Dashboard:**
  ```json
  {
    "success": true,
    "user_id": 428489,
    "days_logged": 9,
    "message": "Productivity analysis successfully processed over 7-day flexible input.",
    "1_main_dashboard": {
      "productivity_status": "Thriving",
      "productivity_score": 67.63,
      "prediction_confidence": 97.52,
      "probabilities": {
        "At Risk": 0.0,
        "Steady": 0.02,
        "Thriving": 0.98
      },
      "fatigue_level": 1.33,
      "completion_rate": 66.67,
      "risk_signal": "NORMAL"
    },
    "2_productivity_analytics_dashboard": {
      "daily_productivity_chart": [
        { "date": "May 24", "score": 58.59 },
        { "date": "May 25", "score": 58.59 },
        { "date": "May 26", "score": 58.59 },
        { "date": "May 27", "score": 58.59 },
        { "date": "May 28", "score": 83.63 },
        { "date": "May 29", "score": 83.01 },
        { "date": "May 30", "score": 67.63 }
      ],
      "weekly_productivity_trend": "Increasing",
      "most_dominant_activity": "Sleep",
      "peak_productive_hours": "09:00-11:00",
      "activity_heatmap": {
        "Monday": 58.59,
        "Tuesday": 58.59,
        "Wednesday": 58.59,
        "Thursday": 83.63,
        "Friday": 70.8,
        "Saturday": 63.11,
        "Sunday": 58.59
      }
    },
    "3_ai_insight_and_recommendation": {
      "condition_insight": "Activity log evaluation predicts a prime condition (Thriving) with a projected productivity score of 67.63% for tomorrow.",
      "performance_cause": "This performance optimization is driven by an ideal daily sleep duration and highly controlled stress management over the past week.",
      "activity_recommendation": "It is recommended to maintain the current operational pacing while consistently integrating active micro-breaks throughout daily tasks.",
      "burnout_warning": "This sustained effort is crucial to stabilize energy levels and mitigate the risk of latent fatigue accumulation moving forward.",
      "recommendation_text": "Activity log evaluation predicts a prime condition (Thriving) with a projected productivity score of 67.63% for tomorrow.\n\nThis performance optimization is driven by an ideal daily sleep duration and highly controlled stress management over the past week.\n\nIt is recommended to maintain the current operational pacing while consistently integrating active micro-breaks throughout daily tasks.\n\nThis sustained effort is crucial to stabilize energy levels and mitigate the risk of latent fatigue accumulation moving forward."
    },
    "4_similar_productivity_history": {
      "top_3_similar_days": [
        { "date": "May 22", "similarity_score": "100%", "historical_productivity_score": 58.59, "historical_category": "Steady" },
        { "date": "May 23", "similarity_score": "100%", "historical_productivity_score": 58.59, "historical_category": "Steady" },
        { "date": "May 24", "similarity_score": "100%", "historical_productivity_score": 58.59, "historical_category": "Steady" }
      ],
      "average_productivity_from_similar_days": 58.59
    },
    "5_activity_input_form_reconciliation": {
      "imputation_status": "Historical profile average applied to empty fields.",
      "reconciled_log_today": {
        "sleep_duration": 8.0,
        "sleep_quality": 8.0,
        "study_work_duration": 6.0,
        "break_duration": 2.10,
        "physical_activity_duration": 28.20,
        "screen_time_duration": 5.62,
        "stress_level": 3.92,
        "mood_score": 6.60,
        "focus_score": 6.25,
        "task_planned": 5.25,
        "task_completed": 3.50,
        "completion_ratio": 0.67,
        "fatigue_accumulation": 1.33,
        "productivity_score": 67.63
      }
    }
  }
  ```

---

## Spesifikasi Data & Feature Engineering

Model LSTM pada arsitektur ini berbasis **Time-Series Sequential**. API membutuhkan input berupa **sekuens log harian selama 7 hari berturut-turut (Timesteps = 7)**.

### 1. Rentang Skala Nilai Input (Fitur Utama)

Berikut adalah daftar seluruh fitur yang digunakan dalam arsitektur data:

| # | Nama Fitur / Atribut | Tipe Data | Rentang Skala | Deskripsi / Representasi Fitur |
| :-: | :--- | :---: | :---: | :--- |
| 1 | `sleep_duration` | Float | `0.0` - `24.0` | Durasi tidur total dalam satuan jam. |
| 2 | `sleep_quality` | Integer | `1` - `10` | Skala kualitas tidur (1: Sangat Buruk, 10: Sangat Nyenyak). |
| 3 | `study_work_duration`| Float | `0.0` - `24.0` | Durasi waktu untuk belajar atau bekerja (jam). |
| 4 | `break_duration` | Float | `0.0` - `24.0` | Total durasi istirahat/jeda di sela aktivitas (jam). |
| 5 | `physical_activity_duration` | Float | `0.0` - `1440.0` | Durasi melakukan aktivitas fisik / olahraga (menit). |
| 6 | `screen_time_duration`| Float | `0.0` - `24.0` | Durasi menatap layar gadget/komputer (jam). |
| 7 | `stress_level` | Integer | `1` - `10` | Skala tingkat stres harian (1: Sangat Tenang, 10: Burnout). |
| 8 | `mood_score` | Integer | `1` - `10` | Skala kondisi suasana hati (1: Sangat Buruk, 10: Bahagia). |
| 9 | `focus_score` | Integer | `1` - `10` | Skala tingkat fokus dan konsentrasi. |
| 10| `task_planned` | Integer | `0` - `100` | Jumlah tugas yang direncanakan hari itu. |
| 11| `task_completed` | Integer | `0` - `100` | Jumlah tugas yang berhasil diselesaikan hari itu. |
| 12| `task_completion_rate`| Float | `0.0` - `1.0` | Rasio penyelesaian tugas (`task_completed` / `task_planned`). |
| 13| `day_of_week` | Integer | `0` - `6` | Hari dalam seminggu (0: Senin, s.d. 6: Minggu). |
| 14| `month` | Integer | `1` - `12` | Bulan kalender berjalan (1: Januari, s.d. 12: Desember). |
| 15| `is_weekend` | Integer | `0` atau `1` | Penanda hari libur akhir pekan (0: Hari Kerja, 1: Weekend). |
| 16| `cumulative_fatigue` | Float | Adaptive | *Dihitung otomatis oleh sistem melalui pipa Feature Engineering.* |

---

### 2. Rumus Rekayasa Fitur (Feature Engineering)

Sebelum data dialirkan ke dalam model LSTM, sistem backend secara otomatis akan melakukan kalkulasi fitur baru untuk menangkap efek kelelahan yang menumpuk.

#### A. Rumus Indeks Kelelahan Harian (*Fatigue Index*)
Setiap hari, tingkat kelelahan dasar (*baseline fatigue*) pengguna dihitung melalui kombinasi linear berbobot dari tingkat stres, rasio durasi kerja terhadap waktu tidur, dan komplemen kualitas tidur harian:

$$FI_t = 0.4 \cdot \left(\frac{Stress_t}{10}\right) + 0.3 \cdot \left(\frac{Work\_Duration_t}{Sleep\_Duration_t + 10^{-5}}\right) + 0.3 \cdot \left(1 - \frac{Sleep\_Quality_t}{10}\right)$$

#### B. Rumus Kelelahan Akumulatif (*Cumulative Fatigue*)
Untuk menangkap efek kelelahan fisik dan mental yang menumpuk dari hari-hari sebelumnya, fitur akhir `cumulative_fatigue` dihitung menggunakan metode penjumlahan jendela bergerak (*Rolling Window Sum*) dengan rentang waktu 3 hari ($W=3$):

$$F_t = \sum_{i=0}^{2} FI_{t-i}$$

---

### 3. Batas Ambang Kategori Kondisi (Output Klasifikasi)

Hasil nilai regresi skor produktivitas besok pagi ($P_{t+1}$) yang telah diubah kembali ke skala aslinya (`target_scaler.pkl`) dipetakan secara otomatis ke dalam 3 batas ambang (*threshold*) kategori kondisi fisik/mental user:

$$\text{Kondisi Besok } (C_{t+1}) = \begin{cases} 
\textbf{At Risk} & \text{jika } P_{t+1} < 55\% \\ 
\textbf{Steady} & \text{jika } 55\% \le P_{t+1} \le 67\% \\ 
\textbf{Thriving} & \text{jika } P_{t+1} > 67\% 
\end{cases}$$

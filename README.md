# Smart Digital Twin: LSTM Multi-Output System (Weekly Sequence)

Sistem Digital Twin Cerdas berbasis kecerdasan buatan (*AI*) dengan **Model LSTM Multi-Output** untuk mengestimasi **Skor Produktivitas** (regresi) dan **Status Kelelahan Pengguna** (klasifikasi 3-kelas: *At Risk*, *Steady*, *Thriving*) secara simultan berdasarkan sekuens data log aktivitas mingguan (7 hari).

Sistem ini dilengkapi dengan layanan rekomendasi kesehatan adaptif berbasis **Google Gemini API (`gemini-2.5-flash`)** dengan mekanisme pertahanan **Auto Fallback & Rate Limit Bypass** otomatis ke *Rule-Based Fallback* jika API Key tidak disediakan, kuota habis (Error 429), atau diblokir.

---

## Pemetaan Kriteria Penilaian Capstone (AI Checklist)

Proyek ini telah dirancang untuk memenuhi **100% kriteria penilaian AI Capstone Project** (kriteria wajib & opsional/nilai tambah):

### 1. Main Quest (Checklist Wajib / MVP)
* **Model Deep Learning Multi-Output**: Dibangun dengan **TensorFlow Functional API** di `src/ai_pipeline/model.py` untuk secara simultan menghasilkan skor produktivitas (regresi) dan status kelelahan (klasifikasi 3-kelas).
* **Komponen Kustom Lanjutan**: Mengimplementasikan 4 komponen kustom:
  - *Custom Layer*: `CustomAttention` di `src/ai_pipeline/components/layers.py` dengan aktivasi `tanh` untuk pembobotan bobot sekuensial dinamis.
  - *Custom Loss Function*: `calculate_multi_objective_loss` di `src/ai_pipeline/components/losses.py` untuk menyeimbangkan MAE & Categorical Crossentropy.
  - *Custom Callbacks*: `ModelCheckpointCallback` dan `EarlyStoppingLRCallback` di `src/ai_pipeline/components/callbacks.py` untuk penyimpanan checkpoint cerdas, penurunan learning rate otomatis, dan early stopping.
* **Format Ekspor Siap Produksi**: Model penuh disimpan dalam format `.keras` di `saved_models/v1/lstmmultioutput.keras`.
* **Kode Inferensi Model**: Terstruktur dengan rapi pada berkas modular `src/services/ai_inference_service.py` untuk memproses scaling data dan prediksi.

### 2. Side Quest (Checklist Opsional / Nilai Tambah)
* **REST API Mandiri**: Dikembangkan menggunakan **FastAPI** asinkron di folder `src/` untuk melayani model inferensi secara efisien.
* **Custom Training Loop (`tf.GradientTape`)**: Pelatihan model tidak menggunakan `.fit()` biasa, melainkan loop kustom penuh di `src/ai_pipeline/train.py` dengan kalkulasi gradien manual via `tf.GradientTape()`.
* **Integrasi Generative AI**: Menggunakan **Google Gemini API** (`gemini-2.5-flash`) dengan mekanisme ketahanan otomatis (*auto-fallback*) jika mengalami limit kuota.
* **Integrasi TensorBoard**: Menyimpan log pelatihan kustom secara dinamis di folder `logs/gradient_tape/` untuk pemantauan kurva metrik.
* **Performa Model Optimal**: Hasil pelatihan menghasilkan akurasi klasifikasi tingkat kelelahan >95% (minimal target 85%) dan MAE <0.02.

---

## Penjelasan Arsitektur Model

Model yang digunakan dalam sistem ini adalah **LSTM Multi-Output untuk Mengestimasi Skor Produktivitas dan Status Kelelahan Berdasarkan Sekuens Aktivitas Mingguan (7 Hari)** dengan dukungan Mekanisme Self-Attention. Model ini termasuk dalam rumpun **Recurrent Neural Network (RNN)** untuk pemrosesan data sekuensial/deret waktu (*time-series*).

### Mengapa Memilih Arsitektur Ini?
1. **Multi-Task Learning (Multi-Output)**: 
   Model dilatih untuk menyelesaikan dua tugas sekaligus (regresi skor produktivitas & klasifikasi tingkat kelelahan) menggunakan representasi fitur yang terbagi (*shared dense layers*). Pendekatan ini meningkatkan efisiensi komputasi dan kapabilitas generalisasi model.
2. **LSTM (Long Short-Term Memory)**: 
   Mampu menangkap dependensi temporal jangka panjang dari pola aktivitas pengguna selama 7 hari berturut-turut tanpa mengalami masalah *vanishing gradient*.
3. **Custom Self-Attention Layer**: 
   Mekanisme perhatian kustom (menggunakan aktivasi `tanh` dan bobot terlatih) yang secara dinamis menilai hari mana dalam sekuens 7-hari yang paling memengaruhi kondisi pengguna saat ini (misalnya, jika pengguna kurang tidur di hari ke-3, layer ini akan memberi bobot lebih tinggi pada hari tersebut).

### Struktur Blok Arsitektur Model:
* **Input Layer**: Menerima tensor 3D berdimensi `(batch_size, 7, 14)` (7 langkah waktu, 14 fitur input).
* **LSTM Layer**: Memiliki 64 unit tersembunyi (*hidden units*) dengan `return_sequences=True` untuk menghasilkan representasi sekuensial bagi layer attention.
* **Custom Attention Layer**: Menghitung bobot atensi, memanipulasi vektor output LSTM, dan mereduksi dimensi sekuens menjadi vektor konteks 2D berdimensi `(batch_size, 64)`.
* **Shared Dense Layer**: Layer padat (32 unit, aktivasi Relu) yang bertindak sebagai ekstraktor fitur bersama untuk kedua output.
* **Output Regresi**: Layer linear (1 unit, aktivasi Sigmoid) menghasilkan estimasi nilai produktivitas dalam skala $[0,1]$ berdasarkan sekuens mingguan.
* **Output Klasifikasi**: Layer padat (3 unit, aktivasi Softmax) menghasilkan nilai probabilitas untuk 3 kelas status kelelahan (*At Risk*, *Steady*, *Thriving*).

---

## Analisis Overfitting & Performa Metrik

Berdasarkan analisis hasil pelatihan (`history_metrics.json`) dan evaluasi test set, **model terbukti tidak mengalami overfitting** dan memiliki tingkat generalisasi yang sangat stabil.

### Tabel Perbandingan Metrik Evaluasi:

| Metrik Kunci | Training Set (Akhir) | Validation Set (Terbaik - Epoch 39) | Test Set (Unseen Data) | Status Target |
| :--- | :---: | :---: | :---: | :---: |
| **Total Loss** | ~0.1036 | **0.0720** | - | Stabil |
| **Classification Accuracy** | 98.57% | **99.20%** | **99.27%** | Lolos (Target $\ge$ 85.00%) |
| **Regression MAE** | 0.0097 | **0.0077** | **0.0076** | Lolos (Target $\le$ 0.0200) |

### Mengapa Model Bebas dari Overfitting?
1. **Regularisasi Cerdas (Callbacks)**:
   * **Weights Rollback**: `EarlyStoppingLRCallback` memantau loss validasi dan secara otomatis mengembalikan bobot model ke Epoch 39 (titik optimal sebelum loss validasi naik kembali).
   * **Dynamic Learning Rate**: Mengurangi learning rate secara dinamis dengan faktor $0.5$ jika performa stagnan, memungkinkan model konvergen ke minimum lokal dengan mulus.
2. **Kesesuaian Kapasitas Model**:
   * Jumlah parameter model disesuaikan secara proporsional dengan ukuran dataset, menghindari penggunaan arsitektur yang terlalu lebar/dalam yang berpotensi menghafal data (*memorization*).
3. **Metrik Konsisten**:
   * Akurasi klasifikasi dan nilai MAE pada data uji (*unseen test data*) hampir identik dengan metrik validasi, mengindikasikan ketahanan model terhadap variasi data baru di produksi.

---

## Desain Arsitektur Folder Modular

Struktur repositori ini disusun secara berlapis (*layered architecture*) berstandar industri:

```text
dtwin-ai/
│
├── .env                        # Konfigurasi Environment (API Key, Model Version, Token Auth)
├── .gitignore                  # Mengabaikan berkas virtual env, keras model, logs, dll.
├── README.md                   # Dokumentasi panduan instalasi & API
├── requirements.txt            # Dependensi Python ter-update
├── evaluate_model.py           # Skrip evaluasi performa model pada test set & kurva training
├── test_prediction_api.py      # Skrip pengujian otomatis komprehensif sistem API FastAPI
│
├── data/                       # Tempat Penyimpanan Dataset
│   └── final_dataset_model_ready.csv  # Dataset latih dan validasi utama
│
├── saved_models/               # Tempat Penyimpanan Model (Mendukung Versioning)
│   └── v1/                     # Folder versi model (v1, v2, dst.)
│       ├── lstmmultioutput.keras # File ekspor model penuh TensorFlow versi 1
│       ├── feature_scaler.pkl  # Objek scaler fitur input
│       ├── target_scaler.pkl   # Objek scaler target regresi
│       ├── history_metrics.json # Riwayat pelatihan loss & metrics
│       ├── confusion_matrix.png # Visualisasi matriks kekacauan klasifikasi
│       └── training_curves.png # Visualisasi kurva loss & accuracy
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
    │   └── prediction_controller.py # Validasi Pydantic, koordinasi service, & formatting respons payload
    │
    ├── schemas/                # Layer DTO Schemas
    │   ├── __init__.py
    │   └── prediction_schema.py# Definisi struktur payload request Pydantic
    │
    └── services/               # Layer 3: Logika Bisnis & Inferensi AI
        ├── __init__.py
        ├── ai_inference_service.py # Feature engineering dinamis & model.predict()
        └── genai_service.py    # Integrasi API Google Gemini (5-paragraf) & Smart Fallback
```

---

## Panduan Cara Mencoba & Menguji

### 1. Prasyarat & Instalasi Dependensi
Pastikan Anda menggunakan Python dengan TensorFlow terpasang (disarankan menggunakan Anaconda environment).
Klon repositori dan jalankan instalasi dependensi berikut:
```bash
pip install -r requirements.txt
```

### 2. Konfigurasi Environment (.env)
Buat file `.env` pada folder root proyek:
```env
GEMINI_API_KEY=AIzaSy... (Masukkan kunci API Gemini Anda di sini)
MODEL_VERSION=v1
SECRET_TOKEN_AI=token_rahasia_anda_disini
```

### 3. Menjalankan Server API FastAPI
Untuk menyalakan server lokal secara interaktif dengan auto-reload:
```bash
-m uvicorn src.app:app --reload
```
Server akan aktif di `http://127.0.0.1:8000/`.

### 4. Menjalankan Skrip Pengujian Otomatis (`test_prediction_api.py`)
Dalam terminal terpisah, Anda dapat menjalankan skrip pengujian API otomatis:
```bash
python test_prediction_api.py
```
Skrip ini akan menguji otorisasi keamanan token, format input sekuensial 7-hari, dan integritas respons model AI serta rekomendasi 5-paragraf dari Gemini.

### 5. Menjalankan Skrip Analisis Pengaruh Temporal & Atensi (`explain_model.py`)
Untuk menganalisis pengaruh historis data 7 hari terhadap estimasi model, serta melihat kontribusi bobot atensi (Custom Attention) per hari:
```bash
python explain_model.py
```
Skrip ini akan:
* Menjalankan **Temporal Ablation Test** (Skenario A vs B) untuk memverifikasi kepekaan model terhadap urutan waktu.
* Memplot grafik rata-rata bobot perhatian (**Custom Attention Weights**) tiap hari dalam sekuens mingguan dan menyimpannya di `saved_models/v1/attention_weights.png`.
* Mengirimkan visualisasi grafik atensi (Images) dan ringkasan laporan uji waktu (Text) ke **TensorBoard** secara otomatis.

### 6. Menjalankan Visualisasi TensorBoard
Untuk melihat visualisasi kurva metrik training, diagram atensi mingguan, dan ringkasan ablation test:
```bash
tensorboard --logdir logs/
```
Buka browser dan buka alamat `http://localhost:6006/`.

---

## Panduan Format JSON Request & Response API

### POST `/api/v1/predict`
Mengirimkan 7 data harian historis beserta preferensi gol pengguna untuk mengestimasi skor produktivitas berdasarkan sekuens aktivitas mingguan (7 hari) dan analisis kelelahan.

* **Headers:**
  ```http
  Authorization: Bearer <SECRET_TOKEN_AI>
  Content-Type: application/json
  ```

* **Payload Request:**
  ```json
  {
    "user_id": "user_active_999",
    "user_goals": {
      "focus_sleep": true,
      "focus_productivity": true,
      "focus_fitness": false,
      "focus_screen_time": false
    },
    "last_7_logs": [
      {
        "log_date": "2026-05-20T00:00:00.000Z",
        "is_weekend": false,
        "sleep_duration": 6.5,
        "sleep_quality": 7,
        "study_work_duration": 8.0,
        "break_duration": 1.0,
        "physical_activity_duration": 30.0,
        "screen_time_duration": 3.5,
        "stress_level": 5,
        "mood_score": 7,
        "focus_score": 7,
        "task_planned": 5,
        "task_completed": 4,
        "completion_ratio": 0.8
      },
      ...
      {
        "log_date": "2026-05-26T00:00:00.000Z",
        "is_weekend": true,
        "sleep_duration": 8.0,
        "sleep_quality": 9,
        "study_work_duration": 2.0,
        "break_duration": 3.0,
        "physical_activity_duration": 60.0,
        "screen_time_duration": 1.5,
        "stress_level": 2,
        "mood_score": 9,
        "focus_score": 9,
        "task_planned": 2,
        "task_completed": 2,
        "completion_ratio": 1.0
      }
    ]
  }
  ```
  *(Catatan: Array `last_7_logs` wajib diisi tepat 7 log harian berurutan).*

* **Contoh Respons Sukses (200 OK):**
  ```json
  {
    "success": true,
    "user_id": "user_active_999",
    "days_logged": 7,
    "message": "Productivity analysis successfully processed over 7-day flexible input.",
    "1_main_dashboard": {
      "productivity_status": "Thriving",
      "productivity_score": 83.25,
      "prediction_confidence": 95.3,
      "probabilities": {
        "At Risk": 0.002,
        "Steady": 0.045,
        "Thriving": 0.953
      },
      "fatigue_level": 18.52,
      "completion_rate": 85.71,
      "risk_signal": "NORMAL"
    },
    "2_productivity_analytics_dashboard": {
      "daily_productivity_chart": [
        { "date": "May 20", "score": 78.5 },
        { "date": "May 21", "score": 79.2 },
        { "date": "May 22", "score": 81.0 },
        { "date": "May 23", "score": 82.5 },
        { "date": "May 24", "score": 83.0 },
        { "date": "May 25", "score": 83.2 },
        { "date": "May 26", "score": 83.25 }
      ],
      "weekly_productivity_trend": "Increasing",
      "most_dominant_activity": "Sleep",
      "peak_productive_hours": "09:00-11:00",
      "activity_heatmap": {
        "Monday": 78.5,
        "Tuesday": 79.2,
        "Wednesday": 81.0,
        "Thursday": 82.5,
        "Friday": 83.0,
        "Saturday": 83.2,
        "Sunday": 83.25
      }
    },
    "3_ai_insight_and_recommendation": {
      "condition_insight": "Current condition is prime (Thriving) with a weekly productivity score of 83.25%...",
      "performance_cause": "This performance optimization is driven by an ideal daily sleep duration...",
      "activity_recommendation": "It is recommended to maintain the current operational pacing while consistently...",
      "tomorrow_prediction": "The projected productivity score for tomorrow is 85.50%...",
      "burnout_warning": "Continued adherence to this routine is crucial to stabilize energy levels and mitigate..."
    },
    "4_similar_productivity_history": {
      "top_3_similar_days": [
        { "date": "May 24", "similarity_score": "99.7%", "historical_productivity_score": 83.0, "historical_category": "Thriving" },
        { "date": "May 25", "similarity_score": "99.9%", "historical_productivity_score": 83.2, "historical_category": "Thriving" },
        { "date": "May 23", "similarity_score": "97.3%", "historical_productivity_score": 81.0, "historical_category": "Thriving" }
      ],
      "average_productivity_from_similar_days": 82.4
    }
  }
  ```

---

## Spesifikasi Data & Feature Engineering

Model LSTM pada arsitektur ini berbasis **Time-Series Sequential**. API membutuhkan input berupa **sekuens log harian selama 7 hari berturut-turut (Timesteps = 7, Features = 14)**.

### 1. Daftar 14 Fitur Input Utama

Berikut adalah daftar seluruh fitur yang digunakan oleh model AI:

| # | Nama Fitur / Atribut | Tipe Data | Representasi / Skala |
| :-: | :--- | :---: | :--- |
| 1 | `is_weekend` | Integer | Penanda hari libur akhir pekan (0: Hari Kerja, 1: Akhir Pekan). |
| 2 | `sleep_duration` | Float | Durasi tidur dalam satuan jam (skala: `0.0` - `24.0`). |
| 3 | `study_work_duration`| Float | Durasi belajar atau bekerja dalam satuan jam (skala: `0.0` - `24.0`). |
| 4 | `break_duration` | Float | Total durasi jeda/istirahat dalam satuan jam (skala: `0.0` - `24.0`). |
| 5 | `exercise_duration` | Float | Durasi melakukan aktivitas fisik / olahraga dalam satuan menit (skala: `0.0` - `1440.0`). |
| 6 | `downtime_duration` | Float | Durasi screen time/waktu santai menatap gadget dalam satuan jam (skala: `0.0` - `24.0`). |
| 7 | `stress_level` | Integer | Skala tingkat stres harian (1: Sangat Tenang, 10: Burnout). |
| 8 | `mood_score` | Integer | Skala kondisi suasana hati (1: Sangat Buruk, 10: Bahagia). |
| 9 | `focus_score` | Integer | Skala tingkat fokus dan konsentrasi (1 - 10). |
| 10| `task_planned` | Integer | Jumlah rencana tugas harian. |
| 11| `task_completed` | Integer | Jumlah tugas yang berhasil diselesaikan. |
| 12| `completion_ratio` | Float | Rasio penyelesaian tugas (`task_completed` / `task_planned`, clipped `0.0` - `1.0`). |
| 13| `fatigue_index` | Float | Indeks kelelahan harian (hasil rekayasa fitur). |
| 14| `cumulative_fatigue` | Float | Kelelahan akumulatif menggunakan metode EWM (Exponential Weighted Mean). |

---

### 2. Rumus Rekayasa Fitur (Feature Engineering)

Sistem secara otomatis menghitung metrik kelelahan dinamis sebelum melakukan scaling data:

#### A. Rumus Indeks Kelelahan Harian (*Fatigue Index*)
Setiap hari, tingkat kelelahan dasar (*baseline fatigue*) pengguna dihitung melalui kombinasi linear dari tingkat stres, rasio durasi kerja, dan downtime:

$$FI_t = 40 \cdot \left(\frac{Stress_t}{8.0}\right) + 30 \cdot \left(\frac{Downtime\_Duration_t}{11.21}\right) + 30 \cdot \left(\frac{Work\_Duration_t}{17.405}\right)$$

*(Hasil dibatasi di rentang $[0, 100]$ menggunakan pembulatan 2 desimal).*

#### B. Rumus Kelelahan Akumulatif (*Cumulative Fatigue*)
Menggunakan **Exponential Weighted Moving Average (EWMA)** dengan $\alpha = 0.1$ untuk menangkap akumulasi kelelahan jangka panjang:

$$CF_t = \alpha \cdot FI_t + (1 - \alpha) \cdot CF_{t-1}$$

*(Hasil dibatasi di rentang $[0, 100]$ menggunakan pembulatan 2 desimal).*

#### C. Rasio Penyelesaian Tugas (*Completion Ratio*)
Rasio penyelesaian tugas harian yang dihitung secara aman untuk menghindari pembagian dengan nol:

$$CR_t = \frac{Task\_Completed_t}{\max(Task\_Planned_t, 1)}$$

*(Hasil dibatasi di rentang $[0.0, 1.0]$).*

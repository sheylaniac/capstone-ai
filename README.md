# Smart Digital Twin: LSTM Multi-Output System

Sistem Digital Twin Cerdas berbasis kecerdasan buatan (*AI*) untuk memprediksi **Skor Produktivitas** (regresi) dan **Status Kelelahan Pengguna** (klasifikasi 3-kelas: *At Risk*, *Steady*, *Thriving*) secara simultan (*multi-output*) berdasarkan sekuens data log aktivitas 7 hari ke belakang.

Sistem ini rencananya mau dilengkapi layanan rekomendasi kesehatan adaptif berbasis **Google Gemini API (`gemini-1.5-flash`)** (menyusul) dengan pertahanan *Rule-Based Fallback* otomatis jika API Key tidak disediakan. 

---

## Desain Arsitektur Folder Modular
Struktur repositori ini disusun secara berlapis (*layered architecture*) berstandar industri:

```text
dtwin/
│
├── .gitignore                  # Mengabaikan berkas virtual env, keras model, logs, dll.
├── README.md                   # Dokumentasi panduan instalasi & API
├── requirements.txt            # Dependensi Python ter-update
├── data_splitter.py            # Skrip pembagi data & pengekspor file numpy test
├── test_dtwin_system.py        # Skrip pengujian otomatis komprehensif (End-to-End)
│
├── data/                       # Tempat Penyimpanan Dataset
│   ├── raw/                    # Dataset mentah (datasetdailylogs.csv)
│   └── processed/              # Hasil data pembagian splitting untuk pengujian
│       ├── x_test.npy          # Fitur input 3D sekuensial
│       ├── y_test_reg.npy      # Skor produktivitas aktual esok hari
│       └── y_test_clf.npy      # Kategori status aktual esok hari (one-hot)
│
├── saved_models/               # Tempat Penyimpanan Model (Mendukung Versioning)
│   └── v1/                     # Folder versi model (v1, v2, dst.) sementara masih v1
│       ├── lstmmultioutput.keras # File ekspor model penuh TensorFlow versi 1
│       └── artifacts/          # Objek scaling scaler versi 1
│           ├── feature_scaler.pkl  
│           └── target_scaler.pkl   
│
├── logs/                       
│   └── fit/                    # File log pelatihan untuk dipantau via TensorBoard
│
└── src/                        # KODE UTAMA BACKEND API
    ├── app.py                  # Entrypoint server FastAPI & lifespan startup loader
    │
    ├── routes/                 # Layer 1: API Routing
    │   ├── __init__.py
    │   └── prediction_routes.py# POST /api/v1/predict
    │
    ├── controllers/            # Layer 2: API Controller & DTO
    │   ├── __init__.py
    │   └── prediction_controller.py # Validasi Pydantic, koordinasi service
    │
    ├── services/               # Layer 3: Logika Bisnis & Inferensi AI
    │   ├── __init__.py
    │   ├── ai_inference_service.py # Feature engineering dinamis & model.predict()
    │   └── genai_service.py    # Integrasi API Google Gemini & Fallback
    │
    └── ai_pipeline/            # AREA MANDIRI: Riset & Pelatihan LSTM
        ├── __init__.py
        ├── model.py            # Konstruksi model via Keras Functional API
        ├── train.py            # Pelatihan kustom loop tf.GradientTape
        │
        └── components/         # Komponen Keras Kustom Lanjutan
            ├── __init__.py
            ├── layers.py       # Custom Layer: CustomAttention
            ├── losses.py       # Custom Loss: Multi-objective Loss Function
            └── callbacks.py    # Custom Callback: Checkpoint saving
```

---

## Langkah Instalasi & Menjalankan API

### 1. Klon Repositori & Instalasi Dependensi
Pastikan python dan virtual environment sudah terpasang, lalu instal dependensi:
```bash
pip install -r requirements.txt
```

### 2. Konfigurasi Kunci API Gemini (Opsional)
Buat berkas `.env` di folder root utama untuk mengaktifkan asisten rekomendasi dinamis:
```env
GEMINI_API_KEY=AIzaSy... (masukkan kunci API Gemini Anda di sini)
MODEL_VERSION=v1
```

### 3. Menjalankan Server API FastAPI
Nyalakan server lokal uvicorn pada root direktori:
```bash
uvicorn src.app:app --reload
```
Server akan aktif di `http://127.0.0.1:8000/`. Anda bisa membuka dokumentasi interaktif Swagger UI di `http://127.0.0.1:8000/docs`.

---

## Skrip Pengujian Otomatis (`test_dtwin_system.py`)

Anda dapat menguji fungsionalitas sistem secara langsung dengan menjalankan skrip pengujian ini:
```bash
python test_dtwin_system.py
```
Skrip ini akan mensimulasikan dua jenis skenario pengujian:
1. **Pengujian Splitting Data**: Mengambil secara acak 7 hari log aktivitas aktual dari data uji splitting (`data/processed/`), memprediksi lewat API, dan membandingkan hasil prediksi dengan data aktual (*ground truth*).
2. **Pengujian Input Manual Skenario**: Mengirimkan profil simulasi ekstrim buatan pengguna (profil *At Risk* / burnout dan profil *Thriving* / bugar) untuk memverifikasi akurasi inferensi model dan kualitas rekomendasi asisten Gemini AI.

---

## Panduan Format JSON Request & Response

### HTTP Request (`POST /api/v1/predict`)
Mengirimkan data runtun waktu 7 hari sekuens log aktivitas.

**Contoh Payload JSON:**
```json
{
  "logs": [
    {"sleep_duration": 7.5, "sleep_quality": 8.0, "study_work_duration": 6.0, "break_duration": 1.0, "physical_activity_duration": 30.0, "screen_time_duration": 3.0, "stress_level": 3.0, "mood_score": 8.0, "focus_score": 8.0, "task_planned": 5, "task_completed": 4, "task_completion_rate": 0.8, "day_of_week": 0, "month": 5, "is_weekend": 0},
    {"sleep_duration": 8.0, "sleep_quality": 8.5, "study_work_duration": 6.5, "break_duration": 1.0, "physical_activity_duration": 40.0, "screen_time_duration": 2.5, "stress_level": 2.0, "mood_score": 8.5, "focus_score": 8.5, "task_planned": 6, "task_completed": 5, "task_completion_rate": 0.83, "day_of_week": 1, "month": 5, "is_weekend": 0},
    {"sleep_duration": 7.0, "sleep_quality": 7.5, "study_work_duration": 7.0, "break_duration": 0.75, "physical_activity_duration": 20.0, "screen_time_duration": 4.0, "stress_level": 4.0, "mood_score": 7.0, "focus_score": 7.0, "task_planned": 7, "task_completed": 5, "task_completion_rate": 0.71, "day_of_week": 2, "month": 5, "is_weekend": 0},
    {"sleep_duration": 7.5, "sleep_quality": 8.0, "study_work_duration": 6.0, "break_duration": 1.0, "physical_activity_duration": 30.0, "screen_time_duration": 3.0, "stress_level": 3.0, "mood_score": 8.0, "focus_score": 8.0, "task_planned": 5, "task_completed": 4, "task_completion_rate": 0.8, "day_of_week": 3, "month": 5, "is_weekend": 0},
    {"sleep_duration": 6.5, "sleep_quality": 7.0, "study_work_duration": 8.0, "break_duration": 0.5, "physical_activity_duration": 0.0, "screen_time_duration": 5.0, "stress_level": 5.0, "mood_score": 6.5, "focus_score": 7.0, "task_planned": 8, "task_completed": 5, "task_completion_rate": 0.625, "day_of_week": 4, "month": 5, "is_weekend": 0},
    {"sleep_duration": 8.5, "sleep_quality": 9.0, "study_work_duration": 2.0, "break_duration": 3.0, "physical_activity_duration": 60.0, "screen_time_duration": 2.0, "stress_level": 1.0, "mood_score": 9.5, "focus_score": 9.0, "task_planned": 2, "task_completed": 2, "task_completion_rate": 1.0, "day_of_week": 5, "month": 5, "is_weekend": 1},
    {"sleep_duration": 8.0, "sleep_quality": 8.5, "study_work_duration": 3.0, "break_duration": 2.0, "physical_activity_duration": 45.0, "screen_time_duration": 2.5, "stress_level": 2.0, "mood_score": 8.5, "focus_score": 8.5, "task_planned": 3, "task_completed": 3, "task_completion_rate": 1.0, "day_of_week": 6, "month": 5, "is_weekend": 1}
  ]
}
```

### HTTP Response JSON
Mengembalikan skor regresi esok hari, prediksi kategori status, probabilitas masing-masing kelas, dan saran dari asisten AI.

**Contoh Respons JSON:**
```json
{
  "predicted_productivity_score": 78.42,
  "predicted_category": "Thriving",
  "probabilities": {
    "At Risk": 0.0012,
    "Steady": 0.0238,
    "Thriving": 0.9750
  },
  "recommendation": "Berdasarkan sekuens log aktivitas Anda, produktivitas Anda esok hari diprediksi berada di tingkat yang sangat tinggi (78.42%) dengan kategori Thriving. Kualitas tidur Anda yang optimal sangat mendukung pemulihan energi fisik. Pertahankan rutmi kerja yang teratur, sisipkan jeda rileksasi pendek secara aktif, dan hindari begadang agar momentum positif ini terus terjaga!"
}
```

## Spesifikasi Data & Feature Engineering

Bagian ini menjelaskan spesifikasi seluruh data input yang dibutuhkan oleh API `dtwin-ai`, rentang skala validasi untuk setiap fitur, serta formula matematika aktual yang digunakan dalam proses transformasi data (*feature engineering*).

---

### 1. Rentang Skala Nilai Input (Fitur Utama & Konfigurasi)
Model LSTM pada arsitektur ini berbasis **Time-Series Sequential**. API membutuhkan input berupa **sekuens log harian selama 7 hari berturut-turut (Timesteps = 7)**. 

Berikut adalah daftar seluruh 17 fitur yang wajib dikirimkan dalam array sekuensial beserta arsitektur tipe data dan batasan skalanya:

| # | Nama Fitur / Atribut | Tipe Data | Rentang Skala | Deskripsi / Representasi Fitur |
| :-: | :--- | :---: | :---: | :--- |
| 1 | `sleep_duration` | Float | `0.0` - `24.0` | Durasi tidur total dalam satuan jam. |
| 2 | `sleep_quality` | Integer | `1` - `10` | Skala kualitas tidur (1: Sangat Buruk, 10: Sangat Nyenyak). |
| 3 | `study_work_duration`| Float | `0.0` - `24.0` | Durasi waktu yang dihabiskan untuk belajar atau bekerja (jam). |
| 4 | `break_duration` | Float | `0.0` - `24.0` | Total durasi istirahat/jeda di sela aktivitas (jam). |
| 5 | `physical_activity_duration` | Float | `0.0` - `24.0` | Durasi melakukan aktivitas fisik atau olahraga (jam). |
| 6 | `screen_time_duration`| Float | `0.0` - `24.0` | Durasi menatap layar gadget/komputer (jam). |
| 7 | `stress_level` | Integer | `1` - `10` | Skala tingkat stres harian (1: Sangat Tenang, 10: Burnout Parah). |
| 8 | `mood_score` | Integer | `1` - `10` | Skala kondisi suasana hati (1: Sangat Buruk, 10: Sangat Bahagia). |
| 9 | `focus_score` | Integer | `1` - `10` | Skala tingkat fokus dan konsentrasi dalam beraktivitas. |
| 10| `task_planned` | Integer | `0` - `100` | Jumlah tugas atau target yang direncanakan hari itu. |
| 11| `task_completed` | Integer | `0` - `100` | Jumlah tugas yang berhasil diselesaikan hari itu. |
| 12| `task_completion_rate`| Float | `0.0` - `1.0` | Rasio penyelesaian tugas (`task_completed` / `task_planned`). |
| 13| `day_of_week` | Integer | `0` - `6` | Representasi hari dalam seminggu (0: Senin, s.d. 6: Minggu). |
| 14| `month` | Integer | `1` - `12` | Angka bulan kalender berjalan (1: Januari, s.d. 12: Desember). |
| 15| `is_weekend` | Integer | `0` atau `1` | Penanda hari libur akhir pekan (0: Hari Kerja, 1: Sabtu/Minggu). |
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

*Keterangan: Jika urutan baris data historis di awal sekuens belum mencapai rentang 3 hari, sistem secara adaptif melakukan kalkulasi berdasarkan subset data harian yang tersedia (min_periods=1).*

---

### 3. Rumus Fungsi Target Prediksi & Skor Produktivitas

Model LSTM memproses sekuens matriks fitur $X$ dari 7 hari ke belakang untuk memprediksi nilai masa depan ($t+1$) berupa skor produktivitas kontinu (regresi) dan label status kondisi (klasifikasi):

$$P_{t+1}, C_{t+1} = f_{LSTM}\Big( \{X_{t-6}, X_{t-5}, \dots, X_t\} \Big)$$

Dimana matriks fitur harian $X_t$ terdiri dari vektor 17 komponen fitur yang telah dinormalisasi menggunakan `MinMaxScaler` (`feature_scaler.pkl`).

#### Penentuan Kategori Kondisi (Output Klasifikasi $C_{t+1}$):
Hasil nilai regresi skor produktivitas besok pagi ($P_{t+1}$) yang telah diubah kembali ke skala aslinya (`target_scaler.pkl`) dipetakan secara otomatis ke dalam 3 batas ambang (*threshold*) kategori kondisi fisik/mental user:

$$\text{Kondisi Besok } (C_{t+1}) = \begin{cases} 
\textbf{At Risk} & \text{jika } P_{t+1} < 55\% \\ 
\textbf{Steady} & \text{jika } 55\% \le P_{t+1} \le 67\% \\ 
\textbf{Thriving} & \text{jika } P_{t+1} > 67\% 
\end{cases}$$
---

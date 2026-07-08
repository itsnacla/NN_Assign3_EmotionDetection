# Sistem Deteksi Emosi & Stres (Emotion & Stress Detection System)

Proyek ini merupakan sistem deteksi emosi dan klasifikasi tingkat stres berbasis kecerdasan buatan menggunakan **Convolutional Neural Network (CNN)** dengan pustaka TensorFlow/Keras. Sistem dapat mengklasifikasikan wajah ke dalam 7 kategori emosi dasar, memetakan emosi tersebut ke dalam kategori biner (**Stres** vs. **Non-Stres**), serta diintegrasikan ke dalam aplikasi web interaktif berbasis Flask yang terhubung dengan Supabase untuk autentikasi dan pencatatan riwayat deteksi.

---

## 🚀 Fitur Utama

1. **Preprocessing Data Otomatis**: Mengekstrak data pixel mentah dari berkas `fer2013.csv` dan mengonversinya menjadi kumpulan gambar `.png` terstruktur sesuai dengan kategori emosinya.
2. **Model CNN Kustom**: Model Deep Learning yang dirancang menggunakan arsitektur CNN berlapis (Conv2D, MaxPooling2D, Dropout, Flatten, dan Dense) untuk klasifikasi 7 emosi wajah.
3. **Deteksi Real-Time**: Deteksi emosi wajah langsung menggunakan umpan kamera (webcam) lokal dengan bantuan OpenCV dan Haar Cascade Classifier.
4. **Metrik & Evaluasi Komprehensif**:
   - Evaluasi klasifikasi emosi 7-kelas (Confusion Matrix).
   - Klasifikasi biner Stres vs. Non-Stress (Stress: *Angry, Fearful, Sad*; Non-Stress: *Disgusted, Happy, Neutral, Surprised*).
   - Visualisasi kurva ROC (Receiver Operating Characteristic) beserta nilai AUC (Area Under Curve).
5. **Deployment Aplikasi Web**:
   - Antarmuka pengguna (dashboard) interaktif berbasis web (Flask, HTML, CSS, JavaScript).
   - Autentikasi Google OAuth via Supabase Auth.
   - Sinkronisasi telemetri logging emosi ke database Supabase, dengan *fallback* otomatis ke berkas CSV lokal (`user_emotions.csv`) apabila koneksi database atau sesi pengguna tidak aktif.
   - Grafik riwayat emosi pengguna secara dinamis di dashboard.

---

## 📂 Struktur Proyek

Berikut adalah struktur folder dan berkas yang menyusun proyek ini:

```text
Emotion-detection/
├── .env                  # Berkas konfigurasi variabel lingkungan (Supabase & Flask)
├── .env.example          # Template konfigurasi variabel lingkungan
├── .gitignore            # Daftar berkas dan folder yang diabaikan oleh Git
├── requirements.txt      # Berkas dependensi Python yang dibutuhkan
│
├── assets/               # Folder penyimpanan berkas aset statis
│   └── haarcascade_frontalface_default.xml  # Model Haar Cascade untuk deteksi wajah
│
├── data/                 # Berkas dataset (diabaikan oleh git)
│   ├── raw/              # Tempat menaruh dataset mentah 'fer2013.csv'
│   ├── train/            # Dataset gambar latihan (.png) yang diekstrak per emosi
│   └── test/             # Dataset gambar uji (.png) yang diekstrak per emosi
│
├── deployment/           # Kode aplikasi web Flask
│   ├── app.py            # Backend server Flask utama
│   ├── auth_google.py    # Integrasi modul autentikasi Google OAuth Supabase
│   ├── static/           # Berkas statis web (CSS & JS)
│   │   ├── css/
│   │   │   └── style.css # Gaya visual dashboard web
│   │   └── js/
│   │       └── main.js   # Logika kamera, pemanggilan API, dan render grafik
│   └── templates/        # Template halaman web HTML
│       ├── index.html    # Halaman dashboard utama
│       └── landing.html  # Halaman login/landing page
│
├── output/               # Output hasil pelatihan & evaluasi (diabaikan oleh git)
│   ├── logs/             # Berkas log lokal (user_emotions.csv & auth_debug.log)
│   └── plots/            # Gambar visualisasi grafik akurasi, confusion matrix, & ROC curve
│
└── src/                  # Kode sumber Python untuk pemrosesan data & model
    ├── evaluate.py       # Script evaluasi performa model di data uji
    ├── train.py          # Script utama untuk melatih model & mendeteksi langsung via webcam
    ├── models/           # Folder penyimpanan file model terlatih
    │   └── model.h5      # Berkas bobot model CNN (.h5) yang disimpan setelah training
    └── preprocessing/
        └── dataset_prepare.py  # Script untuk mengekstrak CSV menjadi gambar PNG
```

---

## 🛠️ Panduan Instalasi & Setup

Ikuti langkah-langkah berikut untuk memasang proyek ini di komputer lokal Anda:

### 1. Prasyarat Sistem
* Python 3.8 hingga 3.11.
* Webcam aktif (untuk fitur deteksi langsung & aplikasi web).
* Koneksi internet untuk integrasi database Supabase.

### 2. Pemasangan Dependensi
Buka terminal/command prompt (disarankan menggunakan PowerShell di Windows atau Bash di Unix) lalu jalankan perintah berikut:

```powershell
# 1. Masuk ke direktori proyek
cd "e:\8th Sem UUM\Neural Network\Assignment03\Emotion-detection"

# 2. Buat Virtual Environment baru
python -m venv .venv

# 3. Aktifkan Virtual Environment
# Di Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Di Linux/macOS:
source .venv/bin/activate

# 4. Perbarui pip dan pasang semua dependensi
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Konfigurasi Variabel Lingkungan (`.env`)
Salin berkas `.env.example` menjadi `.env`:

```powershell
copy .env.example .env
```

Buka berkas `.env` dan masukkan kredensial Supabase Anda:
```ini
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-public-key
FLASK_SECRET_KEY=generate-a-secure-secret-key-here
REDIRECT_URL=http://127.0.0.1:5000/dashboard
```

### 4. Setup Database Supabase
Agar fitur sinkronisasi log di web dashboard berjalan dengan lancar, buatlah tabel baru di proyek Supabase Anda dengan nama `emotion_logs` menggunakan query SQL berikut di editor SQL Supabase:

```sql
create table public.emotion_logs (
  id bigint generated by default as identity primary key,
  timestamp timestamp with time zone default timezone('utc'::text, now()) not null,
  user_id uuid references auth.users(id) on delete cascade not null,
  email text not null,
  emotion text not null,
  confidence double precision not null
);

-- Mengizinkan Row Level Security (RLS)
alter table public.emotion_logs enable row level security;

-- Buat Policy agar user hanya bisa membaca dan memasukkan log milik mereka sendiri
create policy "Allow user insert own logs" 
on public.emotion_logs 
for insert 
with check (auth.uid() = user_id);

create policy "Allow user select own logs" 
on public.emotion_logs 
for select 
using (auth.uid() = user_id);
```

Jangan lupa aktifkan Google OAuth di menu **Authentication -> Providers -> Google** pada dashboard Supabase dan isi Client ID & Client Secret dari Google Cloud Console Anda. Tambahkan `http://127.0.0.1:5000/dashboard` ke dalam **Redirect URLs** di setelan otentikasi Supabase.

---

## 🏃 Cara Menjalankan Aplikasi

### Langkah 1: Ekstraksi Dataset
Pastikan Anda sudah meletakkan file dataset `fer2013.csv` di dalam folder `data/raw/`. Jalankan perintah berikut untuk mengekstrak berkas CSV menjadi sekumpulan gambar PNG:

```bash
python src/preprocessing/dataset_prepare.py
```
*Proses ini akan menghasilkan folder baru berisi gambar-gambar berukuran 48x48 piksel di `data/train/` dan `data/test/`.*

### Langkah 2: Pelatihan Model (Training)
Untuk melatih model CNN dari awal menggunakan data yang sudah diekstrak:

```bash
python src/train.py --mode train
```
*Setelah selesai, berkas model terlatih akan disimpan di `src/models/model.h5` dan grafik akurasi/loss akan disimpan di folder `output/plots/`.*

### Langkah 3: Deteksi Langsung via Webcam (Desktop App)
Anda bisa langsung mencoba kemampuan klasifikasi model menggunakan webcam komputer dengan perintah:

```bash
python src/train.py --mode display
```
*Tekan tombol `q` pada keyboard untuk menutup jendela kamera.*

### Langkah 4: Evaluasi Model
Jalankan evaluasi untuk menghitung performa deteksi emosi 7-kelas dan analisis tingkat stres (stres vs non-stres):

```bash
python src/evaluate.py
```
*Hasil evaluasi berupa confusion matrix dan kurva ROC akan disimpan di folder `output/plots/`, dan berkas ringkasan metrik disimpan di `output/plots/evaluation_metrics_summary.txt`.*

### Langkah 5: Jalankan Server Aplikasi Web
Untuk menjalankan aplikasi web Flask interaktif:

```bash
python deployment/app.py
```
Buka browser Anda dan akses alamat `http://127.0.0.1:5000`. Anda dapat login menggunakan akun Google, memindai wajah via kamera web di halaman web secara real-time, mengunggah file foto statis, dan melihat grafik riwayat emosi yang sinkron ke database.

---

## 📊 Metrik Evaluasi & Pemetaan Stres

Pemetaan kategori stres ditentukan berdasarkan probabilitas emosi wajah yang terdeteksi:
* **Stres**: *Angry* (Marah), *Fearful* (Takut), *Sad* (Sedih)
* **Non-Stres**: *Disgusted* (Jijik), *Happy* (Senang), *Neutral* (Biasa), *Surprised* (Terkejut)

Seluruh data performa pelatihan dan visualisasi metrik model dapat dipantau langsung pada berkas hasil yang berada di folder [output/plots/](file:///e:/8th%20Sem%20UUM/Neural%20Network/Assignment03/Emotion-detection/output/plots).

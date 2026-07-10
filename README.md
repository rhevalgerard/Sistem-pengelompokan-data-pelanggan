# Aplikasi Segmentasi Pelanggan (Streamlit)

Aplikasi ini **memuat model K-Means yang sudah dilatih** (dari notebook
`customer_segmentation_kmeans.ipynb`, dataset Fashion Retail Sales) dan
menerapkannya ke data pelanggan baru. Model **tidak dilatih ulang** setiap
kali dijalankan — ini beda mendasar dari versi web JavaScript sebelumnya.

## Isi folder

- `app.py` — aplikasi Streamlit-nya
- `kmeans_model.joblib` — model K-Means yang sudah dilatih (k=4)
- `scaler.joblib` — StandardScaler yang sudah di-fit saat training
- `pca.joblib` — PCA yang sudah di-fit saat training (untuk visualisasi konsisten)
- `metadata.joblib` — info fitur & konfigurasi yang dipakai model
- `requirements.txt` — daftar library yang dibutuhkan

## Cara menjalankan

```bash
# 1. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install semua library yang dibutuhkan
pip install -r requirements.txt

# 3. Jalankan aplikasi
streamlit run app.py
```

Browser akan otomatis terbuka ke `http://localhost:8501`.

## Cara pakai

1. Upload file CSV dengan kolom yang sama seperti data training:
   `Unit_Price, Quantity, Discount_Amount, Total_Amount, Session_Duration_Minutes,
   Pages_Viewed, Delivery_Time_Days, Customer_Rating, Is_Returning_Customer`
2. Aplikasi otomatis memproses (log-transform, scaling) persis seperti saat training,
   lalu memprediksi segmen tiap pelanggan pakai model yang sudah ada.
3. Lihat visualisasi & unduh hasilnya sebagai CSV baru berisi kolom `Cluster`.

## Penting: model ini khusus untuk skema data Fashion Retail Sales

Model `.joblib` ini dilatih khusus pada dataset **Fashion Retail Sales** (kolom-kolom
di atas). Kalau kamu ingin pakai skema data lain (misalnya dataset RFM Global Fashion
Retail Stores), model perlu dilatih ulang dulu di notebook yang sesuai, baru
`.joblib`-nya diganti di folder ini.

## Deploy supaya bisa diakses orang lain (opsional)

Kalau ingin dipakai orang awam tanpa install apa pun di komputer mereka, folder ini
bisa di-deploy gratis ke **Streamlit Community Cloud** (streamlit.io/cloud):
push folder ini ke repository GitHub, lalu hubungkan repo tersebut di Streamlit Cloud.

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# 1. Memuat data dari file Excel
file_path = 'Data_Chemical_Filtered_Tinstab_MT620.xlsx'
print(f"Sedang membaca file: {file_path}...")
df = pd.read_excel(file_path)

# 2. DATA CLEANING: Mengubah tipe data ke Numerik
# Masalah: Data di Excel seringkali dianggap sebagai "Object" (Teks) oleh Pandas
# karena ada karakter aneh atau format yang salah. KNN tidak bisa menghitung jarak pada Teks.
# Kita harus paksa ubah kolom-kolom target menjadi angka (Float/Int).

# 2. DATA PREPARATION: Memisahkan Kolom Numerik dan Kategori
# Berdasarkan pengecekan, kolom 'APE' (Appearance) adalah Kategori (Isinya 'Clear', dll).
# KNN Imputer Scikit-Learn HANYA bekerja untuk Angka.
# Jadi strategi kita:
# - Kolom 'APE': Isi missing value dengan Modus (Nilai terbanyak muncul)
# - Kolom Lain (Transmission, Tinx, dll): Isi missing value dengan KNN (Rata-rata tetangga)

# Daftar kolom Numerik (Angka)
numeric_columns = [
    'Transmission', 'Tinx', 'RI', 'SG', 'Acid', 
    'Sulfur', 'Water', 'Mono', 'Yellow', 'EH', 'Visco', 'PT'
]

# Daftar kolom Kategori (Teks)
categorical_columns = ['APE']

print("\nSedang memproses imputasi...")

# --- PROSES 1: IMPUTASI KATEGORI (APE) ---
# Mengisi yang kosong dengan nilai yang paling sering muncul (Modus)
for col in categorical_columns:
    mode_val = df[col].mode()[0]  # Ambil nilai terbanyak
    print(f"Mengisi missing value di kolom '{col}' dengan modus: '{mode_val}'")
    df[col] = df[col].fillna(mode_val)

# --- PROSES 2: IMPUTASI NUMERIK (KNN) ---
print("\nMengonversi kolom numerik dan menjalankan KNN...")
for col in numeric_columns:
    # Ubah ke angka, jika error jadi NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Inisialisasi KNN Imputer
imputer = KNNImputer(n_neighbors=5)

# Lakukan KNN hanya pada kolom numerik
imputed_data_array = imputer.fit_transform(df[numeric_columns])

# Masukkan kembali hasil imputasi ke DataFrame
df[numeric_columns] = pd.DataFrame(imputed_data_array, columns=numeric_columns)

# 7. Cek hasil akhir
print("\nSelesai! Jumlah missing value SETELAH imputasi:")
print(df[numeric_columns + categorical_columns].isna().sum())

# --- PROSES 3: ENCODING KATEGORI (APE) ---
print("\n--- PROSES ENCODING ---")
print(f"Nilai unik di kolom 'APE' sebelum encoding:")
print(df['APE'].value_counts())

# Encoding: Clear = 1, nilai lainnya = 0
df['APE_encoded'] = df['APE'].apply(lambda x: 1 if str(x).lower() == 'clear' else 0)

print(f"\nHasil encoding kolom 'APE' menjadi 'APE_encoded':")
print(df[['APE', 'APE_encoded']].value_counts())
print(f"\nDistribusi nilai encoded:")
print(df['APE_encoded'].value_counts())

# 8. Menyimpan hasil
output_file = 'Data_Chemical_Imputed_Encoded.xlsx'
df.to_excel(output_file, index=False)
print(f"\nFile berhasil disimpan: {output_file}")

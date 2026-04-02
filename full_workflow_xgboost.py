import pandas as pd
import numpy as np
# import seaborn as sns # Uncomment di Google Colab jika ingin plot
# import matplotlib.pyplot as plt # Uncomment di Google Colab jika ingin plot
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ==========================================
# BAGIAN 1: LOAD & PREPROCESSING DATA
# ==========================================
print("=== MULAI PROSES: LOAD & IMPUTASI DATA ===")

# 1. Load Data
# Ganti nama file ini jika di Google Colab anda mengupload dengan nama berbeda
file_path = 'Data_Chemical_Filtered_Tinstab_MT620.xlsx' 
try:
    df = pd.read_excel(file_path)
    print(f"Berhasil membaca file: {file_path}")
except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan. Pastikan sudah diupload ke Files (logo folder di kiri Colab).")
    exit()

# 2. Persiapan Imputasi
# Memisahkan kolom numerik dan kategori untuk perlakuan berbeda
numeric_columns = [
    'Transmission', 'Tinx', 'RI', 'SG', 'Acid', 
    'Sulfur', 'Water', 'Mono', 'Yellow', 'EH', 'Visco', 'PT'
]
categorical_columns = ['APE']

# A. Imputasi Kategori (APE) dengan Modus
for col in categorical_columns:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"Imputasi kolom '{col}' dengan modus: {mode_val}")

# B. Imputasi Numerik dengan KNN
print("Melakukan KNN Imputation untuk kolom numerik...")
# Ubah ke angka (coerce error jadi NaN)
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# KNN Process
imputer = KNNImputer(n_neighbors=5)
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
print("KNN Imputation Selesai.\n")

# ==========================================
# BAGIAN 2: LABELING (KLASIFIKASI MANUAL)
# ==========================================
print("=== MULAI PROSES: LABELING DATA ===")

def check_status(row):
    # Logika Spesifikasi Produk
    if not (row['Transmission'] > 98): return 'Not Passed'
    if str(row['APE']).strip() != 'Clear': return 'Not Passed'
    if not (18.8 <= row['Tinx'] <= 19.2): return 'Not Passed'
    if not (1.507 <= row['RI'] <= 1.511): return 'Not Passed'
    if not (1.160 <= row['SG'] <= 1.180): return 'Not Passed'
    if not (row['Acid'] <= 3.0): return 'Not Passed'
    if not (11.5 <= row['Sulfur'] <= 12.5): return 'Not Passed'
    if not (row['Water'] < 3.5): return 'Not Passed'
    if not (19.5 <= row['Mono'] <= 28.5): return 'Not Passed'
    if not (row['Yellow'] < 9.0): return 'Not Passed'
    if not (row['EH'] < 0.7): return 'Not Passed'
    if not (40 <= row['Visco'] <= 80): return 'Not Passed'
    if not (row['PT'] <= 30): return 'Not Passed'
    return 'Passed'

df['Status_Akhir'] = df.apply(check_status, axis=1)
print("Distribusi Kelas Target (Status_Akhir):")
print(df['Status_Akhir'].value_counts())
print("\n")

# ==========================================
# BAGIAN 3: PERSIAPAN MODELING (ENCODING & SPLIT)
# ==========================================
print("=== MULAI PROSES: TRAINING MODEL XGBOOST ===")

# 1. Encoding
# Mengubah data teks menjadi angka agar bisa dibaca mesin
# 'APE' (Clear) -> Angka
le_ape = LabelEncoder()
df['APE_Encoded'] = le_ape.fit_transform(df['APE'].astype(str))

# 'Status_Akhir' (Target) -> Angka (Passed=1, Not Passed=0 biasanya)
le_status = LabelEncoder()
df['Target'] = le_status.fit_transform(df['Status_Akhir'])

# Tampilkan mapping agar paham (misal 0=Not Passed, 1=Passed)
print("Mapping Target:")
for i, item in enumerate(le_status.classes_):
    print(f"{item}  => {i}")

# 2. Tentukan Fitur (X) dan Target (y)
# X adalah data kimia (Input)
# y adalah Status Akhir (Output)
features = numeric_columns + ['APE_Encoded']
X = df[features]
y = df['Target']

# 3. Split Data (80% Training, 20% Validation/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nJumlah Data Training: {len(X_train)}")
print(f"Jumlah Data Validation: {len(X_test)}\n")

# ==========================================
# BAGIAN 4: TRAINING & K-FOLD CROSS VALIDATION
# ==========================================

# 1. Inisialisasi Model XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 2. K-Fold Cross Validation
# Melakukan validasi silang sebanyak 5 kali lipatan (K=5) pada data training
print("Melakukan 5-Fold Cross Validation...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

print(f"Skor Akurasi tiap Fold: {cv_results}")
print(f"Rata-rata Akurasi Cross-Validation: {cv_results.mean():.4f} ({cv_results.mean()*100:.2f}%)")

# 3. Training Model pada seluruh Data Training
print("\nMelatih Model Final pada Data Training 80%...")
model.fit(X_train, y_train)

# ==========================================
# BAGIAN 5: EVALUASI & PREDIKSI
# ==========================================
print("\n=== HASIL EVALUASI PADA DATA VALIDASI (20%) ===")

# 1. Prediksi
y_pred = model.predict(X_test)

# 2. Akurasi
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi Model pada Data Validasi: {acc:.4f} ({acc*100:.2f}%)")

# 3. Classification Report (Presisi, Recall, F1-Score)
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=le_status.classes_))

# 4. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualisasi Confusion Matrix Sederhana (Text)
print("\nConfusion Matrix:")
print("                 Prediksi: Not Passed   Prediksi: Passed")
print(f"Aktual: Not Passed      {cm[0][0]}                   {cm[0][1]}")
print(f"Aktual: Passed          {cm[1][0]}                   {cm[1][1]}")

# Jika di Colab, bisa un-comment baris ini untuk menampilkan plot gambar
# plt.figure(figsize=(6,4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_status.classes_, yticklabels=le_status.classes_)
# plt.xlabel('Prediksi')
# plt.ylabel('Aktual')
# plt.title('Confusion Matrix XGBoost')
# plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("PROSES TRAINING MODEL XGBoost")
print("="*70)

# 1. Load data hasil imputation dan encoding
print("\n[1] Memuat data...")
file_path = 'Data_Chemical_Imputed_Encoded.xlsx'
df = pd.read_excel(file_path)
print(f"Total data: {len(df)} baris")
print(f"Total kolom: {len(df.columns)} kolom")

# 2. Pilih fitur (X) dan target (y)
print("\n[2] Memilih fitur dan target...")

# Fitur numerik yang akan digunakan untuk prediksi
feature_columns = [
    'Transmission', 'Tinx', 'RI', 'SG', 'Acid', 
    'Sulfur', 'Water', 'Mono', 'Yellow', 'EH', 'Visco', 'PT'
]

# Target adalah APE_encoded (1 = Clear, 0 = Tidak Clear)
target_column = 'APE_encoded'

X = df[feature_columns]
y = df[target_column]

print(f"Fitur (X): {feature_columns}")
print(f"Target (y): {target_column}")
print(f"\nDistribusi target:")
print(y.value_counts())

# 3. Split data 80:20 (80% Training, 20% Validasi)
print("\n[3] Melakukan split data 80:20...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2,  # 20% untuk validasi
    random_state=42,  # Untuk reproduktifitas
    stratify=y  # Memastikan proporsi kelas seimbang
)

print(f"Data Training: {len(X_train)} baris ({len(X_train)/len(X)*100:.1f}%)")
print(f"Data Validasi: {len(X_val)} baris ({len(X_val)/len(X)*100:.1f}%)")
print(f"\nDistribusi target di Training:")
print(y_train.value_counts())
print(f"\nDistribusi target di Validasi:")
print(y_val.value_counts())

# 4. Training Model XGBoost
print("\n[4] Training model XGBoost...")
model = XGBClassifier(
    n_estimators=100,  # Jumlah pohon
    max_depth=5,       # Kedalaman maksimal pohon
    learning_rate=0.1, # Learning rate
    random_state=42,
    eval_metric='logloss'
)

# Mulai training
model.fit(X_train, y_train)
print("Training selesai!")

# 5. Evaluasi Model
print("\n[5] Evaluasi Model...")
print("\n--- HASIL PADA DATA TRAINING ---")
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Akurasi Training: {train_accuracy*100:.2f}%")

print("\n--- HASIL PADA DATA VALIDASI ---")
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Akurasi Validasi: {val_accuracy*100:.2f}%")

# Classification Report
print("\n--- CLASSIFICATION REPORT (Validasi) ---")
print(classification_report(y_val, y_val_pred, target_names=['Tidak Clear (0)', 'Clear (1)']))

# Confusion Matrix
print("\n--- CONFUSION MATRIX (Validasi) ---")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)

# 6. Feature Importance
print("\n[6] Feature Importance (Fitur Terpenting)...")
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

# 7. Visualisasi
print("\n[7] Membuat visualisasi...")

# Plot 1: Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance - XGBoost Model')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("Grafik Feature Importance disimpan: feature_importance.png")

# Plot 2: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Tidak Clear (0)', 'Clear (1)'],
            yticklabels=['Tidak Clear (0)', 'Clear (1)'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Data Validasi')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Grafik Confusion Matrix disimpan: confusion_matrix.png")

# 8. Simpan Model
print("\n[8] Menyimpan model...")
import pickle
model_file = 'xgboost_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"Model disimpan: {model_file}")

# 9. Ringkasan Akhir
print("\n" + "="*70)
print("RINGKASAN HASIL TRAINING")
print("="*70)
print(f"Model: XGBoost Classifier")
print(f"Total Data: {len(df)} baris")
print(f"Data Training: {len(X_train)} baris (80%)")
print(f"Data Validasi: {len(X_val)} baris (20%)")
print(f"Jumlah Fitur: {len(feature_columns)}")
print(f"Akurasi Training: {train_accuracy*100:.2f}%")
print(f"Akurasi Validasi: {val_accuracy*100:.2f}%")
print(f"\nFitur Terpenting:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")
print("="*70)
print("\nPROSES SELESAI!")

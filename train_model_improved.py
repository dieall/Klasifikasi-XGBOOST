import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

print("="*70)
print("TRAINING MODEL XGBOOST DENGAN HANDLING IMBALANCED DATA")
print("="*70)

# 1. Load data
print("\n[1] Memuat data...")
file_path = 'Data_Chemical_Imputed_Encoded.xlsx'
df = pd.read_excel(file_path)
print(f"Total data: {len(df)} baris")

# 2. Pilih fitur dan target
feature_columns = [
    'Transmission', 'Tinx', 'RI', 'SG', 'Acid', 
    'Sulfur', 'Water', 'Mono', 'Yellow', 'EH', 'Visco', 'PT'
]
target_column = 'APE_encoded'

X = df[feature_columns]
y = df[target_column]

print(f"\nDistribusi target:")
print(y.value_counts())
print(f"Imbalance ratio: {y.value_counts()[1]}/{y.value_counts()[0]} = {y.value_counts()[1]/y.value_counts()[0]:.1f}:1")

# 3. Split data SEBELUM SMOTE (PENTING!)
print("\n[2] Split data 80:20 SEBELUM balancing...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Data Training: {len(X_train)} baris")
print(f"Data Validasi: {len(X_val)} baris")
print(f"\nDistribusi Training sebelum SMOTE:")
print(y_train.value_counts())

# 4. Apply SMOTE HANYA pada data training
print("\n[3] Applying SMOTE untuk balance data training...")
smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.value_counts()[0]-1))
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nDistribusi Training SETELAH SMOTE:")
print(pd.Series(y_train_balanced).value_counts())
print(f"Data training setelah SMOTE: {len(X_train_balanced)} baris")

# 5. Training Model dengan Regularization
print("\n[4] Training model XGBoost dengan regularization...")
model = XGBClassifier(
    n_estimators=50,          # Kurangi jumlah pohon
    max_depth=3,              # Kurangi kedalaman (lebih sederhana)
    learning_rate=0.05,       # Learning rate lebih kecil
    subsample=0.8,            # Gunakan 80% data per pohon
    colsample_bytree=0.8,     # Gunakan 80% fitur per pohon
    reg_alpha=1,              # L1 regularization
    reg_lambda=1,             # L2 regularization
    min_child_weight=3,       # Minimal sampel di leaf
    random_state=42,
    scale_pos_weight=1,       # Sudah balanced dengan SMOTE
    eval_metric='logloss'
)

model.fit(X_train_balanced, y_train_balanced)
print("Training selesai!")

# 6. Evaluasi dengan Cross-Validation pada data ASLI (tanpa SMOTE)
print("\n[5] Cross-Validation pada data training ASLI (tanpa SMOTE)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
print(f"CV F1-Score (5-Fold): {cv_scores}")
print(f"Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 7. Evaluasi pada data training
print("\n[6] Evaluasi pada Data Training...")
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
print(f"Akurasi Training: {train_accuracy*100:.2f}%")
print(f"F1-Score Training: {train_f1:.4f}")

# 8. Evaluasi pada data validasi
print("\n[7] Evaluasi pada Data Validasi...")
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')
val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')

print(f"Akurasi Validasi: {val_accuracy*100:.2f}%")
print(f"F1-Score Validasi: {val_f1:.4f}")
print(f"Precision Validasi: {val_precision:.4f}")
print(f"Recall Validasi: {val_recall:.4f}")

# Classification Report
print("\n--- CLASSIFICATION REPORT (Validasi) ---")
print(classification_report(y_val, y_val_pred, 
                          target_names=['Tidak Clear (0)', 'Clear (1)'],
                          zero_division=0))

# Confusion Matrix
print("\n--- CONFUSION MATRIX (Validasi) ---")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)

# 9. Feature Importance
print("\n[8] Feature Importance...")
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

# 10. Visualisasi
print("\n[9] Membuat visualisasi...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Feature Importance
axes[0, 0].barh(feature_importance['Feature'], feature_importance['Importance'])
axes[0, 0].set_xlabel('Importance Score')
axes[0, 0].set_ylabel('Features')
axes[0, 0].set_title('Feature Importance - XGBoost Model')

# Plot 2: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['Tidak Clear (0)', 'Clear (1)'],
            yticklabels=['Tidak Clear (0)', 'Clear (1)'])
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_title('Confusion Matrix - Data Validasi')

# Plot 3: Distribusi Data
dist_data = pd.DataFrame({
    'Original Train': y_train.value_counts().sort_index(),
    'After SMOTE': pd.Series(y_train_balanced).value_counts().sort_index(),
    'Validation': y_val.value_counts().sort_index()
})
dist_data.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Distribusi Data sebelum & sesudah SMOTE')
axes[1, 0].set_xlabel('Class')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend(['Training Original', 'Training + SMOTE', 'Validasi'])

# Plot 4: Performance Metrics
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
values = [val_accuracy, val_f1, val_precision, val_recall]
axes[1, 1].bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
axes[1, 1].set_ylim([0, 1.1])
axes[1, 1].set_title('Metrik Performa pada Data Validasi')
axes[1, 1].set_ylabel('Score')
for i, v in enumerate(values):
    axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('model_evaluation_improved.png', dpi=300, bbox_inches='tight')
print("Grafik evaluasi disimpan: model_evaluation_improved.png")

# 11. Simpan Model
print("\n[10] Menyimpan model...")
import pickle
model_file = 'xgboost_model_improved.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"Model disimpan: {model_file}")

# 12. Ringkasan
print("\n" + "="*70)
print("RINGKASAN HASIL TRAINING DENGAN HANDLING IMBALANCED DATA")
print("="*70)
print(f"Model: XGBoost dengan SMOTE + Regularization")
print(f"Total Data: {len(df)} baris")
print(f"Data Training Original: {len(X_train)} baris")
print(f"Data Training + SMOTE: {len(X_train_balanced)} baris")
print(f"Data Validasi: {len(X_val)} baris (TANPA SMOTE)")
print(f"\nPerforma pada Data Validasi:")
print(f"  - Akurasi: {val_accuracy*100:.2f}%")
print(f"  - F1-Score: {val_f1:.4f}")
print(f"  - Precision: {val_precision:.4f}")
print(f"  - Recall: {val_recall:.4f}")
print(f"\nCross-Validation F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"\nFitur Terpenting (Top 3):")
for idx, row in feature_importance.head(3).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")
print("="*70)
print("\nCATATAN:")
print("- SMOTE hanya diterapkan pada data TRAINING")
print("- Data VALIDASI tetap ASLI (tidak di-balance)")
print("- Model dengan regularization untuk hindari overfitting")
print("="*70)

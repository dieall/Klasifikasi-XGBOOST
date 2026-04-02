import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

print("="*70)
print("TRAINING MODEL XGBOOST - VERSI REALISTIS")
print("="*70)

# 1. Load data
print("\n[1] Memuat data...")
file_path = 'Data_Chemical_Imputed_Encoded.xlsx'
df = pd.read_excel(file_path)

# 2. Pilih fitur dan target
feature_columns = [
    'Transmission', 'Tinx', 'RI', 'SG', 'Acid', 
    'Sulfur', 'Water', 'Mono', 'Yellow', 'EH', 'Visco', 'PT'
]
target_column = 'APE_encoded'

X = df[feature_columns]
y = df[target_column]

print(f"Total data: {len(df)} baris")
print(f"Distribusi target: {y.value_counts().to_dict()}")

# 3. Split data dengan stratifikasi
print("\n[2] Split data 80:20...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Data Training: {len(X_train)} baris")
print(f"Data Validasi: {len(X_val)} baris")

# 4. Apply SMOTE dengan sampling_strategy yang lebih konservatif
print("\n[3] Applying SMOTE (Conservative)...")
# Buat rasio 1:5 (20% minority) daripada 1:1
target_ratio = 0.2  # 20% minority class
minority_samples = int(y_train.value_counts()[1] * target_ratio)
smote = SMOTE(
    sampling_strategy={0: minority_samples},
    random_state=42, 
    k_neighbors=min(3, y_train.value_counts()[0]-1)
)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Distribusi Training SETELAH SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# 5. Training Model dengan STRONG Regularization
print("\n[4] Training model dengan STRONG regularization...")
model = XGBClassifier(
    n_estimators=30,          # Sangat sedikit pohon
    max_depth=2,              # Pohon sangat dangkal
    learning_rate=0.01,       # Learning rate sangat kecil
    subsample=0.6,            # Gunakan hanya 60% data
    colsample_bytree=0.6,     # Gunakan hanya 60% fitur
    colsample_bylevel=0.6,    # Sampling di setiap level
    reg_alpha=10,             # L1 regularization TINGGI
    reg_lambda=10,            # L2 regularization TINGGI
    min_child_weight=5,       # Minimal sampel di leaf
    gamma=1,                  # Minimum loss reduction
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_balanced, y_train_balanced)
print("Training selesai!")

# 6. Cross-Validation
print("\n[5] Cross-Validation (5-Fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
print(f"CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 7. Evaluasi
print("\n[6] Evaluasi Model...")
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')
val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
val_recall = recall_score(y_val, y_val_pred, average='weighted')

print(f"\nAkurasi Training: {train_accuracy*100:.2f}%")
print(f"Akurasi Validasi: {val_accuracy*100:.2f}%")
print(f"F1-Score Validasi: {val_f1:.4f}")

# Classification Report
print("\n--- CLASSIFICATION REPORT (Validasi) ---")
print(classification_report(y_val, y_val_pred, 
                          target_names=['Tidak Clear (0)', 'Clear (1)'],
                          zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
print("--- CONFUSION MATRIX (Validasi) ---")
print(cm)
print(f"\nInterpretasi:")
print(f"  True Negatives (TN): {cm[0,0]}")
print(f"  False Positives (FP): {cm[0,1]}")
print(f"  False Negatives (FN): {cm[1,0]}")
print(f"  True Positives (TP): {cm[1,1]}")

# 8. Feature Importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n[7] Feature Importance:")
print(feature_importance)

# 9. Visualisasi Lengkap
print("\n[8] Membuat visualisasi...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Feature Importance (Large)
ax1 = fig.add_subplot(gs[0:2, 0])
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
ax1.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
ax1.set_xlabel('Importance Score', fontsize=11)
ax1.set_ylabel('Features', fontsize=11)
ax1.set_title('Feature Importance - XGBoost Model', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False,
            xticklabels=['Tidak Clear', 'Clear'],
            yticklabels=['Tidak Clear', 'Clear'])
ax2.set_ylabel('Actual', fontsize=10)
ax2.set_xlabel('Predicted', fontsize=10)
ax2.set_title('Confusion Matrix', fontsize=11, fontweight='bold')

# Plot 3: Performance Metrics
ax3 = fig.add_subplot(gs[1, 1])
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
values = [val_accuracy, val_f1, val_precision, val_recall]
bars = ax3.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax3.set_ylim([0, 1.1])
ax3.set_title('Metrik Performa (Validasi)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Score', fontsize=10)
ax3.grid(axis='y', alpha=0.3)
for i, (bar, v) in enumerate(zip(bars, values)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 4: Data Distribution
ax4 = fig.add_subplot(gs[2, 0])
dist_data = {
    'Training\nOriginal': y_train.value_counts().to_dict(),
    'Training\n+ SMOTE': pd.Series(y_train_balanced).value_counts().to_dict(),
    'Validasi': y_val.value_counts().to_dict()
}
x = np.arange(len(dist_data))
width = 0.35
class_0 = [dist_data[k].get(0, 0) for k in dist_data]
class_1 = [dist_data[k].get(1, 0) for k in dist_data]
ax4.bar(x - width/2, class_0, width, label='Tidak Clear (0)', color='#ff7f0e')
ax4.bar(x + width/2, class_1, width, label='Clear (1)', color='#1f77b4')
ax4.set_ylabel('Jumlah Sampel', fontsize=10)
ax4.set_title('Distribusi Data', fontsize=11, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(dist_data.keys(), fontsize=9)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Cross-Validation Scores
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(range(1, len(cv_scores)+1), cv_scores, marker='o', linewidth=2, markersize=8)
ax5.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
ax5.fill_between(range(1, len(cv_scores)+1), 
                 cv_scores.mean() - cv_scores.std(),
                 cv_scores.mean() + cv_scores.std(),
                 alpha=0.2, color='red')
ax5.set_xlabel('Fold', fontsize=10)
ax5.set_ylabel('F1-Score', fontsize=10)
ax5.set_title('Cross-Validation Results', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

# Plot 6: Model Hyperparameters Info
ax6 = fig.add_subplot(gs[0:2, 2])
ax6.axis('off')
info_text = f"""
MODEL CONFIGURATION
{'='*30}

Model: XGBoost Classifier

Hyperparameters:
  • n_estimators: {model.n_estimators}
  • max_depth: {model.max_depth}
  • learning_rate: {model.learning_rate}
  • subsample: {model.subsample}
  • colsample_bytree: {model.colsample_bytree}
  • reg_alpha: {model.reg_alpha}
  • reg_lambda: {model.reg_lambda}
  • min_child_weight: {model.min_child_weight}

Data Split:
  • Training: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)
  • Validasi: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)
  • Training + SMOTE: {len(X_train_balanced)}

Performance:
  • Train Acc: {train_accuracy*100:.2f}%
  • Val Acc: {val_accuracy*100:.2f}%
  • CV F1: {cv_scores.mean():.4f}

Top 3 Features:
"""
for idx, row in feature_importance.head(3).iterrows():
    info_text += f"  • {row['Feature']}: {row['Importance']:.4f}\n"

ax6.text(0.1, 0.95, info_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Plot 7: Training vs Validation Comparison
ax7 = fig.add_subplot(gs[2, 2])
comparison = ['Training', 'Validasi']
train_val_acc = [train_accuracy, val_accuracy]
bars = ax7.bar(comparison, train_val_acc, color=['#2ca02c', '#d62728'])
ax7.set_ylim([0, 1.1])
ax7.set_ylabel('Accuracy', fontsize=10)
ax7.set_title('Training vs Validation', fontsize=11, fontweight='bold')
ax7.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height*100:.2f}%', ha='center', va='bottom', fontsize=10)

plt.suptitle('XGBoost Model - Complete Evaluation Report', 
            fontsize=14, fontweight='bold', y=0.995)

plt.savefig('model_evaluation_complete.png', dpi=300, bbox_inches='tight')
print("Grafik evaluasi lengkap disimpan: model_evaluation_complete.png")

# 10. Simpan Model
import pickle
model_file = 'xgboost_model_final.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel disimpan: {model_file}")

# 11. Ringkasan Final
print("\n" + "="*70)
print("RINGKASAN HASIL TRAINING MODEL")
print("="*70)
print(f"Model: XGBoost dengan SMOTE + Strong Regularization")
print(f"\nData:")
print(f"  • Total: {len(df)} baris")
print(f"  • Training: {len(X_train)} baris (80%)")
print(f"  • Training + SMOTE: {len(X_train_balanced)} baris")
print(f"  • Validasi: {len(X_val)} baris (20%)")
print(f"\nPerforma:")
print(f"  • Akurasi Training: {train_accuracy*100:.2f}%")
print(f"  • Akurasi Validasi: {val_accuracy*100:.2f}%")
print(f"  • F1-Score: {val_f1:.4f}")
print(f"  • Precision: {val_precision:.4f}")
print(f"  • Recall: {val_recall:.4f}")
print(f"  • CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"\nTop 3 Fitur Terpenting:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"  • {row['Feature']}: {row['Importance']:.4f}")
print("\n" + "="*70)
print("CATATAN PENTING:")
print("  • SMOTE hanya diterapkan pada data TRAINING")
print("  • Data VALIDASI tetap ASLI (tidak di-balance)")
print("  • Strong regularization untuk mencegah overfitting")
print("  • Model dilatih dengan hyperparameter konservatif")
print("="*70)

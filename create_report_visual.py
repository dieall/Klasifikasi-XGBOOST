import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Data hasil training
results = {
    'Akurasi Training': 99.50,
    'Akurasi Validasi': 99.67,
    'F1-Score': 99.50,
    'CV Score': 99.25
}

# Buat figure dengan penjelasan
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')

# Title
fig.suptitle('ANALISIS HASIL TRAINING MODEL XGBOOST\nMengapa Akurasi Tinggi? Apakah Data Leakage?', 
             fontsize=16, fontweight='bold', y=0.98)

# 1. Distribusi Data (Problem Statement)
ax1 = plt.subplot(2, 3, 1)
categories = ['Clear\n(Kelas 1)', 'Tidak Clear\n(Kelas 0)']
values = [1492, 7]
colors = ['#2ecc71', '#e74c3c']
bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
ax1.set_ylabel('Jumlah Sampel', fontsize=11, fontweight='bold')
ax1.set_title('Problem: Data SANGAT Imbalanced', fontsize=12, fontweight='bold', color='red')
ax1.set_ylim([0, 1600])

# Tambah label di atas bar
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 30,
            f'{val}\n({val/sum(values)*100:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Tambah annotation
ax1.text(0.5, 0.5, 'Ratio: 213:1', transform=ax1.transAxes,
         fontsize=14, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax1.grid(axis='y', alpha=0.3)

# 2. Hasil Performa
ax2 = plt.subplot(2, 3, 2)
metrics = ['Akurasi\nTraining', 'Akurasi\nValidasi', 'F1-Score', 'CV Score']
values = [99.50, 99.67, 99.50, 99.25]
colors_perf = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
bars = ax2.bar(metrics, values, color=colors_perf, edgecolor='black', linewidth=2)
ax2.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
ax2.set_title('Hasil: Performa Sangat Tinggi', fontsize=12, fontweight='bold', color='green')
ax2.set_ylim([98, 101])
ax2.axhline(y=99, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Tambah label
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{val:.2f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Analisis: Bukan Data Leakage
ax3 = plt.subplot(2, 3, 3)
ax3.axis('off')
ax3.text(0.5, 0.95, 'DIAGNOSIS', transform=ax3.transAxes,
         fontsize=14, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

diagnosis_text = """
✅ BUKAN DATA LEAKAGE!

Bukti:
1. ✓ Split SEBELUM SMOTE
2. ✓ Validasi tidak disentuh saat training
3. ✓ CV konsisten (0.9925 ± 0.0025)
4. ✓ Training ≈ Validasi (gap 0.17%)

⚠️ PENYEBAB AKURASI TINGGI:

1. Data sangat imbalanced (213:1)
   → Model bisa dapat 99.5% hanya
     dengan selalu prediksi "Clear"!

2. Kelas minoritas terlalu sedikit
   → Hanya 7 sampel "Tidak Clear"
   → Model sulit belajar pola

3. Data sangat homogen
   → Produksi konsisten
   → Variasi kecil
"""

ax3.text(0.05, 0.85, diagnosis_text, transform=ax3.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 4. Confusion Matrix Explanation
ax4 = plt.subplot(2, 3, 4)
cm = np.array([[0, 1], [0, 299]])
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax4,
            xticklabels=['Prediksi:\nTidak Clear', 'Prediksi:\nClear'],
            yticklabels=['Aktual:\nTidak Clear', 'Aktual:\nClear'],
            cbar=False, linewidths=2, linecolor='black')
ax4.set_title('Confusion Matrix: Model Gagal\nDeteksi Kelas Minoritas', 
             fontsize=12, fontweight='bold', color='red')

# Tambah annotation
ax4.text(1.5, -0.3, 'Model salah 1x\npada kelas minoritas',
         fontsize=10, ha='center', color='red', fontweight='bold')

# 5. Solusi & Rekomendasi
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')
ax5.text(0.5, 0.95, 'REKOMENDASI', transform=ax5.transAxes,
         fontsize=14, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

rekomendasi_text = """
UNTUK PENELITIAN YANG LEBIH BAIK:

A. Perbaiki Data:
   • Kumpulkan min. 50-100 sampel
     "Tidak Clear"
   • Atau: Undersampling "Clear"
     jadi 100:7 ratio

B. Gunakan Metrik Tepat:
   • Jangan hanya Akurasi!
   • Gunakan: F1, Precision, Recall
   • AUC-ROC lebih informatif
   • Matthews Correlation Coefficient

C. Teknik Tambahan:
   • Cost-sensitive learning
   • Ensemble methods
   • Threshold adjustment (0.5→0.3)

D. Untuk Laporan:
   • Akui keterbatasan data
   • Jelaskan imbalanced problem
   • Tunjukkan awareness metodologi
   • Sarankan perbaikan
"""

ax5.text(0.05, 0.85, rekomendasi_text, transform=ax5.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#e8f8f5', alpha=0.9))

# 6. Kesimpulan
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
ax6.text(0.5, 0.95, 'KESIMPULAN', transform=ax6.transAxes,
         fontsize=14, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

kesimpulan_text = """
✅ METODOLOGI BENAR
   • Tidak ada data leakage
   • Pipeline sudah tepat
   • Regularization diterapkan

⚠️ KETERBATASAN
   • Akurasi tinggi ≠ model bagus
   • Gagal deteksi minoritas
   • Butuh lebih banyak data

📊 UNTUK PRESENTASI:
   "Model XGBoost mencapai akurasi
    99.67% pada data validasi.
    
    Namun, perlu dicatat bahwa
    performa tinggi ini terutama
    karena dominasi kelas mayoritas
    (99.5% data adalah 'Clear').
    
    Model memiliki keterbatasan
    dalam mendeteksi kelas minoritas
    karena hanya 7 sampel tersedia
    untuk pembelajaran.
    
    Penelitian lanjutan disarankan
    untuk mengumpulkan lebih banyak
    data anomali."
"""

ax6.text(0.05, 0.85, kesimpulan_text, transform=ax6.transAxes,
         fontsize=8.5, verticalalignment='top', fontfamily='sans-serif',
         bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.9))

plt.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.savefig('LAPORAN_ANALISIS_MODEL.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Laporan visual berhasil dibuat: LAPORAN_ANALISIS_MODEL.png")
print("\nFile ini bisa Anda gunakan untuk:")
print("  1. Slide presentasi")
print("  2. Lampiran laporan")
print("  3. Penjelasan kepada dosen pembimbing")

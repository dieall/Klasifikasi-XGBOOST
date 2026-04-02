import pandas as pd
import numpy as np

# Load data
df = pd.read_excel('Data_Chemical_Imputed_Encoded.xlsx')

print("="*70)
print("ANALISIS DATA UNTUK DETEKSI POTENSI DATA LEAKAGE")
print("="*70)

# 1. Cek distribusi target
print("\n[1] Distribusi Target (APE_encoded):")
print(df['APE_encoded'].value_counts())
print(f"\nPersentase:")
print(df['APE_encoded'].value_counts(normalize=True) * 100)

# 2. Cek korelasi antara fitur dan target
feature_columns = [
    'Transmission', 'Tinx', 'RI', 'SG', 'Acid', 
    'Sulfur', 'Water', 'Mono', 'Yellow', 'EH', 'Visco', 'PT'
]

print("\n[2] Korelasi Fitur dengan Target (APE_encoded):")
correlations = df[feature_columns + ['APE_encoded']].corr()['APE_encoded'].drop('APE_encoded').sort_values(ascending=False)
print(correlations)

# 3. Cek apakah ada fitur yang terlalu sempurna (korelasi sangat tinggi)
print("\n[3] Fitur dengan Korelasi Tinggi (> 0.7):")
high_corr = correlations[abs(correlations) > 0.7]
if len(high_corr) > 0:
    print("⚠️ WARNING: Ada fitur dengan korelasi sangat tinggi!")
    print(high_corr)
else:
    print("✓ Tidak ada fitur dengan korelasi sangat tinggi")

# 4. Analisis kelas minoritas
print("\n[4] Analisis Data Kelas Minoritas (Tidak Clear):")
minority_data = df[df['APE_encoded'] == 0]
print(f"Jumlah data kelas minoritas: {len(minority_data)}")
print("\nStatistik kelas minoritas:")
print(minority_data[feature_columns].describe())

# 5. Bandingkan dengan kelas mayoritas
print("\n[5] Analisis Data Kelas Mayoritas (Clear):")
majority_data = df[df['APE_encoded'] == 1]
print(f"Jumlah data kelas mayoritas: {len(majority_data)}")

# 6. Cek apakah ada nilai yang sangat berbeda
print("\n[6] Perbedaan Mean antara Kelas 0 dan Kelas 1:")
diff = minority_data[feature_columns].mean() - majority_data[feature_columns].mean()
print(diff.sort_values(ascending=False))

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)
print(f"✓ Imbalance Ratio: {len(majority_data)}/{len(minority_data)} = {len(majority_data)/len(minority_data):.1f}:1")
print(f"✓ Kelas minoritas hanya {len(minority_data)} sampel (terlalu sedikit!)")
print("\n⚠️ MASALAH UTAMA:")
print("  1. Data sangat imbalanced (213:1)")
print("  2. Kelas minoritas terlalu sedikit untuk pembelajaran yang baik")
print("  3. Model mungkin hanya 'menghafal' 7 sampel minoritas")
print("="*70)

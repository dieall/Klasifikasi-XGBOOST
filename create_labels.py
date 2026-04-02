import pandas as pd
import numpy as np

# 1. Memuat Data Hasil Imputasi
file_path = 'Data_Chemical_Imputed.xlsx'
print(f"Sedang membaca file: {file_path}...")
df = pd.read_excel(file_path)

# 2. DEFINISI SPESIFIKASI PRODUK (Thresholds)
# Berdasarkan gambar spesifikasi yang diberikan:
# - Transmission > 98
# - Appearance: Clear
# - Tin Content: 19.0 ± 0.2 -> (18.8 - 19.2)
# - RI @ 25°C: 1.509 ± 0.002 -> (1.507 - 1.511)
# - SG @ 25°C: 1.170 ± 0.01 -> (1.160 - 1.180)
# - Acid Value: ≤ 3.0
# - Sulfur: 12.0 ± 0.5 -> (11.5 - 12.5)
# - Water Content: < 3.5
# - Monomethyltin: 19.5 - 28.5
# - Yellowish Index: < 9.0
# - 2-EH: < 0.7
# - Visco: 40 - 80
# - Pt-Co: ≤ 30

def check_status(row):
    # Kita cek satu per satu. Jika ada SATU SAJA yang gagal, langsung 'Not Passed'.
    
    # 1. Transmission > 98
    if not (row['Transmission'] > 98):
        return 'Not Passed'
    
    # 2. Appearance == 'Clear'
    # Kita ubah ke string dulu biar aman, trim spasi, lalu cek
    ape_val = str(row['APE']).strip()
    if ape_val != 'Clear':
        return 'Not Passed'
    
    # 3. Tin Content (Tinx): 18.8 - 19.2
    if not (18.8 <= row['Tinx'] <= 19.2):
        return 'Not Passed'
    
    # 4. RI: 1.507 - 1.511
    if not (1.507 <= row['RI'] <= 1.511):
        return 'Not Passed'
    
    # 5. SG: 1.160 - 1.180
    if not (1.160 <= row['SG'] <= 1.180):
        return 'Not Passed'
    
    # 6. Acid Value <= 3.0
    if not (row['Acid'] <= 3.0):
        return 'Not Passed'
    
    # 7. Sulfur: 11.5 - 12.5
    if not (11.5 <= row['Sulfur'] <= 12.5):
        return 'Not Passed'
    
    # 8. Water Content < 3.5
    if not (row['Water'] < 3.5):
        return 'Not Passed'
    
    # 9. Monomethyltin (Mono): 19.5 - 28.5 (Rentang)
    if not (19.5 <= row['Mono'] <= 28.5):
        return 'Not Passed'
    
    # 10. Yellowish Index (Yellow) < 9.0
    if not (row['Yellow'] < 9.0):
        return 'Not Passed'
    
    # 11. 2-EH (EH) < 0.7
    if not (row['EH'] < 0.7):
        return 'Not Passed'
    
    # 12. Visco: 40 - 80
    if not (40 <= row['Visco'] <= 80):
        return 'Not Passed'
    
    # 13. Pt-Co (PT) <= 30
    if not (row['PT'] <= 30):
        return 'Not Passed'
    
    # Jika lolos semua filter diatas:
    return 'Passed'

# 3. Menerapkan Logika ke Setiap Baris
print("Sedang melakukan klasifikasi produk...")
# axis=1 artinya kita proses per baris
df['Status_Akhir'] = df.apply(check_status, axis=1)

# 4. Cek Hasil Distribusi
print("\nHasil Distribusi Klasifikasi:")
print(df['Status_Akhir'].value_counts())

# 5. Menyimpan File Hasil
output_file = 'Data_Chemical_Labeled.xlsx'
df.to_excel(output_file, index=False)
print(f"\nSelesai! File berlabel disimpan sebagai: {output_file}")

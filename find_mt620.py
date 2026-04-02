import pandas as pd

file_path = r'c:\Aldi\#KULIAH\Semester 7\Metode Penelitian PRA TA\Coding\Data_Chemical_2026-04-01.xlsx'
df = pd.read_excel(file_path)

found = False
for col in df.columns:
    matches = df[df[col].astype(str).str.contains('MT-620', case=False, na=False)]
    if not matches.empty:
        print(f"Found MT-620 in column '{col}'")
        print(f"Unique values in '{col}' containing 'MT-620': {matches[col].unique()}")
        found = True

if not found:
    print("MT-620 not found in any column.")

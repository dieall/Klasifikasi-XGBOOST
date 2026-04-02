import pandas as pd
df = pd.read_excel('Data_Chemical_2026-04-01.xlsx')
print(f"Total rows: {len(df)}")
print("\nColumns:")
for i, col in enumerate(df.columns):
    print(f"- {col}")

# Search for 'MT-620' and report which columns contain it
found_in = []
for col in df.columns:
    if df[col].astype(str).str.contains('MT-620', case=False, na=False).any():
        found_in.append(col)

print("\n'MT-620' found in these columns:")
for col in found_in:
    count = df[col].astype(str).str.contains('MT-620', case=False, na=False).sum()
    print(f"- {col}: {count} occurrences")

if found_in:
    print("\nFirst 5 rows with 'MT-620':")
    mask = df[found_in[0]].astype(str).str.contains('MT-620', case=False, na=False)
    print(df[mask][['ID'] + found_in].head())

# Also check for 'MT 620' or variations
mask_all = df.apply(lambda row: row.astype(str).str.contains('MT[- ]?620', case=False, na=False).any(), axis=1)
print(f"\nTotal rows matching 'MT-620' pattern: {mask_all.sum()}")

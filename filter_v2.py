import pandas as pd

# Load the file
file_path = 'Data_Chemical_2026-04-01.xlsx'
df = pd.read_excel(file_path)

# Total rows before filtering
original_count = len(df)

# Filter rows where 'Nama' is 'MT-620'
mask = df['Nama'].astype(str).str.contains('MT-620', case=False, na=False)
filtered_df = df[mask].reset_index(drop=True)

# Final count
filtered_count = len(filtered_df)

# Save result to a new file
output_file = 'Data_Chemical_MT620_Only_Final.xlsx'
filtered_df.to_excel(output_file, index=False)

print(f"--- FILTERING COMPLETE ---")
print(f"Original row count: {original_count}")
print(f"Filtered row count (MT-620): {filtered_count}")
print(f"Rows removed: {original_count - filtered_count}")
print(f"Cleaned file saved to: {output_file}")
print("\nFirst 10 rows of result:")
print(filtered_df[['ID', 'Nama Chemical', 'Nama', 'Tanggal', 'Batch']].head(10))

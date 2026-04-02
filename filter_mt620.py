import pandas as pd

# Load the file
file_path = r'Data_Chemical_2026-04-01.xlsx'
df = pd.read_excel(file_path)

# Filter rows where 'Nama' contains 'MT-620'
# Case insensitive as a precaution
filtered_df = df[df['Nama'].astype(str).str.contains('MT-620', case=False, na=False)]

# Save result to a new file
output_file = 'Data_Chemical_MT620_Only.xlsx'
filtered_df.to_excel(output_file, index=False)

print(f"Filtering complete.")
print(f"Rows found: {len(filtered_df)}")
print(f"Results saved to: {output_file}")
print("\nFirst 5 rows of result:")
print(filtered_df[['ID', 'Nama', 'Nama Chemical']].head())

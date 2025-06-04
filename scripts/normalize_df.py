import pandas as pd
import numpy as np

FILE_NAME = "CANON_FEATURES_CHAPITRES_LAST.csv"
NORMALIZED_FILE_NAME = FILE_NAME.replace('.csv', '_NORMALIZED.csv')

# Read the CSV file
data = pd.read_csv(FILE_NAME, index_col=0)
data = data[['MSTTR-100', 'Compressibility', 'Flesch Reading Ease', 'Mean sentence length', 'Mean word length']]

# Select numeric columns only
numeric_data = data.select_dtypes(include=[np.number])

if numeric_data.empty:
    raise ValueError("No numeric data found for normalization.")

# Compute norms to check normalization status
norms = np.linalg.norm(numeric_data.values, axis=1)

# Check if data is already normalized
if np.allclose(norms, 1.0, atol=1e-5):
    print("Data is already normalized.")
else:
    print("Normalizing data...")
    normalized_data = numeric_data.values / norms[:, np.newaxis]
    normalized_df = pd.DataFrame(normalized_data, columns=numeric_data.columns)
    normalized_df.to_csv(NORMALIZED_FILE_NAME, index=True)
    print(f"Normalized data saved to {NORMALIZED_FILE_NAME}")


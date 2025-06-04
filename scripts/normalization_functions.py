
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# function from scripts/normalize_df.py
def normalize_dataframe(df, id_col, output_file):
    numeric_data = df.select_dtypes(include=[np.number])

    if numeric_data.empty:
        raise ValueError("No numeric data found for normalization.")
    
    print("Columns to be normalized:", numeric_data.columns.tolist())

    norms = np.linalg.norm(numeric_data.values, axis=1)

    if np.allclose(norms, 1.0, atol=1e-5):
        print("Data is already normalized.")
        return df

    print("Normalizing data...")
    normalized_data = numeric_data.values / norms[:, np.newaxis]
    normalized_df = pd.DataFrame(normalized_data, columns=numeric_data.columns)

    # Insert ID column at beginning
    if id_col in df.columns:
        normalized_df.insert(0, id_col, df[id_col].values)

    normalized_df.to_csv(output_file, index=False)
    print(f"Normalized data saved to {output_file}")

    return normalized_df


# function to normalize based on features
def standardize_features(df, id_col, output_file):
    scaler = MinMaxScaler()

    # Identify and extract numeric columns (excluding the ID)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric data found for normalization.")

    print("Columns to be normalized:", numeric_cols)

    # Scale numeric data
    scaled_values = scaler.fit_transform(df[numeric_cols])

    # Row-normalize (L2 norm) each row vector
    norms = np.linalg.norm(scaled_values, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1
    row_normalized = scaled_values / norms

    normalized_df = pd.DataFrame(row_normalized, columns=numeric_cols)

    # Add ID column
    normalized_df.insert(0, id_col, df[id_col].values)

    normalized_df.to_csv(output_file, index=False)
    print(f"Standardized data saved to {output_file}")

    return normalized_df


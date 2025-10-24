import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import json
import os

def load_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)

def perform_eda(data):
    """Perform exploratory data analysis: summary stats, info, null values."""
    summary_stats = data.describe(include='all')
    print("Summary Statistics:")
    print(summary_stats)
    print("\nData Info:")
    data.info()
    nulls = data.isnull().sum()
    print("\nNull Values:")
    print(nulls)
    return summary_stats, nulls

def encode_categorical_features(data, quant_columns, output_mapping_path, output_encoded_path):
    """Encode categorical features using LabelEncoder and save mappings."""
    enc_data = data.copy()
    cate_columns = [col for col in enc_data.columns if col not in quant_columns and enc_data[col].dtype == 'object']

    category_map = {}
    for col in cate_columns:
        le = LabelEncoder()
        enc_data[col] = le.fit_transform(enc_data[col])
        category_map[col] = {str(cls): int(label) for cls, label in zip(le.classes_, le.transform(le.classes_))}

    # Clean column names (remove spaces, lowercase, remove special characters)
    def clean_column_name(col):
        import re
        return re.sub(r'[^a-zA-Z0-9_]', '', col.lower().replace(' ', '_'))

    column_mapping = {col: clean_column_name(col) for col in enc_data.columns}
    enc_data = enc_data.rename(columns=column_mapping)

    # Update category map with clean names
    clean_category_map = {clean_column_name(k): v for k, v in category_map.items()}

    # Create directory if not exists
    os.makedirs(os.path.dirname(output_mapping_path), exist_ok=True)
    with open(output_mapping_path, 'w') as f:
        json.dump(clean_category_map, f)

    enc_data.to_csv(output_encoded_path, index=False)
    print(f'Encoded dataset saved with {len(enc_data.columns)} cleaned columns.')
    print('Clean column names:', list(enc_data.columns))
    return enc_data, clean_category_map

def check_duplicates(data):
    """Check for duplicate rows."""
    duplicates = data[data.duplicated()]
    print(f"Number of duplicate rows: {len(duplicates)}")
    return duplicates

def plot_correlation_matrix(data, target_column, output_plot_path):
    """Plot correlation matrix and save."""
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    corr_matrix = data.corr()
    plt.figure(figsize=(15, 15))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Correlations with target variable:')
    print(corr_matrix[target_column].sort_values(ascending=False))
    check_duplicates(data)
    print("Target value counts:")
    print(data[target_column].value_counts())

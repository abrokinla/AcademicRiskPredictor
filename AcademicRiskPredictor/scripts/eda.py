#!/usr/bin/env python3
"""
EDA and Preprocessing Script for Academic Risk Predictor
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import load_data, perform_eda, encode_categorical_features, plot_correlation_matrix

def main():
    # Load data
    data = load_data('data/dataset.csv')

    # Perform EDA
    summary_stats, nulls = perform_eda(data)

    # Define quantitative columns (as in original)
    quant_columns = ['Age [10-15]', 'Age [16-20]', 'Age [21-25]', 'Age [26-30]', 'Age [31-35]']

    # Encode categorical features and save
    enc_data, category_map = encode_categorical_features(data, quant_columns, 'data/category_mapping.json', 'data/stu_risk_data_encoded.csv')

    # Plot correlation matrix
    plot_correlation_matrix(enc_data, 'what_your_current_CGPA', 'plots/correlation_matrix.png')

    print("EDA and preprocessing completed.")

if __name__ == '__main__':
    main()

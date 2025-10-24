#!/usr/bin/env python3
"""
Training Script for Academic Risk Predictor Models
Loads data from PostgreSQL database
"""
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model_evaluation import StuRiskModelEvaluator
from db import SessionLocal, StudentData

def load_data_from_db():
    """Load training data from PostgreSQL database."""
    session = SessionLocal()
    try:
        # Query all records
        records = session.query(StudentData).all()

        # Convert to DataFrame
        data = []
        for record in records:
            record_dict = {
                'gender': record.gender,
                'level': record.level,
                'age': record.age,
                'parent_guardian_support_education_learning': record.parent_guardian_support_education_learning,
                'positive_supportive_relation_with_lecturers': record.positive_supportive_relation_with_lecturers,
                'avoid_engaging_risky_behavior': record.avoid_engaging_risky_behavior,
                'family_status_stable_supportive_in_academic_pursuits': record.family_status_stable_supportive_in_academic_pursuits,
                'parent_guardian_encourages_academic_pursuit': record.parent_guardian_encourages_academic_pursuit,
                'no_significant_family_issues_interfere_academic_pursuit': record.no_significant_family_issues_interfere_academic_pursuit,
                'strong_sense_belonging_university_community': record.strong_sense_belonging_university_community,
                'university_provide_adequate_support_academic_pursuit': record.university_provide_adequate_support_academic_pursuit,
                'feel_safe_comfortable_in_school_environment': record.feel_safe_comfortable_in_school_environment,
                'what_your_current_cgpa': record.what_your_current_cgpa
            }
            data.append(record_dict)

        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} records from database")
        print(f"Columns: {list(df.columns)}")
        return df

    finally:
        session.close()

def main():
    # Load data from database
    try:
        enc_data = load_data_from_db()
    except Exception as e:
        print(f"Error loading data from database: {e}")
        print("Make sure the database is set up and populated with data.")
        return

    # Define features (exact names from database schema)
    features = [
        'parent_guardian_support_education_learning',
        'positive_supportive_relation_with_lecturers',
        'avoid_engaging_risky_behavior',
        'family_status_stable_supportive_in_academic_pursuits',
        'parent_guardian_encourages_academic_pursuit',
        'no_significant_family_issues_interfere_academic_pursuit',
        'strong_sense_belonging_university_community',
        'university_provide_adequate_support_academic_pursuit',
        'feel_safe_comfortable_in_school_environment'
    ]

    # Encode target variable to standardized values (0 for 'Fail', 1 for 'Pass', etc.)
    cgpa_mapping = {
        'Fail': 0,
        'Pass': 1,
        'Third Class': 2,
        'Second Class Lower': 3,
        'Second Class Upper': 4,
        'First Class': 5
    }

    y_encoded = enc_data['what_your_current_cgpa'].map(cgpa_mapping).fillna(-1).astype(int)
    if (y_encoded == -1).any():
        print("Warning: Some CGPA values could not be mapped and were assigned -1")
    y = y_encoded

    X = enc_data[features]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f'Training feature shape: {X_train.shape}')
    print(f'Test feature shape: {X_test.shape}')

    # Define models
    models = {
        'SVM': SVC(probability=True),
        'Logistic Regression': LogisticRegression(multi_class='ovr', max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier()
    }

    # Initialize evaluator
    evaluator = StuRiskModelEvaluator(models, plot_dir='plots')

    # Evaluate models (this trains them)
    results = evaluator.evaluate_models(X_train, X_test, y_train, y_test)

    # Save trained models
    evaluator.save_models('models/')

    print("Model training and evaluation completed. Models saved to 'models/' directory.")

if __name__ == '__main__':
    main()

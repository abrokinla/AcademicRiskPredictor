#!/usr/bin/env python3
"""
Main Pipeline for Academic Risk Predictor
Runs EDA, training, and sample prediction
"""
import sys
import os
from src.data_processing import load_data, perform_eda, encode_categorical_features, plot_correlation_matrix
from src.model_evaluation import StuRiskModelEvaluator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
    print(" Academic Risk Predictor - Database-First Pipeline")
    print("=" * 55)

    # Check database connection and data
    print("Checking database...")
    try:
        from db import SessionLocal, StudentData
        session = SessionLocal()
        count = session.query(StudentData).count()
        session.close()

        if count == 0:
            print("No data found in database. Loading dataset...")
            from db.init_db import load_dataset_to_database
            load_dataset_to_database('data/dataset.csv')
        else:
            print(f"Found {count} records in database")

    except Exception as e:
        print(f"Database error: {e}")
        print("Make sure database is set up. Run: uv run python db/init_db.py")
        return

    print("\nRunning Model Training from Database...")
    # Import training function
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from scripts.train import load_data_from_db, main as train_main

    # Load data using training function
    try:
        enc_data = load_data_from_db()
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return

    # Define features (including demographic and psychosocial factors)
    features = [
        'gender',
        'level',
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

    y = enc_data['what_your_current_cgpa']
    X = enc_data[features]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"    Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Define models
    models = {
        'SVM': SVC(probability=True),
        'Logistic Regression': LogisticRegression(multi_class='ovr', max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier()
    }

    # Initialize evaluator
    evaluator = StuRiskModelEvaluator(models, plot_dir='plots')

    # Evaluate models (trains them)
    results = evaluator.evaluate_models(X_train, X_test, y_train, y_test)

    # Save trained models
    evaluator.save_models('models/')

    print("    Training completed. Models saved.")

    print("\n Making Sample Prediction...")
    # Sample student profile (male, 400L, mostly positive factors)
    input_row = {
        'gender': 1,
        'level': 4,
        'parent_guardian_support_education_learning': 3,
        'positive_supportive_relation_with_lecturers': 3,
        'avoid_engaging_risky_behavior': 3,
        'family_status_stable_supportive_in_academic_pursuits': 3,
        'parent_guardian_encourages_academic_pursuit': 3,
        'no_significant_family_issues_interfere_academic_pursuit': 2,
        'strong_sense_belonging_university_community': 3,
        'university_provide_adequate_support_academic_pursuit': 4,
        'feel_safe_comfortable_in_school_environment': 3
    }

    # Make prediction with best model (Random Forest)
    prediction = evaluator.predict('models/Random_Forest.pkl', input_row)
    print("   Sample Student Profile:")
    print("      Male student (400L), strong family support, good lecturer relations")
    print("      Moderate family issues, high university support")
    print(f"    Predicted CGPA: {prediction}")

    print("\n Model Performance Summary:")
    print(f"    Best Model: Random Forest (Accuracy: {results[results['Model'] == 'Random Forest']['Accuracy'].values[0]*100:.1f}%")
    print(f"      F1 Score: {results[results['Model'] == 'Random Forest']['F1 Score'].values[0]*100:.1f}%")
    print(f"      ROC AUC: {results[results['Model'] == 'Random Forest']['ROC AUC'].values[0]*100:.1f}%")

    print("\n Database-first pipeline completed successfully!")
    print("    Check 'plots/' for visualizations and 'models/' for trained models")
    print("=" * 55)

if __name__ == "__main__":
    main()

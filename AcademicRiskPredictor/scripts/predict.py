#!/usr/bin/env python3
"""
Prediction Script for Academic Risk Predictor
"""
import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model_evaluation import StuRiskModelEvaluator

def main():
    parser = argparse.ArgumentParser(description='Make prediction for student academic risk.')
    parser.add_argument('--model', type=str, default='Random_Forest', choices=['SVM', 'Logistic_Regression', 'Random_Forest', 'Decision_Tree'], help='Model to use for prediction (default: Random_Forest)')
    parser.add_argument('--gender', type=int, default=1, help='Gender (encoded) (default: 1)')
    parser.add_argument('--level', type=int, default=2, help='Level (encoded) (default: 2)')
    parser.add_argument('--parent_support', type=int, default=3, help='Parent/guardian support (default: 3)')
    parser.add_argument('--lecturer_relation', type=int, default=3, help='Positive relation with lecturers (default: 3)')
    parser.add_argument('--risky_behavior', type=int, default=3, help='Avoid engaging in risky behavior (default: 3)')
    parser.add_argument('--family_status', type=int, default=3, help='Family status (default: 3)')
    parser.add_argument('--parent_encouragement', type=int, default=3, help='Parent encouragement (default: 3)')
    parser.add_argument('--family_issues', type=int, default=2, help='Family issues (default: 2)')
    parser.add_argument('--belonging', type=int, default=3, help='Sense of belonging (default: 3)')
    parser.add_argument('--university_support', type=int, default=4, help='University support (default: 4)')
    parser.add_argument('--safe_environment', type=int, default=3, help='Safe environment (default: 3)')

    args = parser.parse_args()

    # Prepare input data
    input_row = {
        'Gender': args.gender,
        'Level': args.level,
        'parent_guardian_support_education_learning': args.parent_support,
        'positive_supportive_relation_with_lecturers': args.lecturer_relation,
        'avoid_engaging _risky_behavior': args.risky_behavior,
        'family_status_stable_supportive_in _academic_pursuits': args.family_status,
        'parent_guardian_encourages_academic_pursuit': args.parent_encouragement,
        'no_significant_family_issues_interfere_academic_pursuit': args.family_issues,
        'strong_sense_belonging_university_community': args.belonging,
        'University_provide_adequate_support_academic_pursuit': args.university_support,
        'feel_safe_comfortable_in_school_environment': args.safe_environment
    }

    # Model path
    model_path = f'models/{args.model}.pkl'

    # Initialize evaluator (not used except for prediction)
    evaluator = StuRiskModelEvaluator({})

    # Make prediction
    prediction = evaluator.predict(model_path, input_row)

    print(f"Predicted CGPA: {prediction}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Database initialization and data loading for Academic Risk Predictor
"""
import pandas as pd
from sqlalchemy.orm import sessionmaker
import sys
from .config import SessionLocal, engine, Base
from .models import Prediction, StudentData

def create_tables():
    """Create all database tables."""
    try:
        # Drop existing tables to ensure clean slate
        Base.metadata.drop_all(bind=engine)
        print("Dropped existing tables (if any)...")

        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")
        print("You can now run the FastAPI server: uv run uvicorn api:app --host 0.0.0.0 --port 8000")
        return True
    except Exception as e:
        print(f"Error creating tables: {e}")
        print("Make sure your PostgreSQL database is running and DATABASE_URL is correct.")
        return False

def clean_column_name(col):
    """Clean column names: lowercase, replace spaces with underscores, remove special chars."""
    import re
    return re.sub(r'[^a-zA-Z0-9_]', '', col.lower().replace(' ', '_'))

def load_dataset_to_database(csv_path):
    """Load CSV dataset into database with clean column names and encoded values."""
    try:
        # Read CSV
        df = pd.read_csv(csv_path)

        # Clean column names
        column_mapping = {col: clean_column_name(col) for col in df.columns}
        df = df.rename(columns=column_mapping)

        # Define Likert scale encoding mapping
        likert_mapping = {
            'Strongly Disagree': 1,
            'Disagree': 2,
            'Neutral': 3,
            'Agree': 4,
            'Strongly Agree': 5
        }

        # Define gender and level mappings
        gender_mapping = {'Male': 1, 'Female': 2}
        level_mapping = {'100L': 1, '200L': 2, '300L': 3, '400L': 4, '500L': 5}

        # Encode categorical columns
        # Gender and Level
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map(gender_mapping).fillna(0).astype(int)
        if 'level' in df.columns:
            df['level'] = df['level'].map(level_mapping).fillna(0).astype(int)

        # Likert scale columns (5-point scale features)
        likert_features = [
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

        for feature in likert_features:
            if feature in df.columns:
                df[feature] = df[feature].map(likert_mapping).fillna(0).astype(int)

        print(f"Loaded {len(df)} rows with encoded data")
        print(f"Columns: {list(df.columns)}")
        print("Sample row:", df.iloc[0].to_dict())

        # Create database session
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Clear existing data
            session.query(StudentData).delete()
            session.commit()

            # Insert data
            records_inserted = 0
            for _, row in df.iterrows():
                record = StudentData(
                    gender=row.get('gender'),
                    level=row.get('level'),
                    age=row.get('age'),
                    parent_guardian_support_education_learning=int(row.get('parent_guardian_support_education_learning', 0)),
                    positive_supportive_relation_with_lecturers=int(row.get('positive_supportive_relation_with_lecturers', 0)),
                    avoid_engaging_risky_behavior=int(row.get('avoid_engaging_risky_behavior', 0)),
                    family_status_stable_supportive_in_academic_pursuits=int(row.get('family_status_stable_supportive_in_academic_pursuits', 0)),
                    parent_guardian_encourages_academic_pursuit=int(row.get('parent_guardian_encourages_academic_pursuit', 0)),
                    no_significant_family_issues_interfere_academic_pursuit=int(row.get('no_significant_family_issues_interfere_academic_pursuit', 0)),
                    strong_sense_belonging_university_community=int(row.get('strong_sense_belonging_university_community', 0)),
                    university_provide_adequate_support_academic_pursuit=int(row.get('university_provide_adequate_support_academic_pursuit', 0)),
                    feel_safe_comfortable_in_school_environment=int(row.get('feel_safe_comfortable_in_school_environment', 0)),
                    what_your_current_cgpa=row.get('what_your_current_cgpa')
                )
                session.add(record)
                records_inserted += 1

            session.commit()
            print(f"Successfully inserted {records_inserted} records into database!")
            return True

        except Exception as e:
            session.rollback()
            print(f"Error loading data: {e}")
            return False
        finally:
            session.close()

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

if __name__ == "__main__":
    if create_tables():
        # Load dataset if tables created successfully
        dataset_path = "data/dataset.csv"
        load_dataset_to_database(dataset_path)

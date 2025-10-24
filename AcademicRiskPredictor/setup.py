#!/usr/bin/env python3
"""
Setup script for Academic Risk Predictor
Handles database connection testing, table creation, and system validation
"""
import os
import sys
import psycopg2
from dotenv import load_dotenv

def check_database_connection():
    """Test database connectivity."""
    print("Checking database connection...")

    database_url = os.getenv("DATABASE_URL")

    # Parse connection string
    try:
        # Basic parsing of postgres://user:pass@host:port/db
        parts = database_url.replace("postgresql://", "").split("/")
        creds_host = parts[0].split("@")
        host_port = creds_host[1].split(":")
        db_name = parts[1]

        conn_params = {
            "host": host_port[0],
            "port": int(host_port[1]) if len(host_port) > 1 else 5432,
            "database": db_name,
            "user": creds_host[0].split(":")[0],
            "password": creds_host[0].split(":")[1] if ":" in creds_host[0] else "",
        }

        # Test connection
        conn = psycopg2.connect(**conn_params)
        conn.close()

        print("Database connection successful!")
        return conn_params

    except Exception as e:
        print(f"Database connection failed: {e}")
        print("Make sure PostgreSQL is running and DATABASE_URL is correct.")
        print("   Example: export DATABASE_URL='postgresql://user:pass@localhost/dbname'")
        return None

def create_database_if_needed(conn_params):
    """Create database if it doesn't exist."""
    print("Checking if database exists...")

    try:
        # Connect to postgres default database to create our db
        temp_conn = psycopg2.connect(
            host=conn_params["host"],
            port=conn_params["port"],
            database="postgres",
            user=conn_params["user"],
            password=conn_params["password"]
        )

        temp_conn.autocommit = True
        cursor = temp_conn.cursor()

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (conn_params["database"],))
        exists = cursor.fetchone()

        if not exists:
            print(f"Creating database '{conn_params['database']}'...")
            cursor.execute(f"CREATE DATABASE {conn_params['database']}")
            print("Database created!")
        else:
            print("Database already exists!")

        cursor.close()
        temp_conn.close()
        return True

    except Exception as e:
        print(f"Error managing database: {e}")
        return False

def setup_tables():
    """Create database tables."""
    print("Setting up database tables...")
    from db.init_db import create_tables
    return create_tables()

def validate_models():
    """Check if ML models are available."""
    print("Validating ML models...")

    import joblib
    model_path = 'models/Random_Forest.pkl'

    try:
        model = joblib.load(model_path)
        print("Random Forest model loaded successfully!")
        return True
    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Make sure you've run database setup and trained models")
        return False

def validate_database_data():
    """Check if database has training data."""
    print("Validating database data...")

    try:
        from db import SessionLocal, StudentData
        session = SessionLocal()
        count = session.query(StudentData).count()
        session.close()

        if count > 0:
            print(f"Found {count} records in database!")
            return True
        else:
            print("No data found in database")
            return False
    except Exception as e:
        print(f"Database validation failed: {e}")
        return False

def check_data_files():
    """Verify that necessary data files exist."""
    print("Checking data files...")

    required_files = [
        'data/dataset.csv',
        'data/category_mapping.json'
    ]

    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)

    if missing:
        print(f"Missing files: {missing}")
        print("Make sure you have run the EDA pipeline")
        return False

    print("All data files present!")
    return True

def main():
    """Main setup orchestration."""
    print("Academic Risk Predictor - System Setup")
    print("=" * 50)

    load_dotenv()

    success_count = 0
    total_checks = 6

    conn_params = check_database_connection()
    if conn_params:
        success_count += 1

        if create_database_if_needed(conn_params):
            success_count += 1
        else:
            return False

        if setup_tables():
            success_count += 1
        else:
            return False

    if check_data_files():
        success_count += 1

    if validate_models():
        success_count += 1

    print("Running final system validation..." ) 
    try:
        import requests
        # Try connecting to FastAPI (if running)
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("FastAPI server is running!")
                success_count += 1
            else:
                print("FastAPI server detected but returning error")
        except requests.exceptions.RequestException:
            print("FastAPI server not running (start with: uv run uvicorn api:app --host 0.0.0.0 --port 8000)")
    except ImportError:
        print("Could not test FastAPI server connectivity")

    print()
    print("=" * 50)
    print(f"Setup Complete: {success_count}/{total_checks} checks passed")

    if success_count >= 5:
        print("System is ready to use!")
        print("\nNext steps:")
        print("1. Start FastAPI backend: uv run uvicorn api:app --host 0.0.0.0 --port 8000")
        print("2. Start Streamlit frontend: uv run streamlit run app.py")
        print("3. Visit: http://localhost:8501 for the web interface")
        return True
    else:
        print("Some issues detected. Please fix the problems above.")
        print("\nCommon fixes:")
        print("- Set DATABASE_URL environment variable")
        print("- Start PostgreSQL service")
        print("- Run: uv run python main.py to train models")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

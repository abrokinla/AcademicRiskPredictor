# Academic Risk Predictor

A comprehensive machine learning system for predicting student academic performance based on psychosocial factors. Uses Random Forest models to assess risk levels and provides educational insights to identify students who may need additional support.

## 🎯 Project Overview

The Academic Risk Predictor leverages advanced ML techniques to analyze 11 psychosocial factors (including demographic data and survey responses) to classify students into CGPA performance classes:

- **High Risk:** Fail, Pass, Third Class (CGPA ≤ 3.99)
- **Medium Risk:** Second Class Lower (CGPA 2.40-2.99)
- **Low Risk:** Second Class Upper, First Class (CGPA ≥ 3.00)

The system helps educators proactively identify at-risk students and allocate resources effectively.

## 🛠️ Tech Stack

- **Language:** Python 3.12+
- **ML Framework:** scikit-learn, NumPy, Pandas, SHAP
- **Deep Learning:** XGBoost, TensorFlow (optional)
- **Database:** PostgreSQL with SQLAlchemy ORM
- **API:** FastAPI with Pydantic validation
- **Frontend:** Streamlit for interactive web interface
- **Deployment:** UV for dependency management, Uvicorn for serving
- **Visualization:** Matplotlib, Seaborn, Plotly

## 📋 Features

### Core ML Pipeline
- **Data Standardization:** Clean encoding of Likert scale responses and categorical variables
- **Multi-Model Training:** SVM, Logistic Regression, Random Forest, Decision Tree
- **Comprehensive Evaluation:** Accuracy, F1-Score, Precision, Recall, ROC-AUC, Confusion Matrices
- **Feature Importance:** SHAP explanations for model interpretability
- **Database-First Architecture:** PostgreSQL as primary data store

### Production Web Application
- **FastAPI Backend:** RESTful API for predictions with automatic logging
- **Streamlit Frontend:** Interactive web interface for assessments
- **Real-time Predictions:** Instant risk assessment with input validation
- **Prediction History:** Database-tracked prediction logs with timestamps
- **Health Monitoring:** API health checks and system status

### Engineering Excellence
- **Clean Architecture:** Modular package structure with separation of concerns
- **Type Safety:** Pydantic models for API validation
- **Environment Management:** UV for reproducible installations
- **Container Ready:** Dockerfile and docker-compose support
- **CLI Tools:** Command-line utilities for each workflow step

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- PostgreSQL database (recommended)
- UV package manager

### Installation

1. **Clone Repository:**
   ```bash
   git clone https://github.com/abrokinla/AcademicRiskPredictor.git
   cd AcademicRiskPredictor
   ```

2. **Install Dependencies:**
   ```bash
   uv sync
   ```

3. **Set Up Database:**
   ```bash
   createdb academic_risk_db
   export DATABASE_URL="postgresql://username:password@localhost/academic_risk_db"
   ```

4. **Run System Setup:**
   ```bash
   uv run python setup.py
   ```

This initializes the database, loads training data, trains models, and validates the system.

## 📊 Workflow

### Data Pipeline
1. **Data Collection** → Google Forms survey from university students
2. **Data Processing** → Clean column names, encode categorical variables
3. **Database Loading** → Store in PostgreSQL with standardized schema
4. **Model Training** → Fetch from DB, train ML models on 11 features
5. **Model Evaluation** → Generate metrics, plots, and SHAP explanations
6. **Web Deployment** → FastAPI backend + Streamlit frontend

### Prediction Process
1. **User Input** → Likert scale responses via Streamlit form
2. **API Request** → Validated data sent to FastAPI backend
3. **Model Prediction** → Random Forest predicts CGPA class
4. **Risk Assessment** → Classifies as High/Medium/Low risk
5. **Database Logging** → Saves prediction with input data and timestamp
6. **Display Results** → Shows CGPA prediction and risk level

### System Architecture
```
Input Data → Feature Selection → Model → Prediction → Database → Display
     ↓            ↓              ↓         ↓           ↓         ↓
  Survey     Clean Features   Trained    CGPA Class  Log        UI
  Forms      (11 factors)     Model      (Strings)   Entry   Response
```

## 🎯 Usage

### Web Application (Recommended)

1. **Start Backend:**
   ```bash
   uv run uvicorn api:app --host 0.0.0.0 --port 8000
   ```

2. **Start Frontend:**
   ```bash
   uv run streamlit run app.py
   ```

3. **Access Interface:**
   - Frontend: `http://localhost:8501`
   - API Docs: `http://localhost:8000/docs`

### Automated Pipeline
```bash
uv run python main.py  # Complete end-to-end pipeline
```

### CLI Tools
```bash
uv run python scripts/train.py    # Train models only
uv run python scripts/predict.py  # Single prediction
uv run python scripts/eda.py      # Data analysis only
```

### API Usage
```python
import requests

data = {
    "gender": 1, "level": 4,
    "parent_guardian_support_education_learning": 3,
    # ... other 8 factors
}

response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
print(f"Predicted CGPA: {result['prediction']}")
```

## 📡 API Reference

### Endpoints

- `POST /predict` - Make student risk prediction
- `GET /predictions?limit=50` - Get prediction history
- `GET /health` - System health status
- `GET /` - API information

### Prediction Input Schema
```json
{
  "gender": 1,
  "level": 4,
  "parent_guardian_support_education_learning": 3,
  "positive_supportive_relation_with_lecturers": 3,
  "avoid_engaging_risky_behavior": 3,
  "family_status_stable_supportive_in_academic_pursuits": 3,
  "parent_guardian_encourages_academic_pursuit": 3,
  "no_significant_family_issues_interfere_academic_pursuit": 2,
  "strong_sense_belonging_university_community": 3,
  "university_provide_adequate_support_academic_pursuit": 4,
  "feel_safe_comfortable_in_school_environment": 3
}
```

### Prediction Response
```json
{
  "prediction": "First Class",
  "encoded_prediction": 5,
  "timestamp": "2024-10-24T09:00:00Z"
}
```

## 🗄️ Database Schema

### Students Table
```sql
CREATE TABLE student_data (
    id SERIAL PRIMARY KEY,
    gender INTEGER,
    level INTEGER,
    parent_guardian_support_education_learning INTEGER,
    positive_supportive_relation_with_lecturers INTEGER,
    avoid_engaging_risky_behavior INTEGER,
    family_status_stable_supportive_in_academic_pursuits INTEGER,
    parent_guardian_encourages_academic_pursuit INTEGER,
    no_significant_family_issues_interfere_academic_pursuit INTEGER,
    strong_sense_belonging_university_community INTEGER,
    university_provide_adequate_support_academic_pursuit INTEGER,
    feel_safe_comfortable_in_school_environment INTEGER,
    what_your_current_cgpa VARCHAR(20)
);
```

### Predictions Table
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    input_data JSONB,
    prediction VARCHAR(50),
    model_version VARCHAR(50)
);
```

## 📁 Project Structure

```
AcademicRiskPredictor/
├── db/                          # Database layer
│   ├── __init__.py             # Module exports
│   ├── config.py               # Database configuration
│   ├── models.py               # SQLAlchemy models
│   └── init_db.py              # Database initialization
├── src/                        # Core application modules
│   ├── __init__.py
│   ├── data_processing.py      # Data loading and preprocessing
│   └── model_evaluation.py     # ML model trainer and predictor
├── scripts/                    # Command-line utilities
│   ├── eda.py                  # Exploratory data analysis
│   ├── train.py                # Model training pipeline
│   └── predict.py              # CLI prediction tool
├── data/                       # Datasets and mapping files
├── models/                     # Saved ML models (generated)
├── plots/                      # Charts and visualizations (generated)
├── api.py                      # FastAPI backend application
├── app.py                      # Streamlit frontend application
├── main.py                     # End-to-end automated pipeline
├── setup.py                    # System setup and validation
├── pyproject.toml              # Project configuration and dependencies
└── README.md
```

## 📊 Data Source & Limitations

### Data Collection
The training dataset consists of Likert scale responses collected through validated Google Forms distributed to university students. The survey captures 11 key factors including demographic information, family support, lecturer relationships, and institutional factors.

### Reliability Notes
While the data source has **high reliability** due to structured academic survey methodology, the following limitations should be considered:

1. **Potential Duplicate Entries:** Despite survey design, it cannot be guaranteed that some students didn't submit multiple responses
2. **Response Authenticity:** Self-reported data may include responses that don't accurately reflect true opinions or situations
3. **Sample Bias:** University-distributed survey may not represent the broader student population
4. **Temporal Factors:** Data collected at a specific point in time may not reflect changing circumstances

### Data Processing
- Automatic column name standardization (lowercase, underscores)
- Handling of missing values through appropriate encoding
- Categorical variable transformation for ML compatibility
- Feature validation to ensure training consistency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with comprehensive tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 📞 Support

For questions or issues:
- Open a [GitHub Issue](https://github.com/abrokinla/AcademicRiskPredictor/issues)
- Review the API documentation at `/docs`
- Check system health at `/health`

---

**Built with ❤️ for student success**

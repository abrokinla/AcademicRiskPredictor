#!/usr/bin/env python3
"""
FastAPI Backend for Academic Risk Predictor
Production-ready API with PostgreSQL integration
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union
import joblib
import json
from db.config import SessionLocal, get_db
from db.models import Prediction
from sqlalchemy.orm import Session

app = FastAPI(
    title="Academic Risk Predictor API",
    description="Production API for predicting student academic risk using psychosocial factors",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class StudentInput(BaseModel):
    """Student psychosocial data input model."""
    gender: int = Field(..., ge=1, le=2, description="Gender (1=Male, 2=Female)")
    level: int = Field(..., ge=1, le=5, description="Academic level")
    parent_guardian_support_education_learning: int = Field(..., ge=1, le=5, description="Parent support level (1-5 scale)")
    positive_supportive_relation_with_lecturers: int = Field(..., ge=1, le=5, description="Lecturer relations (1-5 scale)")
    avoid_engaging_risky_behavior: int = Field(..., ge=1, le=5, description="Risky behavior avoidance (1-5 scale)")
    family_status_stable_supportive_in_academic_pursuits: int = Field(..., ge=1, le=5, description="Family stability (1-5 scale)")
    parent_guardian_encourages_academic_pursuit: int = Field(..., ge=1, le=5, description="Parental encouragement (1-5 scale)")
    no_significant_family_issues_interfere_academic_pursuit: int = Field(..., ge=1, le=5, description="Family issues impact (1-5 scale)")
    strong_sense_belonging_university_community: int = Field(..., ge=1, le=5, description="Sense of belonging (1-5 scale)")
    university_provide_adequate_support_academic_pursuit: int = Field(..., ge=1, le=5, description="University support (1-5 scale)")
    feel_safe_comfortable_in_school_environment: int = Field(..., ge=1, le=5, description="Safety and comfort (1-5 scale)")

class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction: str
    encoded_prediction: Union[int, str] = Field(description="Encoded prediction (int for new models, str for older ones)")
    timestamp: str = None

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    """Load model at startup."""
    global model
    try:
        model_path = 'models/Random_Forest.pkl'
        model = joblib.load(model_path)

        print("Model and mappings loaded successfully!")
    except Exception as e:
        print(f"Error loading model/mappings: {e}")
        model = None

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"message": "Academic Risk Predictor API", "version": "1.0", "status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(student: StudentInput, db: Session = Depends(get_db)):
    """Predict academic risk for a student."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert pydantic input to dict
        input_data = student.dict()
        input_json = input_data.copy()  # Save original

        # Prepare for DataFrame - select only features used in training
        import pandas as pd
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
        df = pd.DataFrame([input_data])[features]

        # Make prediction (column names now match training)
        raw_prediction = model.predict(df)[0]

        # Convert encoded prediction to human-readable (handle both string and int predictions)
        cgpa_classes = ['Fail', 'Pass', 'Third Class', 'Second Class Lower', 'Second Class Upper', 'First Class']
        if isinstance(raw_prediction, str):
            predicted_cgpa = raw_prediction  # Already human-readable
        else:
            predicted_cgpa = cgpa_classes[raw_prediction] if isinstance(raw_prediction, int) and 0 <= raw_prediction < len(cgpa_classes) else "Unknown"

        # Save to database
        db_prediction = Prediction(
            input_data=input_json,
            prediction=predicted_cgpa,
            model_version="random_forest_v1"
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        return PredictionResponse(
            prediction=predicted_cgpa,
            encoded_prediction=raw_prediction,
            timestamp=db_prediction.timestamp.isoformat() if db_prediction.timestamp else None
        )

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")

@app.get("/predictions")
def get_predictions(limit: int = 50, db: Session = Depends(get_db)):
    """Get prediction history."""
    try:
        predictions = db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(limit).all()
        return [
            {
                "id": p.id,
                "timestamp": p.timestamp.isoformat(),
                "input_data": p.input_data,
                "prediction": p.prediction,
                "model_version": p.model_version
            }
            for p in predictions
        ]
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database query error")

@app.get("/health")
def health_check():
    """Detailed health check."""
    health = {"status": "healthy", "services": {}}

    # Check model
    health["services"]["model"] = "loaded" if model else "not loaded"

    # Check database
    try:
        db = SessionLocal()
        db.query(Prediction).first()
        db.close()
        health["services"]["database"] = "connected"
    except Exception as e:
        health["services"]["database"] = f"error: {str(e)}"
        health["status"] = "degraded"

    return health

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

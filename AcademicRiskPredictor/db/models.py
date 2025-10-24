from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, Text
from sqlalchemy.sql import func
from .config import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    input_data = Column(JSON)
    prediction = Column(String(50))
    model_version = Column(String(50), default="random_forest_v1")

class StudentData(Base):
    __tablename__ = "student_data"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    gender = Column(String(10))
    level = Column(String(10))
    age = Column(String(10))
    parent_guardian_support_education_learning = Column(Integer)
    positive_supportive_relation_with_lecturers = Column(Integer)
    avoid_engaging_risky_behavior = Column(Integer)
    family_status_stable_supportive_in_academic_pursuits = Column(Integer)
    parent_guardian_encourages_academic_pursuit = Column(Integer)
    no_significant_family_issues_interfere_academic_pursuit = Column(Integer)
    strong_sense_belonging_university_community = Column(Integer)
    university_provide_adequate_support_academic_pursuit = Column(Integer)
    feel_safe_comfortable_in_school_environment = Column(Integer)
    what_your_current_cgpa = Column(String(20))

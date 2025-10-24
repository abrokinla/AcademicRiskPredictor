"""
Database layer for Academic Risk Predictor
"""
from .config import SessionLocal, get_db, engine, Base
from .models import Prediction, StudentData
from .init_db import create_tables

__all__ = [
    "SessionLocal",
    "get_db",
    "engine",
    "Base",
    "Prediction",
    "StudentData",
    "create_tables"
]

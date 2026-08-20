"""
utils/database.py

SQLAlchemy models + session management for EduLeap.
Defaults to SQLite for local hackathon dev; set DATABASE_URL for Postgres.

Tables:
    students          - basic student record (name minimised per privacy risk mitigation)
    learner_profiles   - one row per (student, topic): level, streaks, misconceptions
    interactions        - log of every diagnostic/tutor/assessment turn, for the analyst
"""

import os
import uuid
import datetime as dt

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, ForeignKey, JSON, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./eduleap.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def _uid() -> str:
    return str(uuid.uuid4())


class Student(Base):
    __tablename__ = "students"

    id = Column(String, primary_key=True, default=_uid)
    display_name = Column(String, nullable=False)   # avoid storing PII beyond a display name
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    profiles = relationship("LearnerProfile", back_populates="student", cascade="all, delete-orphan")
    interactions = relationship("Interaction", back_populates="student", cascade="all, delete-orphan")


class LearnerProfile(Base):
    __tablename__ = "learner_profiles"

    id = Column(String, primary_key=True, default=_uid)
    student_id = Column(String, ForeignKey("students.id"), nullable=False)
    topic = Column(String, nullable=False)

    level = Column(Float, default=1.0)              # 1 (beginner) - 5 (advanced)
    correct_streak = Column(Integer, default=0)
    incorrect_streak = Column(Integer, default=0)
    known_gaps = Column(JSON, default=list)          # e.g. ["common_denominators"]
    misconception_log = Column(JSON, default=list)    # list of {concept, count, last_seen}
    last_updated = Column(DateTime, default=dt.datetime.utcnow)

    student = relationship("Student", back_populates="profiles")


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(String, primary_key=True, default=_uid)
    student_id = Column(String, ForeignKey("students.id"), nullable=False)
    topic = Column(String, nullable=False)
    stage = Column(String, nullable=False)   # diagnostic | tutor | assessment | analysis
    prompt = Column(Text, nullable=True)
    response = Column(Text, nullable=True)
    is_correct = Column(Integer, nullable=True)  # nullable: not every interaction is gradable
    meta = Column(JSON, default=dict)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    student = relationship("Student", back_populates="interactions")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Session: # type: ignore
    """FastAPI dependency: yields a session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

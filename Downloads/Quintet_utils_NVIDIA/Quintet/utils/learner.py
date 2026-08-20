"""
utils/learner.py

Owns all reads/writes to a student's LearnerProfile. Agents never touch the DB
directly for profile data - they go through these functions so the update
rules stay in one place (needed for "Incorrect adaptive decisions" risk
mitigation: explicit, testable rules rather than ad-hoc LLM judgement).
"""

import datetime as dt
from typing import Optional

from sqlalchemy.orm import Session

from utils.database import Student, LearnerProfile

MIN_LEVEL = 1.0
MAX_LEVEL = 5.0
LEVEL_STEP = 0.5
STREAK_TO_LEVEL_UP = 3
STREAK_TO_LEVEL_DOWN = 2


def get_or_create_student(db: Session, display_name: str, student_id: Optional[str] = None) -> Student:
    if student_id:
        existing = db.get(Student, student_id)
        if existing:
            return existing
    student = Student(display_name=display_name)
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


def get_or_create_profile(db: Session, student_id: str, topic: str) -> LearnerProfile:
    profile = (
        db.query(LearnerProfile)
        .filter_by(student_id=student_id, topic=topic)
        .first()
    )
    if profile:
        return profile

    profile = LearnerProfile(student_id=student_id, topic=topic, level=1.0,
                              known_gaps=[], misconception_log=[])
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def set_initial_level(db: Session, profile: LearnerProfile, estimated_level: float, gaps: list[str]) -> LearnerProfile:
    """Called once, right after the diagnostic assessment."""
    profile.level = max(MIN_LEVEL, min(MAX_LEVEL, estimated_level))
    profile.known_gaps = gaps
    profile.last_updated = dt.datetime.utcnow()
    db.commit()
    db.refresh(profile)
    return profile


def record_answer(db: Session, profile: LearnerProfile, is_correct: bool) -> LearnerProfile:
    """
    Explicit rule-based level adjustment (kept separate from the LLM so the
    adaptive decision is deterministic and testable - see proposal section 11).
    """
    if is_correct:
        profile.correct_streak += 1
        profile.incorrect_streak = 0
        if profile.correct_streak >= STREAK_TO_LEVEL_UP:
            profile.level = min(MAX_LEVEL, profile.level + LEVEL_STEP)
            profile.correct_streak = 0
    else:
        profile.incorrect_streak += 1
        profile.correct_streak = 0
        if profile.incorrect_streak >= STREAK_TO_LEVEL_DOWN:
            profile.level = max(MIN_LEVEL, profile.level - LEVEL_STEP)
            profile.incorrect_streak = 0

    profile.last_updated = dt.datetime.utcnow()
    db.commit()
    db.refresh(profile)
    return profile


def add_gap(db: Session, profile: LearnerProfile, gap: str) -> LearnerProfile:
    gaps = set(profile.known_gaps or [])
    gaps.add(gap)
    profile.known_gaps = list(gaps)
    db.commit()
    db.refresh(profile)
    return profile


def resolve_gap(db: Session, profile: LearnerProfile, gap: str) -> LearnerProfile:
    gaps = [g for g in (profile.known_gaps or []) if g != gap]
    profile.known_gaps = gaps
    db.commit()
    db.refresh(profile)
    return profile


def log_misconception(db: Session, profile: LearnerProfile, concept: str) -> LearnerProfile:
    log = profile.misconception_log or []
    for entry in log:
        if entry["concept"] == concept:
            entry["count"] += 1
            entry["last_seen"] = dt.datetime.utcnow().isoformat()
            break
    else:
        log.append({"concept": concept, "count": 1, "last_seen": dt.datetime.utcnow().isoformat()})

    profile.misconception_log = log
    db.commit()
    db.refresh(profile)
    return profile


def recurring_misconceptions(profile: LearnerProfile, min_count: int = 3) -> list[str]:
    """
    Flags concepts a learner keeps getting wrong. This is a *signal for human
    review* only (see proposal 2.1 / 9) - never surfaced as a diagnosis.
    """
    return [e["concept"] for e in (profile.misconception_log or []) if e["count"] >= min_count]

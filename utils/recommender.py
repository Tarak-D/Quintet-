"""
utils/recommender.py

Decides the next concept/difficulty step after each assessment turn.
Deliberately rule-based (not LLM) so the adaptive decision is deterministic,
explainable to judges, and unit-testable (proposal 11: "Incorrect adaptive
decisions" risk mitigation).
"""

from dataclasses import dataclass
from typing import Optional

from utils.database import LearnerProfile

# Minimal prerequisite map for the MVP subject. In a full build this would
# live in the knowledge base / curriculum graph.
PREREQUISITES = {
    "adding_fractions": ["common_denominators"],
    "common_denominators": [],
}


@dataclass
class NextStep:
    action: str          # "teach_prerequisite" | "continue_topic" | "advance_topic"
    topic: str
    difficulty: float
    reason: str


def recommend_next_step(
    profile: LearnerProfile,
    current_topic: str,
    just_answered_correctly: bool,
    detected_gap: Optional[str],
) -> NextStep:
    # 1. If a new prerequisite gap was just detected, address it first.
    if detected_gap and detected_gap != current_topic:
        return NextStep(
            action="teach_prerequisite",
            topic=detected_gap,
            difficulty=max(1.0, profile.level - 1.0),
            reason=f"Learner shows a gap in prerequisite concept '{detected_gap}'.",
        )

    # 2. If known gaps are already recorded and unresolved, keep working them.
    if profile.known_gaps:
        return NextStep(
            action="teach_prerequisite",
            topic=profile.known_gaps[0],
            difficulty=max(1.0, profile.level - 1.0),
            reason="Unresolved prerequisite gap still on the learner profile.",
        )

    # 3. Otherwise progress normally based on the latest answer.
    if just_answered_correctly:
        return NextStep(
            action="advance_topic",
            topic=current_topic,
            difficulty=profile.level,
            reason="Correct answer; continuing at current/updated level.",
        )

    return NextStep(
        action="continue_topic",
        topic=current_topic,
        difficulty=max(1.0, profile.level - 0.5),
        reason="Incorrect answer; re-teaching the same topic at a lower difficulty.",
    )

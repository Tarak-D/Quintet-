"""
utils/agents/analyst.py

Learning Analyst Agent: the "Analyze -> Adapt" half of the loop (proposal
2.1). Combines the deterministic misconception/recommender modules with the
persistent learner profile to decide the next action. Any recurring-pattern
flag is exposed only as a signal for human review (proposal 5 / 9) - it
never emits a diagnostic label.
"""

from typing import Optional

from sqlalchemy.orm import Session

from utils.database import LearnerProfile
from utils import learner as learner_store
from utils.misconception import detect_misconception
from utils.recommender import recommend_next_step, NextStep


class LearningAnalyst:
    def evaluate(
        self,
        db: Session,
        profile: LearnerProfile,
        topic: str,
        question: str,
        student_answer: str,
        correct_answer: str,
    ) -> dict:
        """
        Full analysis pass for one answered question:
          1. determine correctness + misconception (if any)
          2. update the learner profile (level, gaps, misconception log)
          3. recommend the next step
          4. surface any recurring-pattern flag for human review

        Returns a dict summarising all of the above for the orchestrator/API.
        """
        result = detect_misconception(topic, question, student_answer, correct_answer)
        is_correct = result["is_correct"]
        concept = result["concept"]

        profile = learner_store.record_answer(db, profile, is_correct)

        if not is_correct and concept:
            profile = learner_store.log_misconception(db, profile, concept)
            profile = learner_store.add_gap(db, profile, concept)
        elif is_correct and profile.known_gaps and topic in profile.known_gaps:
            profile = learner_store.resolve_gap(db, profile, topic)

        next_step: NextStep = recommend_next_step(
            profile=profile,
            current_topic=topic,
            just_answered_correctly=is_correct,
            detected_gap=concept if not is_correct else None,
        )

        flags = learner_store.recurring_misconceptions(profile)

        return {
            "is_correct": is_correct,
            "misconception": concept,
            "explanation": result.get("explanation", ""),
            "updated_level": profile.level,
            "next_step": {
                "action": next_step.action,
                "topic": next_step.topic,
                "difficulty": next_step.difficulty,
                "reason": next_step.reason,
            },
            "human_review_flags": flags,  # signal only - not a diagnosis
        }

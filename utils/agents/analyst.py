"""
utils/agents/analyst.py

Learning Analyst for EduLeap.

Evaluates a learner's answer, updates the learner profile,
tracks misconceptions, and recommends the next learning action.
"""

import datetime as dt
import logging
from typing import Optional

from sqlalchemy.orm import Session

from utils.database import LearnerProfile

logger = logging.getLogger("eduleap.analyst")


class LearningAnalyst:
    """
    Analyst responsible for:

    1. Checking whether an answer is correct.
    2. Updating learner progress.
    3. Tracking misconceptions.
    4. Updating known learning gaps.
    5. Recommending the next action.
    """

    def evaluate(
        self,
        db: Session,
        profile: LearnerProfile,
        topic: str,
        question: str,
        student_answer: str,
        correct_answer: str,
    ) -> dict:

        # ---------------------------------------------------------
        # 1. Basic answer verification
        # ---------------------------------------------------------

        student = str(student_answer).strip().lower()
        correct = str(correct_answer).strip().lower()

        is_correct = self._check_answer(student, correct)

        # ---------------------------------------------------------
        # 2. Update learner progress
        # ---------------------------------------------------------

        if is_correct:
            profile.correct_streak = (profile.correct_streak or 0) + 1
            profile.incorrect_streak = 0

            # Gradually increase level after successful answers.
            if profile.correct_streak >= 3:
                profile.level = min(
                    5.0,
                    float(profile.level or 1.0) + 0.25
                )

        else:
            profile.incorrect_streak = (profile.incorrect_streak or 0) + 1
            profile.correct_streak = 0

            # Gradually decrease level after repeated mistakes.
            if profile.incorrect_streak >= 3:
                profile.level = max(
                    1.0,
                    float(profile.level or 1.0) - 0.25
                )

        # ---------------------------------------------------------
        # 3. Misconception / gap handling
        # ---------------------------------------------------------

        misconception = None

        if not is_correct:
            misconception = self._identify_misconception(
                topic=topic,
                question=question,
                student_answer=student_answer,
                correct_answer=correct_answer,
            )

            self._add_misconception(
                profile,
                misconception,
            )

            self._add_gap(
                profile,
                misconception,
            )

        else:
            # Successful answer can gradually remove a gap.
            self._resolve_gap_after_success(profile, topic)

        # ---------------------------------------------------------
        # 4. Update timestamp
        # ---------------------------------------------------------

        profile.last_updated = dt.datetime.utcnow()

        db.add(profile)
        db.commit()
        db.refresh(profile)

        # ---------------------------------------------------------
        # 5. Decide next action
        # ---------------------------------------------------------

        next_action = self._next_action(
            profile=profile,
            topic=topic,
            is_correct=is_correct,
        )

        return {
            "is_correct": is_correct,
            "student_answer": student_answer,
            "correct_answer": correct_answer,
            "topic": topic,
            "level": profile.level,
            "correct_streak": profile.correct_streak,
            "incorrect_streak": profile.incorrect_streak,
            "misconception": misconception,
            "known_gaps": profile.known_gaps or [],
            "misconception_log": profile.misconception_log or [],
            "next_action": next_action,
        }

    # =============================================================
    # ANSWER CHECKING
    # =============================================================

    @staticmethod
    def _check_answer(student_answer: str, correct_answer: str) -> bool:
        """
        Simple normalized answer comparison.

        This intentionally keeps verification deterministic.
        """

        if not student_answer or not correct_answer:
            return False

        student = student_answer.strip().lower()
        correct = correct_answer.strip().lower()

        # Exact match
        if student == correct:
            return True

        # Remove common punctuation differences
        normalized_student = (
            student
            .replace(".", "")
            .replace(",", "")
            .replace(" ", "")
        )

        normalized_correct = (
            correct
            .replace(".", "")
            .replace(",", "")
            .replace(" ", "")
        )

        return normalized_student == normalized_correct

    # =============================================================
    # MISCONCEPTION DETECTION
    # =============================================================

    @staticmethod
    def _identify_misconception(
        topic: str,
        question: str,
        student_answer: str,
        correct_answer: str,
    ) -> str:
        """
        Creates a stable misconception label.

        The current backend does not yet contain a dedicated
        misconception classifier interface, so we use the topic
        as the learning-gap identifier.

        This keeps the system functional without inventing
        database dependencies.
        """

        topic_clean = str(topic).strip()

        if topic_clean:
            return topic_clean

        return "unknown_concept"

    # =============================================================
    # MISCONCEPTION LOG
    # =============================================================

    @staticmethod
    def _add_misconception(
        profile: LearnerProfile,
        misconception: str,
    ) -> None:

        log = profile.misconception_log

        if not isinstance(log, list):
            log = []

        now = dt.datetime.utcnow().isoformat()

        # Search for an existing misconception
        existing = None

        for item in log:
            if isinstance(item, dict) and item.get("concept") == misconception:
                existing = item
                break

        if existing:
            existing["count"] = int(existing.get("count", 0)) + 1
            existing["last_seen"] = now

        else:
            log.append(
                {
                    "concept": misconception,
                    "count": 1,
                    "last_seen": now,
                }
            )

        # Keep the log reasonably sized.
        profile.misconception_log = log[-50:]

    # =============================================================
    # KNOWN GAPS
    # =============================================================

    @staticmethod
    def _add_gap(
        profile: LearnerProfile,
        misconception: str,
    ) -> None:

        gaps = profile.known_gaps

        if not isinstance(gaps, list):
            gaps = []

        if misconception not in gaps:
            gaps.append(misconception)

        # Keep only the most relevant gaps.
        profile.known_gaps = gaps[-10:]

    # =============================================================
    # GAP RESOLUTION
    # =============================================================

    @staticmethod
    def _resolve_gap_after_success(
        profile: LearnerProfile,
        topic: str,
    ) -> None:

        gaps = profile.known_gaps

        if not isinstance(gaps, list):
            return

        # Only remove a gap after a successful answer
        # when it corresponds to the current topic.
        if topic in gaps:
            gaps.remove(topic)

        profile.known_gaps = gaps

    # =============================================================
    # NEXT ACTION
    # =============================================================

    @staticmethod
    def _next_action(
        profile: LearnerProfile,
        topic: str,
        is_correct: bool,
    ) -> str:

        gaps = profile.known_gaps or []

        if not is_correct:

            if gaps:
                return "teach_prerequisite"

            return "review_topic"

        if gaps:
            return "teach_prerequisite"

        if profile.correct_streak >= 3:
            return "increase_difficulty"

        return "continue_topic"
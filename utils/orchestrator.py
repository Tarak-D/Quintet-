"""
utils/orchestrator.py

AI Orchestrator: the coordination layer described in proposal 3.2 / 4.
Exposes a small, high-level API that utils/api.py (FastAPI) calls, so the
web layer never talks to individual agents directly.

Flow implemented here:
    Student Input -> Learner Analysis -> AI Planning -> Agent Execution
    -> Verification -> Adaptation -> Final Output
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from utils import learner as learner_store
from utils.database import Interaction
from utils.agents.diagnostic import DiagnosticAgent
from utils.agents.tutor import TutorAgent
from utils.agents.assessment import AssessmentAgent
from utils.agents.analyst import LearningAnalyst

logger = logging.getLogger("eduleap.orchestrator")


class AIOrchestrator:
    def __init__(self):
        self.diagnostic = DiagnosticAgent()
        self.tutor = TutorAgent()
        self.assessment = AssessmentAgent()
        self.analyst = LearningAnalyst()

    # ---------- Stage 1: diagnostic ----------

    def start_diagnostic(self, topic: str) -> list[dict]:
        return self.diagnostic.generate_diagnostic(topic)

    def submit_diagnostic(
        self, db: Session, student_id: str, topic: str, responses: list[dict]
    ) -> dict:
        """
        responses: [{"question": str, "targets": str, "correct": bool}, ...]
        Creates/initialises the learner profile from the diagnostic result.
        """
        estimate = self.diagnostic.estimate_level(topic, responses)

        profile = learner_store.get_or_create_profile(db, student_id, topic)
        profile = learner_store.set_initial_level(db, profile, estimate["level"], estimate["gaps"])

        db.add(Interaction(
            student_id=student_id, topic=topic, stage="diagnostic",
            prompt=str(responses), response=str(estimate),
        ))
        db.commit()

        return {
            "level": profile.level,
            "gaps": profile.known_gaps,
        }

    # ---------- Stage 2: tutor + practice loop ----------

    def teach(
        self, db: Session, student_id: str, topic: str,
        struggling: bool = False, prior_explanation_summary: Optional[str] = None,
    ) -> dict:
        """Learner Analysis -> Planning -> Agent Execution -> Final Output (teach step)."""
        profile = learner_store.get_or_create_profile(db, student_id, topic)

        # If the profile already has an unresolved gap, teach that instead
        # of the requested topic (adaptation happening at the planning stage).
        teach_topic = profile.known_gaps[0] if profile.known_gaps else topic

        explanation = self.tutor.explain(
            topic=teach_topic,
            level=profile.level,
            struggling=struggling,
            prior_explanation_summary=prior_explanation_summary,
        )

        db.add(Interaction(
            student_id=student_id, topic=teach_topic, stage="tutor",
            response=explanation,
        ))
        db.commit()

        return {"topic_taught": teach_topic, "explanation": explanation, "level": profile.level}

    def ask_question(self, db: Session, student_id: str, topic: str) -> dict:
        profile = learner_store.get_or_create_profile(db, student_id, topic)
        focus_topic = profile.known_gaps[0] if profile.known_gaps else topic

        q = self.assessment.generate_question(
            topic=focus_topic, level=profile.level,
            target_gap=profile.known_gaps[0] if profile.known_gaps else None,
        )

        db.add(Interaction(
            student_id=student_id, topic=focus_topic, stage="assessment",
            prompt=q.get("question"),
        ))
        db.commit()

        return q  # {"question", "correct_answer", "targets"}

    # ---------- Stage 3: verification + adaptation ----------

    def submit_answer(
        self, db: Session, student_id: str, topic: str,
        question: str, correct_answer: str, student_answer: str,
    ) -> dict:
        """Verification (grade) -> Adaptation (analyst) -> Final Output (next step)."""
        profile = learner_store.get_or_create_profile(db, student_id, topic)

        analysis = self.analyst.evaluate(
            db=db, profile=profile, topic=topic,
            question=question, student_answer=student_answer, correct_answer=correct_answer,
        )

        db.add(Interaction(
            student_id=student_id, topic=topic, stage="analysis",
            prompt=question, response=student_answer,
            is_correct=int(analysis["is_correct"]),
            meta=analysis,
        ))
        db.commit()

        return analysis

    # ---------- Convenience: full "next action" decision ----------

    def get_next_action(self, db: Session, student_id: str, topic: str) -> dict:
        profile = learner_store.get_or_create_profile(db, student_id, topic)
        if profile.known_gaps:
            return {"action": "teach_prerequisite", "topic": profile.known_gaps[0], "level": profile.level}
        return {"action": "continue_topic", "topic": topic, "level": profile.level}


# module-level singleton used by the API layer
orchestrator = AIOrchestrator()

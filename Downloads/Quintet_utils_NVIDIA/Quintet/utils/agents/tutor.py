"""
utils/agents/tutor.py

Tutor Agent: generates explanations calibrated to the learner's level, and
switches teaching strategy (e.g. breaking a concept into a simpler
prerequisite) when the learner is struggling, instead of repeating the same
explanation (proposal 2.2 / 7).
"""

from typing import Optional

from utils.llm_client import call_llm
from utils.rag import retrieve_context

LEVEL_DESCRIPTIONS = {
    1: "complete beginner - use very simple language and a concrete, everyday example",
    2: "early learner - simple language, one worked example",
    3: "developing - can handle a bit more detail and a second practice example",
    4: "confident - can handle more formal explanation and notation",
    5: "advanced - concise explanation, focus on edge cases/nuance",
}


def _level_bucket(level: float) -> int:
    return max(1, min(5, round(level)))


class TutorAgent:
    def explain(
        self,
        topic: str,
        level: float,
        struggling: bool = False,
        prior_explanation_summary: Optional[str] = None,
    ) -> str:
        """
        Generates one explanation. If `struggling` is True, the prompt
        instructs the model to change strategy rather than repeat itself.
        """
        context = retrieve_context(f"{topic} explanation for learners", topic=topic, k=4)
        bucket = _level_bucket(level)

        system = (
            "You are a warm, patient AI tutor for school-age children in "
            "low-resource settings. Explain the given concept clearly and "
            f"at this level: {LEVEL_DESCRIPTIONS[bucket]}. "
            "Ground your explanation in the provided reference material "
            "where possible, and never invent facts not supported by it or "
            "by standard curriculum knowledge. Keep the explanation short "
            "enough to read in under a minute."
        )

        if struggling and prior_explanation_summary:
            system += (
                " The learner is still struggling after a previous "
                "explanation. Do NOT repeat that explanation - instead, "
                "break the concept into a simpler prerequisite step, use a "
                "different example or analogy, or approach it visually/"
                "concretely instead of abstractly."
            )

        user_parts = [f"Topic: {topic}"]
        if prior_explanation_summary:
            user_parts.append(f"Previous explanation given (avoid repeating): {prior_explanation_summary}")
        user_parts.append(f"Reference material:\n{context if context else '(none found - use general subject knowledge)'}")
        user = "\n\n".join(user_parts)

        return call_llm(system, user, max_tokens=500, temperature=0.5)

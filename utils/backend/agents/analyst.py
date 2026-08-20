"""
Learning Analyst Agent
----------------------
Maintains the persistent learner profile: updates mastery after each
response, detects recurring misconceptions, and recommends the next
concept/difficulty. This is the agent that closes the loop:

    ... -> Verification -> [Learning Analyst] -> Adaptation -> Final Output
"""

from __future__ import annotations
from typing import Dict, Any

from utils.llm_client import chat_json
from utils.database import get_learner_profile, save_learner_profile


ANALYST_SYSTEM_PROMPT = """You are the Learning Analyst inside EduLeap. \
Given a learner's recent question/answer history and mastery map, \
recommend the single next concept and difficulty level to move to. Treat \
any repeated concerning pattern only as a signal for human review, never \
as a medical or educational diagnosis. Return STRICT JSON only."""

# A concept missed this many times in a row is flagged as a recurring
# misconception, which triggers tutor.decompose_prerequisite() instead of
# the tutor repeating the same explanation.
MISCONCEPTION_THRESHOLD = 2


def update_profile(
    learner_id: str, evaluation: Dict[str, Any], concept_tag: str
) -> Dict[str, Any]:
    """Persist the outcome of one answered question into the learner profile.

    `evaluation` is the dict returned by assessment.evaluate_response().
    Returns the updated profile.
    """
    profile = get_learner_profile(learner_id) or _default_profile(learner_id)

    profile["history"].append(
        {
            "concept_tag": concept_tag,
            "correct": evaluation["correct"],
            "misconception": evaluation.get("misconception"),
        }
    )

    if evaluation["correct"]:
        profile["consecutive_misses"][concept_tag] = 0
        profile["mastery"][concept_tag] = min(
            1.0, profile["mastery"].get(concept_tag, 0.5) + 0.15
        )
    else:
        profile["consecutive_misses"][concept_tag] = (
            profile["consecutive_misses"].get(concept_tag, 0) + 1
        )
        profile["mastery"][concept_tag] = max(
            0.0, profile["mastery"].get(concept_tag, 0.5) - 0.1
        )

    save_learner_profile(learner_id, profile)
    return profile


def detect_recurring_misconception(profile: Dict[str, Any], concept_tag: str) -> bool:
    """True if `concept_tag` has been missed enough times in a row to
    trigger the Tutor Agent's prerequisite-decomposition strategy."""
    return profile["consecutive_misses"].get(concept_tag, 0) >= MISCONCEPTION_THRESHOLD


def recommend_next_step(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the LLM to recommend the next concept + difficulty based on
    the learner's recent history and mastery map.

    Returns:
        {
          "next_concept": str,
          "next_difficulty": "easy" | "medium" | "hard",
          "reason": str,
          "flag_for_human_review": bool
        }
    """
    recent_history = profile["history"][-8:]
    mastery = profile["mastery"]

    user_prompt = f"""
Recent history (most recent last): {recent_history}
Current mastery per concept (0-1): {mastery}

Recommend the next concept and difficulty. Set flag_for_human_review to \
true only if the same concept has been missed 3+ times across this \
history despite the tutor changing strategy - this is a signal for a \
human educator to look in, not a diagnosis.

Return JSON exactly:
{{
  "next_concept": "...",
  "next_difficulty": "easy",
  "reason": "...",
  "flag_for_human_review": false
}}
"""
    return chat_json(
        [
            {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )


def _default_profile(learner_id: str) -> Dict[str, Any]:
    return {
        "learner_id": learner_id,
        "history": [],
        "mastery": {},
        "consecutive_misses": {},
    }

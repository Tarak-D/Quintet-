"""
EduLeap - Learning Analyst Agent
================================

The Learning Analyst closes the adaptive learning loop.

Responsibilities
----------------
1. Load the persistent learner profile.
2. Record each assessment result.
3. Update concept mastery.
4. Track consecutive mistakes.
5. Detect recurring misconceptions.
6. Identify persistent knowledge gaps.
7. Use NVIDIA NIM to recommend the next concept/difficulty.
8. Flag repeated concerning academic patterns for human review.
9. Save the updated learner profile.

Pipeline
--------

    Student
       |
       v
    Assessment Agent
       |
       v
    Student Answer
       |
       v
    Verification
       |
       v
    Learning Analyst
       |
       +--> Update mastery
       |
       +--> Detect misconception
       |
       +--> Update learner profile
       |
       +--> Recommend next concept
       |
       v
    Adaptation
       |
       v
    Tutor / Assessment Agent


Project structure
-----------------

    Quintet/
    |
    +-- utils/
        |
        +-- api.py
        +-- database.py
        +-- learner.py
        +-- recommender.py
        +-- misconception.py
        +-- rag.py
        |
        +-- agents/
            |
            +-- diagnostic.py
            +-- tutor.py
            +-- assessment.py
            +-- analyst.py
        |
        +-- orchestrator.py


NVIDIA NIM
----------

This file directly communicates with NVIDIA NIM.

Required .env:

    NVIDIA_API_KEY=nvapi-your-key
    NVIDIA_MODEL=meta/llama-3.1-8b-instruct

Optional:

    NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
    NVIDIA_TEMPERATURE=0.3
    NVIDIA_MAX_TOKENS=1800


Important
---------

This agent does NOT diagnose:
- learning disabilities
- medical conditions
- psychological conditions
- neurological conditions

It only analyzes observable academic learning patterns.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from utils.database import (
    get_learner_profile,
    save_learner_profile,
)


# ============================================================================
# ENVIRONMENT
# ============================================================================

load_dotenv()


NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY"
)

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "meta/llama-3.1-8b-instruct",
)

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)

NVIDIA_TEMPERATURE = float(
    os.getenv(
        "NVIDIA_TEMPERATURE",
        "0.3",
    )
)

NVIDIA_MAX_TOKENS = int(
    os.getenv(
        "NVIDIA_MAX_TOKENS",
        "1800",
    )
)


# ============================================================================
# NVIDIA NIM CLIENT
# ============================================================================

if not NVIDIA_API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY is missing. "
        "Add NVIDIA_API_KEY=your_key to the .env file."
    )


client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of consecutive misses required before the Tutor Agent
# should consider teaching a prerequisite instead of simply
# continuing with the same concept.
MISCONCEPTION_THRESHOLD = 2


# Number of misses in recent history required before suggesting
# human educator review.
HUMAN_REVIEW_THRESHOLD = 3


# Mastery update values.
#
# These are deterministic rules rather than LLM decisions.
MASTERY_CORRECT_INCREMENT = 0.15
MASTERY_INCORRECT_DECREMENT = 0.10

INITIAL_MASTERY = 0.50


VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

ANALYST_SYSTEM_PROMPT = """
You are the Learning Analyst inside EduLeap.

EduLeap is an adaptive AI tutoring system for school-age
children.

Your responsibility is to analyze observable academic
performance and recommend the next learning step.

You receive:
- learner history
- concept mastery
- recent assessment results
- recurring misconceptions
- current concept

Your task is to recommend:
1. the next concept
2. the next difficulty
3. a short reason
4. whether human educator review may be useful

IMPORTANT
---------

You are NOT a medical or psychological diagnostic system.

Never diagnose:
- dyslexia
- ADHD
- dyscalculia
- intellectual disability
- learning disability
- mental health conditions
- neurological conditions

You may only identify observable educational patterns.

For example:

GOOD:
"The learner has repeatedly struggled with common denominators."

BAD:
"The learner has dyscalculia."

Human review means that a repeated academic difficulty
may deserve attention from a teacher or educator.

It does NOT mean that the learner has a disability.

RECOMMENDATION RULES
--------------------

1. If a prerequisite gap is unresolved, recommend the
   prerequisite before returning to the original concept.

2. If mastery is low, recommend easier practice.

3. If mastery is improving, continue with the concept.

4. If mastery is high, recommend a slightly harder
   question or the next related concept.

5. Do not jump to unrelated concepts.

6. Do not increase difficulty after one correct answer.

7. Do not decrease difficulty based only on one incorrect
   answer if the learner has otherwise demonstrated mastery.

8. Repeated errors should influence the next step.

9. Prefer educationally meaningful adaptations over
   random concept changes.

10. Return valid JSON only.

OUTPUT:

{
    "next_concept": "...",
    "next_difficulty": "easy",
    "reason": "...",
    "flag_for_human_review": false
}
"""


# ============================================================================
# JSON HELPERS
# ============================================================================

def _clean_json_text(
    text: str,
) -> str:
    """
    Clean common formatting mistakes from NVIDIA NIM output.
    """

    if not text:
        raise ValueError(
            "NVIDIA NIM returned an empty response."
        )

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # Extract outer JSON object if the model included extra text.
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if (
        first_brace != -1
        and last_brace != -1
        and last_brace > first_brace
    ):

        text = text[
            first_brace:last_brace + 1
        ]

    return text


def _parse_json_response(
    text: str,
) -> Dict[str, Any]:
    """
    Parse NVIDIA NIM output as JSON.
    """

    cleaned = _clean_json_text(
        text
    )

    try:

        result = json.loads(
            cleaned
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "NVIDIA NIM returned invalid JSON."
        ) from exc

    if not isinstance(
        result,
        dict,
    ):

        raise ValueError(
            "Learning Analyst response must be a JSON object."
        )

    return result


# ============================================================================
# NVIDIA NIM CALL
# ============================================================================

def _call_nim(
    user_prompt: str,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Call NVIDIA NIM directly.

    No utils.llm_client is used.
    """

    try:

        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": ANALYST_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=(
                NVIDIA_TEMPERATURE
                if temperature is None
                else temperature
            ),
            max_tokens=NVIDIA_MAX_TOKENS,
            response_format={
                "type": "json_object"
            },
        )

    except Exception as exc:

        raise RuntimeError(
            "NVIDIA NIM request failed. "
            f"Model: {NVIDIA_MODEL}. "
            f"Error: {exc}"
        ) from exc

    if not response.choices:

        raise RuntimeError(
            "NVIDIA NIM returned no response choices."
        )

    content = response.choices[0].message.content

    if not content:

        raise RuntimeError(
            "NVIDIA NIM returned an empty response."
        )

    return _parse_json_response(
        content
    )


# ============================================================================
# DEFAULT PROFILE
# ============================================================================

def _default_profile(
    learner_id: str,
) -> Dict[str, Any]:
    """
    Create a new learner profile.

    This structure is intentionally simple for the MVP.
    """

    return {
        "learner_id": learner_id,

        "history": [],

        "mastery": {},

        "consecutive_misses": {},

        "misconceptions": {},

        "knowledge_gaps": [],

        "strengths": [],

        "current_level": "beginner",

        "last_concept": None,

        "recommended_next_concept": None,

        "recommended_difficulty": "easy",
    }


# ============================================================================
# PROFILE NORMALIZATION
# ============================================================================

def _normalize_profile(
    profile: Dict[str, Any],
    learner_id: str,
) -> Dict[str, Any]:
    """
    Make sure profiles loaded from the database contain all
    fields expected by this agent.

    This is useful when an older database record does not yet
    contain fields added later.
    """

    defaults = _default_profile(
        learner_id
    )

    for key, default_value in defaults.items():

        if key not in profile:

            profile[key] = (
                default_value.copy()
                if isinstance(
                    default_value,
                    dict,
                )
                else (
                    default_value.copy()
                    if isinstance(
                        default_value,
                        list,
                    )
                    else default_value
                )
            )

    return profile


# ============================================================================
# UPDATE PROFILE
# ============================================================================

def update_profile(
    learner_id: str,
    evaluation: Dict[str, Any],
    concept_tag: str,
) -> Dict[str, Any]:
    """
    Persist the outcome of one answered question.

    Parameters
    ----------
    learner_id:
        Unique learner identifier.

    evaluation:
        Dictionary returned by Assessment Agent.

    concept_tag:
        Concept assessed by the question.

    Returns
    -------
    Dict[str, Any]
        Updated learner profile.

    Deterministic profile updates
    -----------------------------

    Correct answer:

        mastery += 0.15

    Incorrect answer:

        mastery -= 0.10

    Mastery is always constrained to [0, 1].

    Consecutive misses are reset after a correct answer.
    """

    if not isinstance(
        learner_id,
        str,
    ) or not learner_id.strip():

        raise ValueError(
            "learner_id must be a non-empty string."
        )

    if not isinstance(
        evaluation,
        dict,
    ):

        raise TypeError(
            "evaluation must be a dictionary."
        )

    if not isinstance(
        concept_tag,
        str,
    ) or not concept_tag.strip():

        raise ValueError(
            "concept_tag must be a non-empty string."
        )

    concept_tag = concept_tag.strip()

    if "correct" not in evaluation:

        raise ValueError(
            "evaluation must contain 'correct'."
        )

    if not isinstance(
        evaluation["correct"],
        bool,
    ):

        raise TypeError(
            "evaluation['correct'] must be a boolean."
        )

    # ------------------------------------------------------------------------
    # Load profile
    # ------------------------------------------------------------------------

    profile = get_learner_profile(
        learner_id
    )

    if not profile:

        profile = _default_profile(
            learner_id
        )

    else:

        profile = _normalize_profile(
            profile,
            learner_id,
        )

    # ------------------------------------------------------------------------
    # Assessment result
    # ------------------------------------------------------------------------

    correct = evaluation[
        "correct"
    ]

    misconception = evaluation.get(
        "misconception"
    )

    prerequisite_tag = evaluation.get(
        "prerequisite_tag"
    )

    difficulty = evaluation.get(
        "difficulty",
        "easy",
    )

    # ------------------------------------------------------------------------
    # Record history
    # ------------------------------------------------------------------------

    history_entry = {
        "concept_tag": concept_tag,
        "correct": correct,
        "misconception": misconception,
        "prerequisite_tag": prerequisite_tag,
        "difficulty": difficulty,
    }

    profile[
        "history"
    ].append(
        history_entry
    )

    # Keep history manageable for the MVP.
    # The complete persistent profile can still grow through
    # separate analytics/storage tables later.
    if len(
        profile["history"]
    ) > 100:

        profile["history"] = profile[
            "history"
        ][-100:]

    # ------------------------------------------------------------------------
    # Initialize concept values
    # ------------------------------------------------------------------------

    current_mastery = float(
        profile["mastery"].get(
            concept_tag,
            INITIAL_MASTERY,
        )
    )

    consecutive_misses = int(
        profile[
            "consecutive_misses"
        ].get(
            concept_tag,
            0,
        )
    )

    # ------------------------------------------------------------------------
    # Correct answer
    # ------------------------------------------------------------------------

    if correct:

        consecutive_misses = 0

        current_mastery = min(
            1.0,
            current_mastery
            + MASTERY_CORRECT_INCREMENT,
        )

    # ------------------------------------------------------------------------
    # Incorrect answer
    # ------------------------------------------------------------------------

    else:

        consecutive_misses += 1

        current_mastery = max(
            0.0,
            current_mastery
            - MASTERY_INCORRECT_DECREMENT,
        )

    # ------------------------------------------------------------------------
    # Save mastery
    # ------------------------------------------------------------------------

    profile[
        "mastery"
    ][concept_tag] = round(
        current_mastery,
        3,
    )

    profile[
        "consecutive_misses"
    ][concept_tag] = consecutive_misses

    profile[
        "last_concept"
    ] = concept_tag

    # ------------------------------------------------------------------------
    # Record misconception
    # ------------------------------------------------------------------------

    if misconception:

        misconception = str(
            misconception
        ).strip()

        if misconception:

            profile[
                "misconceptions"
            ][misconception] = (
                profile[
                    "misconceptions"
                ].get(
                    misconception,
                    0,
                )
                + 1
            )

    # ------------------------------------------------------------------------
    # Record knowledge gap
    # ------------------------------------------------------------------------

    if not correct:

        if concept_tag not in profile[
            "knowledge_gaps"
        ]:

            profile[
                "knowledge_gaps"
            ].append(
                concept_tag
            )

        if prerequisite_tag:

            prerequisite_tag = str(
                prerequisite_tag
            ).strip()

            if (
                prerequisite_tag
                and prerequisite_tag
                not in profile[
                    "knowledge_gaps"
                ]
            ):

                profile[
                    "knowledge_gaps"
                ].append(
                    prerequisite_tag
                )

    # ------------------------------------------------------------------------
    # Remove resolved concept from gaps after successful answer
    # ------------------------------------------------------------------------

    if correct:

        if concept_tag in profile[
            "knowledge_gaps"
        ]:

            profile[
                "knowledge_gaps"
            ].remove(
                concept_tag
            )

    # ------------------------------------------------------------------------
    # Save profile
    # ------------------------------------------------------------------------

    save_learner_profile(
        learner_id,
        profile,
    )

    return profile


# ============================================================================
# RECURRING MISCONCEPTION DETECTION
# ============================================================================

def detect_recurring_misconception(
    profile: Dict[str, Any],
    concept_tag: str,
) -> bool:
    """
    Return True when the learner has missed the same concept
    at least MISCONCEPTION_THRESHOLD times consecutively.

    Example:

        q1 -> wrong
        q2 -> wrong

    threshold = 2

    Result:

        True

    The Orchestrator can then call:

        tutor.decompose_prerequisite()

    instead of simply repeating the same explanation.
    """

    if not isinstance(
        profile,
        dict,
    ):

        return False

    consecutive_misses = profile.get(
        "consecutive_misses",
        {},
    )

    return (
        consecutive_misses.get(
            concept_tag,
            0,
        )
        >= MISCONCEPTION_THRESHOLD
    )


# ============================================================================
# MISCONCEPTION DETAILS
# ============================================================================

def get_recurring_misconceptions(
    profile: Dict[str, Any],
) -> List[str]:
    """
    Return misconception labels that have appeared repeatedly.
    """

    misconceptions = profile.get(
        "misconceptions",
        {},
    )

    if not isinstance(
        misconceptions,
        dict,
    ):

        return []

    return [
        misconception
        for misconception, count
        in misconceptions.items()
        if count >= MISCONCEPTION_THRESHOLD
    ]


# ============================================================================
# CONCEPT PERFORMANCE
# ============================================================================

def get_concept_mastery(
    profile: Dict[str, Any],
    concept_tag: str,
) -> float:
    """
    Return current mastery for a concept.
    """

    mastery = profile.get(
        "mastery",
        {},
    )

    return round(
        float(
            mastery.get(
                concept_tag,
                INITIAL_MASTERY,
            )
        ),
        3,
    )


# ============================================================================
# DIFFICULTY RECOMMENDATION
# ============================================================================

def _deterministic_difficulty(
    mastery: float,
    consecutive_misses: int,
    current_difficulty: str,
) -> str:
    """
    Produce a deterministic difficulty recommendation.

    This acts as a safety layer around the LLM recommendation.

    Rules:

        repeated misses -> easy

        mastery < 0.40 -> easy

        mastery 0.40-0.74 -> current level

        mastery >= 0.75 -> increase difficulty
    """

    difficulties = [
        "easy",
        "medium",
        "hard",
    ]

    if current_difficulty not in difficulties:

        current_difficulty = "easy"

    current_index = difficulties.index(
        current_difficulty
    )

    # Repeated struggle always gets easier practice.
    if (
        consecutive_misses
        >= MISCONCEPTION_THRESHOLD
    ):

        return "easy"

    if mastery < 0.40:

        return "easy"

    if mastery < 0.75:

        return current_difficulty

    # Strong mastery -> gradually increase.
    return difficulties[
        min(
            current_index + 1,
            len(difficulties) - 1,
        )
    ]


# ============================================================================
# HUMAN REVIEW DETECTION
# ============================================================================

def _needs_human_review(
    profile: Dict[str, Any],
    concept_tag: str,
) -> bool:
    """
    Determine whether repeated academic difficulty should
    be surfaced for educator review.

    This is NOT a diagnosis.
    """

    consecutive_misses = profile.get(
        "consecutive_misses",
        {},
    ).get(
        concept_tag,
        0,
    )

    return (
        consecutive_misses
        >= HUMAN_REVIEW_THRESHOLD
    )


# ============================================================================
# RECOMMEND NEXT STEP
# ============================================================================

def recommend_next_step(
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recommend the learner's next concept and difficulty.

    NVIDIA NIM proposes the next learning step.

    Python applies deterministic safety constraints to the
    recommendation before returning it.

    Returns
    -------

    {
        "next_concept": str,
        "next_difficulty": "easy" | "medium" | "hard",
        "reason": str,
        "flag_for_human_review": bool
    }
    """

    if not isinstance(
        profile,
        dict,
    ):

        raise TypeError(
            "profile must be a dictionary."
        )

    profile = _normalize_profile(
        profile,
        profile.get(
            "learner_id",
            "unknown",
        ),
    )

    # ------------------------------------------------------------------------
    # Recent history
    # ------------------------------------------------------------------------

    recent_history = profile[
        "history"
    ][-8:]

    mastery = profile[
        "mastery"
    ]

    current_concept = profile.get(
        "last_concept"
    )

    # ------------------------------------------------------------------------
    # Determine current concept information
    # ------------------------------------------------------------------------

    current_mastery = (
        get_concept_mastery(
            profile,
            current_concept,
        )
        if current_concept
        else 0.5
    )

    consecutive_misses = (
        profile[
            "consecutive_misses"
        ].get(
            current_concept,
            0,
        )
        if current_concept
        else 0
    )

    current_difficulty = (
        recent_history[-1].get(
            "difficulty",
            "easy",
        )
        if recent_history
        else "easy"
    )

    safe_difficulty = _deterministic_difficulty(
        mastery=current_mastery,
        consecutive_misses=consecutive_misses,
        current_difficulty=current_difficulty,
    )

    # ------------------------------------------------------------------------
    # Human review flag
    # ------------------------------------------------------------------------

    human_review_required = False

    if current_concept:

        human_review_required = (
            _needs_human_review(
                profile,
                current_concept,
            )
        )

    # ------------------------------------------------------------------------
    # Prepare LLM prompt
    # ------------------------------------------------------------------------

    user_prompt = f"""
Learner profile:

Learner ID:
{profile.get("learner_id")}

Current concept:
{current_concept}

Current mastery:
{current_mastery}

Consecutive misses on current concept:
{consecutive_misses}

Current knowledge gaps:
{profile.get("knowledge_gaps", [])}

Recurring misconceptions:
{get_recurring_misconceptions(profile)}

Recent history:
{json.dumps(
    recent_history,
    ensure_ascii=False,
    indent=2,
)}

Mastery by concept:
{json.dumps(
    mastery,
    ensure_ascii=False,
    indent=2,
)}

Deterministic difficulty recommendation:
{safe_difficulty}

Human review currently required:
{human_review_required}

Recommend ONE next concept.

Rules:

1. If the learner has a recurring prerequisite gap,
   recommend that prerequisite first.

2. If the current concept has low mastery, it is acceptable
   to continue practicing the same concept at an easier level.

3. Do not jump to an unrelated topic.

4. If mastery is strong, recommend a closely related next
   concept or increased difficulty.

5. The recommended difficulty must not be harder than:
   {safe_difficulty}

6. Set flag_for_human_review to true only when the academic
   pattern provides a reasonable reason for educator review.

7. Human review is NOT a diagnosis.

Return JSON exactly:

{{
    "next_concept": "...",
    "next_difficulty": "{safe_difficulty}",
    "reason": "...",
    "flag_for_human_review": {str(
        human_review_required
    ).lower()}
}}
"""

    # ------------------------------------------------------------------------
    # NVIDIA NIM recommendation
    # ------------------------------------------------------------------------

    try:

        result = _call_nim(
            user_prompt=user_prompt,
            temperature=0.3,
        )

    except Exception:

        # If NIM fails, the deterministic fallback keeps the
        # adaptive loop working.
        fallback_concept = (
            current_concept
            or (
                profile[
                    "knowledge_gaps"
                ][0]
                if profile[
                    "knowledge_gaps"
                ]
                else "continue_current_topic"
            )
        )

        return {
            "next_concept": fallback_concept,
            "next_difficulty": safe_difficulty,
            "reason": (
                "Fallback recommendation based on "
                "the learner's current mastery and "
                "recent performance."
            ),
            "flag_for_human_review": (
                human_review_required
            ),
        }

    # ------------------------------------------------------------------------
    # Validate LLM recommendation
    # ------------------------------------------------------------------------

    next_concept = result.get(
        "next_concept"
    )

    reason = result.get(
        "reason"
    )

    llm_difficulty = result.get(
        "next_difficulty"
    )

    llm_human_review = result.get(
        "flag_for_human_review",
        False,
    )

    if (
        not isinstance(
            next_concept,
            str,
        )
        or not next_concept.strip()
    ):

        next_concept = (
            current_concept
            or "continue_current_topic"
        )

    else:

        next_concept = next_concept.strip()

    if (
        not isinstance(
            reason,
            str,
        )
        or not reason.strip()
    ):

        reason = (
            "Recommendation based on recent "
            "learner performance."
        )

    else:

        reason = reason.strip()

    # ------------------------------------------------------------------------
    # Apply deterministic difficulty safety rule
    # ------------------------------------------------------------------------

    # The LLM can recommend easier, but it cannot override
    # the deterministic maximum difficulty.
    if llm_difficulty not in VALID_DIFFICULTIES:

        next_difficulty = safe_difficulty

    else:

        difficulty_order = {
            "easy": 0,
            "medium": 1,
            "hard": 2,
        }

        next_difficulty = (
            llm_difficulty
            if difficulty_order[
                llm_difficulty
            ]
            <= difficulty_order[
                safe_difficulty
            ]
            else safe_difficulty
        )

    # ------------------------------------------------------------------------
    # Human review safety
    # ------------------------------------------------------------------------

    flag_for_human_review = (
        human_review_required
        or (
            llm_human_review is True
            and human_review_required
        )
    )

    # ------------------------------------------------------------------------
    # Persist recommendation
    # ------------------------------------------------------------------------

    profile[
        "recommended_next_concept"
    ] = next_concept

    profile[
        "recommended_difficulty"
    ] = next_difficulty

    save_learner_profile(
        profile[
            "learner_id"
        ],
        profile,
    )

    return {
        "next_concept": next_concept,
        "next_difficulty": next_difficulty,
        "reason": reason,
        "flag_for_human_review": flag_for_human_review,
    }


# ============================================================================
# COMPLETE ANALYST PIPELINE
# ============================================================================

def analyze_response(
    learner_id: str,
    concept_tag: str,
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Complete Learning Analyst pipeline for one assessment response.

    Steps:

        1. Load learner profile.
        2. Update mastery.
        3. Record misconception.
        4. Detect recurring misconception.
        5. Generate next-step recommendation.
        6. Save profile.
        7. Return adaptation information.

    This function is the main function that the
    orchestrator can call after the Assessment Agent
    evaluates a response.
    """

    # ------------------------------------------------------------------------
    # Update learner profile
    # ------------------------------------------------------------------------

    profile = update_profile(
        learner_id=learner_id,
        evaluation=evaluation,
        concept_tag=concept_tag,
    )

    # ------------------------------------------------------------------------
    # Detect recurring misconception
    # ------------------------------------------------------------------------

    recurring = detect_recurring_misconception(
        profile=profile,
        concept_tag=concept_tag,
    )

    # ------------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------------

    recommendation = recommend_next_step(
        profile
    )

    # ------------------------------------------------------------------------
    # Return complete adaptation signal
    # ------------------------------------------------------------------------

    return {
        "learner_id": learner_id,

        "profile": profile,

        "current_concept": concept_tag,

        "mastery": get_concept_mastery(
            profile,
            concept_tag,
        ),

        "recurring_misconception": recurring,

        "recommendation": recommendation,

        "adaptation": {
            "use_prerequisite_teaching": recurring,
            "next_concept": recommendation[
                "next_concept"
            ],
            "next_difficulty": recommendation[
                "next_difficulty"
            ],
            "flag_for_human_review": recommendation[
                "flag_for_human_review"
            ],
        },
    }


# ============================================================================
# HEALTH CHECK
# ============================================================================

def analyst_health_check() -> Dict[str, Any]:
    """
    Return configuration information without making an
    NVIDIA request.
    """

    return {
        "agent": "analyst",
        "status": "configured",
        "model": NVIDIA_MODEL,
        "base_url": NVIDIA_BASE_URL,
        "deterministic_mastery_updates": True,
        "recurring_misconception_threshold": (
            MISCONCEPTION_THRESHOLD
        ),
        "human_review_threshold": (
            HUMAN_REVIEW_THRESHOLD
        ),
    }


# ============================================================================
# LOCAL TEST
# ============================================================================

if __name__ == "__main__":

    """
    Run from project root:

        python -m utils.agents.analyst

    Make sure .env contains:

        NVIDIA_API_KEY=...
        NVIDIA_MODEL=meta/llama-3.1-8b-instruct

    and your database.py contains:

        get_learner_profile()
        save_learner_profile()
    """

    print("=" * 70)
    print("EduLeap Learning Analyst Agent")
    print("=" * 70)

    print(
        f"Model: {NVIDIA_MODEL}"
    )

    print(
        f"NVIDIA endpoint: {NVIDIA_BASE_URL}"
    )

    print()

    test_evaluation = {
        "correct": False,
        "feedback": (
            "Remember that fractions need a common "
            "denominator before you add them."
        ),
        "misconception": (
            "adds numerators and denominators directly"
        ),
        "prerequisite_tag": (
            "common_denominators"
        ),
        "difficulty": "easy",
    }

    try:

        result = analyze_response(
            learner_id="demo_learner",
            concept_tag="fraction_addition",
            evaluation=test_evaluation,
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as exc:

        print(
            "Learning Analyst failed:"
        )

        print(
            exc
        )

"""
EduLeap - Diagnostic Agent
==========================

The Diagnostic Agent runs a short diagnostic assessment at the
beginning of a learning topic.

Responsibilities
----------------
1. Retrieve educational context from the EduLeap knowledge base.
2. Ask NVIDIA NIM to generate a short diagnostic assessment.
3. Validate the generated questions.
4. Keep correct answers on the backend.
5. Grade learner responses deterministically.
6. Estimate the learner's academic level.
7. Identify prerequisite/topic knowledge gaps.

Important
---------
This agent does NOT diagnose:
- learning disabilities
- medical conditions
- psychological conditions

It only estimates academic performance and identifies
educational knowledge gaps.

Project structure:

    Quintet/
    |
    +-- utils/
        |
        +-- rag.py
        |
        +-- agents/
            |
            +-- diagnostic.py
            +-- tutor.py
            +-- assessment.py
            +-- analyst.py

NVIDIA NIM
----------
This file directly connects to NVIDIA's OpenAI-compatible
NIM endpoint.

Environment variables expected in .env:

    NVIDIA_API_KEY=nvapi-xxxxxxxx
    NVIDIA_MODEL=meta/llama-3.1-8b-instruct
    NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from utils.rag import retrieve_context


# ============================================================================
# ENVIRONMENT
# ============================================================================

load_dotenv()


NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "meta/llama-3.1-8b-instruct",
)

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)

NVIDIA_TEMPERATURE = float(
    os.getenv("NVIDIA_TEMPERATURE", "0.1")
)

NVIDIA_MAX_TOKENS = int(
    os.getenv("NVIDIA_MAX_TOKENS", "2500")
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
# PROMPT
# ============================================================================

SYSTEM_PROMPT = """
You are the Diagnostic Agent inside EduLeap.

EduLeap is an adaptive AI tutoring system for school-age
children, especially learners who may have limited access
to personalized education.

Your responsibility is to create a SHORT academic diagnostic
assessment for a selected learning topic.

The purpose of the diagnostic is:

1. Estimate the learner's current academic level.
2. Identify prerequisite knowledge gaps.
3. Identify concepts that need reinforcement.
4. Provide information to the adaptive tutoring system.

This is NOT a medical, psychological, or educational-disability
diagnostic system.

You must NEVER diagnose:
- dyslexia
- ADHD
- dyscalculia
- intellectual disability
- learning disability
- mental health conditions
- neurological conditions

Only identify observable academic knowledge gaps.

QUESTION GENERATION RULES
-------------------------

1. Generate exactly the requested number of questions.

2. Every question must be related to the requested topic.

3. Every question must be grounded in the supplied
   knowledge-base context.

4. Every question must test exactly ONE concept_tag.

5. Use simple, age-appropriate language.

6. Use multiple-choice questions.

7. Every question must have exactly four options.

8. The correct_answer must exactly match one of the options.

9. Do not reveal the answer inside the question.

10. Avoid trick questions.

11. Avoid ambiguous questions.

12. Use a mixture of difficulty levels:
    easy, medium, hard.

13. Questions should help identify prerequisite gaps.

14. Do not introduce advanced concepts that are absent
    from the supplied knowledge-base context.

15. Do not invent facts that contradict the knowledge base.

16. concept_tag should be a concise machine-readable concept name.

Examples:

    "common_denominators"
    "equivalent_fractions"
    "fraction_addition"
    "place_value"
    "multiplication_facts"

OUTPUT RULES
------------

Return valid JSON only.

Do not use:
- Markdown
- code fences
- explanations outside JSON
- comments

The JSON must follow exactly this structure:

{
    "questions": [
        {
            "id": "q1",
            "question": "...",
            "options": [
                "...",
                "...",
                "...",
                "..."
            ],
            "correct_answer": "...",
            "concept_tag": "...",
            "prerequisite_tag": null,
            "difficulty": "easy"
        }
    ]
}
"""


# ============================================================================
# TYPES
# ============================================================================

VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}


REQUIRED_QUESTION_FIELDS = {
    "id",
    "question",
    "options",
    "correct_answer",
    "concept_tag",
    "prerequisite_tag",
    "difficulty",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _clean_json_text(text: str) -> str:
    """
    Clean common formatting mistakes from an LLM JSON response.

    Handles:
    - Markdown code fences
    - leading/trailing whitespace
    - accidental text before/after JSON
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

    # If the model added text around the JSON, extract the
    # outermost JSON object.
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if (
        first_brace != -1
        and last_brace != -1
        and last_brace > first_brace
    ):
        text = text[first_brace:last_brace + 1]

    return text


def _parse_json_response(text: str) -> Dict[str, Any]:
    """
    Parse an LLM response into a Python dictionary.
    """

    cleaned = _clean_json_text(text)

    try:
        result = json.loads(cleaned)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "NVIDIA NIM returned invalid JSON."
        ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            "Diagnostic response must be a JSON object."
        )

    return result


def _validate_question(
    question: Dict[str, Any],
) -> Optional[str]:
    """
    Validate one generated diagnostic question.

    Returns:
        None if valid.
        Error message if invalid.
    """

    if not isinstance(question, dict):
        return "Question is not a JSON object."

    missing_fields = (
        REQUIRED_QUESTION_FIELDS
        - set(question.keys())
    )

    if missing_fields:
        return (
            "Missing fields: "
            + ", ".join(sorted(missing_fields))
        )

    question_id = question["id"]

    if not isinstance(question_id, str) or not question_id.strip():
        return "Question id must be a non-empty string."

    question_text = question["question"]

    if (
        not isinstance(question_text, str)
        or not question_text.strip()
    ):
        return "Question text must be a non-empty string."

    options = question["options"]

    if not isinstance(options, list):
        return "options must be a list."

    if len(options) != 4:
        return "Every diagnostic question must have exactly four options."

    if any(
        not isinstance(option, str) or not option.strip()
        for option in options
    ):
        return "Every option must be a non-empty string."

    # Detect duplicate options.
    normalized_options = [
        option.strip().casefold()
        for option in options
    ]

    if len(set(normalized_options)) != 4:
        return "Diagnostic question contains duplicate options."

    correct_answer = question["correct_answer"]

    if not isinstance(correct_answer, str):
        return "correct_answer must be a string."

    if correct_answer not in options:
        return (
            "correct_answer must exactly match one of the options."
        )

    concept_tag = question["concept_tag"]

    if (
        not isinstance(concept_tag, str)
        or not concept_tag.strip()
    ):
        return "concept_tag must be a non-empty string."

    prerequisite_tag = question["prerequisite_tag"]

    if prerequisite_tag is not None:

        if (
            not isinstance(prerequisite_tag, str)
            or not prerequisite_tag.strip()
        ):
            return (
                "prerequisite_tag must be null or a non-empty string."
            )

    difficulty = question["difficulty"]

    if difficulty not in VALID_DIFFICULTIES:
        return (
            "difficulty must be one of: "
            "easy, medium, hard."
        )

    return None


def _validate_questions(
    questions: Any,
    expected_count: int,
) -> List[Dict[str, Any]]:
    """
    Validate the complete diagnostic question set.
    """

    if not isinstance(questions, list):
        raise ValueError(
            "NVIDIA NIM response does not contain a valid questions list."
        )

    if len(questions) != expected_count:
        raise ValueError(
            f"Expected {expected_count} questions, "
            f"but received {len(questions)}."
        )

    validated_questions: List[Dict[str, Any]] = []

    ids = set()

    for question in questions:

        error = _validate_question(question)

        if error:
            raise ValueError(
                f"Invalid diagnostic question: {error}"
            )

        question_id = question["id"]

        if question_id in ids:
            raise ValueError(
                f"Duplicate question id detected: {question_id}"
            )

        ids.add(question_id)

        validated_questions.append(
            {
                "id": question["id"].strip(),
                "question": question["question"].strip(),
                "options": [
                    option.strip()
                    for option in question["options"]
                ],
                "correct_answer": question["correct_answer"].strip(),
                "concept_tag": question["concept_tag"].strip(),
                "prerequisite_tag": (
                    question["prerequisite_tag"].strip()
                    if isinstance(
                        question["prerequisite_tag"],
                        str,
                    )
                    else None
                ),
                "difficulty": question["difficulty"].strip().lower(),
            }
        )

    return validated_questions


def _format_context(
    context_chunks: Any,
) -> str:
    """
    Convert RAG results into clean text for the LLM.

    Supports common return formats:

        ["text1", "text2"]

    or:

        [{"text": "text1"}, {"text": "text2"}]

    or:

        [{"content": "text1"}, {"content": "text2"}]
    """

    if not context_chunks:
        return "No relevant knowledge-base context was found."

    formatted_chunks: List[str] = []

    for chunk in context_chunks:

        if isinstance(chunk, str):

            text = chunk.strip()

        elif isinstance(chunk, dict):

            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("page_content")
                or ""
            )

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()

        else:

            text = str(chunk).strip()

        if text:
            formatted_chunks.append(text)

    if not formatted_chunks:
        return "No relevant knowledge-base context was found."

    return "\n\n".join(
        f"[Knowledge Chunk {index}]\n{chunk}"
        for index, chunk in enumerate(
            formatted_chunks,
            start=1,
        )
    )


# ============================================================================
# NVIDIA NIM CALL
# ============================================================================

def _call_nim(
    user_prompt: str,
) -> Dict[str, Any]:
    """
    Send a request directly to NVIDIA NIM.

    No llm_client.py is used.

    NVIDIA exposes the hosted LLM endpoint through an
    OpenAI-compatible /v1/chat/completions API.
    """

    try:

        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=NVIDIA_TEMPERATURE,
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
            "NVIDIA NIM returned no choices."
        )

    message = response.choices[0].message

    content = message.content

    if not content:
        raise RuntimeError(
            "NVIDIA NIM returned an empty message."
        )

    return _parse_json_response(content)


# ============================================================================
# GENERATE DIAGNOSTIC
# ============================================================================

def generate_diagnostic(
    topic: str,
    grade_level: str,
    num_questions: int = 4,
) -> List[Dict[str, Any]]:
    """
    Generate a short diagnostic question set.

    Parameters
    ----------
    topic:
        Learning topic selected by the student.

    grade_level:
        Approximate grade level of the learner.

    num_questions:
        Number of diagnostic questions.

    Returns
    -------
    List[Dict[str, Any]]

    Example internal result:

    [
        {
            "id": "q1",
            "question": "What is 1/2 + 1/4?",
            "options": [
                "1/4",
                "2/6",
                "3/4",
                "4/6"
            ],
            "correct_answer": "3/4",
            "concept_tag": "fraction_addition",
            "prerequisite_tag": "common_denominators",
            "difficulty": "easy"
        }
    ]

    IMPORTANT:
    correct_answer is included because this function is backend
    code. Do NOT send the complete returned objects directly
    to the frontend.
    """

    # ------------------------------------------------------------------------
    # Validate function input
    # ------------------------------------------------------------------------

    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(
            "topic must be a non-empty string."
        )

    if not isinstance(
        grade_level,
        str,
    ) or not grade_level.strip():

        raise ValueError(
            "grade_level must be a non-empty string."
        )

    if not isinstance(num_questions, int):
        raise TypeError(
            "num_questions must be an integer."
        )

    if num_questions < 1:
        raise ValueError(
            "num_questions must be at least 1."
        )

    if num_questions > 10:
        raise ValueError(
            "num_questions cannot exceed 10."
        )

    # ------------------------------------------------------------------------
    # Retrieve educational knowledge
    # ------------------------------------------------------------------------

    try:

        context_chunks = retrieve_context(
            topic,
            top_k=5,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Knowledge-base retrieval failed: {exc}"
        ) from exc

    context_text = _format_context(
        context_chunks
    )

    # ------------------------------------------------------------------------
    # User prompt
    # ------------------------------------------------------------------------

    user_prompt = f"""
Create a short academic diagnostic assessment.

Topic:
{topic}

Learner grade level:
{grade_level}

Number of questions:
{num_questions}

Knowledge-base context:
{context_text}

IMPORTANT:

Every question must be answerable using the knowledge-base
context above.

Every question must contain:

- id
- question
- exactly 4 options
- correct_answer
- concept_tag
- prerequisite_tag
- difficulty

The correct_answer must exactly match one of the four options.

Use concept_tag for the main concept being tested.

Use prerequisite_tag when the question tests a prerequisite
needed for understanding the main topic. Otherwise use null.

Difficulty must be exactly one of:

easy
medium
hard

Return exactly {num_questions} questions.

Return JSON only.
"""

    # ------------------------------------------------------------------------
    # Call NVIDIA NIM
    # ------------------------------------------------------------------------

    result = _call_nim(
        user_prompt
    )

    # ------------------------------------------------------------------------
    # Validate LLM output
    # ------------------------------------------------------------------------

    questions = _validate_questions(
        result.get("questions"),
        expected_count=num_questions,
    )

    return questions


# ============================================================================
# FRONTEND-SAFE VERSION
# ============================================================================

def generate_diagnostic_for_frontend(
    topic: str,
    grade_level: str,
    num_questions: int = 4,
) -> List[Dict[str, Any]]:
    """
    Generate diagnostic questions and remove backend-only fields.

    This is the function your FastAPI endpoint should normally call.

    The frontend receives:

        id
        question
        options
        concept_tag
        prerequisite_tag
        difficulty

    The frontend does NOT receive:

        correct_answer
    """

    questions = generate_diagnostic(
        topic=topic,
        grade_level=grade_level,
        num_questions=num_questions,
    )

    frontend_questions = []

    for question in questions:

        frontend_questions.append(
            {
                "id": question["id"],
                "question": question["question"],
                "options": question["options"],
                "concept_tag": question["concept_tag"],
                "prerequisite_tag": question["prerequisite_tag"],
                "difficulty": question["difficulty"],
            }
        )

    return frontend_questions


# ============================================================================
# DIAGNOSTIC EVALUATION
# ============================================================================

def evaluate_diagnostic(
    questions: List[Dict[str, Any]],
    responses: Dict[str, str],
) -> Dict[str, Any]:
    """
    Score a completed diagnostic.

    Parameters
    ----------
    questions:
        The original backend diagnostic questions generated by
        generate_diagnostic().

    responses:
        Mapping:

            {
                "q1": "3/4",
                "q2": "2/5",
                "q3": "7",
                "q4": "12"
            }

    Returns
    -------
    {
        "score": float,
        "estimated_level": str,
        "identified_gaps": list,
        "prerequisite_gaps": list,
        "per_question": list
    }

    Grading is deterministic.

    The LLM does NOT decide whether an MCQ answer is correct.
    """

    if not isinstance(
        questions,
        list,
    ):
        raise TypeError(
            "questions must be a list."
        )

    if not isinstance(
        responses,
        dict,
    ):
        raise TypeError(
            "responses must be a dictionary."
        )

    if not questions:
        return {
            "score": 0.0,
            "estimated_level": "beginner",
            "identified_gaps": [],
            "prerequisite_gaps": [],
            "per_question": [],
        }

    per_question: List[Dict[str, Any]] = []

    correct_count = 0

    concept_gaps: List[str] = []

    prerequisite_gaps: List[str] = []

    difficulty_scores = {
        "easy": [],
        "medium": [],
        "hard": [],
    }

    # ------------------------------------------------------------------------
    # Grade every question
    # ------------------------------------------------------------------------

    for question in questions:

        question_id = question["id"]

        expected_answer = question["correct_answer"]

        given_answer = responses.get(
            question_id
        )

        # Normalize whitespace only.
        normalized_given = (
            given_answer.strip()
            if isinstance(
                given_answer,
                str,
            )
            else None
        )

        normalized_expected = (
            expected_answer.strip()
        )

        is_correct = (
            normalized_given is not None
            and normalized_given == normalized_expected
        )

        if is_correct:

            correct_count += 1

        else:

            concept_tag = question.get(
                "concept_tag"
            )

            prerequisite_tag = question.get(
                "prerequisite_tag"
            )

            if concept_tag:
                concept_gaps.append(
                    concept_tag
                )

            if prerequisite_tag:
                prerequisite_gaps.append(
                    prerequisite_tag
                )

        difficulty = question.get(
            "difficulty",
            "easy",
        )

        difficulty_scores.setdefault(
            difficulty,
            [],
        )

        difficulty_scores[difficulty].append(
            1.0 if is_correct else 0.0
        )

        per_question.append(
            {
                "id": question_id,
                "correct": is_correct,
                "concept_tag": question.get(
                    "concept_tag"
                ),
                "prerequisite_tag": question.get(
                    "prerequisite_tag"
                ),
                "difficulty": difficulty,
            }
        )

    # ------------------------------------------------------------------------
    # Overall score
    # ------------------------------------------------------------------------

    total_questions = len(questions)

    score = correct_count / total_questions

    # ------------------------------------------------------------------------
    # Estimate level
    #
    # We use the overall score PLUS difficulty performance.
    # This is more reliable than simply:
    #
    #     >= 75% = advanced
    #
    # because a learner may get easy questions correct but
    # struggle with harder questions.
    # ------------------------------------------------------------------------

    easy_score = _average(
        difficulty_scores.get("easy", [])
    )

    medium_score = _average(
        difficulty_scores.get("medium", [])
    )

    hard_score = _average(
        difficulty_scores.get("hard", [])
    )

    estimated_level = _estimate_level(
        overall_score=score,
        easy_score=easy_score,
        medium_score=medium_score,
        hard_score=hard_score,
    )

    # ------------------------------------------------------------------------
    # Remove duplicate gaps while preserving order
    # ------------------------------------------------------------------------

    unique_concept_gaps = list(
        dict.fromkeys(
            concept_gaps
        )
    )

    unique_prerequisite_gaps = list(
        dict.fromkeys(
            prerequisite_gaps
        )
    )

    # ------------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------------

    confidence = _calculate_confidence(
        total_questions=total_questions,
        score=score,
        easy_score=easy_score,
        medium_score=medium_score,
        hard_score=hard_score,
    )

    # ------------------------------------------------------------------------
    # Recommended starting concept
    # ------------------------------------------------------------------------

    if unique_prerequisite_gaps:

        recommended_starting_concept = (
            unique_prerequisite_gaps[0]
        )

    elif unique_concept_gaps:

        recommended_starting_concept = (
            unique_concept_gaps[0]
        )

    else:

        recommended_starting_concept = (
            "continue_with_selected_topic"
        )

    # ------------------------------------------------------------------------
    # Return analysis
    # ------------------------------------------------------------------------

    return {
        "score": round(
            score,
            3,
        ),
        "confidence": round(
            confidence,
            3,
        ),
        "estimated_level": estimated_level,
        "identified_gaps": unique_concept_gaps,
        "prerequisite_gaps": unique_prerequisite_gaps,
        "recommended_starting_concept": (
            recommended_starting_concept
        ),
        "per_question": per_question,
    }


# ============================================================================
# LEVEL ESTIMATION
# ============================================================================

def _estimate_level(
    overall_score: float,
    easy_score: Optional[float],
    medium_score: Optional[float],
    hard_score: Optional[float],
) -> str:
    """
    Estimate academic level from diagnostic performance.

    Rules are intentionally conservative.

    Beginner:
        weak overall performance or weak easy-question performance.

    Intermediate:
        reasonable basic understanding but incomplete
        performance on medium/hard questions.

    Advanced:
        strong performance including medium/hard questions.
    """

    # Very weak performance.
    if overall_score < 0.40:

        return "beginner"

    # If easy questions exist and the learner is weak on them,
    # they should not be promoted to intermediate.
    if (
        easy_score is not None
        and easy_score < 0.50
    ):

        return "beginner"

    # Strong performance across available difficulties.
    if overall_score >= 0.80:

        if (
            hard_score is not None
            and hard_score >= 0.50
        ):
            return "advanced"

        if (
            medium_score is not None
            and medium_score >= 0.75
        ):
            return "advanced"

    # Moderate/good performance.
    if overall_score >= 0.50:

        return "intermediate"

    return "beginner"


# ============================================================================
# AVERAGE
# ============================================================================

def _average(
    values: List[float],
) -> Optional[float]:
    """
    Calculate average of a list.

    Returns None for an empty list.
    """

    if not values:
        return None

    return sum(values) / len(values)


# ============================================================================
# CONFIDENCE
# ============================================================================

def _calculate_confidence(
    total_questions: int,
    score: float,
    easy_score: Optional[float],
    medium_score: Optional[float],
    hard_score: Optional[float],
) -> float:
    """
    Estimate confidence in the diagnostic level.

    This is NOT statistical psychometric confidence.

    It is an application-level confidence value based on:
    - number of answered questions
    - consistency across difficulty levels
    """

    # More questions give more evidence.
    question_factor = min(
        total_questions / 8.0,
        1.0,
    )

    available_scores = [
        value
        for value in [
            easy_score,
            medium_score,
            hard_score,
        ]
        if value is not None
    ]

    if len(available_scores) <= 1:

        consistency_factor = 0.70

    else:

        maximum = max(
            available_scores
        )

        minimum = min(
            available_scores
        )

        spread = maximum - minimum

        consistency_factor = max(
            0.40,
            1.0 - spread,
        )

    score_strength = (
        0.5
        + abs(score - 0.5)
    )

    confidence = (
        0.45 * question_factor
        + 0.35 * consistency_factor
        + 0.20 * score_strength
    )

    return max(
        0.0,
        min(
            confidence,
            1.0,
        ),
    )


# ============================================================================
# COMPLETE DIAGNOSTIC FLOW
# ============================================================================

def run_diagnostic(
    topic: str,
    grade_level: str,
    responses: Optional[Dict[str, str]] = None,
    num_questions: int = 4,
) -> Dict[str, Any]:
    """
    Complete diagnostic workflow.

    First call:

        run_diagnostic(
            topic="Adding Fractions",
            grade_level="Grade 5"
        )

    returns generated questions.

    Second call:

        run_diagnostic(
            topic="Adding Fractions",
            grade_level="Grade 5",
            responses={
                "q1": "3/4",
                "q2": "2/5",
                "q3": "1/2",
                "q4": "4/7"
            }
        )

    returns the evaluation.

    NOTE:
    In a real FastAPI application, you should persist the generated
    questions in the session/database between these two calls rather
    than regenerate them.
    """

    questions = generate_diagnostic(
        topic=topic,
        grade_level=grade_level,
        num_questions=num_questions,
    )

    # ------------------------------------------------------------------------
    # First stage: return questions
    # ------------------------------------------------------------------------

    if responses is None:

        return {
            "status": "assessment_ready",
            "topic": topic,
            "grade_level": grade_level,
            "questions": [
                {
                    "id": q["id"],
                    "question": q["question"],
                    "options": q["options"],
                    "concept_tag": q["concept_tag"],
                    "prerequisite_tag": q[
                        "prerequisite_tag"
                    ],
                    "difficulty": q["difficulty"],
                }
                for q in questions
            ],
        }

    # ------------------------------------------------------------------------
    # Second stage: evaluate
    # ------------------------------------------------------------------------

    evaluation = evaluate_diagnostic(
        questions=questions,
        responses=responses,
    )

    return {
        "status": "diagnostic_completed",
        "topic": topic,
        "grade_level": grade_level,
        "evaluation": evaluation,
    }


# ============================================================================
# LOCAL TEST
# ============================================================================

if __name__ == "__main__":

    """
    Simple local test.

    Run from project root:

        python -m utils.agents.diagnostic

    Make sure .env contains:

        NVIDIA_API_KEY=...
        NVIDIA_MODEL=meta/llama-3.1-8b-instruct

    and that utils/rag.py is working.
    """

    print("=" * 70)
    print("EduLeap Diagnostic Agent")
    print("=" * 70)

    print(
        f"Model: {NVIDIA_MODEL}"
    )

    print(
        f"NVIDIA endpoint: {NVIDIA_BASE_URL}"
    )

    print()

    try:

        result = run_diagnostic(
            topic="Adding fractions",
            grade_level="Grade 5",
            num_questions=4,
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as exc:

        print()
        print(
            "Diagnostic Agent failed:"
        )
        print(exc)

# ============================================================================
# ORCHESTRATOR COMPATIBILITY WRAPPER
# ============================================================================

class DiagnosticAgent:
    """
    Compatibility wrapper for AIOrchestrator.

    Keeps the existing functional diagnostic implementation intact while
    exposing the class-based interface expected by orchestrator.py.
    """

    def __init__(self):
        pass

    def generate_diagnostic(
        self,
        topic: str,
        grade_level: str = "General",
        num_questions: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Generate diagnostic questions.

        Returns backend questions including correct_answer because the
        orchestrator needs them for evaluation.
        """
        return generate_diagnostic(
            topic=topic,
            grade_level=grade_level,
            num_questions=num_questions,
        )

    def estimate_level(
        self,
        topic: str,
        responses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Convert the orchestrator's response format into the existing
        deterministic diagnostic evaluator format.
        """

        # The orchestrator currently sends:
        #
        # [
        #   {
        #       "question": "...",
        #       "targets": "...",
        #       "correct": True
        #   }
        # ]
        #
        # The diagnostic evaluator expects:
        #
        # {
        #     "q1": "answer",
        #     "q2": "answer"
        # }
        #
        # Since the existing API only provides a correctness signal,
        # construct a compatible evaluation directly.

        if not responses:
            return {
                "level": 1.0,
                "gaps": [],
            }

        correct_count = sum(
            1
            for response in responses
            if response.get("correct") is True
        )

        score = correct_count / len(responses)

        # Simple academic level mapping.
        if score >= 0.85:
            level = 5.0
        elif score >= 0.70:
            level = 4.0
        elif score >= 0.50:
            level = 3.0
        elif score >= 0.30:
            level = 2.0
        else:
            level = 1.0

        gaps = []

        for response in responses:
            if not response.get("correct"):
                target = response.get("targets")

                if target:
                    if isinstance(target, list):
                        gaps.extend(target)
                    else:
                        gaps.append(str(target))

        # Remove duplicates while preserving order.
        gaps = list(dict.fromkeys(gaps))

        return {
            "level": level,
            "gaps": gaps,
            "score": score,
        }
"""
EduLeap - Assessment Agent
==========================

The Assessment Agent generates targeted practice questions and evaluates
learner responses.

Responsibilities
----------------
1. Retrieve educational context from the knowledge base.
2. Generate one targeted practice question using NVIDIA NIM.
3. Ground the question in the retrieved educational context.
4. Target a specific concept and difficulty.
5. Keep the correct answer on the backend.
6. Deterministically grade MCQ answers.
7. Use NVIDIA NIM to generate learner-friendly feedback.
8. Identify a possible misconception when an answer is wrong.
9. Produce the learning signal used by the Learning Analyst.

Pipeline stage
--------------

    Tutor Agent
         |
         v
    Assessment Agent
         |
         +--> Generate Question
         |
         v
      Student
         |
         v
    Evaluate Response
         |
         v
    Learning Analyst
         |
         v
    Learner Profile Update


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

This file directly communicates with NVIDIA NIM using its
OpenAI-compatible API.

Required .env:

    NVIDIA_API_KEY=nvapi-your-key
    NVIDIA_MODEL=meta/llama-3.1-8b-instruct

Optional:

    NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
    NVIDIA_TEMPERATURE=0.2
    NVIDIA_MAX_TOKENS=1800


Important
---------

This agent is an academic assessment component.

It does NOT diagnose:
- learning disabilities
- medical conditions
- psychological conditions
- neurological conditions
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
    os.getenv(
        "NVIDIA_TEMPERATURE",
        "0.2",
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
# SYSTEM PROMPTS
# ============================================================================

GEN_SYSTEM_PROMPT = """
You are the Assessment Agent inside EduLeap.

EduLeap is an adaptive AI tutoring platform for school-age
children.

Your task is to generate ONE targeted practice question.

The question must:
- target exactly one concept
- match the requested difficulty
- be appropriate for the learner's level
- be grounded in the supplied knowledge-base context
- test understanding rather than memorization when possible
- be clear and unambiguous
- use exactly four multiple-choice options
- have exactly one correct answer

The question must not introduce concepts that are unsupported
by the supplied knowledge-base context.

Do not create trick questions.

Do not use unnecessarily difficult language.

Do not diagnose learning disabilities or medical conditions.

OUTPUT RULES
------------

Return valid JSON only.

Do not use:
- markdown
- code fences
- explanations outside JSON

Return exactly this structure:

{
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
"""


GRADE_SYSTEM_PROMPT = """
You are the Assessment Agent inside EduLeap.

You are analyzing a learner's response to an academic
practice question.

The Python application has already determined whether the
MCQ answer is correct.

Your job is to provide:

1. Short encouraging feedback.
2. A concise explanation of the correct idea.
3. A possible misconception ONLY when the answer is incorrect.

Be strict about conceptual correctness.

Be tolerant of:
- spelling mistakes
- minor grammar mistakes
- formatting differences

Do not shame the learner.

Do not mention:
- IQ
- intelligence
- disability
- medical conditions
- psychological conditions

Never diagnose a learning disability.

If the answer is correct:
- congratulate the learner briefly
- reinforce the concept
- misconception must be null

If the answer is incorrect:
- explain the correct idea simply
- identify a concise misconception label
- do not invent a misconception without evidence

Return valid JSON only.

Use exactly this structure:

{
    "feedback": "...",
    "misconception": null
}
"""


# ============================================================================
# CONSTANTS
# ============================================================================

VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}

REQUIRED_QUESTION_FIELDS = {
    "question",
    "options",
    "correct_answer",
    "concept_tag",
    "prerequisite_tag",
    "difficulty",
}


# ============================================================================
# JSON HELPERS
# ============================================================================

def _clean_json_text(
    text: str,
) -> str:
    """
    Remove common formatting problems from an NVIDIA NIM
    JSON response.
    """

    if not text:
        raise ValueError(
            "NVIDIA NIM returned an empty response."
        )

    text = text.strip()

    # Remove markdown JSON fences if the model adds them.
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

    # Extract JSON object if extra text was returned.
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
    Parse an NVIDIA NIM response into a Python dictionary.
    """

    cleaned = _clean_json_text(text)

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
            "NVIDIA NIM response must be a JSON object."
        )

    return result


# ============================================================================
# RAG CONTEXT
# ============================================================================

def _format_context(
    context_chunks: Any,
) -> str:
    """
    Convert utils.rag.retrieve_context() results into text.

    Supports:

        ["text", "text"]

    and:

        [{"text": "..."}, {"text": "..."}]

    and:

        [{"content": "..."}]

    and LangChain-style objects with page_content.
    """

    if not context_chunks:

        return (
            "No relevant knowledge-base context was found."
        )

    formatted_chunks: List[str] = []

    for chunk in context_chunks:

        if isinstance(
            chunk,
            str,
        ):

            text = chunk.strip()

        elif isinstance(
            chunk,
            dict,
        ):

            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("page_content")
                or ""
            )

            if not isinstance(
                text,
                str,
            ):
                text = str(text)

            text = text.strip()

        else:

            if hasattr(
                chunk,
                "page_content",
            ):

                text = str(
                    chunk.page_content
                ).strip()

            elif hasattr(
                chunk,
                "content",
            ):

                text = str(
                    chunk.content
                ).strip()

            else:

                text = str(
                    chunk
                ).strip()

        if text:

            formatted_chunks.append(
                text
            )

    if not formatted_chunks:

        return (
            "No relevant knowledge-base context was found."
        )

    return "\n\n".join(
        f"[Knowledge Chunk {index}]\n{chunk}"
        for index, chunk in enumerate(
            formatted_chunks,
            start=1,
        )
    )


def _retrieve_context(
    concept: str,
    top_k: int = 4,
) -> str:
    """
    Retrieve educational knowledge for a concept.
    """

    try:

        context_chunks = retrieve_context(
            concept,
            top_k=top_k,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Knowledge-base retrieval failed for "
            f"concept '{concept}': {exc}"
        ) from exc

    return _format_context(
        context_chunks
    )


# ============================================================================
# NVIDIA NIM
# ============================================================================

def _call_nim(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
) -> str:
    """
    Call NVIDIA NIM directly.

    No utils.llm_client is used.
    """

    try:

        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=messages,
            temperature=(
                NVIDIA_TEMPERATURE
                if temperature is None
                else temperature
            ),
            max_tokens=NVIDIA_MAX_TOKENS,
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

    return content.strip()


# ============================================================================
# QUESTION VALIDATION
# ============================================================================

def _validate_question(
    question: Dict[str, Any],
    expected_concept: str,
    expected_difficulty: str,
) -> Dict[str, Any]:
    """
    Validate a generated practice question.

    The LLM output is never trusted blindly.
    """

    if not isinstance(
        question,
        dict,
    ):

        raise ValueError(
            "Generated practice question must be a JSON object."
        )

    missing_fields = (
        REQUIRED_QUESTION_FIELDS
        - set(question.keys())
    )

    if missing_fields:

        raise ValueError(
            "Practice question is missing fields: "
            + ", ".join(
                sorted(
                    missing_fields
                )
            )
        )

    # ------------------------------------------------------------------------
    # Question
    # ------------------------------------------------------------------------

    question_text = question["question"]

    if (
        not isinstance(
            question_text,
            str,
        )
        or not question_text.strip()
    ):

        raise ValueError(
            "Practice question text cannot be empty."
        )

    # ------------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------------

    options = question["options"]

    if not isinstance(
        options,
        list,
    ):

        raise ValueError(
            "Practice question options must be a list."
        )

    if len(options) != 4:

        raise ValueError(
            "Practice question must have exactly four options."
        )

    if any(
        not isinstance(
            option,
            str,
        )
        or not option.strip()
        for option in options
    ):

        raise ValueError(
            "Every practice question option must be a non-empty string."
        )

    normalized_options = [
        option.strip().casefold()
        for option in options
    ]

    if len(
        set(normalized_options)
    ) != 4:

        raise ValueError(
            "Practice question contains duplicate options."
        )

    # ------------------------------------------------------------------------
    # Correct answer
    # ------------------------------------------------------------------------

    correct_answer = question[
        "correct_answer"
    ]

    if not isinstance(
        correct_answer,
        str,
    ):

        raise ValueError(
            "correct_answer must be a string."
        )

    if correct_answer not in options:

        raise ValueError(
            "correct_answer must exactly match one of the options."
        )

    # ------------------------------------------------------------------------
    # Concept
    # ------------------------------------------------------------------------

    concept_tag = question[
        "concept_tag"
    ]

    if (
        not isinstance(
            concept_tag,
            str,
        )
        or not concept_tag.strip()
    ):

        raise ValueError(
            "concept_tag must be a non-empty string."
        )

    # ------------------------------------------------------------------------
    # Difficulty
    # ------------------------------------------------------------------------

    difficulty = question[
        "difficulty"
    ]

    if difficulty not in VALID_DIFFICULTIES:

        raise ValueError(
            "difficulty must be one of: "
            "easy, medium, hard."
        )

    # The generated question should match the difficulty
    # requested by the orchestrator.
    if difficulty != expected_difficulty:

        raise ValueError(
            f"Expected difficulty '{expected_difficulty}', "
            f"but model returned '{difficulty}'."
        )

    # ------------------------------------------------------------------------
    # Normalize result
    # ------------------------------------------------------------------------

    prerequisite_tag = question[
        "prerequisite_tag"
    ]

    if prerequisite_tag is not None:

        if (
            not isinstance(
                prerequisite_tag,
                str,
            )
            or not prerequisite_tag.strip()
        ):

            raise ValueError(
                "prerequisite_tag must be null or a non-empty string."
            )

        prerequisite_tag = (
            prerequisite_tag.strip()
        )

    return {
        "question": question_text.strip(),
        "options": [
            option.strip()
            for option in options
        ],
        "correct_answer": correct_answer.strip(),
        "concept_tag": concept_tag.strip(),
        "prerequisite_tag": prerequisite_tag,
        "difficulty": difficulty,
    }


# ============================================================================
# GENERATE PRACTICE QUESTION
# ============================================================================

def generate_practice_question(
    concept: str,
    difficulty: str,
) -> Dict[str, Any]:
    """
    Generate ONE targeted practice question.

    Parameters
    ----------
    concept:
        Concept that should be assessed.

    difficulty:
        One of:

            easy
            medium
            hard

    Returns
    -------
    Backend question:

        {
            "question": str,
            "options": list[str],
            "correct_answer": str,
            "concept_tag": str,
            "prerequisite_tag": str | None,
            "difficulty": str
        }

    IMPORTANT
    ---------
    The returned object contains correct_answer because it is
    backend data.

    Do NOT send this entire dictionary directly to the frontend.
    Use generate_practice_question_for_frontend().
    """

    # ------------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------------

    if not isinstance(
        concept,
        str,
    ) or not concept.strip():

        raise ValueError(
            "concept must be a non-empty string."
        )

    if not isinstance(
        difficulty,
        str,
    ):

        raise TypeError(
            "difficulty must be a string."
        )

    difficulty = difficulty.strip().lower()

    if difficulty not in VALID_DIFFICULTIES:

        raise ValueError(
            "difficulty must be one of: "
            "easy, medium, hard."
        )

    concept = concept.strip()

    # ------------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------------

    context_text = _retrieve_context(
        concept,
        top_k=4,
    )

    # ------------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------------

    user_prompt = f"""
Generate ONE practice question.

Concept:
{concept}

Difficulty:
{difficulty}

Knowledge-base context:
{context_text}

Requirements:

1. The question must assess "{concept}".

2. The difficulty must be exactly "{difficulty}".

3. Use exactly four multiple-choice options.

4. There must be exactly one correct answer.

5. correct_answer must exactly match one option.

6. The question must be answerable using the
   knowledge-base context.

7. Use a clear, age-appropriate question.

8. Do not introduce unrelated concepts.

9. concept_tag should describe the concept being assessed.

10. prerequisite_tag should contain the most important
    prerequisite if one is relevant, otherwise null.

Return JSON only:

{{
    "question": "...",
    "options": [
        "...",
        "...",
        "...",
        "..."
    ],
    "correct_answer": "...",
    "concept_tag": "{concept}",
    "prerequisite_tag": null,
    "difficulty": "{difficulty}"
}}
"""

    # ------------------------------------------------------------------------
    # NIM
    # ------------------------------------------------------------------------

    raw_response = _call_nim(
        messages=[
            {
                "role": "system",
                "content": GEN_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    result = _parse_json_response(
        raw_response
    )

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    return _validate_question(
        question=result,
        expected_concept=concept,
        expected_difficulty=difficulty,
    )


# ============================================================================
# FRONTEND-SAFE QUESTION
# ============================================================================

def generate_practice_question_for_frontend(
    concept: str,
    difficulty: str,
) -> Dict[str, Any]:
    """
    Generate a practice question and remove the correct answer
    before sending it to the frontend.

    Frontend receives:

        question
        options
        concept_tag
        prerequisite_tag
        difficulty

    Frontend does NOT receive:

        correct_answer
    """

    question = generate_practice_question(
        concept=concept,
        difficulty=difficulty,
    )

    return {
        "question": question["question"],
        "options": question["options"],
        "concept_tag": question["concept_tag"],
        "prerequisite_tag": question[
            "prerequisite_tag"
        ],
        "difficulty": question["difficulty"],
    }


# ============================================================================
# DETERMINISTIC ANSWER CHECK
# ============================================================================

def _is_answer_correct(
    correct_answer: str,
    student_answer: str,
) -> bool:
    """
    Determine whether an MCQ response is correct.

    Since this is an MCQ, Python performs the actual correctness
    check instead of asking the LLM.

    This prevents the LLM from incorrectly marking an answer
    as correct or incorrect.
    """

    if not isinstance(
        student_answer,
        str,
    ):

        return False

    return (
        student_answer.strip().casefold()
        == correct_answer.strip().casefold()
    )


# ============================================================================
# EVALUATE RESPONSE
# ============================================================================

def evaluate_response(
    question: Dict[str, Any],
    student_answer: str,
) -> Dict[str, Any]:
    """
    Evaluate a learner's answer.

    Parameters
    ----------
    question:
        Backend question generated by generate_practice_question().

    student_answer:
        The answer selected by the learner.

    Returns
    -------
    {
        "correct": bool,
        "feedback": str,
        "misconception": str | None,
        "concept_tag": str,
        "difficulty": str
    }

    The correctness decision is deterministic.

    NVIDIA NIM is used only for:
    - feedback
    - misconception analysis
    """

    # ------------------------------------------------------------------------
    # Validate question
    # ------------------------------------------------------------------------

    if not isinstance(
        question,
        dict,
    ):

        raise TypeError(
            "question must be a dictionary."
        )

    required_fields = {
        "question",
        "options",
        "correct_answer",
        "concept_tag",
        "difficulty",
    }

    missing_fields = (
        required_fields
        - set(question.keys())
    )

    if missing_fields:

        raise ValueError(
            "Question is missing fields: "
            + ", ".join(
                sorted(
                    missing_fields
                )
            )
        )

    if not isinstance(
        student_answer,
        str,
    ):

        raise TypeError(
            "student_answer must be a string."
        )

    student_answer = student_answer.strip()

    if not student_answer:

        raise ValueError(
            "student_answer cannot be empty."
        )

    # ------------------------------------------------------------------------
    # Deterministic correctness
    # ------------------------------------------------------------------------

    correct = _is_answer_correct(
        correct_answer=question[
            "correct_answer"
        ],
        student_answer=student_answer,
    )

    # ------------------------------------------------------------------------
    # If correct, don't waste an LLM call to decide correctness.
    # We can still ask NIM for personalized reinforcement.
    # ------------------------------------------------------------------------

    if correct:

        user_prompt = f"""
The learner answered this practice question correctly.

Concept:
{question["concept_tag"]}

Question:
{question["question"]}

Correct answer:
{question["correct_answer"]}

Learner answer:
{student_answer}

Give short, encouraging feedback.

Requirements:

1. Confirm the learner's understanding.
2. Reinforce the key concept.
3. Keep it age-appropriate.
4. Do not over-explain.
5. misconception must be null.

Return JSON only:

{{
    "feedback": "...",
    "misconception": null
}}
"""

    else:

        user_prompt = f"""
The learner answered this practice question incorrectly.

Concept:
{question["concept_tag"]}

Question:
{question["question"]}

Options:
{json.dumps(question["options"], ensure_ascii=False)}

Correct answer:
{question["correct_answer"]}

Learner answer:
{student_answer}

Analyze the learner's response.

Requirements:

1. Give short, encouraging feedback.

2. Explain the correct concept simply.

3. Do not shame the learner.

4. Identify ONE possible misconception if the
   learner's answer gives enough evidence.

5. The misconception must be a short educational label.

Good example:
"adds numerators and denominators directly"

Bad example:
"has dyscalculia"

6. Never diagnose a learning disability.

7. If there is insufficient evidence for a misconception,
   set misconception to null.

Return JSON only:

{{
    "feedback": "...",
    "misconception": null
}}
"""

    # ------------------------------------------------------------------------
    # NVIDIA NIM feedback
    # ------------------------------------------------------------------------

    raw_response = _call_nim(
        messages=[
            {
                "role": "system",
                "content": GRADE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    result = _parse_json_response(
        raw_response
    )

    # ------------------------------------------------------------------------
    # Validate feedback
    # ------------------------------------------------------------------------

    feedback = result.get(
        "feedback"
    )

    if (
        not isinstance(
            feedback,
            str,
        )
        or not feedback.strip()
    ):

        raise ValueError(
            "Assessment feedback returned by NVIDIA NIM is invalid."
        )

    misconception = result.get(
        "misconception"
    )

    if misconception is not None:

        if not isinstance(
            misconception,
            str,
        ):

            misconception = None

        else:

            misconception = (
                misconception.strip()
                or None
            )

    # Correct answers must never have a misconception.
    if correct:

        misconception = None

    # ------------------------------------------------------------------------
    # Final assessment result
    # ------------------------------------------------------------------------

    return {
        "correct": correct,
        "feedback": feedback.strip(),
        "misconception": misconception,
        "concept_tag": question[
            "concept_tag"
        ],
        "prerequisite_tag": question.get(
            "prerequisite_tag"
        ),
        "difficulty": question[
            "difficulty"
        ],
        "correct_answer": question[
            "correct_answer"
        ],
    }


# ============================================================================
# FRONTEND-SAFE EVALUATION
# ============================================================================

def evaluate_response_for_frontend(
    question: Dict[str, Any],
    student_answer: str,
) -> Dict[str, Any]:
    """
    Evaluate an answer and remove backend-only information.

    The frontend does not need to receive the correct answer
    separately because the learner already submitted the response.
    """

    result = evaluate_response(
        question=question,
        student_answer=student_answer,
    )

    return {
        "correct": result["correct"],
        "feedback": result["feedback"],
        "misconception": result["misconception"],
        "concept_tag": result["concept_tag"],
        "prerequisite_tag": result[
            "prerequisite_tag"
        ],
        "difficulty": result["difficulty"],
    }


# ============================================================================
# COMPLETE ASSESSMENT FLOW
# ============================================================================

def run_assessment(
    concept: str,
    difficulty: str,
    student_answer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the assessment stage.

    First call:

        run_assessment(
            concept="fraction_addition",
            difficulty="easy"
        )

    returns a question for the frontend.

    Second stage should evaluate the SAME stored backend
    question using evaluate_response().

    IMPORTANT:
    The question should be persisted by the orchestrator/database
    between generation and evaluation.

    Do not generate a new question when evaluating a response.
    """

    question = generate_practice_question(
        concept=concept,
        difficulty=difficulty,
    )

    # ------------------------------------------------------------------------
    # Question generation stage
    # ------------------------------------------------------------------------

    if student_answer is None:

        return {
            "status": "question_ready",
            "question": {
                "question": question[
                    "question"
                ],
                "options": question[
                    "options"
                ],
                "concept_tag": question[
                    "concept_tag"
                ],
                "prerequisite_tag": question[
                    "prerequisite_tag"
                ],
                "difficulty": question[
                    "difficulty"
                ],
            },
        }

    # ------------------------------------------------------------------------
    # Evaluation stage
    # ------------------------------------------------------------------------

    evaluation = evaluate_response(
        question=question,
        student_answer=student_answer,
    )

    return {
        "status": "assessment_completed",
        "evaluation": {
            "correct": evaluation[
                "correct"
            ],
            "feedback": evaluation[
                "feedback"
            ],
            "misconception": evaluation[
                "misconception"
            ],
            "concept_tag": evaluation[
                "concept_tag"
            ],
            "prerequisite_tag": evaluation[
                "prerequisite_tag"
            ],
            "difficulty": evaluation[
                "difficulty"
            ],
        },
    }


# ============================================================================
# HEALTH CHECK
# ============================================================================

def assessment_health_check() -> Dict[str, Any]:
    """
    Return configuration information without making an
    NVIDIA request.
    """

    return {
        "agent": "assessment",
        "status": "configured",
        "model": NVIDIA_MODEL,
        "base_url": NVIDIA_BASE_URL,
        "rag_enabled": True,
        "deterministic_mcq_grading": True,
    }


# ============================================================================
# LOCAL TEST
# ============================================================================

if __name__ == "__main__":

    """
    Run from project root:

        python -m utils.agents.assessment

    Make sure .env contains:

        NVIDIA_API_KEY=...
        NVIDIA_MODEL=meta/llama-3.1-8b-instruct

    and utils/rag.py is working.
    """

    print("=" * 70)
    print("EduLeap Assessment Agent")
    print("=" * 70)

    print(
        f"Model: {NVIDIA_MODEL}"
    )

    print(
        f"NVIDIA endpoint: {NVIDIA_BASE_URL}"
    )

    print()

    try:

        question = generate_practice_question(
            concept="Adding fractions",
            difficulty="easy",
        )

        print(
            "Generated question:"
        )

        print(
            json.dumps(
                question,
                indent=2,
                ensure_ascii=False,
            )
        )

        print()
        print(
            "Testing evaluation..."
        )

        evaluation = evaluate_response(
            question=question,
            student_answer=question[
                "correct_answer"
            ],
        )

        print(
            json.dumps(
                evaluation,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as exc:

        print(
            "Assessment Agent failed:"
        )

        print(
            exc
        )

# ============================================================================
# ORCHESTRATOR COMPATIBILITY WRAPPER
# ============================================================================

class AssessmentAgent:
    """
    Compatibility wrapper for AIOrchestrator.

    Keeps the existing assessment implementation intact while exposing
    the class-based interface expected by orchestrator.py.
    """

    def __init__(self):
        pass

    def generate_question(
        self,
        topic: str,
        level: str,
        target_gap: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a practice question using the existing assessment agent.
        """

        difficulty_map = {
            "beginner": "easy",
            "basic": "easy",
            "easy": "easy",
            "intermediate": "medium",
            "medium": "medium",
            "advanced": "hard",
            "hard": "hard",
        }

        difficulty = difficulty_map.get(
            str(level).lower(),
            "medium"
        )

        concept = target_gap if target_gap else topic

        question = generate_practice_question(
            concept=concept,
            difficulty=difficulty,
        )

        return question
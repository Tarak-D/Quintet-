"""
utils/misconception.py

Detects *why* an answer was wrong, not just that it was wrong. Combines a
small rule-based pattern table (fast, deterministic, easy to test/demo) with
an LLM fallback for anything the rules don't cover.

Output is always a "signal", per proposal 2.1 / 5 - never framed as a
diagnosis of a learning disability.
"""

from typing import Optional

from utils.llm_client import call_llm_json

# A few illustrative rule-based patterns for the MVP subject (fractions).
# Each entry: (topic, pattern-check) -> concept tag.
RULES = {
    "fractions": [
        {
            "concept": "common_denominators",
            "check": lambda q, a: "denominator" in q.lower() and _looks_like_added_numerators_only(q, a),
        },
    ],
}


def _looks_like_added_numerators_only(question: str, answer: str) -> bool:
    """
    Very small heuristic used only as a hackathon-demo shortcut: flags the
    classic 1/2 + 1/3 -> 2/5 style mistake pattern. Real deployment would
    replace this with a proper symbolic answer-checker.
    """
    return False  # placeholder hook - wire up a real answer parser here


def detect_misconception(
    topic: str,
    question: str,
    student_answer: str,
    correct_answer: str,
) -> dict:
    """
    Returns: {"is_correct": bool, "concept": str | None, "explanation": str}
    """
    is_correct = student_answer.strip().lower() == correct_answer.strip().lower()
    if is_correct:
        return {"is_correct": True, "concept": None, "explanation": ""}

    for rule in RULES.get(topic, []):
        if rule["check"](question, student_answer):
            return {
                "is_correct": False,
                "concept": rule["concept"],
                "explanation": f"Matched rule-based pattern for {rule['concept']}.",
            }

    # Fallback: ask the LLM to classify the likely misconception.
    system = (
        "You are an educational diagnostics assistant. Given a question, a "
        "student's wrong answer, and the correct answer, identify the most "
        "likely underlying misconception in a few words (e.g. "
        "'common_denominators', 'sign_error', 'place_value'). "
        "Respond ONLY as JSON: {\"concept\": string, \"explanation\": string}. "
        "Do not speculate about the student's abilities or diagnose any "
        "condition - describe only the mathematical/conceptual error."
    )
    user = (
        f"Topic: {topic}\n"
        f"Question: {question}\n"
        f"Student answer: {student_answer}\n"
        f"Correct answer: {correct_answer}"
    )
    result = call_llm_json(system, user, max_tokens=200)

    return {
        "is_correct": False,
        "concept": result.get("concept", "unknown"),
        "explanation": result.get("explanation", ""),
    }

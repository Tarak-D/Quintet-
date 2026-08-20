"""
Assessment Agent
----------------
Generates targeted practice questions for a concept/difficulty and grades
student responses, producing the signal the Learning Analyst uses to
detect misconceptions and update the learner profile.

Pipeline stage: Agent Execution -> [Assessment Agent] -> Verification
"""

from __future__ import annotations
from typing import Dict, Any

from utils.llm_client import chat_json
from utils.rag import retrieve_context


GEN_SYSTEM_PROMPT = """You are the Assessment Agent inside EduLeap. You \
write ONE practice question at a time, targeted at a specific concept and \
difficulty, grounded in the given knowledge base context. Return STRICT \
JSON only, no markdown fences."""

GRADE_SYSTEM_PROMPT = """You are the Assessment Agent inside EduLeap, now \
grading a learner's answer. Be lenient with phrasing/typos but strict on \
conceptual correctness. If the answer is wrong, identify the specific \
misconception in a short label (e.g. "adds numerators and denominators \
directly"). Return STRICT JSON only."""


def generate_practice_question(concept: str, difficulty: str) -> Dict[str, Any]:
    """Create one practice question targeted at `concept`/`difficulty`.

    Returns:
        {
          "question": str,
          "options": List[str],       # MCQ options for fast grading
          "correct_answer": str,
          "concept_tag": str,
          "difficulty": str
        }
    """
    context_chunks = retrieve_context(concept, top_k=4)
    context_text = "\n".join(f"- {c}" for c in context_chunks) or "No KB context found."

    user_prompt = f"""
Concept: {concept}
Difficulty: {difficulty}

Knowledge base context:
{context_text}

Return JSON exactly in this shape:
{{
  "question": "...",
  "options": ["...", "...", "...", "..."],
  "correct_answer": "...",
  "concept_tag": "{concept}",
  "difficulty": "{difficulty}"
}}
"""
    return chat_json(
        [
            {"role": "system", "content": GEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )


def evaluate_response(question: Dict[str, Any], student_answer: str) -> Dict[str, Any]:
    """Grade a student's answer to `question`.

    Returns:
        {
          "correct": bool,
          "feedback": str,              # short, encouraging feedback
          "misconception": str | None   # short label if the answer is wrong
        }
    """
    user_prompt = f"""
Question: {question["question"]}
Options: {question.get("options")}
Correct answer: {question["correct_answer"]}
Student's answer: {student_answer}

Return JSON exactly:
{{
  "correct": true,
  "feedback": "...",
  "misconception": null
}}
(set correct to false and misconception to a short label if wrong)
"""
    return chat_json(
        [
            {"role": "system", "content": GRADE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

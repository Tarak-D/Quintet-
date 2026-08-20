"""
Diagnostic Agent
----------------
Runs the short diagnostic assessment used at the start of a topic to
estimate a learner's current level and surface an initial prerequisite
gap. This is the first agent in the loop:

    Student Input -> [Diagnostic Agent] -> Learner Analysis -> ...

It does NOT make any medical/educational diagnosis - only an academic
level estimate and a list of concept tags the learner is weak on.
"""

from __future__ import annotations
from typing import List, Dict, Any

from utils.llm_client import chat_json
from utils.rag import retrieve_context  # grounds questions in the KB


SYSTEM_PROMPT = """You are the Diagnostic Agent inside EduLeap, an adaptive \
AI tutoring system for learning-disadvantaged children. Your job is to \
write a SHORT diagnostic assessment (not a full exam) that estimates a \
learner's starting level for a topic and reveals which prerequisite \
concept, if any, they are missing.

Rules:
- Keep language simple and age-appropriate.
- Every question must map to exactly one concept_tag grounded in the \
  provided knowledge base context.
- Return STRICT JSON only, matching the schema given in the user message. \
  No prose, no markdown fences.
- Never diagnose a medical or learning disability - you only estimate \
  academic level and knowledge gaps.
"""


def generate_diagnostic(
    topic: str, grade_level: str, num_questions: int = 4
) -> List[Dict[str, Any]]:
    """Generate a short diagnostic question set for `topic`.

    Returns a list of question dicts:
        {
          "id": str,
          "question": str,
          "options": List[str],     # 4 options, MCQ for fast auto-grading
          "correct_answer": str,    # must exactly match one of options
          "concept_tag": str,       # prerequisite/topic concept being probed
          "difficulty": "easy" | "medium" | "hard"
        }
    """
    context_chunks = retrieve_context(topic, top_k=5)
    context_text = "\n".join(f"- {c}" for c in context_chunks) or "No KB context found."

    user_prompt = f"""
Topic: {topic}
Learner grade level (approx): {grade_level}
Number of questions: {num_questions}

Knowledge base context (use to ground concept_tags and correctness):
{context_text}

Return JSON exactly in this shape:
{{
  "questions": [
    {{
      "id": "q1",
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_answer": "...",
      "concept_tag": "...",
      "difficulty": "easy"
    }}
  ]
}}
"""
    result = chat_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    return result.get("questions", [])


def evaluate_diagnostic(
    questions: List[Dict[str, Any]], responses: Dict[str, str]
) -> Dict[str, Any]:
    """Score the diagnostic and estimate level + initial gap.

    `responses` maps question id -> student's chosen answer text.

    Returns:
        {
          "score": float,                  # 0-1
          "estimated_level": "beginner" | "intermediate" | "advanced",
          "identified_gaps": [str, ...],    # concept_tags answered wrong
          "per_question": [{"id", "correct": bool, "concept_tag"}]
        }
    """
    per_question = []
    correct_count = 0
    gaps: List[str] = []

    for q in questions:
        qid = q["id"]
        given = responses.get(qid)
        is_correct = given is not None and given == q["correct_answer"]
        if is_correct:
            correct_count += 1
        else:
            gaps.append(q["concept_tag"])
        per_question.append(
            {"id": qid, "correct": is_correct, "concept_tag": q["concept_tag"]}
        )

    total = len(questions) or 1
    score = correct_count / total

    if score >= 0.75:
        level = "advanced"
    elif score >= 0.4:
        level = "intermediate"
    else:
        level = "beginner"

    seen = set()
    unique_gaps = [g for g in gaps if not (g in seen or seen.add(g))]

    return {
        "score": score,
        "estimated_level": level,
        "identified_gaps": unique_gaps,
        "per_question": per_question,
    }

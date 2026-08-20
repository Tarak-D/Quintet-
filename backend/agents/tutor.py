"""
Tutor Agent
-----------
Generates level-appropriate explanations and switches teaching strategy
(e.g. breaking a concept into a simpler prerequisite) when a learner is
repeatedly struggling, instead of repeating the same explanation.

Pipeline stage: AI Planning -> [Tutor Agent execution] -> Verification
"""

from __future__ import annotations
from typing import Optional

from utils.llm_client import chat
from utils.rag import retrieve_context


SYSTEM_PROMPT = """You are the Tutor Agent inside EduLeap, an adaptive AI \
tutoring platform for children in low-resource learning environments.

Your teaching style:
- Explain ONE concept at a time, in simple, encouraging, age-appropriate \
  language.
- Ground every explanation in the provided knowledge base context; do not \
  invent facts outside it.
- If told the learner is struggling, do NOT repeat the same explanation. \
  Instead, break the concept down into a simpler prerequisite step, use a \
  more concrete example, or change the analogy.
- Keep explanations short (roughly 80-150 words) followed by exactly one \
  worked example.
- Never mention grades, scores, IQ, or make the learner feel judged.
"""


def explain_concept(
    concept: str,
    level: str,
    struggling: bool = False,
    prior_explanation: Optional[str] = None,
) -> str:
    """Generate an explanation for `concept` calibrated to `level`.

    If `struggling` is True, the agent is told the previous explanation
    (`prior_explanation`) did not work and must change strategy rather
    than restate it - this is what makes the loop "adaptive" rather than
    a generic Q&A chatbot.
    """
    context_chunks = retrieve_context(concept, top_k=4)
    context_text = "\n".join(f"- {c}" for c in context_chunks) or "No KB context found."

    strategy_note = ""
    if struggling:
        strategy_note = f"""
The learner struggled after this previous explanation:
\"\"\"{prior_explanation or "(none recorded)"}\"\"\"
Do NOT repeat it. Break the concept into a simpler prerequisite, or use a
different concrete/visual example.
"""

    user_prompt = f"""
Concept to teach: {concept}
Learner level: {level}
{strategy_note}
Knowledge base context:
{context_text}

Write the explanation now, followed by exactly one worked example.
"""

    return chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )


def decompose_prerequisite(concept: str, level: str) -> str:
    """Explicitly teach the prerequisite concept a learner is missing.

    Called when the Learning Analyst flags a recurring misconception
    (see analyst.detect_recurring_misconception) - this is the concrete
    behaviour behind the proposal's "breaks a concept into a simpler
    prerequisite" example (fraction addition -> common denominators).
    """
    context_chunks = retrieve_context(concept, top_k=4)
    context_text = "\n".join(f"- {c}" for c in context_chunks) or "No KB context found."

    user_prompt = f"""
The learner keeps failing questions on: {concept}
Learner level: {level}

Knowledge base context:
{context_text}

1. Name the single prerequisite concept most likely missing.
2. Teach ONLY that prerequisite with a short, concrete example.
3. End with one sentence bridging back to "{concept}".
"""
    return chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )

"""
utils/agents/assessment.py

Assessment Agent: generates targeted practice questions and grades the
learner's response (proposal 2.2 / 3.2). Grading is grounded in the KB and
returns a strict correct answer string so misconception.py can compare
deterministically rather than relying on the LLM's own judgement of
correctness.
"""

from typing import Optional

from utils.llm_client import call_llm_json
from utils.rag import retrieve_context


class AssessmentAgent:
    def generate_question(self, topic: str, level: float, target_gap: Optional[str] = None) -> dict:
        """
        Returns {"question": str, "correct_answer": str, "targets": str}
        targets = the concept/gap this question is meant to probe.
        """
        focus = target_gap or topic
        context = retrieve_context(f"{focus} practice question", topic=topic, k=4)

        system = (
            "You are an assessment-question generator for a children's AI "
            "tutor. Write ONE practice question that specifically probes "
            "the given concept, appropriate for the given difficulty level "
            "(1=beginner .. 5=advanced). Keep it short and unambiguous, "
            "with a single clearly correct short-form answer (a number, "
            "word, or short phrase - not an essay).\n"
            "Respond ONLY as JSON: {\"question\": str, \"correct_answer\": str, "
            "\"targets\": str}."
        )
        user = (
            f"Topic: {topic}\n"
            f"Focus concept: {focus}\n"
            f"Difficulty level (1-5): {level}\n"
            f"Reference material:\n{context if context else '(none found)'}"
        )
        return call_llm_json(system, user, max_tokens=300)

    def evaluate_answer(self, question: str, correct_answer: str, student_answer: str) -> dict:
        """
        Lightweight semantic grading pass, used when a plain string compare
        would be too strict (e.g. '1/2' vs 'one half'). Returns
        {"is_correct": bool, "feedback": str}.
        """
        system = (
            "You grade a single short-answer response for a child's math/"
            "learning app. Decide if the student's answer is mathematically/"
            "conceptually equivalent to the correct answer, even if worded "
            "differently. Be encouraging in your feedback, 1-2 sentences, "
            "and never comment on the student's intelligence or ability.\n"
            "Respond ONLY as JSON: {\"is_correct\": bool, \"feedback\": str}."
        )
        user = (
            f"Question: {question}\n"
            f"Correct answer: {correct_answer}\n"
            f"Student answer: {student_answer}"
        )
        return call_llm_json(system, user, max_tokens=200)

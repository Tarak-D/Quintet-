"""
utils/agents/diagnostic.py

Diagnostic Agent: runs the short initial assessment (proposal 7) used to
estimate a learner's starting level and surface an initial prerequisite gap,
before the main tutoring loop begins.
"""

from typing import List, Dict

from utils.llm_client import call_llm_json
from utils.rag import retrieve_context


class DiagnosticAgent:
    NUM_QUESTIONS = 3

    def generate_diagnostic(self, topic: str) -> List[Dict]:
        """
        Returns a short list of {question, options?, answer} items covering
        the topic and its immediate prerequisites, grounded in the KB.
        """
        context = retrieve_context(f"{topic} fundamentals and prerequisites", topic=topic, k=4)

        system = (
            "You are an educational diagnostic-assessment designer for "
            "school-age children. Write a SHORT diagnostic quiz (not a full "
            "exam) that estimates a learner's starting level for the given "
            "topic and checks whether they know its key prerequisite "
            "concepts. Keep language simple and age-appropriate.\n"
            "Respond ONLY as JSON: a list of objects, each with "
            "'question', 'options' (list of 4 short strings), "
            "'correct_option', and 'targets' (the prerequisite or topic "
            "concept this question checks, e.g. 'common_denominators')."
        )
        user = (
            f"Topic: {topic}\n"
            f"Number of questions: {self.NUM_QUESTIONS}\n"
            f"Reference material:\n{context if context else '(no KB material found, use general knowledge)'}"
        )
        result = call_llm_json(system, user, max_tokens=700)

        if isinstance(result, list):
            return result
        # in case the model wraps it, e.g. {"questions": [...]}
        return result.get("questions", []) if isinstance(result, dict) else []

    def estimate_level(self, topic: str, responses: List[Dict]) -> Dict:
        """
        responses: [{"question": str, "targets": str, "correct": bool}, ...]
        Returns {"level": float 1-5, "gaps": [concept, ...]}
        """
        correct_count = sum(1 for r in responses if r.get("correct"))
        total = max(1, len(responses))
        ratio = correct_count / total

        # simple deterministic mapping: ratio -> starting level (1-5)
        level = 1.0 + round(ratio * 4, 1)

        gaps = [r["targets"] for r in responses if not r.get("correct") and r.get("targets")]
        # dedupe, preserve order
        seen = set()
        gaps = [g for g in gaps if not (g in seen or seen.add(g))]

        return {"level": level, "gaps": gaps}

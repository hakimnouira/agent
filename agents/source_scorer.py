from agents.llm_selector import get_best_llm

class SourceScorerAgent:
    def __init__(self):
        self.llm = get_best_llm("scoring")

    def score_source(self, source_type, source_name):
        prompt = (
            "You are a rigorous fact-checking source rater. Rate the reputation and factual reliability of this source/domain "
            f"('{source_name}') for factual reporting, using ONLY a number from 1 (very low credibility) to 5 (very high credibility). "
            "AGAIN: Reply only with a number. Use 5 for domains like (nasa.gov, bbc.com, reuters.com), 1 for anonymous/unfamiliar domains or social sites. "
            f"Source Type: {source_type}\nSource Name: {source_name}\nScore (number only):"
        )
        result = self.llm.invoke(prompt)
        try:
            score = float(str(result).strip())
        except Exception:
            score = 2.5
        return score

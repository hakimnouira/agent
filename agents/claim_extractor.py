from agents.llm_selector import get_best_llm

class ClaimExtractorAgent:
    def __init__(self):
        self.llm = get_best_llm("claim_extraction")

    def extract_claims(self, article_text):
        prompt =(
            "You are an expert fact-checking assistant. Your job is to extract ONLY verifiable, objective, discrete factual statements from the following news article. "
            "Do not include opinions, inferences, or duplicated information. Each fact must be atomic and suitable for independent verification. "
            "List each claim on a new line, with no numbering, and omit any empty lines. If there are no factual claims, return 'NONE'.\n\n"
            f"Article:\n{article_text}\n\nClaims:\n"
        )
        result = self.llm.invoke(prompt)
        return [line.strip() for line in str(result).split('\n') if line.strip()]

from agents.llm_selector import get_best_llm


class CrossVerifierAgent:
    def __init__(self):
        self.llm = get_best_llm("fact_verification")

    def verify_claim(self, claim, evidence):
        prompt = (
            "You are an expert fact-checking AI. Carefully analyze the CLAIM and EVIDENCE below.\n\n"
            "Determine the relationship:\n"
            "- **support**: The evidence confirms or provides factual backing for the claim, even if worded differently\n"
            "- **contradict**: The evidence directly contradicts or disproves the claim\n"
            "- **unrelated**: The evidence has no clear relevance to the claim\n\n"
            "Guidelines:\n"
            "- Paraphrases and synonyms count as support if the meaning matches\n"
            "- Focus on semantic meaning, not exact wording\n"
            "- Only output ONE word: support, contradict, or unrelated\n\n"
            f"CLAIM: {claim}\n"
            f"EVIDENCE: {evidence}\n\n"
            "Verdict (one word only):"
        )
        
        result = self.llm.invoke(prompt)
        
        # Extract content from LangChain response
        if hasattr(result, "content"):
            verdict = result.content.strip().lower()
        elif isinstance(result, dict) and "content" in result:
            verdict = result["content"].strip().lower()
        else:
            verdict = str(result).strip().lower()
        
        # Ensure only valid outputs
        if verdict not in ['support', 'contradict', 'unrelated']:
            print(f"Warning: Invalid verdict '{verdict}', defaulting to 'unrelated'")
            verdict = 'unrelated'
        
        return verdict

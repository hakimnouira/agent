from agents.llm_selector import get_best_llm

class CrossVerifierAgent:
    def __init__(self):
        self.llm = get_best_llm("fact_verification")

    def verify_claim(self, claim, evidence):
        prompt = (
            "You are a highly strict AI for claim verification. Examine the CLAIM and EVIDENCE below.\n"
            "Output ONE word: 'support' if the evidence directly confirms the claim, "
            "'contradict' if the evidence directly disproves it, or 'unrelated' if the evidence has no relevance. "
            "You must ONLY use these three responses. Be strict: Only choose 'support' for direct, explicit confirmation.\n\n"
            f"CLAIM: {claim}\n"
            f"EVIDENCE: {evidence}\n\n"
            "Answer (support/contradict/unrelated):"
        )
        result = self.llm.invoke(prompt)
        if isinstance(result, str):
            verdict = result.strip().lower()
        else:
            verdict = str(result).strip().lower()
        return verdict

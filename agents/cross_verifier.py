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
        
        if hasattr(result, "content"):
            verdict = result.content.strip().lower()
        elif isinstance(result, dict) and "content" in result:
            verdict = result["content"].strip().lower()
        else:
            verdict = str(result).strip().lower()
        
        if verdict not in ['support', 'contradict', 'unrelated']:
            print(f"Warning: Invalid verdict '{verdict}', defaulting to 'unrelated'")
            verdict = 'unrelated'
        
        return verdict
    
    def verify_claim_with_explanation(self, claim, evidence):
        """XAI: Returns verdict with detailed explanation."""
        # Get explanation from LLM
        explanation_prompt = (
            "You are an expert fact-checker. Analyze the CLAIM and EVIDENCE, then:\n"
            "1. State your verdict: support, contradict, or unrelated\n"
            "2. Explain WHY in 1-2 sentences\n\n"
            f"CLAIM: {claim}\n"
            f"EVIDENCE: {evidence}\n\n"
            "Format:\n"
            "Verdict: [support/contradict/unrelated]\n"
            "Explanation: [Your reasoning]\n"
        )
        
        result = self.llm.invoke(explanation_prompt)
        
        if hasattr(result, "content"):
            response = result.content
        elif isinstance(result, dict) and "content" in result:
            response = result["content"]
        else:
            response = str(result)
        
        # Parse response
        lines = response.strip().split('\n')
        verdict = 'unrelated'
        explanation = 'No explanation provided'
        
        for line in lines:
            if line.lower().startswith('verdict:'):
                verdict = line.split(':', 1)[1].strip().lower()
            elif line.lower().startswith('explanation:'):
                explanation = line.split(':', 1)[1].strip()
        
        if verdict not in ['support', 'contradict', 'unrelated']:
            verdict = 'unrelated'
        
        return {
            'verdict': verdict,
            'explanation': explanation,
            'claim': claim,
            'evidence': evidence[:200] + '...' if len(evidence) > 200 else evidence
        }

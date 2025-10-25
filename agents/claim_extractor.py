from agents.llm_selector import get_best_llm

class ClaimExtractorAgent:
    def __init__(self):
        self.llm = get_best_llm("claim_extraction")

    def extract_claims(self, article_text):
        prompt = (
            "You are an expert fact-checking assistant. Extract ONLY verifiable, objective, and discrete factual statements from the news article below.\n\n"
            "Requirements:\n"
            "- Each claim must be atomic (one fact per line)\n"
            "- Claims must be independently verifiable through credible sources\n"
            "- Do NOT include opinions, speculation, or inferences\n"
            "- Do NOT include duplicated information\n"
            "- Do NOT number the claims\n"
            "- List each claim on a separate line\n"
            "- If no verifiable factual claims exist, respond with exactly: NONE\n\n"
            f"Article:\n{article_text}\n\n"
            "Extracted Claims (one per line):\n"
        )
        
        result = self.llm.invoke(prompt)
        
        if hasattr(result, "content"):
            text = result.content
        elif isinstance(result, dict) and "content" in result:
            text = result["content"]
        else:
            text = str(result)
        
        claims = [
            line.strip() 
            for line in text.split('\n') 
            if line.strip() and line.strip().upper() != 'NONE'
        ]
        
        return claims if claims else None
    
    def extract_claims_with_explanation(self, article_text):
        """XAI: Returns claims with explanations."""
        claims = self.extract_claims(article_text)
        
        if not claims:
            return {
                'claims': [],
                'explanation': 'No verifiable factual claims found in the text. The content may be purely opinion-based or lacks concrete statements.'
            }
        
        explanation = f"Identified {len(claims)} verifiable claim(s) from the article. These are atomic, fact-based statements that can be independently verified."
        
        return {
            'claims': claims,
            'explanation': explanation,
            'claim_details': [
                {
                    'claim': claim,
                    'reason': 'Contains verifiable factual assertion'
                }
                for claim in claims
            ]
        }

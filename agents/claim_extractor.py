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
        
        # Extract content from LangChain response
        if hasattr(result, "content"):
            text = result.content
        elif isinstance(result, dict) and "content" in result:
            text = result["content"]
        else:
            text = str(result)
        
        # Robust splitting and filtering
        claims = [
            line.strip() 
            for line in text.split('\n') 
            if line.strip() and line.strip().upper() != 'NONE'
        ]
        
        return claims if claims else None

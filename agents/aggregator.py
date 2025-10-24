from agents.llm_selector import get_best_llm

class AggregatorAgent:
    def __init__(self):
        self.llm = get_best_llm("aggregation")

    def aggregate(self, support_score, source_score, verdict):
        prompt = (
           "You are an expert in credibility aggregation for news claims. Given the support score (from evidence), "
            "the source credibility score, and the verdict label ('support', 'contradict', or 'unrelated'),"
            "output a FINAL credibility score for this claim, as a single float from 1 (unreliable) to 5 (highly credible). "
            "Weight 'support' highest, 'contradict' as lowest, and 'unrelated' as low. Your answer must be ONLY the number.\n"
            f"Support Score: {support_score}\n"
            f"Source Score: {source_score}\n"
            f"Verdict: {verdict}\n"
            "Final Credibility Score (number only):"
        )
        # always use messages for chat APIs
        message = [{"role": "user", "content": prompt}]
        result = self.llm.invoke(message)
        # robust extraction for latest LangChain
        try:
            # Most chat models return an object with .content
            if hasattr(result, "content"):
                text = result.content
            elif isinstance(result, dict) and "content" in result:
                text = result["content"]
            else:
                text = str(result)
            final_score = float(text.strip())
        except Exception:
            final_score = (float(support_score) + float(source_score)) / 2
        return final_score

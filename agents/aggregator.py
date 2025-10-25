from agents.llm_selector import get_best_llm

class AggregatorAgent:
    def __init__(self):
        self.llm = get_best_llm("aggregation")
    
    def aggregate(self, support_score, source_score, verdict):
        prompt = (
            f"Given:\n"
            f"- Evidence support score: {support_score}/5\n"
            f"- Source credibility score: {source_score}/5\n"
            f"- Verdict: {verdict}\n\n"
            f"Calculate a final credibility score (1-5) by weighing both factors.\n"
            f"Return ONLY a number between 1.0 and 5.0"
        )
        result = self.llm.invoke(prompt)
        
        if hasattr(result, "content"):
            text = result.content
        else:
            text = str(result)
        
        try:
            score = float(text.strip())
            return max(1.0, min(5.0, score))
        except:
            return (support_score + source_score) / 2
    
    def aggregate_with_explanation(self, support_score, source_score, verdict):
        """XAI: Returns final score with breakdown."""
        final_score = self.aggregate(support_score, source_score, verdict)
        
        # Calculate contributions
        support_contribution = (support_score / 5.0) * 50  # 50% weight
        source_contribution = (source_score / 5.0) * 50    # 50% weight
        
        explanation = f"Final score combines evidence quality ({support_score}/5) and source credibility ({source_score}/5)"
        
        verdict_impact = {
            'support': 'Positive: Evidence directly confirms the claim',
            'contradict': 'Negative: Evidence contradicts the claim',
            'unrelated': 'Neutral: Limited relevant evidence found'
        }
        
        return {
            'final_score': final_score,
            'final_percentage': int((final_score / 5.0) * 100),
            'explanation': explanation,
            'breakdown': {
                'evidence_quality': {
                    'score': support_score,
                    'contribution': f"{support_contribution:.1f}%",
                    'verdict': verdict,
                    'impact': verdict_impact.get(verdict, 'Unknown')
                },
                'source_credibility': {
                    'score': source_score,
                    'contribution': f"{source_contribution:.1f}%"
                }
            }
        }

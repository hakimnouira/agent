from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import tempfile

from agents.claim_extractor import ClaimExtractorAgent
from agents.evidence_retriever import EvidenceRetrieverAgent
from agents.cross_verifier import CrossVerifierAgent
from agents.source_scorer import SourceScorerAgent
from agents.aggregator import AggregatorAgent
from agents.web_retriever import WebRetrieverAgent
from agents.image_to_text import ImageToTextAgent
from urllib.parse import urlparse
import os
import tempfile
from dotenv import load_dotenv
load_dotenv(override=True)
# Initialize FastAPI app
app = FastAPI(title="Fact Checking API with XAI", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Social platforms to filter
SOCIAL_PLATFORMS = [
    "youtube.com", "youtu.be", "instagram.com", 
    "facebook.com", "fb.com", "m.facebook.com",
    "twitter.com", "x.com", "reddit.com", "tiktok.com",
    "pinterest.com", "linkedin.com", "snapchat.com",
    "quora.com", "medium.com", "tumblr.com"
]

def is_social_platform(url):
    if not url:
        return True
    return any(platform in url.lower() for platform in SOCIAL_PLATFORMS)

def format_source_for_model(url):
    if not url:
        return "unknown"
    domain = urlparse(url).netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

# Initialize agents
print("Initializing agents...")
claim_agent = ClaimExtractorAgent()
retriever_agent = EvidenceRetrieverAgent()
verifier_agent = CrossVerifierAgent()
source_agent = SourceScorerAgent()
aggregator_agent = AggregatorAgent()
web_agent = WebRetrieverAgent()
image_agent = ImageToTextAgent()
print("Agents initialized successfully!")

# Request/Response Models
class TextVerificationRequest(BaseModel):
    text: str
    include_explanation: bool = True

class VerificationResponse(BaseModel):
    claims: List[str]
    best_evidence: str
    best_url: str
    source_domain: str
    source_credibility_score: float
    verdict: str
    final_credibility_score: float
    all_sources: List[dict]
    explanation: Optional[Dict[str, Any]] = None

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fact Checking API with XAI is running"}

@app.post("/verify/text", response_model=VerificationResponse)
def verify_text(request: TextVerificationRequest):
    try:
        # Extract claims WITH explanation (with fallback)
        if request.include_explanation and hasattr(claim_agent, 'extract_claims_with_explanation'):
            claim_result = claim_agent.extract_claims_with_explanation(request.text)
            if not claim_result['claims']:
                raise HTTPException(status_code=400, detail="No claims extracted")
            claims = claim_result['claims']
            claim_explanation = {
                'extraction': claim_result['explanation'],
                'claims_analyzed': len(claims)
            }
        else:
            claims = claim_agent.extract_claims(request.text)
            if not claims:
                raise HTTPException(status_code=400, detail="No claims extracted")
            claim_explanation = {'extraction': f'Extracted {len(claims)} claim(s)', 'claims_analyzed': len(claims)}
        
        claim = claims[0]
        web_results = web_agent.get_live_evidence(claim)
        
        # Filter valid sources
        valid_sources = []
        skipped_social = 0
        for result in web_results:
            url = result.get("link", "")
            if is_social_platform(url):
                skipped_social += 1
                continue
            valid_sources.append(result)
            if len(valid_sources) >= 5:
                break
        
        if not valid_sources:
            raise HTTPException(status_code=404, detail="No valid news sources found")
        
        # Verify each source
        verdict_map = {'support': 1, 'contradict': 0, 'unrelated': -1}
        best_score = -1
        best_evidence = ""
        best_url = ""
        best_verdict = ""
        best_verdict_explanation = ""
        all_sources_data = []
        
        for result in valid_sources:
            snippet = result.get("snippet", "")
            url = result.get("link", "")
            
            # Get verdict with explanation if available
            if request.include_explanation and hasattr(verifier_agent, 'verify_claim_with_explanation'):
                verdict_result = verifier_agent.verify_claim_with_explanation(claim, snippet)
                verdict = verdict_result['verdict']
                verdict_explanation = verdict_result['explanation']
            else:
                verdict_result = verifier_agent.verify_claim(claim, snippet)
                if hasattr(verdict_result, "content"):
                    verdict = verdict_result.content.strip().lower()
                elif isinstance(verdict_result, dict) and "content" in verdict_result:
                    verdict = verdict_result["content"].strip().lower()
                else:
                    verdict = str(verdict_result).strip().lower()
                verdict_explanation = f"Verdict: {verdict}"
            
            verdict_score = verdict_map.get(verdict, -1)
            
            all_sources_data.append({
                "url": url,
                "snippet": snippet,
                "verdict": verdict,
                "explanation": verdict_explanation if request.include_explanation else None
            })
            
            if verdict_score > best_score:
                best_score = verdict_score
                best_evidence = snippet
                best_url = url
                best_verdict = verdict
                if request.include_explanation:
                    best_verdict_explanation = verdict_explanation
        
        # Fallback
        if not best_url and valid_sources:
            first = valid_sources[0]
            best_url = first.get("link", "")
            best_evidence = first.get("snippet", "")
            best_verdict = "unrelated"
        
        # Score source WITH explanation (with fallback)
        formatted_source = format_source_for_model(best_url)
        if request.include_explanation and hasattr(source_agent, 'score_source_with_explanation'):
            source_result = source_agent.score_source_with_explanation("Web", formatted_source)
            source_score = source_result['score']
            source_explanation = source_result
        else:
            source_score = source_agent.score_source("Web", formatted_source)
            source_explanation = {
                'score': source_score,
                'explanation': f'Source credibility: {source_score}/5',
                'contributing_factors': ['Domain reputation'],
                'is_trusted': source_score >= 4.0
            }
        
        # Calculate final score WITH explanation (with fallback)
        support_score = 4 if 'support' in best_verdict else 1
        
        if request.include_explanation and hasattr(aggregator_agent, 'aggregate_with_explanation'):
            aggregation_result = aggregator_agent.aggregate_with_explanation(
                support_score, source_score, best_verdict
            )
            final_score = aggregation_result['final_score']
            aggregation_explanation = aggregation_result
        else:
            final_score = aggregator_agent.aggregate(support_score, source_score, best_verdict)
            aggregation_explanation = {
                'final_score': final_score,
                'explanation': f'Combined evidence ({support_score}/5) and source credibility ({source_score}/5)',
                'breakdown': {
                    'evidence_quality': {'score': support_score, 'verdict': best_verdict},
                    'source_credibility': {'score': source_score}
                }
            }
        
        # Build explanation object
        explanation = None
        if request.include_explanation:
            explanation = {
                'claim_extraction': claim_explanation,
                'evidence_retrieval': {
                    'total_sources_found': len(web_results),
                    'social_platforms_filtered': skipped_social,
                    'valid_news_sources': len(valid_sources)
                },
                'best_evidence_selection': {
                    'chosen_source': best_url,
                    'reason': f"Highest verdict score ({best_score})",
                    'verdict_explanation': best_verdict_explanation
                },
                'source_credibility': source_explanation,
                'final_calculation': aggregation_explanation
            }
        
        return VerificationResponse(
            claims=claims,
            best_evidence=best_evidence,
            best_url=best_url,
            source_domain=formatted_source,
            source_credibility_score=source_score,
            verdict=best_verdict,
            final_credibility_score=final_score,
            all_sources=all_sources_data,
            explanation=explanation
        )
    
    except Exception as e:
        import traceback
        print("Error details:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify/image", response_model=VerificationResponse)
async def verify_image(file: UploadFile = File(...), include_explanation: bool = True):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            text = image_agent.extract_text_from_file(tmp_file_path)
            if not text:
                raise HTTPException(status_code=400, detail="No text extracted from image")
            
            request = TextVerificationRequest(text=text, include_explanation=include_explanation)
            return verify_text(request)
        finally:
            os.unlink(tmp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

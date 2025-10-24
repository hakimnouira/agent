from agents.claim_extractor import ClaimExtractorAgent
from agents.evidence_retriever import EvidenceRetrieverAgent
from agents.cross_verifier import CrossVerifierAgent
from agents.source_scorer import SourceScorerAgent
from agents.aggregator import AggregatorAgent
from agents.web_retriever import WebRetrieverAgent
import torch
from urllib.parse import urlparse

print("\n=== Fact Verification Pipeline Test ===")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
print("\nEvaluating top web results for the best support...")

best_score = -1  # -1 = unrelated, 0 = contradict, 1 = support
best_evidence = ""
best_url = ""
best_verdict = ""
verdict_map = {'support': 1, 'contradict': 0, 'unrelated': -1}
test_article = """NASA's Artemis I mission landed astronauts on Mars in December 2022 with the Orion spacecraft and returned safely to Earth after 25 days."""

print("\nInitializing agents...")
claim_agent = ClaimExtractorAgent()
retriever_agent = EvidenceRetrieverAgent()
verifier_agent = CrossVerifierAgent()
source_agent = SourceScorerAgent()
aggregator_agent = AggregatorAgent()
web_agent = WebRetrieverAgent()
print("\nAgents initialized successfully.")
print("\nExtracting claims from article...")
TRUSTED_DOMAINS = ["nasa.gov", "nytimes.com", "reuters.com", "nature.com","www.nasa.gov"]

claims = claim_agent.extract_claims(test_article)
print("\nExtracted Claims:", claims)

if claims:
    claim = claims[0]
    print("\n--- Processing Claim ---")
    print("Claim:", claim)

    web_results = web_agent.get_live_evidence(claim)
    print("\nTop Web Results:")

    # Select best snippet/domain from top 5 results
    best_evidence = ""
    best_url = ""
    for result in web_results[:4]:  # check top 5, or full list
        snippet = result.get("snippet", "")
        url = result.get("link", "")
        verdict = verifier_agent.verify_claim(claim, snippet)
        print(f"Snippet: {snippet}\nURL: {url}\nVerdict: {verdict}")
        verdict_score = verdict_map.get(verdict, -1)
    # Prioritize 'support', if multiple supports prefer a trustworthy domain
    if verdict_score > best_score or (verdict_score == best_score and "nasa" in url):
        best_score = verdict_score
        best_evidence = snippet
        best_url = url
        best_verdict = verdict

    source_name = urlparse(best_url).netloc if best_url else "unknown"
    print("\nBest supporting evidence selected:")
    print("Snippet:", best_evidence)
    print("URL:", best_url)
    print("Source detected:", source_name)

    source_score = source_agent.score_source("Web", source_name)
    print("Source Credibility Score:", source_score)

    support_score = 4 if best_verdict == "support" else 1
    final_score = aggregator_agent.aggregate(support_score, source_score, best_verdict)
    print("Verification Verdict:", best_verdict)
    print("\nFinal Credibility Score:", final_score)

else:
    print("\nNo claims extracted from article.")
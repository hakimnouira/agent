from agents.claim_extractor import ClaimExtractorAgent
from agents.evidence_retriever import EvidenceRetrieverAgent
from agents.cross_verifier import CrossVerifierAgent
from agents.source_scorer import SourceScorerAgent
from agents.aggregator import AggregatorAgent
from agents.web_retriever import WebRetrieverAgent
from agents.image_to_text import ImageToTextAgent
import torch
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

print("\n=== Fact Verification Pipeline Test ===")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# Define social/video platforms to filter out
SOCIAL_PLATFORMS = [
    "youtube.com", "youtu.be", "instagram.com", 
    "facebook.com", "fb.com", "m.facebook.com",
    "twitter.com", "x.com", "reddit.com", "tiktok.com",
    "pinterest.com", "linkedin.com", "snapchat.com",
    "quora.com", "medium.com", "tumblr.com"
]

def is_social_platform(url):
    """Check if URL is from a social/video platform."""
    if not url:
        return True
    url_lower = url.lower()
    return any(platform in url_lower for platform in SOCIAL_PLATFORMS)

def format_source_for_model(url):
    """Format URL to domain only for DeBERTa model."""
    if not url:
        return "unknown"
    domain = urlparse(url).netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

# ========================================
# Configuration
# ========================================
INPUT_TYPE = "text"  # "text" or "image"
IMAGE_PATH = "test1.png"
TEST_TEXT = "US military flew supersonic B-1 bombers close to Venezuela"

# ========================================
# Initialize Agents
# ========================================
print("\nInitializing agents...")
claim_agent = ClaimExtractorAgent()
retriever_agent = EvidenceRetrieverAgent()
verifier_agent = CrossVerifierAgent()
source_agent = SourceScorerAgent()
aggregator_agent = AggregatorAgent()
web_agent = WebRetrieverAgent()
image_agent = ImageToTextAgent()
print("✓ Agents initialized successfully\n")

# Trusted domains for prioritization
TRUSTED_DOMAINS = ["nasa.gov", "bbc.com", "nytimes.com", "reuters.com", "nature.com", "apnews.com"]

# ========================================
# Handle Input (Text or Image)
# ========================================
if INPUT_TYPE == "image":
    print(f"Extracting text from image: {IMAGE_PATH}")
    test_article = image_agent.extract_text_from_file(IMAGE_PATH)
    print(f"Extracted text:\n{test_article}\n")
    if not test_article:
        print("❌ Error: No text extracted from image. Exiting.")
        exit(1)
else:
    test_article = TEST_TEXT

# ========================================
# Extract Claims
# ========================================
print("Extracting claims from article...")
claims = claim_agent.extract_claims(test_article)
print(f"✓ Extracted {len(claims) if claims else 0} claim(s):", claims, "\n")

if not claims:
    print("❌ No claims extracted from article. Exiting.")
    exit(1)

# ========================================
# Process First Claim
# ========================================
claim = claims[0]
print("="*60)
print(f"PROCESSING CLAIM: {claim}")
print("="*60 + "\n")

# Get web evidence
web_results = web_agent.get_live_evidence(claim)
print(f"Received {len(web_results)} total web results\n")

# ========================================
# Filter Valid Sources
# ========================================
print("Filtering web results (skipping social media)...")
valid_sources = []
skipped_count = 0

for result in web_results:
    url = result.get("link", "")
    
    if is_social_platform(url):
        skipped_count += 1
        print(f"  ⊗ Skipped: {url}")
        continue
    
    valid_sources.append(result)
    print(f"  ✓ Valid: {url}")
    
    if len(valid_sources) >= 5:
        break

print(f"\n{'='*60}")
print(f"Valid Sources: {len(valid_sources)} | Filtered: {skipped_count}")
print(f"{'='*60}\n")

if not valid_sources:
    print("❌ No valid news sources found!")
    exit(1)

# ========================================
# Verify Each Source
# ========================================
print("Verifying claims against sources...\n")

verdict_map = {'support': 1, 'contradict': 0, 'unrelated': -1}
best_score = -1
best_evidence = ""
best_url = ""
best_author = ""
best_verdict = ""

for result in valid_sources:
    snippet = result.get("snippet", "")
    url = result.get("link", "")
    author = result.get("author") or result.get("source") or "Unknown"
    
    verdict_result = verifier_agent.verify_claim(claim, snippet)
    
    # Extract verdict
    if hasattr(verdict_result, "content"):
        verdict = verdict_result.content.strip().lower()
    elif isinstance(verdict_result, dict) and "content" in verdict_result:
        verdict = verdict_result["content"].strip().lower()
    else:
        verdict = str(verdict_result).strip().lower()
    
    # Ensure valid verdict
    if verdict not in verdict_map:
        print(f"⚠️ Invalid verdict '{verdict}' for {url}, defaulting to 'unrelated'")
        verdict = 'unrelated'
    
    verdict_score = verdict_map[verdict]
    
    print(f"Source: {url}")
    print(f"  Snippet: {snippet[:100]}...")
    print(f"  Verdict: {verdict.upper()}")
    print()
    
    # Select best source
    is_trusted = any(trusted in url for trusted in TRUSTED_DOMAINS)
    if verdict_score > best_score or (verdict_score == best_score and is_trusted):
        best_score = verdict_score
        best_evidence = snippet
        best_url = url
        best_author = author
        best_verdict = verdict

# Fallback to first source if none selected
if not best_url and valid_sources:
    print("⚠️ No 'support' verdict found, using first valid source\n")
    first = valid_sources[0]
    best_url = first.get("link", "")
    best_evidence = first.get("snippet", "")
    best_author = first.get("author") or first.get("source") or "Unknown"
    best_verdict = "unrelated"

# ========================================
# Score Source & Calculate Final Score
# ========================================
formatted_source = format_source_for_model(best_url)

print("="*60)
print("BEST EVIDENCE SELECTED")
print("="*60)
print(f"URL: {best_url}")
print(f"Domain: {formatted_source}")
print(f"Author: {best_author}")
print(f"Verdict: {best_verdict.upper()}")
print(f"Evidence: {best_evidence[:150]}...")
print("="*60 + "\n")

# Score source credibility
source_score = source_agent.score_source("Web", formatted_source)
print(f"Source Credibility Score: {source_score:.2f}/5.0")

# Calculate final score
support_score = 4 if 'support' in best_verdict else 1
final_score = aggregator_agent.aggregate(support_score, source_score, best_verdict)

print(f"Evidence Support Score: {support_score}/5")
print(f"Final Credibility Score: {final_score:.2f}/5.0")
print("="*60)

import os

# Model paths
REPUTATION_MODEL_PATH = "./models/deberta_reputation_model_export"

# Knowledge base settings
KB_PERSIST_DIR = "./knowledge_base/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM settings - LOCAL MODELS ONLY (no API keys needed)
DEFAULT_LLM_MODEL = "google/flan-t5-base"  # Free, runs locally
# Alternative: "microsoft/DialoGPT-medium" or any other local model

# Pipeline settings
MAX_CLAIMS_PER_ARTICLE = 5
MAX_EVIDENCE_DOCS = 3
AGGREGATION_WEIGHTS = {
    "evidence_support": 0.4,
    "source_credibility": 0.4,
    "cross_verification": 0.2
}

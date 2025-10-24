from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import KB_PERSIST_DIR, EMBEDDING_MODEL

class EvidenceRetrieverAgent:
    def __init__(self):
        # Initialize embedding model
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.db = Chroma(
            persist_directory=KB_PERSIST_DIR,
            embedding_function=embeddings
        )
        # Create retriever
        self.retriever = self.db.as_retriever()

    def get_evidence(self, claim, max_docs=3):
        # Updated for new LangChain core version
        docs = self.retriever.invoke(claim)
        return docs[:max_docs] if docs else []

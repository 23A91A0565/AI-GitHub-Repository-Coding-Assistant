import os
from typing import List, Dict, Any
from langchain_core.documents import Document

# Try importing HuggingFace embeddings (Local & 100% Free)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embeddings_model = None

    def get_embeddings_model(self):
        """
        Lazy loader for the embeddings model to avoid slow startup times until requested.
        Uses sentence-transformers/all-MiniLM-L6-v2 (384 vector dimensions, free & local).
        """
        if self._embeddings_model is None:
            try:
                # Initialize local HuggingFace embeddings model
                self._embeddings_model = HuggingFaceEmbeddings(
                    model_name=self.model_name
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize HuggingFace embeddings model ({self.model_name}): {str(e)}"
                )
        return self._embeddings_model

    def generate_chunk_embeddings(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Generates vector embeddings for a list of Document chunks.
        Returns total vectors, vector dimension count, sample vector values, and chunk stats.
        """
        if not chunks:
            raise ValueError("No document chunks provided for embedding generation.")

        embeddings_model = self.get_embeddings_model()
        
        # Extract page_content text from each chunk
        texts = [chunk.page_content for chunk in chunks]

        # Generate dense vector embeddings for all chunk texts
        vectors = embeddings_model.embed_documents(texts)

        vector_dimensions = len(vectors[0]) if vectors else 0
        sample_vector = vectors[0][:8] if vectors else []  # First 8 float values of sample vector

        return {
            "total_chunks_embedded": len(vectors),
            "vector_dimensions": vector_dimensions,
            "model_used": self.model_name,
            "sample_vector_preview": [round(v, 4) for v in sample_vector],
            "vectors": vectors  # Raw vector arrays used in Phase 7 FAISS
        }

embedding_service = EmbeddingService()

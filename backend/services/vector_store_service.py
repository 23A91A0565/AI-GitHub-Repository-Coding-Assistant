import os
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

class VectorStoreService:
    def __init__(self, base_dir: str = "data/vector_stores"):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_repo_store_path(self, owner: str, repo_name: str) -> str:
        folder_name = f"{owner}_{repo_name}_faiss"
        return os.path.join(self.base_dir, folder_name)

    def build_and_save_index(
        self,
        owner: str,
        repo_name: str,
        chunks: List[Document],
        embeddings_model
    ) -> Dict[str, Any]:
        """
        Phase 7 Indexing: Takes list of Document chunks and Embedding model,
        computes vectors, constructs a local FAISS vector store, and saves it to disk.
        """
        if not chunks:
            raise ValueError("Cannot build FAISS index with empty document chunks.")

        save_path = self._get_repo_store_path(owner, repo_name)

        # FAISS.from_documents computes vector embeddings for each chunk and constructs the index
        faiss_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings_model
        )

        # Save local FAISS index files (index.faiss & index.pkl)
        faiss_store.save_local(save_path)

        return {
            "status": "indexed",
            "total_chunks_indexed": len(chunks),
            "save_directory": save_path,
            "owner": owner,
            "repo_name": repo_name
        }

    def search_similarity(
        self,
        owner: str,
        repo_name: str,
        query: str,
        top_k: int,
        embeddings_model
    ) -> List[Dict[str, Any]]:
        """
        Phase 7 Querying: Converts developer question string into query vector,
        searches the local FAISS index, and retrieves Top-K most similar code chunks.
        """
        save_path = self._get_repo_store_path(owner, repo_name)

        if not os.path.exists(save_path):
            raise FileNotFoundError(
                f"No FAISS vector store index found for {owner}/{repo_name}. "
                "Please build the FAISS index first."
            )

        # Load FAISS vector index from local disk
        # allow_dangerous_deserialization=True is safe here as we generated the pkl locally
        faiss_store = FAISS.load_local(
            save_path,
            embeddings_model,
            allow_dangerous_deserialization=True
        )

        # Perform Top-K similarity search with score
        results_with_scores: List[Tuple[Document, float]] = faiss_store.similarity_search_with_score(
            query=query,
            k=top_k
        )

        formatted_results = []
        for doc, score in results_with_scores:
            formatted_results.append({
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "file_path": doc.metadata.get("file_path", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "similarity_score": round(float(score), 4),  # Distance score (lower/closer is better)
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        return formatted_results

vector_store_service = VectorStoreService()

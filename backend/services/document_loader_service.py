import os
from typing import List, Dict, Any
from langchain_core.documents import Document

class DocumentLoaderService:
    def load_repository_documents(
        self, repo_data: Dict[str, Any]
    ) -> List[Document]:
        """
        Converts filtered repository source files into LangChain Document instances.
        Each Document contains the raw text/code as page_content and structured metadata.
        """
        documents: List[Document] = []
        files = repo_data.get("files", [])
        owner = repo_data.get("owner", "unknown")
        repo_name = repo_data.get("repo_name", "unknown")

        for file_info in files:
            abs_path = file_info["absolute_path"]
            rel_path = file_info["file_path"]
            file_name = file_info["file_name"]
            extension = file_info["extension"]
            file_size = file_info["size_bytes"]

            try:
                # Read text file content safely using UTF-8 with fallback
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                # Skip completely empty files
                if not content.strip():
                    continue

                # Standardized metadata dictionary attached to every document
                metadata = {
                    "file_name": file_name,
                    "file_path": rel_path,
                    "file_type": extension.lstrip("."),
                    "repo_name": repo_name,
                    "owner": owner,
                    "size_bytes": file_size,
                    "source": rel_path  # LangChain standard source identifier
                }

                # Construct LangChain Document object
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)

            except Exception as e:
                print(f"Warning: Could not read file {abs_path}: {str(e)}")
                continue

        return documents

document_loader_service = DocumentLoaderService()

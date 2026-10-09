from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkingService:
    def __init__(self, default_chunk_size: int = 1000, default_chunk_overlap: int = 200):
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap

    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Document]:
        """
        Splits a list of LangChain Document objects into smaller chunks.
        Inherits metadata from parent Document and appends chunk-specific metadata.
        """
        c_size = chunk_size if chunk_size is not None else self.default_chunk_size
        c_overlap = chunk_overlap if chunk_overlap is not None else self.default_chunk_overlap

        # Initialize LangChain RecursiveCharacterTextSplitter
        # Separators prioritize paragraph breaks (\n\n), line breaks (\n), spaces, and characters
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        # split_documents automatically copies parent metadata to each chunk
        raw_chunks = splitter.split_documents(documents)

        # Track chunk indices per file
        file_chunk_counters: Dict[str, int] = {}
        chunked_documents: List[Document] = []

        for chunk in raw_chunks:
            file_path = chunk.metadata.get("file_path", "unknown")
            chunk_idx = file_chunk_counters.get(file_path, 0)
            file_chunk_counters[file_path] = chunk_idx + 1

            # Enrich chunk metadata
            updated_metadata = dict(chunk.metadata)
            updated_metadata["chunk_index"] = chunk_idx
            updated_metadata["chunk_id"] = f"{file_path}#chunk-{chunk_idx}"
            updated_metadata["chunk_size_chars"] = len(chunk.page_content)

            chunked_documents.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=updated_metadata
                )
            )

        return chunked_documents

chunking_service = ChunkingService()

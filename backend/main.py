import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from services.llm_service import llm_service
from services.github_service import github_service
from services.document_loader_service import document_loader_service
from services.chunking_service import chunking_service
from services.embedding_service import embedding_service
from services.vector_store_service import vector_store_service
from services.rag_service import rag_service
from services.tools_service import tools_service

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI GitHub Repository Coding Assistant API",
    version="1.0.0",
    description="Backend API for indexing and querying GitHub repositories with RAG & Tool Calling"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class ChatRequest(BaseModel):
    message: str = Field(..., description="User prompt or coding question")
    system_instruction: Optional[str] = Field(None, description="Optional custom system prompt")

class ChatResponse(BaseModel):
    reply: str
    model: str

class IngestRepoRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")

class FileMetadata(BaseModel):
    file_name: str
    file_path: str
    extension: str
    size_bytes: int

class IngestRepoResponse(BaseModel):
    owner: str
    repo_name: str
    repo_url: str
    total_files: int
    total_size_kb: float
    file_extensions: Dict[str, int]
    preview_files: List[FileMetadata]

class LoadDocumentsRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")

class DocumentMetadataPreview(BaseModel):
    file_name: str
    file_path: str
    file_type: str
    repo_name: str
    character_count: int
    line_count: int
    sample_content: str

class LoadDocumentsResponse(BaseModel):
    total_documents: int
    total_characters: int
    documents_preview: List[DocumentMetadataPreview]

class ChunkDocumentsRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")
    chunk_size: Optional[int] = Field(1000, description="Target character size per chunk")
    chunk_overlap: Optional[int] = Field(200, description="Character overlap between consecutive chunks")

class ChunkMetadataPreview(BaseModel):
    chunk_id: str
    file_name: str
    file_path: str
    chunk_index: int
    character_count: int
    sample_content: str

class ChunkDocumentsResponse(BaseModel):
    total_parent_documents: int
    total_chunks: int
    average_chunk_size: float
    chunk_size_used: int
    chunk_overlap_used: int
    chunks_preview: List[ChunkMetadataPreview]

class GenerateEmbeddingsRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")
    chunk_size: Optional[int] = Field(1000, description="Target character size per chunk")
    chunk_overlap: Optional[int] = Field(200, description="Character overlap between consecutive chunks")

class GenerateEmbeddingsResponse(BaseModel):
    total_chunks_embedded: int
    vector_dimensions: int
    model_used: str
    sample_vector_preview: List[float]
    sample_chunk_id: str
    sample_chunk_text: str

class BuildFaissIndexRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")
    chunk_size: Optional[int] = Field(1000, description="Target character size per chunk")
    chunk_overlap: Optional[int] = Field(200, description="Character overlap between consecutive chunks")

class BuildFaissIndexResponse(BaseModel):
    status: str
    total_chunks_indexed: int
    save_directory: str
    owner: str
    repo_name: str

class SearchFaissRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")
    query: str = Field(..., description="Developer search question or phrase")
    top_k: Optional[int] = Field(4, description="Number of top matching code chunks to retrieve")

class SearchResultChunk(BaseModel):
    chunk_id: str
    file_name: str
    file_path: str
    chunk_index: int
    similarity_score: float
    content: str

class SearchFaissResponse(BaseModel):
    query: str
    top_k: int
    total_matches: int
    retrieved_chunks: List[SearchResultChunk]

class ChatHistoryMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")

class RagQueryRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")
    user_question: str = Field(..., description="Developer question about the codebase")
    history: Optional[List[ChatHistoryMessage]] = Field([], description="Prior conversation history turns")
    top_k: Optional[int] = Field(4, description="Number of retrieved chunks to include in RAG context")

class RagQueryResponse(BaseModel):
    answer: str
    sources: List[str]
    model_used: str
    retrieved_chunks_count: int
    retrieved_chunks: List[SearchResultChunk]

# Phase 12 Tool Schemas
class SearchCodeToolRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")
    keyword: str = Field(..., description="Keyword string to search in source code")

class GetFileContentToolRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")
    file_path: str = Field(..., description="Relative file path (e.g. src/app.py)")

class RepoStructureToolRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")

@app.get("/")
def read_root():
    return {"message": "Welcome to AI GitHub Repository Coding Assistant API"}

@app.get("/api/health")
def check_health():
    return {
        "status": "ok",
        "message": "Backend connected successfully"
    }

@app.post("/api/chat", response_model=ChatResponse)
def basic_chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")
    
    try:
        reply = llm_service.generate_response(
            user_message=request.message,
            system_instruction=request.system_instruction
        )
        return ChatResponse(
            reply=reply,
            model=llm_service.model_name
        )
    except ValueError as ve:
        raise HTTPException(status_code=401, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while calling the LLM API: {str(e)}"
        )

@app.post("/api/ingest-repo", response_model=IngestRepoResponse)
def ingest_repository(request: IngestRepoRequest):
    if not request.repo_url.strip():
        raise HTTPException(status_code=400, detail="Repository URL cannot be empty.")
    
    try:
        local_path = github_service.clone_repository(request.repo_url)
        repo_data = github_service.inspect_repository(local_path, request.repo_url)
        preview = repo_data["files"][:30]
        
        return IngestRepoResponse(
            owner=repo_data["owner"],
            repo_name=repo_data["repo_name"],
            repo_url=repo_data["repo_url"],
            total_files=repo_data["total_files"],
            total_size_kb=repo_data["total_size_kb"],
            file_extensions=repo_data["file_extensions"],
            preview_files=[
                FileMetadata(
                    file_name=f["file_name"],
                    file_path=f["file_path"],
                    extension=f["extension"],
                    size_bytes=f["size_bytes"]
                ) for f in preview
            ]
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process GitHub repository: {str(e)}"
        )

@app.post("/api/load-documents", response_model=LoadDocumentsResponse)
def load_documents(request: LoadDocumentsRequest):
    if not request.repo_url.strip():
        raise HTTPException(status_code=400, detail="Repository URL cannot be empty.")

    try:
        local_path = github_service.clone_repository(request.repo_url)
        repo_data = github_service.inspect_repository(local_path, request.repo_url)
        documents = document_loader_service.load_repository_documents(repo_data)
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        preview_list = []
        for doc in documents[:10]:
            lines = doc.page_content.splitlines()
            sample = "\n".join(lines[:4])
            preview_list.append(
                DocumentMetadataPreview(
                    file_name=doc.metadata["file_name"],
                    file_path=doc.metadata["file_path"],
                    file_type=doc.metadata["file_type"],
                    repo_name=doc.metadata["repo_name"],
                    character_count=len(doc.page_content),
                    line_count=len(lines),
                    sample_content=sample
                )
            )

        return LoadDocumentsResponse(
            total_documents=len(documents),
            total_characters=total_chars,
            documents_preview=preview_list
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load repository documents: {str(e)}"
        )

@app.post("/api/chunk-documents", response_model=ChunkDocumentsResponse)
def chunk_documents(request: ChunkDocumentsRequest):
    if not request.repo_url.strip():
        raise HTTPException(status_code=400, detail="Repository URL cannot be empty.")

    try:
        local_path = github_service.clone_repository(request.repo_url)
        repo_data = github_service.inspect_repository(local_path, request.repo_url)
        documents = document_loader_service.load_repository_documents(repo_data)
        
        chunks = chunking_service.chunk_documents(
            documents=documents,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        total_chunks = len(chunks)
        avg_size = (sum(len(c.page_content) for c in chunks) / total_chunks) if total_chunks > 0 else 0

        preview_list = []
        for chunk in chunks[:12]:
            lines = chunk.page_content.splitlines()
            sample = "\n".join(lines[:5])
            preview_list.append(
                ChunkMetadataPreview(
                    chunk_id=chunk.metadata.get("chunk_id", ""),
                    file_name=chunk.metadata.get("file_name", ""),
                    file_path=chunk.metadata.get("file_path", ""),
                    chunk_index=chunk.metadata.get("chunk_index", 0),
                    character_count=len(chunk.page_content),
                    sample_content=sample
                )
            )

        return ChunkDocumentsResponse(
            total_parent_documents=len(documents),
            total_chunks=total_chunks,
            average_chunk_size=round(avg_size, 1),
            chunk_size_used=request.chunk_size or 1000,
            chunk_overlap_used=request.chunk_overlap or 200,
            chunks_preview=preview_list
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to chunk documents: {str(e)}"
        )

@app.post("/api/generate-embeddings", response_model=GenerateEmbeddingsResponse)
def generate_embeddings(request: GenerateEmbeddingsRequest):
    if not request.repo_url.strip():
        raise HTTPException(status_code=400, detail="Repository URL cannot be empty.")

    try:
        local_path = github_service.clone_repository(request.repo_url)
        repo_data = github_service.inspect_repository(local_path, request.repo_url)
        documents = document_loader_service.load_repository_documents(repo_data)
        chunks = chunking_service.chunk_documents(
            documents=documents,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        result = embedding_service.generate_chunk_embeddings(chunks)

        sample_chunk = chunks[0] if chunks else None
        sample_text = sample_chunk.page_content[:150] if sample_chunk else ""
        sample_id = sample_chunk.metadata.get("chunk_id", "") if sample_chunk else ""

        return GenerateEmbeddingsResponse(
            total_chunks_embedded=result["total_chunks_embedded"],
            vector_dimensions=result["vector_dimensions"],
            model_used=result["model_used"],
            sample_vector_preview=result["sample_vector_preview"],
            sample_chunk_id=sample_id,
            sample_chunk_text=sample_text
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {str(e)}"
        )

@app.post("/api/build-faiss-index", response_model=BuildFaissIndexResponse)
def build_faiss_index(request: BuildFaissIndexRequest):
    if not request.repo_url.strip():
        raise HTTPException(status_code=400, detail="Repository URL cannot be empty.")

    try:
        local_path = github_service.clone_repository(request.repo_url)
        repo_data = github_service.inspect_repository(local_path, request.repo_url)
        documents = document_loader_service.load_repository_documents(repo_data)
        chunks = chunking_service.chunk_documents(
            documents=documents,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        embeddings_model = embedding_service.get_embeddings_model()

        result = vector_store_service.build_and_save_index(
            owner=repo_data["owner"],
            repo_name=repo_data["repo_name"],
            chunks=chunks,
            embeddings_model=embeddings_model
        )

        return BuildFaissIndexResponse(
            status=result["status"],
            total_chunks_indexed=result["total_chunks_indexed"],
            save_directory=result["save_directory"],
            owner=result["owner"],
            repo_name=result["repo_name"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build FAISS index: {str(e)}"
        )

@app.post("/api/search-faiss", response_model=SearchFaissResponse)
def search_faiss(request: SearchFaissRequest):
    if not request.repo_url.strip():
        raise HTTPException(status_code=400, detail="Repository URL cannot be empty.")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        url_info = github_service.validate_github_url(request.repo_url)
        embeddings_model = embedding_service.get_embeddings_model()
        top_k = request.top_k or 4

        results = vector_store_service.search_similarity(
            owner=url_info["owner"],
            repo_name=url_info["repo"],
            query=request.query,
            top_k=top_k,
            embeddings_model=embeddings_model
        )

        return SearchFaissResponse(
            query=request.query,
            top_k=top_k,
            total_matches=len(results),
            retrieved_chunks=[
                SearchResultChunk(
                    chunk_id=r["chunk_id"],
                    file_name=r["file_name"],
                    file_path=r["file_path"],
                    chunk_index=r["chunk_index"],
                    similarity_score=r["similarity_score"],
                    content=r["content"]
                ) for r in results
            ]
        )
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail=str(fnfe))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search FAISS index: {str(e)}"
        )

@app.post("/api/rag-query", response_model=RagQueryResponse)
def rag_query(request: RagQueryRequest):
    if not request.repo_url.strip():
        raise HTTPException(status_code=400, detail="Repository URL cannot be empty.")
    if not request.user_question.strip():
        raise HTTPException(status_code=400, detail="User question cannot be empty.")

    try:
        top_k = request.top_k or 4
        history_dicts = [
            {"role": h.role, "content": h.content} for h in (request.history or [])
        ]

        result = rag_service.answer_question(
            repo_url=request.repo_url,
            user_question=request.user_question,
            history=history_dicts,
            top_k=top_k
        )

        chunks_list = [
            SearchResultChunk(
                chunk_id=r.get("chunk_id", ""),
                file_name=r.get("file_name", ""),
                file_path=r.get("file_path", ""),
                chunk_index=r.get("chunk_index", 0),
                similarity_score=r.get("similarity_score", 0.0),
                content=r.get("content", "")
            ) for r in result["retrieved_chunks"]
        ]

        return RagQueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            model_used=result["model_used"],
            retrieved_chunks_count=len(chunks_list),
            retrieved_chunks=chunks_list
        )
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail=str(fnfe))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process RAG query: {str(e)}"
        )

# Phase 12: Tool Endpoints
@app.post("/api/tools/search-code")
def tool_search_code(request: SearchCodeToolRequest):
    """Tool 1: Search exact keyword occurrences across repository code."""
    try:
        return tools_service.search_code(request.repo_url, request.keyword)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tools/get-file-content")
def tool_get_file_content(request: GetFileContentToolRequest):
    """Tool 2: Retrieve full raw text content of a target file by path."""
    try:
        return tools_service.get_file_content(request.repo_url, request.file_path)
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail=str(fnfe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tools/repo-structure")
def tool_repo_structure(request: RepoStructureToolRequest):
    """Tool 3: Retrieve high-level file tree structure of the repository."""
    try:
        return tools_service.get_repository_structure(request.repo_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run("main:app", host=host, port=port, reload=True)

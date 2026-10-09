from typing import Dict, List, Any, Optional
from services.vector_store_service import vector_store_service
from services.embedding_service import embedding_service
from services.llm_service import llm_service
from services.github_service import github_service

class RAGService:
    def answer_question(
        self,
        repo_url: str,
        user_question: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Grounded RAG Pipeline with Repository Structure Metadata & Conversation Memory.
        Enables answering file-structure questions as well as semantic code queries.
        """
        url_info = github_service.validate_github_url(repo_url)
        owner = url_info["owner"]
        repo_name = url_info["repo"]
        history = history or []

        # Step 1: Inspect repository to get file tree structure metadata
        local_path = github_service.clone_repository(repo_url)
        repo_data = github_service.inspect_repository(local_path, repo_url)
        file_tree = [f["file_path"] for f in repo_data.get("files", [])]
        file_tree_str = "\n".join([f"- {f}" for f in file_tree[:40]])

        # Step 2: Contextualize search query for vector retrieval
        search_query = user_question
        if history:
            recent_turns = history[-4:]
            history_summary = " ".join([f"{msg['role']}: {msg['content']}" for msg in recent_turns])
            search_query = f"{history_summary} Current Question: {user_question}"

        # Step 3: Retrieve Top-K code chunks from FAISS
        embeddings_model = embedding_service.get_embeddings_model()
        retrieved_results = vector_store_service.search_similarity(
            owner=owner,
            repo_name=repo_name,
            query=search_query,
            top_k=top_k,
            embeddings_model=embeddings_model
        )

        # Step 4: Format retrieved code snippets & sources
        context_blocks = []
        unique_sources: List[str] = []

        for idx, item in enumerate(retrieved_results, 1):
            file_path = item.get("file_path", "unknown file")
            content = item.get("content", "")
            
            if file_path not in unique_sources:
                unique_sources.append(file_path)

            context_blocks.append(
                f"--- CODE SNIPPET #{idx} [File: {file_path}] ---\n{content}\n"
            )

        formatted_context = "\n".join(context_blocks)

        # Step 5: Format conversation history dialogue
        formatted_history = ""
        if history:
            history_lines = []
            for msg in history[-6:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_lines.append(f"{role}: {msg.get('content', '')}")
            formatted_history = "PRIOR CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"

        # Step 6: Construct System Instruction & RAG Prompt with File Tree Context
        system_instruction = (
            "You are an expert AI Coding Assistant analyzing a GitHub repository.\n"
            "You are provided with both:\n"
            "1. The REPOSITORY FILE TREE (list of files in the codebase).\n"
            "2. RETRIEVED CODE SNIPPETS from vector similarity search.\n\n"
            "Answering Rules:\n"
            "- If the user asks about file existence, project structure, or where files are located, refer to the REPOSITORY FILE TREE.\n"
            "- If the user asks about code logic, functions, or implementation, use the RETRIEVED CODE SNIPPETS.\n"
            "- Use the prior conversation history to resolve follow-up references like 'that class', 'it', or 'the previous file'.\n"
            "- If the answer cannot be determined from the repository file tree or code snippets, state clearly: "
            "'Based on the provided repository context, this information is not available.'"
        )

        rag_prompt = (
            f"REPOSITORY FILE TREE ({repo_data['total_files']} files total):\n"
            f"{file_tree_str}\n\n"
            f"RETRIEVED CODE SNIPPETS:\n"
            f"{formatted_context}\n\n"
            f"{formatted_history}"
            f"CURRENT USER QUESTION:\n"
            f"{user_question}"
        )

        # Step 7: Generate LLM Response
        answer = llm_service.generate_response(
            user_message=rag_prompt,
            system_instruction=system_instruction
        )

        return {
            "answer": answer,
            "sources": unique_sources,
            "retrieved_chunks": retrieved_results,
            "model_used": llm_service.model_name
        }

rag_service = RAGService()

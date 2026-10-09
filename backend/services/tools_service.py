import os
import re
from typing import Dict, List, Any
from services.github_service import github_service

class ToolsService:
    """
    Phase 12: Basic Tool Calling Engine.
    Exposes explicit functional tools that the AI or backend can invoke on demand:
    1. search_code(keyword)
    2. get_file_content(file_path)
    3. get_repository_structure()
    """

    def search_code(self, repo_url: str, keyword: str) -> Dict[str, Any]:
        """
        Tool 1: Searches for exact keyword occurrences across all repository source files.
        """
        url_info = github_service.validate_github_url(repo_url)
        local_path = github_service.clone_repository(repo_url)
        repo_data = github_service.inspect_repository(local_path, repo_url)

        matches = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        for file_info in repo_data.get("files", []):
            abs_path = file_info["absolute_path"]
            rel_path = file_info["file_path"]

            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    if pattern.search(line):
                        matches.append({
                            "file_path": rel_path,
                            "line_number": line_num,
                            "line_content": line.strip()
                        })
                        if len(matches) >= 30:  # Cap at 30 matches max
                            break
            except Exception:
                continue

        return {
            "tool": "search_code",
            "keyword": keyword,
            "total_matches": len(matches),
            "matches": matches
        }

    def get_file_content(self, repo_url: str, file_path: str) -> Dict[str, Any]:
        """
        Tool 2: Fetches full or partial text content of a specific file by its path.
        """
        local_path = github_service.clone_repository(repo_url)
        clean_path = file_path.strip().lstrip("/").replace("\\", "/")
        target_file_path = os.path.join(local_path, clean_path)

        if not os.path.exists(target_file_path):
            raise FileNotFoundError(f"File '{file_path}' not found in repository.")

        try:
            with open(target_file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            lines = content.splitlines()
            return {
                "tool": "get_file_content",
                "file_path": clean_path,
                "total_lines": len(lines),
                "total_characters": len(content),
                "content": content
            }
        except Exception as e:
            raise RuntimeError(f"Could not read file '{file_path}': {str(e)}")

    def get_repository_structure(self, repo_url: str) -> Dict[str, Any]:
        """
        Tool 3: Generates a high-level directory tree structure of the repository.
        """
        local_path = github_service.clone_repository(repo_url)
        repo_data = github_service.inspect_repository(local_path, repo_url)

        file_tree = [f["file_path"] for f in repo_data.get("files", [])]

        return {
            "tool": "get_repository_structure",
            "owner": repo_data["owner"],
            "repo_name": repo_data["repo_name"],
            "total_files": repo_data["total_files"],
            "file_tree": file_tree[:50]  # First 50 files for structure view
        }

tools_service = ToolsService()

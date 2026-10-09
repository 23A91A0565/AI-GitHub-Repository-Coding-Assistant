import os
import re
import shutil
import git
from typing import Dict, List, Any

# Supported source code and documentation file extensions
SUPPORTED_EXTENSIONS = {
    ".py", ".java", ".js", ".ts", ".jsx", ".tsx",
    ".md", ".json", ".xml", ".yml", ".yaml", ".txt",
    ".html", ".css", ".go", ".rs", ".c", ".cpp", ".h",
    ".cs", ".php", ".rb", ".sh", ".sql"
}

# Directories that must be ignored during repository scanning
IGNORED_DIRECTORIES = {
    ".git", "node_modules", "target", "build", "dist",
    "venv", ".venv", "env", "__pycache__", ".idea", ".vscode",
    "coverage", ".next", ".nuxt", "bin", "obj"
}

# Maximum allowed size for an individual source file (1 MB)
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024

class GitHubService:
    def __init__(self, data_dir: str = "data/repos"):
        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

    def validate_github_url(self, repo_url: str) -> Dict[str, str]:
        """
        Validate and parse a GitHub repository URL.
        Example: https://github.com/fastapi/fastapi -> owner: fastapi, repo: fastapi
        """
        # Clean URL
        clean_url = repo_url.strip().rstrip("/")
        if clean_url.endswith(".git"):
            clean_url = clean_url[:-4]

        pattern = r"^https?://github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)$"
        match = re.match(pattern, clean_url)

        if not match:
            raise ValueError(
                "Invalid GitHub URL format. Expected format: https://github.com/owner/repository"
            )

        owner, repo = match.groups()
        return {
            "owner": owner,
            "repo": repo,
            "clean_url": f"https://github.com/{owner}/{repo}.git"
        }

    def clone_repository(self, repo_url: str) -> str:
        """
        Clones or updates a public GitHub repository locally into backend/data/repos.
        Returns the absolute local path to the cloned repository.
        """
        url_info = self.validate_github_url(repo_url)
        target_dir_name = f"{url_info['owner']}_{url_info['repo']}"
        target_path = os.path.join(self.data_dir, target_dir_name)

        # Clone fresh if directory does not exist
        if not os.path.exists(target_path):
            try:
                git.Repo.clone_from(url_info["clean_url"], target_path, depth=1)
            except Exception as e:
                raise RuntimeError(f"Failed to clone GitHub repository: {str(e)}")
        else:
            # If repo already cloned, attempt a quick fetch/pull
            try:
                repo_obj = git.Repo(target_path)
                origin = repo_obj.remotes.origin
                origin.pull()
            except Exception:
                # If pull fails, re-clone fresh
                shutil.rmtree(target_path, ignore_errors=True)
                git.Repo.clone_from(url_info["clean_url"], target_path, depth=1)

        return target_path

    def inspect_repository(self, repo_path: str, repo_url: str) -> Dict[str, Any]:
        """
        Scans local repo directory, filters noise & binary files, and returns metadata & file statistics.
        """
        url_info = self.validate_github_url(repo_url)
        processed_files: List[Dict[str, Any]] = []
        extension_counts: Dict[str, int] = {}
        total_size_bytes = 0

        for root, dirs, files in os.walk(repo_path):
            # Prune ignored directories in-place so os.walk skips them entirely
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES and not d.startswith(".")]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                
                # Filter by supported extensions
                if ext in SUPPORTED_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, repo_path).replace("\\", "/")
                    
                    try:
                        file_size = os.path.getsize(full_path)
                        
                        # Skip oversized files (e.g. huge minified bundles)
                        if file_size > MAX_FILE_SIZE_BYTES or file_size == 0:
                            continue

                        extension_counts[ext] = extension_counts.get(ext, 0) + 1
                        total_size_bytes += file_size
                        
                        processed_files.append({
                            "file_name": file,
                            "file_path": rel_path,
                            "extension": ext,
                            "size_bytes": file_size,
                            "absolute_path": full_path
                        })
                    except OSError:
                        continue

        return {
            "owner": url_info["owner"],
            "repo_name": url_info["repo"],
            "repo_url": repo_url,
            "local_path": repo_path,
            "total_files": len(processed_files),
            "total_size_kb": round(total_size_bytes / 1024, 2),
            "file_extensions": extension_counts,
            "files": processed_files
        }

github_service = GitHubService()

import os
import json
import subprocess
import httpx
from typing import Dict, Any, List, Optional

class ToolManager:
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = os.path.abspath(workspace_root)
        self.allowed_domains = ["github.com", "docs.python.org", "pypi.org", "developer.mozilla.org", "wikipedia.org"]

    def set_allowed_domains(self, domains: List[str]):
        self.allowed_domains = domains

    def add_allowed_domain(self, domain: str):
        if domain and domain not in self.allowed_domains:
            self.allowed_domains.append(domain)

    def _is_safe_path(self, full_path: str) -> bool:
        try:
            return os.path.commonpath([full_path, self.workspace_root]) == self.workspace_root
        except ValueError:
            return False

    def classify_tool_risk(self, tool_name: str) -> str:
        low_risk = ["read_file", "list_files", "search_workspace", "internet_search", "git_status", "git_diff", "git_log"]
        consequential = ["write_file", "patch_file", "git_branch", "git_commit", "git_rollback"]
        high_risk = ["run_terminal_cmd"]

        if tool_name in low_risk:
            return "low"
        elif tool_name in consequential:
            return "consequential"
        elif tool_name in high_risk:
            return "high"
        return "consequential"

    def read_file(self, filepath: str) -> Dict[str, Any]:
        full_path = os.path.abspath(os.path.join(self.workspace_root, filepath))
        if not self._is_safe_path(full_path):
            return {"success": False, "error": "Access outside workspace denied."}
        if not os.path.exists(full_path):
            return {"success": False, "error": f"File not found: {filepath}"}
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return {"success": True, "filepath": filepath, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_files(self, rel_dir: str = ".") -> Dict[str, Any]:
        full_path = os.path.abspath(os.path.join(self.workspace_root, rel_dir))
        if not self._is_safe_path(full_path):
            return {"success": False, "error": "Access outside workspace denied."}
        try:
            items = []
            for entry in os.listdir(full_path):
                if entry.startswith(".") or entry in ["node_modules", "__pycache__", "venv"]:
                    continue
                p = os.path.join(full_path, entry)
                items.append({
                    "name": entry,
                    "is_dir": os.path.isdir(p),
                    "path": os.path.relpath(p, self.workspace_root)
                })
            return {"success": True, "items": items}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_workspace(self, query: str) -> Dict[str, Any]:
        results = []
        try:
            for root, dirs, files in os.walk(self.workspace_root):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["node_modules", "__pycache__", "venv"]]
                for file in files:
                    if file.startswith("."):
                        continue
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, self.workspace_root)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            for idx, line in enumerate(f, 1):
                                if query.lower() in line.lower():
                                    results.append({
                                        "filepath": rel_path,
                                        "line": idx,
                                        "content": line.strip()
                                    })
                                    if len(results) >= 50:
                                        break
                    except Exception:
                        pass
            return {"success": True, "query": query, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def internet_search(self, query: str, domain_filter: Optional[str] = None) -> Dict[str, Any]:
        domains = self.allowed_domains
        if domain_filter:
            domains = [domain_filter]

        return {
            "success": True,
            "query": query,
            "allowed_domains": domains,
            "results": [
                {
                    "title": f"Documentation resource for '{query}'",
                    "url": f"https://{domains[0] if domains else 'docs.python.org'}/search?q={query}",
                    "snippet": f"Official reference material regarding {query} within approved domain policy."
                }
            ]
        }

    def git_status(self) -> Dict[str, Any]:
        try:
            res = subprocess.check_output(["git", "status", "--porcelain"], cwd=self.workspace_root, text=True)
            branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=self.workspace_root, text=True).strip()
            return {"success": True, "branch": branch, "changes": res.strip().split("\n") if res.strip() else []}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def git_diff(self) -> Dict[str, Any]:
        try:
            res = subprocess.check_output(["git", "diff"], cwd=self.workspace_root, text=True)
            return {"success": True, "diff": res}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def write_file(self, filepath: str, content: str) -> Dict[str, Any]:
        full_path = os.path.abspath(os.path.join(self.workspace_root, filepath))
        if not self._is_safe_path(full_path):
            return {"success": False, "error": "Access outside workspace denied."}
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "filepath": filepath, "bytes_written": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_terminal_cmd(self, command: str) -> Dict[str, Any]:
        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "success": res.returncode == 0,
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

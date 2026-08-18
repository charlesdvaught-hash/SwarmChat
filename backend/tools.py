import logging
import os
import sys
import json
import shutil
import subprocess
import py_compile
import time
import httpx
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ToolManager:
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = os.path.abspath(workspace_root)
        self.bot_workspaces_dir = os.path.join(self.workspace_root, ".swarmchat", "workspaces")
        os.makedirs(self.bot_workspaces_dir, exist_ok=True)

        self.allowed_domains = [
            "github.com",
            "docs.python.org",
            "pypi.org",
            "developer.mozilla.org",
            "wikipedia.org",
            "huggingface.co",
            "hf.co"
        ]

    def set_allowed_domains(self, domains: List[str]):
        self.allowed_domains = domains

    def add_allowed_domain(self, domain: str):
        if domain and domain not in self.allowed_domains:
            self.allowed_domains.append(domain)

    def get_bot_workspace_dir(self, bot_id: str) -> str:
        bot_dir = os.path.join(self.bot_workspaces_dir, bot_id)
        os.makedirs(bot_dir, exist_ok=True)
        return bot_dir

    def _is_safe_path(self, full_path: str, root_dir: Optional[str] = None) -> bool:
        target_root = root_dir or self.workspace_root
        try:
            return os.path.commonpath([full_path, target_root]) == target_root
        except ValueError:
            return False

    def classify_tool_risk(self, tool_name: str) -> str:
        low_risk = [
            "read_file", "list_files", "search_workspace", "internet_search", "web_fetch", "web_open",
            "search_huggingface", "git_status", "git_diff", "git_log", "run_python", "run_tests", "condense_workspace_code"
        ]
        consequential = [
            "write_file", "patch_file", "copy_file", "git_branch", "git_commit", "git_rollback",
            "bot_workspace_write", "bot_workspace_merge"
        ]
        high_risk = ["run_terminal_cmd"]

        if tool_name in low_risk:
            return "low"
        elif tool_name in consequential:
            return "consequential"
        elif tool_name in high_risk:
            return "high"
        return "consequential"

    def patch_file(
        self,
        filepath: str,
        search_block: str,
        replace_block: str,
        bot_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Applies a targeted search-and-replace block edit to a workspace or bot file after pre-validation."""
        root = self.get_bot_workspace_dir(bot_id) if bot_id else self.workspace_root
        full_path = os.path.abspath(os.path.join(root, filepath))
        if not os.path.exists(full_path):
            full_path = os.path.abspath(os.path.join(self.workspace_root, filepath))

        if not self._is_safe_path(full_path, self.workspace_root) and not self._is_safe_path(full_path, self.bot_workspaces_dir):
            return {"success": False, "error": f"Access outside workspace denied for '{filepath}'."}

        if not os.path.exists(full_path):
            return {"success": False, "error": f"Target file '{filepath}' does not exist."}

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if search_block not in content:
                # Try targeted line-level normalized whitespace comparison
                lines = content.splitlines(keepends=True)
                search_lines = search_block.splitlines()
                matched_idx = -1
                for idx in range(len(lines) - len(search_lines) + 1):
                    sub = [l.rstrip("\r\n").rstrip() for l in lines[idx:idx+len(search_lines)]]
                    target_sub = [l.rstrip() for l in search_lines]
                    if sub == target_sub:
                        matched_idx = idx
                        break

                if matched_idx != -1:
                    rep_str = replace_block if replace_block.endswith("\n") else replace_block + "\n"
                    lines[matched_idx:matched_idx+len(search_lines)] = [rep_str]
                    new_content = "".join(lines)
                else:
                    return {
                        "success": False,
                        "error": f"Search block not found in '{filepath}'. Please check line indentation and content exactness."
                    }
            else:
                new_content = content.replace(search_block, replace_block, 1)

            # Pre-validate syntax before committing patch
            temp_path = f"{full_path}.patch_test"
            with open(temp_path, "w", encoding="utf-8") as tf:
                tf.write(new_content)

            syntax_check = self.validate_file_syntax(temp_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if not syntax_check.get("valid"):
                return {
                    "success": False,
                    "error": f"Patch rejected due to syntax error after application: {syntax_check.get('error')}"
                }

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return {
                "success": True,
                "filepath": filepath,
                "bot_id": bot_id,
                "patched": True,
                "bytes_written": len(new_content)
            }
        except OSError as e:
            logger.warning("Patch failed for %s: %s", filepath, e)
            return {"success": False, "error": f"Failed to patch '{filepath}': {e}"}

    async def web_fetch(self, url: str, max_chars: int = 4000) -> Dict[str, Any]:
        """Fetches web page content, extracts clean text, and truncates to max_chars as a structured observation."""
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            return {"success": False, "error": "Invalid URL format. URL must start with http:// or https://"}

        try:
            async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "SwarmChat-Agent/1.0"})
        except httpx.HTTPError as e:
            logger.warning("Web fetch failed for URL '%s': %s", url, e)
            return {"success": False, "url": url, "error": f"HTTP fetch request failed: {e}"}

        if resp.status_code != 200:
            return {"success": False, "url": url, "error": f"Web fetch returned HTTP {resp.status_code}."}

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove scripts, styles, and non-content elements
            for elem in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                elem.decompose()

            title = soup.title.string.strip() if soup.title and soup.title.string else url
            text = soup.get_text(separator="\n", strip=True)

            # Compact consecutive newlines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

            truncated = clean_text[:max_chars] + ("... [truncated page content]" if len(clean_text) > max_chars else "")

            return {
                "success": True,
                "url": url,
                "title": title,
                "content": truncated,
                "char_count": len(clean_text)
            }
        except Exception as e:
            logger.warning("Parsing web fetch content failed for '%s': %s", url, e)
            return {"success": False, "url": url, "error": f"Failed to parse web page content: {e}"}

    async def web_open(self, url: str, max_chars: int = 4000) -> Dict[str, Any]:
        """Alias for web_fetch."""
        return await self.web_fetch(url, max_chars)

    def run_python(self, filepath: str, bot_id: str, timeout: int = 10) -> Dict[str, Any]:
        """Runs a python file inside the bot's workspace sandbox without shell=True."""
        bot_dir = self.get_bot_workspace_dir(bot_id)
        full_path = os.path.abspath(os.path.join(bot_dir, filepath))
        if not os.path.exists(full_path):
            full_path = os.path.abspath(os.path.join(self.workspace_root, filepath))

        if not os.path.exists(full_path):
            return {"success": False, "error": f"File '{filepath}' not found."}

        try:
            res = subprocess.run(
                [sys.executable, full_path],
                cwd=bot_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": res.returncode == 0,
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "filepath": filepath,
                "bot_id": bot_id,
                "error": None if res.returncode == 0 else f"Process exited with return code {res.returncode}"
            }
        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "timed_out": True,
                "error": f"Execution timed out after {timeout}s",
                "stdout": e.stdout if isinstance(e.stdout, str) else "",
                "stderr": e.stderr if isinstance(e.stderr, str) else ""
            }
        except Exception as e:
            return {"success": False, "error": f"Execution failed: {e}"}

    def run_tests(self, bot_id: str, test_path: Optional[str] = None, timeout: int = 15) -> Dict[str, Any]:
        """Runs pytest on bot's workspace sandbox without shell=True."""
        bot_dir = self.get_bot_workspace_dir(bot_id)
        cmd = [sys.executable, "-m", "pytest"]
        if test_path:
            cmd.append(test_path)

        try:
            res = subprocess.run(
                cmd,
                cwd=bot_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": res.returncode == 0,
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "bot_id": bot_id,
                "error": None if res.returncode == 0 else f"Pytest exited with return code {res.returncode}"
            }
        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "timed_out": True,
                "error": f"Pytest timed out after {timeout}s",
                "stdout": e.stdout if isinstance(e.stdout, str) else "",
                "stderr": e.stderr if isinstance(e.stderr, str) else ""
            }
        except Exception as e:
            return {"success": False, "error": f"Pytest execution failed: {e}"}

    def read_file(self, filepath: str, bot_id: Optional[str] = None) -> Dict[str, Any]:
        root = self.get_bot_workspace_dir(bot_id) if bot_id else self.workspace_root
        full_path = os.path.abspath(os.path.join(root, filepath))
        if not os.path.exists(full_path):
            # Fallback to main workspace if not in bot workspace
            full_path = os.path.abspath(os.path.join(self.workspace_root, filepath))

        if not self._is_safe_path(full_path, self.workspace_root) and not self._is_safe_path(full_path, self.bot_workspaces_dir):
            return {"success": False, "error": "Access outside workspace denied."}
        if not os.path.exists(full_path):
            return {"success": False, "error": f"File not found: {filepath}"}
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return {"success": True, "filepath": filepath, "content": content}
        except OSError as e:
            logger.warning("Failed to read %s: %s", full_path, e)
            return {"success": False, "error": f"Failed to read '{filepath}': {e}"}

    def list_files(self, rel_dir: str = ".", bot_id: Optional[str] = None) -> Dict[str, Any]:
        root = self.get_bot_workspace_dir(bot_id) if bot_id else self.workspace_root
        full_path = os.path.abspath(os.path.join(root, rel_dir))
        if not os.path.exists(full_path):
            full_path = os.path.abspath(os.path.join(self.workspace_root, rel_dir))

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
        except OSError as e:
            logger.warning("Failed to list %s: %s", full_path, e)
            return {"success": False, "error": f"Failed to list '{rel_dir}': {e}"}

    def search_workspace(self, query: str) -> Dict[str, Any]:
        results = []
        skipped: List[Dict[str, str]] = []
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
                    except OSError as e:
                        logger.debug("Skipping unreadable file %s during search: %s", rel_path, e)
                        skipped.append({"filepath": rel_path, "error": str(e)})
            return {
                "success": True,
                "query": query,
                "results": results,
                "skipped_files": skipped,
                "partial": bool(skipped)
            }
        except OSError as e:
            logger.warning("Workspace search failed: %s", e)
            return {"success": False, "error": f"Workspace search failed: {e}"}

    async def internet_search(self, query: str, domain_filter: Optional[str] = None) -> Dict[str, Any]:
        domains = self.allowed_domains
        if domain_filter:
            domains = [domain_filter]

        if "huggingface" in query.lower() or "hf" in query.lower():
            return await self.search_huggingface(query)

        results = []
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.get(f"https://html.duckduckgo.com/html/?q={query}")
        except httpx.HTTPError as e:
            logger.warning("Internet search request failed for '%s': %s", query, e)
            return {
                "success": False,
                "query": query,
                "allowed_domains": domains,
                "error": f"Internet search request failed: {e}"
            }

        if resp.status_code != 200:
            logger.warning("Internet search returned HTTP %s for '%s'", resp.status_code, query)
            return {
                "success": False,
                "query": query,
                "allowed_domains": domains,
                "error": f"Internet search returned HTTP {resp.status_code}."
            }

        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            logger.error("beautifulsoup4 is required for internet search: %s", e)
            return {
                "success": False,
                "query": query,
                "allowed_domains": domains,
                "error": "beautifulsoup4 is not installed; cannot parse search results."
            }

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__url", limit=5):
            href = a.get("href", "")
            title = a.get_text(strip=True)
            results.append({"title": title, "url": href, "snippet": title})

        return {
            "success": True,
            "query": query,
            "allowed_domains": domains,
            "results": results
        }

    async def search_huggingface(self, query: str, limit: int = 5) -> Dict[str, Any]:
        clean_q = query.replace("huggingface", "").replace("hf", "").strip() or query
        url = f"https://huggingface.co/api/models?search={clean_q}&limit={limit}&full=true"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
        except httpx.HTTPError as e:
            logger.warning("HuggingFace search request failed for '%s': %s", clean_q, e)
            return {
                "success": False,
                "query": clean_q,
                "count": 0,
                "models": [],
                "error": f"HuggingFace search request failed: {e}"
            }

        if resp.status_code != 200:
            logger.warning("HuggingFace search returned HTTP %s for '%s'", resp.status_code, clean_q)
            return {
                "success": False,
                "query": clean_q,
                "count": 0,
                "models": [],
                "error": f"HuggingFace API returned HTTP {resp.status_code}: {resp.text[:200]}"
            }

        try:
            models_data = resp.json()
        except ValueError as e:
            logger.warning("HuggingFace search returned non-JSON payload: %s", e)
            return {
                "success": False,
                "query": clean_q,
                "count": 0,
                "models": [],
                "error": f"HuggingFace API returned an unparseable payload: {e}"
            }

        formatted = []
        for m in models_data:
            m_id = m.get("id", "")
            tags = m.get("tags", [])
            formatted.append({
                "model_id": m_id,
                "url": f"https://huggingface.co/{m_id}",
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "pipeline_tag": m.get("pipeline_tag", "text-generation"),
                "is_gguf": any("gguf" in t.lower() for t in tags) or "gguf" in m_id.lower(),
                "tags": tags[:6]
            })

        return {
            "success": True,
            "query": clean_q,
            "count": len(formatted),
            "models": formatted
        }

    def _run_git(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=30
        )

    def git_status(self) -> Dict[str, Any]:
        try:
            status = self._run_git(["status", "--porcelain"])
            if status.returncode != 0:
                return {"success": False, "error": f"git status failed: {status.stderr.strip()}"}
            branch = self._run_git(["branch", "--show-current"])
            if branch.returncode != 0:
                return {"success": False, "error": f"git branch failed: {branch.stderr.strip()}"}
            return {
                "success": True,
                "branch": branch.stdout.strip(),
                "changes": status.stdout.strip().split("\n") if status.stdout.strip() else []
            }
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning("git status could not be run: %s", e)
            return {"success": False, "error": f"git status could not be run: {e}"}

    def git_diff(self) -> Dict[str, Any]:
        try:
            res = self._run_git(["diff"])
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning("git diff could not be run: %s", e)
            return {"success": False, "error": f"git diff could not be run: {e}"}
        if res.returncode != 0:
            return {"success": False, "error": f"git diff failed: {res.stderr.strip()}"}
        return {"success": True, "diff": res.stdout}

    def validate_file_syntax(self, full_path: str) -> Dict[str, Any]:
        """Validates Python/JSON file syntax prior to merging."""
        if full_path.endswith(".py"):
            try:
                py_compile.compile(full_path, doraise=True)
                return {"valid": True}
            except py_compile.PyCompileError as e:
                return {"valid": False, "error": f"Python syntax error: {e}"}
            except OSError as e:
                return {"valid": False, "error": f"Could not compile '{full_path}': {e}"}
        elif full_path.endswith(".json"):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    json.load(f)
                return {"valid": True}
            except json.JSONDecodeError as e:
                return {"valid": False, "error": f"JSON syntax error: {e}"}
            except OSError as e:
                return {"valid": False, "error": f"Could not read '{full_path}': {e}"}
        return {"valid": True}

    def condense_workspace_code(self, bot_id: str, target_filename: str = "main.py") -> Dict[str, Any]:
        """Condenses all generated code snippets in bot workspace sandbox into a single unified file."""
        bot_dir = self.get_bot_workspace_dir(bot_id)
        if not os.path.exists(bot_dir):
            return {"success": False, "error": f"Workspace sandbox for bot '{bot_id}' does not exist."}

        code_blocks = []
        for file_entry in sorted(os.listdir(bot_dir)):
            if file_entry == target_filename or file_entry.startswith("."):
                continue
            full_p = os.path.join(bot_dir, file_entry)
            if os.path.isfile(full_p) and (file_entry.endswith(".py") or file_entry.endswith(".js") or file_entry.endswith(".ts") or file_entry.endswith(".txt")):
                try:
                    with open(full_p, "r", encoding="utf-8", errors="ignore") as f:
                        c = f.read().strip()
                        if c:
                            code_blocks.append(f"# --- Source Snippet: {file_entry} ---\n{c}")
                except OSError as e:
                    logger.warning("Could not read snippet %s during condensing: %s", full_p, e)

        if not code_blocks:
            return {"success": False, "error": f"No code snippets found to condense in workspace '{bot_id}'."}

        condensed_content = "\n\n".join(code_blocks) + "\n"
        w_res = self.bot_workspace_write(bot_id=bot_id, filepath=target_filename, content=condensed_content)
        if w_res.get("success"):
            return {
                "success": True,
                "bot_id": bot_id,
                "target_filename": target_filename,
                "snippets_condensed_count": len(code_blocks),
                "bytes_written": len(condensed_content)
            }
        return w_res

    def bot_workspace_write(self, bot_id: str, filepath: str, content: str) -> Dict[str, Any]:
        """Writes file to the bot's isolated workspace sandbox."""
        bot_dir = self.get_bot_workspace_dir(bot_id)
        full_path = os.path.abspath(os.path.join(bot_dir, filepath))
        if not self._is_safe_path(full_path, bot_dir):
            return {"success": False, "error": "Access outside bot workspace denied."}
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Syntax verification
            syntax_res = self.validate_file_syntax(full_path)
            res = {
                "success": syntax_res["valid"],
                "filepath": filepath,
                "bytes_written": len(content),
                "bot_id": bot_id,
                "syntax_valid": syntax_res["valid"],
                "syntax_error": syntax_res.get("error")
            }
            if not syntax_res["valid"]:
                res["error"] = syntax_res.get("error", "Syntax validation failed.")
            return res
        except OSError as e:
            logger.warning("Sandbox write failed for %s/%s: %s", bot_id, filepath, e)
            return {"success": False, "error": f"Sandbox write failed for '{filepath}': {e}"}

    def bot_workspace_merge_to_main(self, bot_id: str, filepath: str) -> Dict[str, Any]:
        """Validates and merges a file from bot workspace into main repository with git commit / rollback."""
        bot_dir = self.get_bot_workspace_dir(bot_id)
        src_path = os.path.abspath(os.path.join(bot_dir, filepath))
        dest_path = os.path.abspath(os.path.join(self.workspace_root, filepath))

        if not os.path.exists(src_path):
            return {"success": False, "error": f"File '{filepath}' does not exist in bot workspace '{bot_id}'."}

        syntax_res = self.validate_file_syntax(src_path)
        if not syntax_res["valid"]:
            return {"success": False, "error": f"Merge rejected due to syntax error: {syntax_res.get('error')}"}

        # Keep a backup of the destination so a failed merge can be rolled back.
        backup_path: Optional[str] = None
        if os.path.exists(dest_path):
            backup_path = f"{dest_path}.swarmchat-backup-{int(time.time() * 1000)}"
            try:
                shutil.copy2(dest_path, backup_path)
            except OSError as e:
                logger.warning("Could not back up %s before merge: %s", dest_path, e)
                return {"success": False, "error": f"Could not back up '{filepath}' before merge: {e}"}

        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)
        except OSError as e:
            rollback_error = self._restore_backup(dest_path, backup_path)
            logger.warning("Merge of %s failed: %s", filepath, e)
            return {
                "success": False,
                "error": f"Merge failed with error: {e}",
                "rolled_back": rollback_error is None,
                "rollback_error": rollback_error
            }

        git_error: Optional[str] = None
        committed = False
        try:
            add_res = self._run_git(["add", filepath])
            if add_res.returncode != 0:
                git_error = f"git add failed: {add_res.stderr.strip()}"
            else:
                commit_res = self._run_git(["commit", "-m", f"Incremental bot update by {bot_id}: {filepath}"])
                committed = commit_res.returncode == 0
                if not committed:
                    git_error = f"git commit failed: {(commit_res.stderr or commit_res.stdout).strip()}"
        except (subprocess.SubprocessError, OSError) as e:
            git_error = f"git commit could not be run: {e}"

        if git_error:
            logger.warning("Merge of %s was written but not committed: %s", filepath, git_error)

        if backup_path and os.path.exists(backup_path):
            try:
                os.remove(backup_path)
            except OSError as e:
                logger.debug("Could not remove merge backup %s: %s", backup_path, e)

        return {
            "success": True,
            "filepath": filepath,
            "bot_id": bot_id,
            "git_committed": committed,
            "git_error": git_error
        }

    def _restore_backup(self, dest_path: str, backup_path: Optional[str]) -> Optional[str]:
        """Restores dest_path from backup_path. Returns an error string when rollback itself failed."""
        if not backup_path:
            return None
        try:
            shutil.move(backup_path, dest_path)
            return None
        except OSError as e:
            logger.error("Rollback of %s from %s failed: %s", dest_path, backup_path, e)
            return f"Rollback failed; backup retained at '{backup_path}': {e}"

    def copy_file(self, src: str, dest: str, bot_id: Optional[str] = None) -> Dict[str, Any]:
        """Copies or clones a file within the workspace or bot sandbox."""
        root = self.get_bot_workspace_dir(bot_id) if bot_id else self.workspace_root
        full_src = os.path.abspath(os.path.join(root, src))
        if not os.path.exists(full_src):
            full_src = os.path.abspath(os.path.join(self.workspace_root, src))

        full_dest = os.path.abspath(os.path.join(root, dest))
        if not self._is_safe_path(full_src) or not self._is_safe_path(full_dest):
            return {"success": False, "error": "Copy operations outside workspace are denied."}

        if not os.path.exists(full_src):
            return {"success": False, "error": f"Source file '{src}' does not exist."}

        try:
            os.makedirs(os.path.dirname(full_dest), exist_ok=True)
            shutil.copy2(full_src, full_dest)
            return {"success": True, "src": src, "dest": dest}
        except OSError as e:
            logger.warning("Copy %s -> %s failed: %s", full_src, full_dest, e)
            return {"success": False, "error": f"Copy '{src}' -> '{dest}' failed: {e}"}

    def write_file(self, filepath: str, content: str, bot_id: Optional[str] = None) -> Dict[str, Any]:
        """Standard write tool. If bot_id is provided, writes to bot workspace first."""
        if bot_id:
            w_res = self.bot_workspace_write(bot_id, filepath, content)
            if not w_res.get("success"):
                return w_res
            # Auto-merge to main if syntax valid
            m_res = self.bot_workspace_merge_to_main(bot_id, filepath)
            return m_res

        full_path = os.path.abspath(os.path.join(self.workspace_root, filepath))
        if not self._is_safe_path(full_path):
            return {"success": False, "error": "Access outside workspace denied."}
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "filepath": filepath, "bytes_written": len(content)}
        except OSError as e:
            logger.warning("Write of %s failed: %s", full_path, e)
            return {"success": False, "error": f"Write of '{filepath}' failed: {e}"}

    def run_terminal_cmd(self, command: str, bot_id: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
        """Runs terminal command inside per-bot workspace CWD without shell=True, enforcing allowlist."""
        cwd = self.get_bot_workspace_dir(bot_id) if bot_id else self.workspace_root
        import shlex
        try:
            cmd_args = shlex.split(command)
        except Exception as e:
            return {"success": False, "error": f"Failed to parse command: {e}"}

        if not cmd_args:
            return {"success": False, "error": "Empty command provided."}

        binary = cmd_args[0].lower()
        allowed_binaries = ["git", "python", "python3", "pytest", "pip", "node", "npm", "ls", "dir", "echo", "cat"]
        if binary not in allowed_binaries:
            return {
                "success": False,
                "error": f"Command binary '{binary}' is not in the allowed list ({', '.join(allowed_binaries)})."
            }

        try:
            res = subprocess.run(
                cmd_args,
                shell=False,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired as e:
            logger.warning("Terminal command timed out after %ss: %s", timeout, command)
            return {
                "success": False,
                "timed_out": True,
                "error": f"Command timed out after {timeout}s.",
                "stdout": e.stdout if isinstance(e.stdout, str) else "",
                "stderr": e.stderr if isinstance(e.stderr, str) else ""
            }
        except OSError as e:
            logger.warning("Terminal command could not be started: %s", e)
            return {"success": False, "error": f"Command could not be started: {e}"}

        return {
            "success": res.returncode == 0,
            "returncode": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "error": None if res.returncode == 0 else f"Command exited with code {res.returncode}."
        }

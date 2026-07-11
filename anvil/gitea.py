"""Minimal Gitea REST client (stdlib ``urllib`` only) for the CI/CD loop.

Lets ANVIL and Lara file / list / comment / close **issues** and open / merge
**pull requests** on the self-hosted Gitea repo — the machinery behind the
issue → fix → test → promote → monitor loop (see docs/cicd.md).

Auth is the ``GITEA_TOKEN`` from ``.env`` (``cfg.gitea_token``). The API base and
``owner/repo`` are taken from config when set, else derived from the ``gitea``
git remote. Defensive: the token is never returned or logged, and a failed call
raises ``GiteaError`` with a clean message rather than leaking internals.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class GiteaError(RuntimeError):
    pass


def _resolve(cfg) -> Optional[Dict[str, str]]:
    """Return {api, repo, token} or None when not configured. ``repo`` is
    'owner/name'; ``api`` is the '/api/v1' base."""
    token = (getattr(cfg, "gitea_token", "") or os.environ.get("GITEA_TOKEN", "")).strip()
    if not token:
        return None
    base = (getattr(cfg, "gitea_url", "") or "").strip().rstrip("/")
    repo = (getattr(cfg, "gitea_repo", "") or "").strip().strip("/")
    if not base or not repo:
        remote = getattr(cfg, "forge_push_remote", "gitea")
        try:
            out = subprocess.run(["git", "remote", "get-url", remote],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            out = ""
        m = re.match(r"(https?://)(?:[^@/]+@)?([^/]+)/(.+?)(?:\.git)?/?$", out)
        if m:
            base = base or (m.group(1) + m.group(2))
            repo = repo or m.group(3)
    if not base or not repo:
        return None
    return {"api": base.rstrip("/") + "/api/v1", "repo": repo, "token": token}


class GiteaClient:
    """Thin authenticated client for one repo. ``.ok`` is False when unconfigured
    (no token / can't resolve the repo) — callers should check it first."""

    def __init__(self, cfg):
        r = _resolve(cfg)
        self.ok = r is not None
        if r:
            self.api = r["api"]
            self.repo = r["repo"]
            self._token = r["token"]

    # -- transport ------------------------------------------------------- #
    def _req(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        if not self.ok:
            raise GiteaError("Gitea is not configured (no GITEA_TOKEN / remote)")
        url = f"{self.api}/repos/{self.repo}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Authorization": f"token {self._token}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8", "replace")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            body_txt = exc.read().decode("utf-8", "replace")[:300]
            # never echo the URL (it's clean of the token, but stay conservative)
            raise GiteaError(f"Gitea {method} {path} -> HTTP {exc.code}: {body_txt}") from None
        except urllib.error.URLError as exc:
            raise GiteaError(f"Gitea unreachable: {exc.reason}") from None

    # -- labels (create-issue takes NAMES; Gitea wants IDs) -------------- #
    def _label_ids(self, names: List[str]) -> List[int]:
        if not names:
            return []
        try:
            existing = {l.get("name", "").lower(): l.get("id")
                        for l in self._req("GET", "/labels")}
        except GiteaError:
            return []
        ids = []
        for n in names:
            lid = existing.get(str(n).lower())
            if lid is None:
                try:                       # create the label on first use
                    lid = self._req("POST", "/labels",
                                    {"name": str(n), "color": "#5b8def"}).get("id")
                except GiteaError:
                    lid = None
            if lid is not None:
                ids.append(lid)
        return ids

    # -- issues ---------------------------------------------------------- #
    def create_issue(self, title: str, body: str = "",
                     labels: Optional[List[str]] = None,
                     ref: str = "") -> dict:
        payload: Dict[str, Any] = {"title": title, "body": body}
        ids = self._label_ids(labels or [])
        if ids:
            payload["labels"] = ids
        if ref:                              # associate the issue with its branch/tag
            payload["ref"] = str(ref)
        return self._req("POST", "/issues", payload)

    def add_dependency(self, number: int, blocker: int) -> None:
        """Mark issue ``number`` as DEPENDING ON (blocked by) issue ``blocker`` —
        the real Gitea dependency link, not just a comment. Gitea wants the blocker
        identified by owner+repo+index, not the bare index."""
        owner, _, name = self.repo.partition("/")
        try:
            self._req("POST", f"/issues/{int(number)}/dependencies",
                      {"owner": owner, "repo": name, "index": int(blocker)})
        except GiteaError:
            pass

    def list_dependencies(self, number: int) -> List[dict]:
        """Issues that ``number`` DEPENDS ON (its blockers)."""
        out = self._req("GET", f"/issues/{int(number)}/dependencies")
        return out if isinstance(out, list) else []

    def list_issues(self, labels: Optional[List[str]] = None,
                    state: str = "open") -> List[dict]:
        q = f"/issues?state={state}&type=issues"
        if labels:
            q += "&labels=" + ",".join(labels)
        out: List[dict] = []
        page = 1
        while True:
            chunk = self._req("GET", f"{q}&page={page}&limit=50")
            if not isinstance(chunk, list):
                break
            out.extend(chunk)
            if len(chunk) < 50:
                break
            page += 1
        return out

    def comment_issue(self, number: int, body: str) -> dict:
        return self._req("POST", f"/issues/{int(number)}/comments", {"body": body})

    def close_issue(self, number: int) -> dict:
        return self._req("PATCH", f"/issues/{int(number)}", {"state": "closed"})

    def get_issue(self, number: int) -> dict:
        return self._req("GET", f"/issues/{int(number)}")

    def set_assignees(self, number: int, logins: List[str]) -> dict:
        """Replace the issue's assignees (empty list clears them). Assignment is how
        Lara signals whose turn it is — herself while working, the creator for a
        question, the operator when she needs help — instead of status-label churn."""
        return self._req("PATCH", f"/issues/{int(number)}",
                         {"assignees": [str(x) for x in logins]})

    def list_comments(self, number: int) -> List[dict]:
        out = self._req("GET", f"/issues/{int(number)}/comments")
        return out if isinstance(out, list) else []

    def add_labels(self, number: int, names: List[str]) -> None:
        ids = self._label_ids(names)
        if ids:
            self._req("POST", f"/issues/{int(number)}/labels", {"labels": ids})

    def remove_label(self, number: int, name: str) -> None:
        try:
            existing = {l.get("name", "").lower(): l.get("id")
                        for l in self._req("GET", "/labels")}
        except GiteaError:
            return
        lid = existing.get(str(name).lower())
        if lid is not None:
            try:
                self._req("DELETE", f"/issues/{int(number)}/labels/{lid}")
            except GiteaError:
                pass

    # -- pull requests (the promotion gate) ------------------------------ #
    def create_pull(self, head: str, base: str, title: str, body: str = "") -> dict:
        return self._req("POST", "/pulls",
                         {"head": head, "base": base, "title": title, "body": body})

    def merge_pull(self, number: int, method: str = "merge") -> dict:
        # method: "merge" | "rebase" | "squash"
        return self._req("POST", f"/pulls/{int(number)}/merge", {"Do": method})

    def list_pulls(self, state: str = "open") -> List[dict]:
        out = self._req("GET", f"/pulls?state={state}")
        return out if isinstance(out, list) else []

    def find_open_pull(self, head: str, base: str) -> Optional[dict]:
        """The open PR from ``head`` into ``base``, if one exists (idempotent promotion)."""
        for p in self.list_pulls("open"):
            if (p.get("head") or {}).get("ref") == head \
                    and (p.get("base") or {}).get("ref") == base:
                return p
        return None

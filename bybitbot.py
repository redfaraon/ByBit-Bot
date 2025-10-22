# -*- coding: utf-8 -*-
"""Entry point and fallback wrapper for bybitbot_impl."""
import importlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BOT_VERSION = os.getenv("BYBITBOT_VERSION", "2025.10.22.3")
CHANGELOG_FILE = REPO_ROOT / "CHANGELOG.txt"
FALLBACK_HISTORY_FILE = REPO_ROOT / "fallback_history.json"
FALLBACK_HISTORY_FILE = REPO_ROOT / "fallback_history.json"


def _resolve_commit_limit(raw_value: str | None) -> int:
    try:
        value = int(raw_value) if raw_value is not None else 8
    except ValueError:
        value = 8
    return max(1, value)


CHANGELOG_COMMIT_LIMIT = _resolve_commit_limit(os.getenv("BYBITBOT_CHANGELOG_COMMITS"))
FALLBACK_COMMIT_CANDIDATE_LIMIT = _resolve_commit_limit(os.getenv("BYBITBOT_FALLBACK_COMMIT_LIMIT", "12"))
DEFAULT_STABLE_BRANCH = (os.getenv("BYBITBOT_STABLE_BRANCH") or "stable").strip() or "stable"


def _build_commit_changelog(limit: int | None = None):
    limit = CHANGELOG_COMMIT_LIMIT if limit is None else _resolve_commit_limit(str(limit))
    cmd = ["git", "log", "-n", str(limit), "--pretty=format:%cs %h %s"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
    except Exception as exc:
        header = ""
        bullets = [f"- git log unavailable: {exc}"]
        return limit, header, bullets
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        header = "No recent commits available."
        bullets = ["- No commits to display."]
        return limit, header, bullets
    header = f"Recent commits (last {limit})"
    bullets = [f"- {line}" for line in lines]
    return limit, header, bullets


def _format_changelog_entry(version: str, header: str, bullets: list[str]) -> str:
    body_lines = []
    if header:
        body_lines.append(header)
    body_lines.extend(bullets or ["- No commits to display."])
    body = "\n".join(body_lines)
    return f"{version}\n{body}"


def _update_changelog_file(version: str, header: str, bullets: list[str]) -> None:
    entry = _format_changelog_entry(version, header, bullets)
    if CHANGELOG_FILE.exists():
        content = CHANGELOG_FILE.read_text(encoding="utf-8").strip()
        if content:
            entries = content.split("\n\n")
        else:
            entries = []
    else:
        entries = []
    if entries:
        if entries[-1].startswith(version):
            entries[-1] = entry
        else:
            entries.append(entry)
    else:
        entries = [entry]
    CHANGELOG_FILE.write_text("\n\n".join(entries).strip() + "\n\n", encoding="utf-8")


_commit_limit, CHANGELOG_HEADER, CHANGELOG_LINES = _build_commit_changelog()
CURRENT_CHANGELOG = "\n".join(
    [CHANGELOG_HEADER] + CHANGELOG_LINES if CHANGELOG_HEADER else CHANGELOG_LINES
)
_update_changelog_file(BOT_VERSION, CHANGELOG_HEADER, CHANGELOG_LINES)
LATEST_VERSION = BOT_VERSION


def _parse_version_tuple(version: str) -> tuple:
    parts = []
    for chunk in version.split('.'):
        digits = ''.join(ch for ch in chunk if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def _load_fallback_history() -> dict:
    try:
        raw = FALLBACK_HISTORY_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"commits": {}, "backups": {}}
    except Exception:
        return {"commits": {}, "backups": {}}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"commits": {}, "backups": {}}
    if not isinstance(data, dict):
        return {"commits": {}, "backups": {}}
    data.setdefault("commits", {})
    data.setdefault("backups", {})
    data.setdefault("branches", {})
    if not isinstance(data["commits"], dict):
        data["commits"] = {}
    if not isinstance(data["backups"], dict):
        data["backups"] = {}
    if not isinstance(data["branches"], dict):
        data["branches"] = {}
    stable_branch = (data.get("stable_branch") or "").strip()
    if not stable_branch:
        data["stable_branch"] = DEFAULT_STABLE_BRANCH
    return data


def _save_fallback_history(history: dict) -> None:
    try:
        FALLBACK_HISTORY_FILE.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _list_past_commits(limit: int = FALLBACK_COMMIT_CANDIDATE_LIMIT) -> list[str]:
    cmd = [
        "git",
        "rev-list",
        "--max-count",
        str(max(0, limit)),
        "--skip",
        "1",
        "HEAD",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
    except Exception:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _current_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
    except Exception:
        return None
    head = result.stdout.strip()
    return head or None


def _materialize_commit_script(commit_hash: str) -> Path | None:
    target = commit_hash.strip()
    if not target:
        return None
    cmd = ["git", "show", f"{target}:bybitbot_impl.py"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
    except Exception:
        return None
    content = result.stdout
    if not content:
        return None
    backups_dir = REPO_ROOT / "backups"
    backups_dir.mkdir(exist_ok=True)
    script_path = backups_dir / f"bybitbot_impl_commit_{target}.py"
    try:
        script_path.write_text(content, encoding="utf-8")
    except Exception:
        return None
    return script_path


def _materialize_branch_script(branch_name: str) -> Path | None:
    target = branch_name.strip()
    if not target:
        return None

    def try_show(ref: str) -> Path | None:
        cmd = ["git", "show", f"{ref}:bybitbot_impl.py"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=REPO_ROOT,
            )
        except Exception:
            return None
        content = result.stdout
        if not content:
            return None
        backups_dir = REPO_ROOT / "backups"
        backups_dir.mkdir(exist_ok=True)
        safe_target = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in ref)
        script_path = backups_dir / f"bybitbot_impl_branch_{safe_target}.py"
        try:
            script_path.write_text(content, encoding="utf-8")
        except Exception:
            return None
        return script_path

    candidate_refs: list[str] = []
    commit_attempts: list[str] = []

    if "/" in target:
        candidate_refs.append(target)
    else:
        commit_attempts.append(target)
        candidate_refs.append(target)
        candidate_refs.append(f"origin/{target}")

    for ref in commit_attempts + candidate_refs:
        path = try_show(ref)
        if path:
            return path

    if "/" not in target:
        try:
            subprocess.run(
                ["git", "fetch", "--quiet", "origin", target],
                check=True,
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
        else:
            for ref in (target, f"origin/{target}"):
                path = try_show(ref)
                if path:
                    return path

    for ref in candidate_refs:
        try:
            subprocess.run(
                ["git", "fetch", "--quiet", "origin", ref],
                check=True,
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
        else:
            path = try_show(ref)
            if path:
                return path

    return None


def _log_fallback_event(source_desc: str, source_ref: str, target_desc: str, target_ref: str, context: str) -> None:
    message = (
        f"[BOOT] Fallback executed ({context}): "
        f"{source_desc} {source_ref} -> {target_desc} {target_ref}"
    )
    print(message, file=sys.stderr)


def _update_current_branch() -> None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
    except Exception:
        return
    branch = result.stdout.strip()
    if not branch or branch == "HEAD":
        return
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin", branch],
            check=True,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return
    try:
        subprocess.run(
            ["git", "merge", "--ff-only", f"origin/{branch}"],
            check=True,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print(f"[BOOT] Failed to fast-forward branch {branch}; continuing with local HEAD.", file=sys.stderr)
    except Exception:
        pass

    return None


def _run_script_candidate(script_path: Path, version_label: str, reason: str, source: str, *, fallback_context: str | None = None) -> bool:
    _, fb_header, fb_lines = _build_commit_changelog()
    fallback_changelog = "\n".join([fb_header] + fb_lines if fb_header else fb_lines)
    print(
        f"[BOOT] Falling back to {source} due to {reason}",
        file=sys.stderr,
    )
    if fallback_context:
        _log_fallback_event("HEAD", _current_head() or "unknown", source, version_label, fallback_context)
    env = os.environ.copy()
    env["BYBITBOT_CHANGELOG_VERSION"] = version_label
    env["BYBITBOT_CHANGELOG_TEXT"] = fallback_changelog
    env["BYBITBOT_EXPECTED_VERSION"] = LATEST_VERSION
    env["BYBITBOT_SOURCE_LABEL"] = source
    env["BYBITBOT_SOURCE_REF"] = version_label
    if fallback_context:
        env["BYBITBOT_FALLBACK_CONTEXT"] = fallback_context
    else:
        env.pop("BYBITBOT_FALLBACK_CONTEXT", None)
    result = subprocess.run([sys.executable, str(script_path)], env=env)
    return result.returncode == 0


def _iter_backups():
    backups_dir = Path(__file__).with_name("backups")
    if not backups_dir.exists():
        return []
    candidates = []
    patterns = [
        "bybitbot_v*.py",
        "bybit_intraday_30m_5pairs_v*.py",
        "bybitbot_impl_v*.py",
    ]
    seen = set()
    for pattern in patterns:
        for path in backups_dir.glob(pattern):
            try:
                version_str = path.stem.split("_v", 1)[-1]
            except Exception:
                continue
            key = (version_str, path)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((_parse_version_tuple(version_str), version_str, path))
    candidates.sort(reverse=True)
    return [p for _, _, p in candidates]


def _run_current():
    os.environ["BYBITBOT_CHANGELOG_VERSION"] = BOT_VERSION
    os.environ["BYBITBOT_CHANGELOG_TEXT"] = CURRENT_CHANGELOG
    os.environ["BYBITBOT_EXPECTED_VERSION"] = LATEST_VERSION
    current_head = _current_head() or "unknown"
    os.environ["BYBITBOT_SOURCE_LABEL"] = "HEAD"
    os.environ["BYBITBOT_SOURCE_REF"] = current_head
    os.environ.pop("BYBITBOT_FALLBACK_CONTEXT", None)
    module = importlib.import_module("bybitbot_impl")
    if hasattr(module, "apply_metadata"):
        module.apply_metadata(BOT_VERSION, CURRENT_CHANGELOG, LATEST_VERSION)
    else:
        if hasattr(module, "BOT_VERSION"):
            module.BOT_VERSION = BOT_VERSION
        if hasattr(module, "CHANGELOG_TEXT"):
            module.CHANGELOG_TEXT = CURRENT_CHANGELOG
    if hasattr(module, "main"):
        module.main()
    else:
        raise AttributeError("bybitbot_impl.main not found")


def _run_backups(reason: str) -> bool:
    history = _load_fallback_history()
    commit_history = history.setdefault("commits", {})
    backup_history = history.setdefault("backups", {})
    branch_history = history.setdefault("branches", {})
    stable_branch = (history.get("stable_branch") or DEFAULT_STABLE_BRANCH).strip()
    if not stable_branch:
        stable_branch = DEFAULT_STABLE_BRANCH
        history["stable_branch"] = stable_branch
    if stable_branch:
        script_path = _materialize_branch_script(stable_branch)
        if script_path:
            version_label = f"branch.{stable_branch}"
            source_label = f"{stable_branch} branch"
            head_hash = _current_head() or "unknown"
            success = _run_script_candidate(script_path, version_label, reason, source_label, fallback_context=f"{head_hash[:8]} -> branch")
            branch_history[stable_branch] = "success" if success else "failed"
            if success:
                history["stable_branch"] = stable_branch
                history.pop("stable_commit", None)
                history.pop("stable_backup", None)
                _save_fallback_history(history)
                return True
            else:
                _save_fallback_history(history)
        else:
            branch_history[stable_branch] = "missing"
            _save_fallback_history(history)
    stable_commit = history.get("stable_commit")
    stable_backup = history.get("stable_backup")
    head_hash = _current_head()

    head_short = head_hash[:8] if head_hash else "unknown"
    if stable_commit and stable_commit != head_hash:
        script_path = _materialize_commit_script(stable_commit)
        if script_path:
            version_label = f"commit.{stable_commit[:8]}"
            source_label = f"stable commit {stable_commit[:8]}"
            context = f"{head_short} -> {stable_commit[:8]}"
            success = _run_script_candidate(script_path, version_label, reason, source_label, fallback_context=context)
            commit_history[stable_commit] = "success" if success else "failed"
            if success:
                history["stable_commit"] = stable_commit
                history.pop("stable_backup", None)
            else:
                history["stable_commit"] = None
            _save_fallback_history(history)
            if success:
                return True

    if stable_backup:
        backup_path = REPO_ROOT / "backups" / stable_backup
        if backup_path.exists():
            context = f"{head_short} -> backup:{stable_backup}"
            success = _run_script_candidate(backup_path, stable_backup, reason, stable_backup, fallback_context=context)
            backup_history[stable_backup] = "success" if success else "failed"
            if success:
                history["stable_backup"] = stable_backup
                history["stable_commit"] = None
                _save_fallback_history(history)
                return True
            else:
                history["stable_backup"] = None
                _save_fallback_history(history)

    commit_hashes = _list_past_commits()
    for commit_hash in commit_hashes:
        if commit_hash == head_hash:
            continue
        if commit_history.get(commit_hash) == "failed":
            continue
        script_path = _materialize_commit_script(commit_hash)
        if not script_path:
            continue
        version_label = f"commit.{commit_hash[:8]}"
        source_label = f"commit {commit_hash[:8]}"
        context = f"{head_short} -> {commit_hash[:8]}"
        success = _run_script_candidate(script_path, version_label, reason, source_label, fallback_context=context)
        commit_history[commit_hash] = "success" if success else "failed"
        if success:
            history["stable_commit"] = commit_hash
            history.pop("stable_backup", None)
        _save_fallback_history(history)
        if success:
            return True

    backups = _iter_backups()
    if not backups:
        print("[BOOT] No backups available.", file=sys.stderr)
        return False
    for candidate in backups:
        backup_key = candidate.name
        if backup_history.get(backup_key) == "failed":
            continue
        candidate_version = candidate.stem.split('_v', 1)[-1]
        context = f"{head_short} -> backup:{backup_key}"
        success = _run_script_candidate(candidate, candidate_version, reason, candidate.name, fallback_context=context)
        backup_history[backup_key] = "success" if success else "failed"
        if success:
            history["stable_commit"] = None
            history["stable_backup"] = backup_key
        _save_fallback_history(history)
        if success:
            return True
    print("[BOOT] All backups failed.", file=sys.stderr)
    return False


def main():
    _update_current_branch()
    history = _load_fallback_history()
    head_hash = _current_head()
    head_status = None
    if head_hash:
        head_status = history.setdefault("commits", {}).get(head_hash)
    if head_hash and head_status == "failed":
        reason = f"HEAD {head_hash[:8]} previously failed"
        if not _run_backups(reason):
            raise RuntimeError("No viable fallback available")
        return

    try:
        _run_current()
    except Exception as exc:
        traceback.print_exc()
        if head_hash:
            history.setdefault("commits", {})[head_hash] = "failed"
            _save_fallback_history(history)
        if not _run_backups(str(exc)):
            raise
    else:
        if head_hash:
            history.setdefault("commits", {})[head_hash] = "success"
            history["stable_commit"] = head_hash
            history.pop("stable_backup", None)
            _save_fallback_history(history)


if __name__ == "__main__":
    main()

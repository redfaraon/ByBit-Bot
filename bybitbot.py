# -*- coding: utf-8 -*-
"""Entry point and fallback wrapper for bybitbot_impl."""
import argparse
import importlib
import json
import os
import random
import time
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent
BOT_VERSION = os.getenv("BYBITBOT_VERSION", "12.0")
CHANGELOG_FILE = REPO_ROOT / "CHANGELOG.txt"
STATE_DIR = Path(os.getenv("BYBITBOT_STATE_DIR", REPO_ROOT))

def _refresh_state_paths() -> None:
    global STATE_DIR, FALLBACK_HISTORY_FILE, CYCLE_STATE_FILE
    state_dir_raw = os.getenv("BYBITBOT_STATE_DIR")
    try:
        STATE_DIR = (Path(state_dir_raw).expanduser().resolve() if state_dir_raw else REPO_ROOT)
    except Exception:
        STATE_DIR = REPO_ROOT
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    FALLBACK_HISTORY_FILE = STATE_DIR / "fallback_history.json"
    CYCLE_STATE_FILE = STATE_DIR / "cycle_state.json"

_refresh_state_paths()

USERS_DIR = REPO_ROOT / "users"
USERS_CONFIG_FILE = USERS_DIR / "users.json"
USERS_DEFAULT_SECRET = "secrets.env"

_ENGINE_AUTOSTART_PROCESS = None
_ENGINE_AUTOSTART_LOG = None


def _resolve_commit_limit(raw_value: str | None) -> int:
    try:
        value = int(raw_value) if raw_value is not None else 8
    except ValueError:
        value = 8
    return max(1, value)


CHANGELOG_COMMIT_LIMIT = _resolve_commit_limit(os.getenv("BYBITBOT_CHANGELOG_COMMITS"))
FALLBACK_COMMIT_CANDIDATE_LIMIT = _resolve_commit_limit(os.getenv("BYBITBOT_FALLBACK_COMMIT_LIMIT", "12"))
DEFAULT_STABLE_BRANCH = (os.getenv("BYBITBOT_STABLE_BRANCH") or "stable").strip() or "stable"
DEFAULT_LEGACY_BRANCH = (os.getenv("BYBITBOT_LEGACY_BRANCH") or "legacy").strip() or "legacy"
DEFAULT_CURRENT_TAG = (os.getenv("BYBITBOT_CURRENT_TAG") or "current").strip() or "current"
FAULT_TAG_NAME = (os.getenv("BYBITBOT_FAULT_TAG") or "fault").strip() or "fault"


FALLBACK_PROBE_INTERVAL = max(1, int(os.getenv("BYBITBOT_FALLBACK_PROBE_INTERVAL", "10")))


@dataclass
class BackupCandidate:
    script_path: Path
    version_label: str
    source_label: str
    reason: str
    context: str | None
    cycle_kind: str
    cycle_mode: str
    cycle_counter: int
    commit_hash: str | None = None
    commit_message: str | None = None
    commit_timestamp: str | None = None
    on_result: Callable[[bool], None] | None = None

    def finalize(self, success: bool) -> None:
        if self.on_result:
            self.on_result(success)


@dataclass
class UserProfile:
    user_id: str
    label: str
    enabled: bool
    env_overrides: dict[str, str]
    env_files: list[Path]
    state_dir: Path


def _normalize_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_user_registry() -> dict[str, UserProfile]:
    if not USERS_CONFIG_FILE.exists():
        return {}
    try:
        payload = json.loads(USERS_CONFIG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[USERS] Failed to parse {USERS_CONFIG_FILE}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(payload, dict):
        print(f"[USERS] Invalid registry format in {USERS_CONFIG_FILE}", file=sys.stderr)
        return {}
    registry: dict[str, UserProfile] = {}
    users_iter = payload.get("users") if isinstance(payload.get("users"), list) else []
    for entry in users_iter:
        if not isinstance(entry, dict):
            continue
        user_id = str(entry.get("id") or "").strip()
        if not user_id:
            continue
        enabled = bool(entry.get("enabled", True))
        label = str(entry.get("label") or user_id).strip() or user_id
        env_overrides_raw = entry.get("env") or {}
        if not isinstance(env_overrides_raw, dict):
            env_overrides_raw = {}
        env_overrides = {str(k): str(v) for k, v in env_overrides_raw.items() if v is not None}
        env_files: list[Path] = []
        extra_files = entry.get("env_files") or []
        if isinstance(extra_files, (list, tuple)):
            for candidate in extra_files:
                try:
                    env_files.append(_normalize_path(candidate))
                except Exception:
                    continue
        default_secret = USERS_DIR / user_id / USERS_DEFAULT_SECRET
        if default_secret not in env_files:
            env_files.append(default_secret)
        state_dir_raw = entry.get("state_dir")
        state_dir = _normalize_path(state_dir_raw) if isinstance(state_dir_raw, str) and state_dir_raw else (REPO_ROOT / "runtime" / user_id).resolve()
        registry[user_id] = UserProfile(
            user_id=user_id,
            label=label,
            enabled=enabled,
            env_overrides=env_overrides,
            env_files=env_files,
            state_dir=state_dir,
        )
    return registry


def _apply_user_profile(profile: UserProfile) -> None:
    try:
        profile.state_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"[USERS] Failed to create state dir {profile.state_dir}: {exc}", file=sys.stderr)
    os.environ["BYBITBOT_MULTIUSER"] = "1"
    os.environ["BYBITBOT_USER_ID"] = profile.user_id
    os.environ["BYBITBOT_USER_LABEL"] = profile.label
    os.environ["BYBITBOT_STATE_DIR"] = str(profile.state_dir)
    for key, value in profile.env_overrides.items():
        os.environ[key] = value
    for env_path in profile.env_files:
        try:
            values = dotenv_values(env_path)
        except Exception as exc:
            print(f"[USERS] Cannot load {env_path}: {exc}", file=sys.stderr)
            continue
        if not values:
            continue
        for key, value in values.items():
            if value is None:
                continue
            os.environ[key] = value
    _refresh_state_paths()


def _engine_autostart_enabled() -> bool:
    raw_value = str(os.getenv("BYBITBOT_AUTOSTART_ENGINE", "1")).strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def _start_background_engine() -> None:
    """Launch bybit_engine.py --loop --quiet in the background when requested."""
    global _ENGINE_AUTOSTART_PROCESS, _ENGINE_AUTOSTART_LOG
    if not _engine_autostart_enabled():
        return
    if _ENGINE_AUTOSTART_PROCESS and _ENGINE_AUTOSTART_PROCESS.poll() is None:
        return
    script_path = REPO_ROOT / "bybit_engine.py"
    if not script_path.exists():
        return
    runtime_root = Path(os.getenv("BYBITBOT_STATE_DIR", REPO_ROOT / "runtime"))
    log_dir = runtime_root / "engine"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    log_path = log_dir / "engine-autostart.log"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        log_file = log_path.open("a", encoding="utf-8")
    except Exception as exc:
        print(f"[ENGINE] Failed to open {log_path}: {exc}", file=sys.stderr)
        log_file = None
    else:
        log_file.write(f"\n=== Engine autostart @ {timestamp} ===\n")
        log_file.flush()
    env = os.environ.copy()
    env.setdefault("BYBIT_ENGINE_AUTOSTART", "1")
    cmd = [sys.executable, str(script_path), "--loop", "--quiet"]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file if log_file else None,
            stderr=log_file if log_file else None,
            env=env,
        )
    except Exception as exc:
        if log_file:
            log_file.write(f"[ENGINE] Failed to start: {exc}\n")
            log_file.flush()
            log_file.close()
        print(f"[ENGINE] Autostart failed: {exc}", file=sys.stderr)
        return
    _ENGINE_AUTOSTART_PROCESS = proc
    _ENGINE_AUTOSTART_LOG = log_file


def _stop_background_engine() -> None:
    global _ENGINE_AUTOSTART_PROCESS, _ENGINE_AUTOSTART_LOG
    proc = _ENGINE_AUTOSTART_PROCESS
    if proc:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass
    if _ENGINE_AUTOSTART_LOG:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            _ENGINE_AUTOSTART_LOG.write(f"=== Engine autostop @ {timestamp} ===\n")
            _ENGINE_AUTOSTART_LOG.flush()
        except Exception:
            pass
        try:
            _ENGINE_AUTOSTART_LOG.close()
        except Exception:
            pass
    _ENGINE_AUTOSTART_PROCESS = None
    _ENGINE_AUTOSTART_LOG = None


def _parse_args():
    parser = argparse.ArgumentParser(description="ByBit Bot launcher")
    parser.add_argument("--user", help="Run using the specified user profile id")
    parser.add_argument("--list-users", action="store_true", help="List configured user profiles and exit")
    parser.add_argument("--check-user-logs", action="store_true", help="Check that each configured user has a per-user bybit.log in runtime/<user>/bybit.log")
    return parser.parse_args()


def _list_user_profiles(registry: dict[str, UserProfile]) -> None:
    if not registry:
        print("No user profiles configured. Create users/users.json based on users/users.example.json.")
        return
    print("Configured user profiles:")
    for profile in registry.values():
        status = "enabled" if profile.enabled else "disabled"
        print(f"- {profile.user_id} ({profile.label}) [{status}] -> state_dir={profile.state_dir}")
        for env_file in profile.env_files:
            print(f"    secrets: {env_file}")


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
    data.setdefault("tags", {})
    if not isinstance(data["tags"], dict):
        data["tags"] = {}
    stable_branch = (data.get("stable_branch") or "").strip()
    if not stable_branch:
        data["stable_branch"] = DEFAULT_STABLE_BRANCH
    if not isinstance(data.get("fallback_active"), bool):
        data["fallback_active"] = False
    if not isinstance(data.get("fallback_cycles"), int):
        data["fallback_cycles"] = 0
    if not isinstance(data.get("fallback_probe_interval"), int):
        data["fallback_probe_interval"] = FALLBACK_PROBE_INTERVAL
    data.setdefault("fallback_last_head", None)
    data.setdefault("fallback_source", None)
    data.setdefault("fallback_target", None)
    data.setdefault("fallback_failed_head", None)
    return data


def _save_fallback_history(history: dict) -> None:
    try:
        FALLBACK_HISTORY_FILE.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_cycle_state() -> dict:
    try:
        raw = CYCLE_STATE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"total_cycles": 0, "fallback_cycles": 0}
    except Exception:
        return {"total_cycles": 0, "fallback_cycles": 0}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"total_cycles": 0, "fallback_cycles": 0}
    if not isinstance(data, dict):
        return {"total_cycles": 0, "fallback_cycles": 0}
    total_cycles = data.get("total_cycles")
    fallback_cycles = data.get("fallback_cycles")
    try:
        total_cycles = int(total_cycles)
    except (TypeError, ValueError):
        total_cycles = 0
    try:
        fallback_cycles = int(fallback_cycles)
    except (TypeError, ValueError):
        fallback_cycles = 0
    data["total_cycles"] = total_cycles
    data["fallback_cycles"] = fallback_cycles
    return data




def _record_fallback(
    history: dict,
    *,
    head_hash: str | None,
    source_label: str,
    target_label: str,
    origin_head: str | None = None,
) -> None:
    history["fallback_active"] = True
    history["fallback_cycles"] = 0
    if head_hash:
        history["fallback_last_head"] = head_hash
    if origin_head:
        history["fallback_failed_head"] = origin_head
    else:
        history.setdefault("fallback_failed_head", head_hash)
    history["fallback_source"] = source_label
    history["fallback_target"] = target_label
    history.setdefault("fallback_probe_interval", FALLBACK_PROBE_INTERVAL)
    _save_fallback_history(history)
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
    _sync_backup_resources(backups_dir)
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
        _sync_backup_resources(backups_dir)
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


def _resolve_ref_commit(ref: str | None) -> str | None:
    target = (ref or "").strip()
    if not target:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", target],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _sanitize_commit_message(message: str | None) -> str | None:
    if not message:
        return None
    first_line = message.strip().splitlines()[0].strip()
    return first_line or None


def _resolve_commit_metadata(ref: str | None) -> tuple[str | None, str | None, str | None]:
    target = (ref or "").strip()
    if not target:
        return None, None, None
    cmd = ["git", "show", "-s", "--format=%H%x1f%s%x1f%cI", target]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
    except Exception:
        return None, None, None
    raw = result.stdout.strip()
    if not raw:
        return None, None, None
    parts = raw.split("\x1f")
    commit_hash = parts[0].strip() if parts else ""
    commit_msg = parts[1] if len(parts) > 1 else ""
    commit_ts = parts[2] if len(parts) > 2 else ""
    return (
        commit_hash or None,
        _sanitize_commit_message(commit_msg),
        commit_ts.strip() or None,
    )


def _resolve_commit_details(ref: str | None) -> tuple[str | None, str | None]:
    commit_hash, commit_message, _ = _resolve_commit_metadata(ref)
    return commit_hash, commit_message


def _build_backup_candidates(
    history: dict,
    *,
    head_hash: str | None,
    reason: str,
    cycle_kind: str,
    cycle_counter: int,
    record_fallback: bool,
) -> list[BackupCandidate]:
    commit_history = history.setdefault("commits", {})
    history.setdefault("backups", {})
    branch_history = history.setdefault("branches", {})
    tag_history = history.setdefault("tags", {})
    stable_branch = (history.get("stable_branch") or DEFAULT_STABLE_BRANCH).strip() or DEFAULT_STABLE_BRANCH
    legacy_branch = (history.get("legacy_branch") or DEFAULT_LEGACY_BRANCH).strip() or DEFAULT_LEGACY_BRANCH
    current_tag = (history.get("current_tag") or DEFAULT_CURRENT_TAG).strip() or DEFAULT_CURRENT_TAG
    head_short = head_hash[:8] if head_hash else "unknown"

    candidates: list[BackupCandidate] = []

    def add_branch(branch_name: str, mode_label: str) -> None:
        name = (branch_name or "").strip()
        if not name:
            return
        script_path = _materialize_branch_script(name)
        if not script_path:
            branch_history[name] = "missing"
            _save_fallback_history(history)
            return
        commit_hash, commit_message, commit_timestamp = _resolve_commit_metadata(name)
        if not commit_hash:
            branch_history[name] = "missing"
            _save_fallback_history(history)
            return
        version_label = f"branch.{name}"
        base_source_label = f"{name} branch"
        if cycle_kind == "normal":
            source_label = f"{base_source_label} (routine)"
            context = None
            candidate_reason = f"{reason} ({mode_label})"
        else:
            source_label = base_source_label
            context = f"{head_short} -> branch:{name}"
            candidate_reason = reason

        def on_result(
            success: bool,
            branch=name,
            label=mode_label,
            source=base_source_label,
            version=version_label,
            fallback_commit=commit_hash,
            failing_head=head_hash,
        ) -> None:
            branch_history[branch] = "success" if success else "failed"
            if success:
                if label == "stable":
                    history["stable_branch"] = branch
                elif label == "legacy":
                    history["legacy_branch"] = branch
                history.pop("stable_commit", None)
                history.pop("stable_backup", None)
                history["fallback_branch_next"] = "legacy" if label == "stable" else "stable"
                if record_fallback:
                    _record_fallback(
                        history,
                        head_hash=fallback_commit,
                        source_label=source,
                        target_label=version,
                        origin_head=failing_head,
                    )
                else:
                    _save_fallback_history(history)
            else:
                _save_fallback_history(history)

        candidates.append(
            BackupCandidate(
                script_path=script_path,
                version_label=version_label,
                source_label=source_label,
                reason=candidate_reason,
                context=context,
                cycle_kind=cycle_kind,
                cycle_mode=mode_label,
                cycle_counter=cycle_counter,
                commit_hash=commit_hash,
                commit_message=commit_message,
                commit_timestamp=commit_timestamp,
                on_result=on_result,
            )
        )

    def add_current_tag(tag_name: str) -> None:
        name = (tag_name or "").strip()
        if not name:
            return
        commit_hash = _resolve_ref_commit(name)
        if not commit_hash:
            return
        script_path = _materialize_commit_script(commit_hash)
        if not script_path:
            return
        _, commit_message, commit_timestamp = _resolve_commit_metadata(commit_hash)
        version_label = f"tag.{name}"
        source_label = f"{name} tag"
        if cycle_kind == "normal":
            context = None
            candidate_reason = f"{reason} ({name})"
        else:
            context = f"{head_short} -> tag:{name}"
            candidate_reason = reason

        def on_result(
            success: bool,
            tag=name,
            commit=commit_hash,
            version=version_label,
            source=source_label,
            failing_head=head_hash,
        ) -> None:
            tag_history[tag] = "success" if success else "failed"
            if success:
                history["stable_commit"] = commit
                history.pop("stable_backup", None)
                if record_fallback:
                    _record_fallback(
                        history,
                        head_hash=commit,
                        source_label=source,
                        target_label=version,
                        origin_head=failing_head,
                    )
                else:
                    _save_fallback_history(history)
            else:
                _save_fallback_history(history)

        candidates.append(
            BackupCandidate(
                script_path=script_path,
                version_label=version_label,
                source_label=source_label,
                reason=candidate_reason,
                context=context,
                cycle_kind=cycle_kind,
                cycle_mode="current",
                cycle_counter=cycle_counter,
                commit_hash=commit_hash,
                commit_message=commit_message,
                commit_timestamp=commit_timestamp,
                on_result=on_result,
            )
        )

    def add_random_commit() -> None:
        fault_commit = _resolve_ref_commit(FAULT_TAG_NAME) if FAULT_TAG_NAME else None
        commit_candidates = [
            commit
            for commit in _list_past_commits()
            if commit_history.get(commit) != "failed"
        ]
        if head_hash:
            commit_candidates = [commit for commit in commit_candidates if commit != head_hash]
        if fault_commit:
            commit_candidates = [commit for commit in commit_candidates if commit != fault_commit]
        if not commit_candidates:
            return
        selected_commit = random.choice(commit_candidates)
        script_path = _materialize_commit_script(selected_commit)
        if not script_path:
            commit_history[selected_commit] = "missing"
            _save_fallback_history(history)
            return
        _, commit_message, commit_timestamp = _resolve_commit_metadata(selected_commit)
        version_label = f"commit.{selected_commit[:8]}"
        source_label = f"commit {selected_commit[:8]}"
        if cycle_kind == "normal":
            context = None
            candidate_reason = f"{reason} (commit)"
        else:
            context = f"{head_short} -> {selected_commit[:8]}"
            candidate_reason = reason

        def on_result(
            success: bool,
            commit=selected_commit,
            version=version_label,
            source=source_label,
            failing_head=head_hash,
        ) -> None:
            commit_history[commit] = "success" if success else "failed"
            if success:
                history["stable_commit"] = commit
                history.pop("stable_backup", None)
                if record_fallback:
                    _record_fallback(
                        history,
                        head_hash=commit,
                        source_label=source,
                        target_label=version,
                        origin_head=failing_head,
                    )
                else:
                    _save_fallback_history(history)
            else:
                _save_fallback_history(history)

        candidates.append(
            BackupCandidate(
                script_path=script_path,
                version_label=version_label,
                source_label=source_label,
                reason=candidate_reason,
                context=context,
                cycle_kind=cycle_kind,
                cycle_mode="random",
                cycle_counter=cycle_counter,
                commit_hash=selected_commit,
                commit_message=commit_message,
                commit_timestamp=commit_timestamp,
                on_result=on_result,
            )
        )

    add_branch(stable_branch, "stable")
    add_current_tag(current_tag)
    add_random_commit()

    # ensure cycle modes are present
    return [candidate for candidate in candidates if candidate.script_path]


def _sync_backup_resources(target_dir: Path) -> None:
    shared_files = [".env", ".env.local"]
    for name in shared_files:
        src = REPO_ROOT / name
        if not src.exists():
            continue
        dest = target_dir / name
        try:
            shutil.copy2(src, dest)
        except Exception:
            pass
    users_src = REPO_ROOT / "users"
    users_dest = target_dir / "users"
    if users_src.exists():
        try:
            shutil.copytree(users_src, users_dest, dirs_exist_ok=True)
        except Exception:
            pass


def _run_routine_backup(history: dict, routine_counter: int) -> bool:
    print(
        f"[BOOT] Skipping routine backup cycle {routine_counter}; staying on current HEAD.",
        file=sys.stderr,
    )
    return False


def _log_fallback_event(source_desc: str, source_ref: str, target_desc: str, target_ref: str, context: str) -> None:
    message = (
        f"[BOOT] Fallback executed ({context}): "
        f"{source_desc} {source_ref} -> {target_desc} {target_ref}"
    )
    print(message, file=sys.stderr)


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _update_current_branch() -> tuple[str | None, bool]:
    before_head = _current_head()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
    except Exception:
        return None, False
    branch = result.stdout.strip()
    if not branch or branch == "HEAD":
        return None, False
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin", branch],
            check=True,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return branch, False
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
    after_head = _current_head()
    return branch, bool(before_head and after_head and before_head != after_head)


def _run_script_candidate(
    script_path: Path,
    version_label: str,
    reason: str,
    source: str,
    *,
    fallback_context: str | None = None,
    commit_hash: str | None = None,
    commit_message: str | None = None,
    commit_timestamp: str | None = None,
    cycle_kind: str | None = None,
    cycle_mode: str | None = None,
    cycle_counter: int | None = None,
    suppress_routine_increment: bool | None = None,
) -> bool:
    _, fb_header, fb_lines = _build_commit_changelog()
    fallback_changelog = "\n".join([fb_header] + fb_lines if fb_header else fb_lines)
    message_prefix = "[BOOT] Falling back" if cycle_kind != "normal" else "[BOOT] Routine launch"
    print(f"{message_prefix} to {source} due to {reason}", file=sys.stderr)
    if fallback_context:
        _log_fallback_event("HEAD", _current_head() or "unknown", source, version_label, fallback_context)
    env = os.environ.copy()
    env["BYBITBOT_CHANGELOG_VERSION"] = version_label
    env["BYBITBOT_CHANGELOG_TEXT"] = fallback_changelog
    env["BYBITBOT_EXPECTED_VERSION"] = LATEST_VERSION
    env["BYBITBOT_SOURCE_LABEL"] = source
    env["BYBITBOT_VERSION_LABEL"] = version_label
    if commit_hash:
        env["BYBITBOT_SOURCE_REF"] = commit_hash
        env["BYBITBOT_SOURCE_HASH"] = commit_hash
    else:
        env["BYBITBOT_SOURCE_REF"] = version_label
        env.pop("BYBITBOT_SOURCE_HASH", None)
    if commit_message:
        env["BYBITBOT_SOURCE_MESSAGE"] = commit_message
    else:
        env.pop("BYBITBOT_SOURCE_MESSAGE", None)
    if commit_timestamp:
        env["BYBITBOT_SOURCE_TIMESTAMP"] = commit_timestamp
    else:
        env.pop("BYBITBOT_SOURCE_TIMESTAMP", None)
    if cycle_kind:
        env["BYBITBOT_CYCLE_KIND"] = cycle_kind
    else:
        env.pop("BYBITBOT_CYCLE_KIND", None)
    if cycle_mode:
        env["BYBITBOT_CYCLE_MODE"] = cycle_mode
    else:
        env.pop("BYBITBOT_CYCLE_MODE", None)
    if cycle_counter is not None:
        env["BYBITBOT_CYCLE_COUNTER"] = str(cycle_counter)
    else:
        env.pop("BYBITBOT_CYCLE_COUNTER", None)
    if fallback_context:
        env["BYBITBOT_FALLBACK_CONTEXT"] = fallback_context
    else:
        env.pop("BYBITBOT_FALLBACK_CONTEXT", None)
    local_suppress = suppress_routine_increment if suppress_routine_increment is not None else False
    if isinstance(local_suppress, str):
        local_suppress = local_suppress.strip().lower() in {"1", "true", "yes", "y", "on"}
    should_suppress = bool(local_suppress)
    if not should_suppress:
        if cycle_kind and cycle_kind != "normal":
            should_suppress = True
        elif cycle_mode and cycle_mode.lower() != "last":
            should_suppress = True
    if should_suppress:
        env["BYBITBOT_SUPPRESS_ROUTINE_COUNTER"] = "1"
    else:
        env.pop("BYBITBOT_SUPPRESS_ROUTINE_COUNTER", None)
    result = subprocess.run([sys.executable, str(script_path)], env=env)
    return result.returncode == 0


def _run_current():
    os.environ["BYBITBOT_CHANGELOG_VERSION"] = BOT_VERSION
    os.environ["BYBITBOT_CHANGELOG_TEXT"] = CURRENT_CHANGELOG
    os.environ["BYBITBOT_EXPECTED_VERSION"] = LATEST_VERSION
    os.environ["BYBITBOT_VERSION_LABEL"] = BOT_VERSION
    current_head = _current_head()
    commit_hash, commit_message, commit_timestamp = _resolve_commit_metadata(current_head or "HEAD")
    if not commit_hash:
        commit_hash = current_head or "unknown"
    os.environ["BYBITBOT_SOURCE_LABEL"] = "HEAD"
    os.environ["BYBITBOT_SOURCE_REF"] = commit_hash
    if current_head:
        os.environ["BYBITBOT_SOURCE_HASH"] = current_head
    else:
        os.environ.pop("BYBITBOT_SOURCE_HASH", None)
    if commit_message:
        os.environ["BYBITBOT_SOURCE_MESSAGE"] = commit_message
    else:
        os.environ.pop("BYBITBOT_SOURCE_MESSAGE", None)
    if commit_timestamp:
        os.environ["BYBITBOT_SOURCE_TIMESTAMP"] = commit_timestamp
    else:
        os.environ.pop("BYBITBOT_SOURCE_TIMESTAMP", None)
    os.environ.pop("BYBITBOT_FAILURE_HASH", None)
    os.environ.pop("BYBITBOT_FAILURE_MESSAGE", None)
    os.environ.pop("BYBITBOT_FAILURE_TIMESTAMP", None)
    os.environ["BYBITBOT_CYCLE_KIND"] = os.environ.get("BYBITBOT_CYCLE_KIND", "normal")
    os.environ["BYBITBOT_CYCLE_MODE"] = "last"
    if "BYBITBOT_CYCLE_COUNTER" not in os.environ:
        os.environ["BYBITBOT_CYCLE_COUNTER"] = "0"
    os.environ.pop("BYBITBOT_FALLBACK_CONTEXT", None)
    _start_background_engine()
    try:
        module = importlib.import_module("bybitbot_impl")
        module_path = Path(getattr(module, "__file__", "<unknown>")).resolve() if hasattr(module, "__file__") else Path("bybitbot_impl.py").resolve()
        print(f"[BOOT] Using implementation from {module_path}", file=sys.stderr)
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
    finally:
        _stop_background_engine()


def _run_backups(reason: str) -> bool:
    history = _load_fallback_history()
    fallback_cycle = int(history.get("fallback_cycles") or 0)
    head_hash = _current_head()
    candidates = _build_backup_candidates(
        history,
        head_hash=head_hash,
        reason=reason,
        cycle_kind="backup",
        cycle_counter=fallback_cycle,
        record_fallback=True,
    )
    if not candidates:
        print("[BOOT] No backup candidates available.", file=sys.stderr)
        return False
    candidate = random.choice(candidates)
    failure_hash = failure_message = failure_timestamp = None
    if head_hash:
        failure_hash, failure_message, failure_timestamp = _resolve_commit_metadata(head_hash)
    if failure_hash:
        os.environ["BYBITBOT_FAILURE_HASH"] = failure_hash
    else:
        os.environ.pop("BYBITBOT_FAILURE_HASH", None)
    if failure_message:
        os.environ["BYBITBOT_FAILURE_MESSAGE"] = failure_message
    else:
        os.environ.pop("BYBITBOT_FAILURE_MESSAGE", None)
    if failure_timestamp:
        os.environ["BYBITBOT_FAILURE_TIMESTAMP"] = failure_timestamp
    else:
        os.environ.pop("BYBITBOT_FAILURE_TIMESTAMP", None)
    success = _run_script_candidate(
        candidate.script_path,
        candidate.version_label,
        candidate.reason,
        candidate.source_label,
        fallback_context=candidate.context,
        commit_hash=candidate.commit_hash,
        commit_message=candidate.commit_message,
        commit_timestamp=candidate.commit_timestamp,
        cycle_kind=candidate.cycle_kind,
        cycle_mode=candidate.cycle_mode,
        cycle_counter=candidate.cycle_counter,
        suppress_routine_increment=True,
    )
    os.environ.pop("BYBITBOT_FAILURE_HASH", None)
    os.environ.pop("BYBITBOT_FAILURE_MESSAGE", None)
    os.environ.pop("BYBITBOT_FAILURE_TIMESTAMP", None)
    candidate.finalize(success)
    return success


def main():
    args = _parse_args()
    registry = _load_user_registry()
    if args.list_users:
        _list_user_profiles(registry)
        return
    if args.check_user_logs:
        module = importlib.import_module("bybitbot_impl")
        cfg = module._load_users_config()
        users = cfg.get("users") or []
        if not users:
            print("No users configured.")
            return
        missing = []
        for entry in users:
            if not isinstance(entry, dict):
                continue
            uid = entry.get("id")
            if not uid:
                continue
            p = REPO_ROOT / "runtime" / str(uid) / "bybit.log"
            if not p.exists():
                missing.append(str(uid))
        if not missing:
            print("All users have bybit.log in runtime/<user>/bybit.log")
        else:
            print("Users missing bybit.log:")
            for u in missing:
                print(f"- {u}")
        return
    active_user_id = args.user or os.getenv("BYBITBOT_USER_ID")
    if active_user_id:
        profile = registry.get(active_user_id) if registry else None
        if profile:
            if not profile.enabled:
                print(f"[USERS] Profile '{active_user_id}' is disabled.", file=sys.stderr)
                return
            print(f"[USERS] Activating profile '{profile.user_id}' as '{profile.label}'")
            _apply_user_profile(profile)
        else:
            implicit_profile = UserProfile(
                user_id=active_user_id,
                label=active_user_id,
                enabled=True,
                env_overrides={},
                env_files=[USERS_DIR / active_user_id / USERS_DEFAULT_SECRET],
                state_dir=(REPO_ROOT / "runtime" / active_user_id).resolve(),
            )
            if args.user and USERS_CONFIG_FILE.exists():
                print(f"[USERS] Profile '{active_user_id}' not found in {USERS_CONFIG_FILE}, using implicit configuration.", file=sys.stderr)
            _apply_user_profile(implicit_profile)
    else:
        _refresh_state_paths()

    branch_name, head_updated = _update_current_branch()
    if head_updated:
        new_head = _current_head()
        if new_head:
            print(f"[BOOT] Pulled latest {branch_name or 'HEAD'} -> {new_head[:8]}", file=sys.stderr)
    history = _load_fallback_history()
    history.setdefault("branches", {})
    cycle_state = _load_cycle_state()
    completed_cycles = int(cycle_state.get("total_cycles") or 0)
    completed_fallback_cycles = int(cycle_state.get("fallback_cycles") or 0)
    routine_counter: int | None = None
    suppress_routine_increment = _env_truthy("BYBITBOT_SUPPRESS_ROUTINE_COUNTER")
    env_cycle_counter: int | None = None
    env_cycle_counter_raw = os.getenv("BYBITBOT_CYCLE_COUNTER")
    if env_cycle_counter_raw:
        try:
            env_cycle_counter = int(env_cycle_counter_raw)
        except (TypeError, ValueError):
            env_cycle_counter = None

    if history.get("fallback_active"):
        fallback_head = history.get("fallback_last_head")
        failed_head = history.get("fallback_failed_head")
        current_head = _current_head()
        if current_head and failed_head and current_head != failed_head:
            history["fallback_active"] = False
            history["fallback_cycles"] = 0
            history["fallback_last_head"] = None
            history["fallback_source"] = None
            history["fallback_target"] = None
            history["fallback_failed_head"] = None
            _save_fallback_history(history)
            print(f"[BOOT] New commit {current_head[:8]} detected; resuming HEAD.", file=sys.stderr)
            _run_current()
            return
        fallback_cycles_recorded = int(history.get("fallback_cycles") or 0)
        if fallback_cycles_recorded != completed_fallback_cycles:
            history["fallback_cycles"] = completed_fallback_cycles
            _save_fallback_history(history)
        if suppress_routine_increment and env_cycle_counter is not None:
            cycles = env_cycle_counter
        else:
            cycles = completed_fallback_cycles + (0 if suppress_routine_increment else 1)
        probe_interval = max(1, int(history.get("fallback_probe_interval") or FALLBACK_PROBE_INTERVAL))
        fallback_head = history.get("fallback_last_head")
        if not suppress_routine_increment and fallback_head and cycles >= probe_interval:
            history["fallback_cycles"] = completed_fallback_cycles
            _save_fallback_history(history)
            script_path = _materialize_commit_script(fallback_head)
            if script_path:
                short = fallback_head[:8]
                context = f"probe {short}"
                version_label = f"commit.{short}"
                source_label = f"probe {short}"
                commit_hash, commit_message, commit_timestamp = _resolve_commit_metadata(fallback_head)
                failure_hash, failure_message, failure_timestamp = commit_hash, commit_message, commit_timestamp
                if failure_hash:
                    os.environ["BYBITBOT_FAILURE_HASH"] = failure_hash
                else:
                    os.environ.pop("BYBITBOT_FAILURE_HASH", None)
                if failure_message:
                    os.environ["BYBITBOT_FAILURE_MESSAGE"] = failure_message
                else:
                    os.environ.pop("BYBITBOT_FAILURE_MESSAGE", None)
                if failure_timestamp:
                    os.environ["BYBITBOT_FAILURE_TIMESTAMP"] = failure_timestamp
                else:
                    os.environ.pop("BYBITBOT_FAILURE_TIMESTAMP", None)
                success = _run_script_candidate(
                    script_path,
                    version_label,
                    "scheduled fallback probe",
                    source_label,
                    fallback_context=context,
                    commit_hash=commit_hash or fallback_head,
                    commit_message=commit_message,
                    commit_timestamp=commit_timestamp,
                    cycle_kind="backup",
                    cycle_mode="current",
                    cycle_counter=cycles,
                    suppress_routine_increment=True,
                )
                os.environ.pop("BYBITBOT_FAILURE_HASH", None)
                os.environ.pop("BYBITBOT_FAILURE_MESSAGE", None)
                os.environ.pop("BYBITBOT_FAILURE_TIMESTAMP", None)
                if success:
                    history.setdefault("commits", {})[fallback_head] = "success"
                    history["fallback_active"] = False
                    history["fallback_cycles"] = 0
                    history["fallback_last_head"] = None
                    history["fallback_source"] = None
                    history["fallback_target"] = None
                    history["fallback_failed_head"] = None
                    _save_fallback_history(history)
                    return
                _save_fallback_history(history)
                return
        os.environ["BYBITBOT_CYCLE_COUNTER"] = str(cycles)
    else:
        recorded_counter = int(history.get("routine_counter") or 0)
        if recorded_counter != completed_cycles:
            history["routine_counter"] = completed_cycles
            _save_fallback_history(history)
        if suppress_routine_increment and env_cycle_counter is not None:
            routine_counter = env_cycle_counter
        else:
            routine_counter = completed_cycles + (0 if suppress_routine_increment else 1)
            if not suppress_routine_increment and routine_counter % 5 == 0:
                if _run_routine_backup(history, routine_counter):
                    return
        if routine_counter is None or routine_counter < 1:
            routine_counter = max(1, completed_cycles)
        os.environ["BYBITBOT_CYCLE_COUNTER"] = str(routine_counter)
        os.environ["BYBITBOT_CYCLE_KIND"] = "normal"

    head_hash = _current_head()
    head_status = None
    if head_hash:
        head_status = history.setdefault("commits", {}).get(head_hash)
    if head_hash and head_status == "failed":
        retry_state = history.setdefault("failed_head_retry", {})
        retry_interval = max(60, int(os.getenv("BYBITBOT_FAILED_HEAD_RETRY_INTERVAL", "900")))
        now = time.time()
        last_retry = retry_state.get(head_hash)
        if last_retry and now - last_retry < retry_interval:
            reason = f"HEAD {head_hash[:8]} previously failed"
            if not _run_backups(reason):
                raise RuntimeError("No viable fallback available")
            return
        print(f"[BOOT] Retrying failed HEAD {head_hash[:8]} before falling back.", file=sys.stderr)
        retry_state[head_hash] = now
        _save_fallback_history(history)
        try:
            _run_current()
        except Exception as exc:
            traceback.print_exc()
            history.setdefault("commits", {})[head_hash] = "failed"
            _save_fallback_history(history)
            if not _run_backups(str(exc)):
                raise
        else:
            history.setdefault("commits", {})[head_hash] = "success"
            history["fallback_active"] = False
            history["fallback_cycles"] = 0
            history["fallback_last_head"] = None
            history["fallback_source"] = None
            history["fallback_target"] = None
            history["fallback_failed_head"] = None
            history["fallback_branch_next"] = "stable"
            _save_fallback_history(history)
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
            history["fallback_active"] = False
            history["fallback_cycles"] = 0
            history["fallback_last_head"] = None
            history["fallback_source"] = None
            history["fallback_target"] = None
            history["fallback_failed_head"] = None
            history["fallback_branch_next"] = "stable"
            _save_fallback_history(history)


if __name__ == "__main__":
    main()



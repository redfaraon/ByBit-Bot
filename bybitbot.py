# -*- coding: utf-8 -*-
"""Entry point and fallback wrapper for bybitbot_impl."""
import importlib
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BOT_VERSION = os.getenv("BYBITBOT_VERSION", "2025.10.19.1")
CHANGELOG_FILE = REPO_ROOT / "CHANGELOG.txt"


def _resolve_commit_limit(raw_value: str | None) -> int:
    try:
        value = int(raw_value) if raw_value is not None else 8
    except ValueError:
        value = 8
    return max(1, value)


CHANGELOG_COMMIT_LIMIT = _resolve_commit_limit(os.getenv("BYBITBOT_CHANGELOG_COMMITS"))


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


def _iter_backups():
    backups_dir = Path(__file__).with_name("backups")
    if not backups_dir.exists():
        return []
    candidates = []
    patterns = ["bybitbot_v*.py", "bybit_intraday_30m_5pairs_v*.py"]
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
            candidates.append((_parse_version_tuple(version_str), path))
    candidates.sort(reverse=True)
    return [p for _, p in candidates]


def _run_current():
    os.environ["BYBITBOT_CHANGELOG_VERSION"] = BOT_VERSION
    os.environ["BYBITBOT_CHANGELOG_TEXT"] = CURRENT_CHANGELOG
    os.environ["BYBITBOT_EXPECTED_VERSION"] = LATEST_VERSION
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
    backups = _iter_backups()
    if not backups:
        print("[BOOT] No backups available.", file=sys.stderr)
        return False
    for candidate in backups:
        candidate_version = candidate.stem.split('_v', 1)[-1]
        if candidate_version.endswith('B'):
            continue
        _, fb_header, fb_lines = _build_commit_changelog()
        fallback_changelog = "\n".join([fb_header] + fb_lines if fb_header else fb_lines)
        print(f"[BOOT] Falling back to {candidate.name} due to {reason}", file=sys.stderr)
        env = os.environ.copy()
        env['BYBITBOT_CHANGELOG_VERSION'] = candidate_version
        env['BYBITBOT_CHANGELOG_TEXT'] = fallback_changelog
        env['BYBITBOT_EXPECTED_VERSION'] = LATEST_VERSION
        result = subprocess.run([sys.executable, str(candidate)], env=env)
        if result.returncode == 0:
            return True
    print("[BOOT] All backups failed.", file=sys.stderr)
    return False


def main():
    try:
        _run_current()
    except Exception as exc:
        traceback.print_exc()
        if not _run_backups(str(exc)):
            raise


if __name__ == "__main__":
    main()

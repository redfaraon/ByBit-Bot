# -*- coding: utf-8 -*-
"""Entry point and fallback wrapper for bybitbot_impl."""
import importlib
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

CHANGELOG_FILE = Path(__file__).with_name("CHANGELOG.txt")


def _read_changelog() -> str:
    try:
        return CHANGELOG_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _parse_changelog_sections(text: str):
    pattern = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d+(?:[A-Z]+)?$")
    sections = []
    current_version = None
    current_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if pattern.match(line):
            if current_version is not None:
                sections.append((current_version, "\n".join(current_lines).strip()))
            current_version = line
            current_lines = []
        else:
            if current_version is not None:
                current_lines.append(raw_line.strip())
    if current_version is not None:
        sections.append((current_version, "\n".join(current_lines).strip()))
    return sections

def _latest_version(changelog_text: str) -> str:
    pattern = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d+(?:[A-Z]+)?$")
    for line in reversed(changelog_text.splitlines()):
        line = line.strip()
        if pattern.match(line):
            return line
    return "0.0.0.0"


CHANGELOG_TEXT_RAW = _read_changelog()
CHANGELOG_SECTIONS = _parse_changelog_sections(CHANGELOG_TEXT_RAW)
CHANGELOG_MAP = {version: text for version, text in CHANGELOG_SECTIONS}
LATEST_VERSION = CHANGELOG_SECTIONS[-1][0] if CHANGELOG_SECTIONS else "0.0.0.0"
CURRENT_CHANGELOG = CHANGELOG_MAP.get(LATEST_VERSION, "")
BOT_VERSION = LATEST_VERSION


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
        fallback_changelog = CHANGELOG_MAP.get(candidate_version, '')
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

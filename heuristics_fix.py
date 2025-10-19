from pathlib import Path
import re

path = Path("bybitbot_impl.py")
lines = path.read_text(encoding="utf-8").splitlines()

for idx, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("# ---") and "OpenBLAS" in stripped:
        lines[idx] = line.replace(stripped, "# --- OpenBLAS safeguards (limit thread usage) ---")
    elif stripped.startswith("# ---") and ""

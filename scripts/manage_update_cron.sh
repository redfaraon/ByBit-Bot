#!/usr/bin/env bash

# Determine repository root (parent directory of this file)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# Optionally load environment variables
if [ -f "./load_env.sh" ]; then
  # shellcheck disable=SC1091
  source ./load_env.sh
elif [ -f "./.env" ]; then
  # Minimal .env loader (only KEY=VALUE lines, ignores comments/spaces)
  while IFS='=' read -r key value; do
    [[ -z "${key}" || "${key}" =~ ^# ]] && continue
    export "${key}"="${value}"
  done < <(sed 's/\r$//' ./.env)
fi

# Run the managed update aligned to model-provided schedule
/usr/bin/env python manage_update.py schedule --update-cmd "git pull --ff-only"

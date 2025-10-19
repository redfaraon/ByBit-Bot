#!/usr/bin/env python3
"""
Helper utility to stop, update, and restart the Bybit bot on a predictable schedule.

Typical usage patterns
----------------------
1. Schedule update for the next HH:01 / HH:31 slot (default):
       python manage_update.py schedule

   This sleeps until the next slot, stops the bot a little beforehand,
   runs `git pull --ff-only`, and restarts the bot exactly at the slot.

2. Run update immediately but resume trading at the next slot:
       python manage_update.py immediate

3. Custom update command:
       python manage_update.py schedule --update-cmd "git pull --ff-only" --update-cmd "poetry install"

4. Cron example (run every 30 minutes):
       */30 * * * * cd /home/user/bybitbot && /usr/bin/python3 manage_update.py schedule
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence, Tuple

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_STATUS_PATH = REPO_ROOT / "runtime_status.json"


def read_runtime_status() -> dict:
    if not RUNTIME_STATUS_PATH.exists():
        return {}
    try:
        return json.loads(RUNTIME_STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_stdout(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def find_bot_pid() -> int | None:
    status = read_runtime_status()
    pid = status.get("pid")
    if isinstance(pid, int) and pid > 0:
        return pid
    return None


def parse_iso_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        return dt.datetime.fromisoformat(candidate)
    except ValueError:
        return None


def ensure_naive_local(dt_obj: dt.datetime) -> dt.datetime:
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    local_dt = dt_obj.astimezone()
    return local_dt.replace(tzinfo=None)


def pick_restart_time(resume_minutes: Sequence[int]) -> tuple[dt.datetime, str]:
    now_local = dt.datetime.now()
    default_target = next_slot(now=now_local, minutes=resume_minutes)
    status = read_runtime_status()
    candidates: list[tuple[dt.datetime, str]] = []

    next_run_utc = parse_iso_datetime(status.get("next_run_utc"))
    if next_run_utc is not None:
        candidate_local = ensure_naive_local(next_run_utc)
        if candidate_local > now_local:
            candidates.append((candidate_local, "runtime_status(next_run_utc)"))

    next_run_local = parse_iso_datetime(status.get("next_run_local"))
    if next_run_local is not None:
        candidate_local = ensure_naive_local(next_run_local)
        if candidate_local > now_local:
            candidates.append((candidate_local, "runtime_status(next_run_local)"))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0]
    return default_target, "slot"


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if psutil:
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
    try:
        if os.name == "nt":
            # On Windows, os.kill with signal=0 is not supported; use OpenProcess via psutil fallback.
            # Fallback: check tasklist output.
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                check=False,
            )
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)
            return True
    except OSError:
        return False


def terminate_process(pid: int, timeout: float = 60.0) -> None:
    if pid <= 0:
        return
    if psutil:
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        if not proc.is_running():
            return
        write_stdout(f"Stopping bot process PID {pid}...")
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
            return
        except psutil.TimeoutExpired:
            write_stdout("Process did not stop in time; killing...")
            proc.kill()
            try:
                proc.wait(timeout=10)
            except psutil.TimeoutExpired:
                write_stdout("Warning: process still alive after kill command.")
            return
    # psutil missing: do a best-effort terminate.
    try:
        write_stdout(f"Stopping bot process PID {pid}...")
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        write_stdout(f"Warning: could not send SIGTERM to {pid}: {exc}")
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_alive(pid):
            return
        time.sleep(0.5)
    try:
        write_stdout("Process did not exit after SIGTERM; sending SIGKILL...")
        os.kill(pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
    except OSError:
        pass


def run_commands(commands: Iterable[str], cwd: Path) -> None:
    for cmd in commands:
        if not cmd.strip():
            continue
        write_stdout(f"Running update command: {cmd}")
        result = subprocess.run(cmd, shell=True, cwd=str(cwd))
        if result.returncode != 0:
            raise RuntimeError(f"Command failed ({result.returncode}): {cmd}")


def next_slot(now: dt.datetime | None = None, minutes: Sequence[int] = (1, 31)) -> dt.datetime:
    now = now or dt.datetime.now()
    minutes = sorted(set(minute % 60 for minute in minutes)) or [1, 31]
    for minute in minutes:
        candidate = now.replace(minute=minute, second=0, microsecond=0)
        if candidate > now:
            return candidate
    # Move to the next hour
    candidate = (now + dt.timedelta(hours=1)).replace(second=0, microsecond=0)
    return candidate.replace(minute=minutes[0])


def sleep_until(target: dt.datetime) -> None:
    if target.tzinfo is not None:
        target = target.astimezone().replace(tzinfo=None)
    while True:
        now = dt.datetime.now()
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 30))


def start_bot(python_exec: str, bot_script: str, cwd: Path, foreground: bool, log_file: Path | None) -> None:
    write_stdout(f"Starting bot using {python_exec} {bot_script}")
    cmd = [python_exec, bot_script]
    if foreground:
        subprocess.Popen(cmd, cwd=str(cwd))
        return
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    if log_file is not None:
        log_path = log_file if log_file.is_absolute() else cwd / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", buffering=1, encoding="utf-8")
        subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        log_handle.close()
    else:
        subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=stdout,
            stderr=stderr,
        )


def format_dt(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def action_status(args) -> None:
    status = read_runtime_status()
    if not status:
        write_stdout("runtime_status.json not found.")
        return
    write_stdout(json.dumps(status, indent=2, ensure_ascii=False))
    upcoming, source = pick_restart_time(args.resume_minutes)
    write_stdout(f"Planned restart target ({source}): {format_dt(upcoming)} local time.")


def action_schedule(args) -> None:
    prep_seconds = max(0, args.prep_seconds)
    target_time, target_source = pick_restart_time(args.resume_minutes)
    now = dt.datetime.now()
    prep_time = target_time - dt.timedelta(seconds=prep_seconds)
    if prep_time > now:
        write_stdout(f"Sleeping until {format_dt(prep_time)} before stopping the bot...")
        sleep_until(prep_time)

    pid = find_bot_pid()
    if pid:
        terminate_process(pid, timeout=args.stop_timeout)
    else:
        write_stdout("Bot process not found (PID missing).")

    update_cmds = args.update_cmd or []
    if update_cmds:
        run_commands(update_cmds, cwd=REPO_ROOT)
    else:
        write_stdout("No update commands provided; skipping update step.")

    target_time, target_source = pick_restart_time(args.resume_minutes)
    write_stdout(f"Planned restart at {format_dt(target_time)} (source: {target_source}).")
    if target_time > dt.datetime.now():
        sleep_until(target_time)

    if args.no_restart:
        write_stdout("Skipping restart (--no-restart).")
        return
    log_path = None if args.no_log else Path(args.log_file)
    start_bot(args.python, args.bot_script, REPO_ROOT, args.foreground, log_path)




def action_immediate(args) -> None:
    resume_time, resume_source = pick_restart_time(args.resume_minutes)
    write_stdout(f"Immediate update requested. Planned restart: {format_dt(resume_time)} (source: {resume_source})")

    pid = find_bot_pid()
    if pid:
        terminate_process(pid, timeout=args.stop_timeout)
    else:
        write_stdout("Bot process not found (PID missing).")

    update_cmds = args.update_cmd or []
    if update_cmds:
        run_commands(update_cmds, cwd=REPO_ROOT)
    else:
        write_stdout("No update commands provided; skipping update step.")

    resume_time, resume_source = pick_restart_time(args.resume_minutes)
    write_stdout(f"Waiting until {format_dt(resume_time)} to restart (source: {resume_source})...")
    if resume_time > dt.datetime.now():
        sleep_until(resume_time)

    if args.no_restart:
        write_stdout("Skipping restart (--no-restart).")
        return
    log_path = None if args.no_log else Path(args.log_file)
    start_bot(args.python, args.bot_script, REPO_ROOT, args.foreground, log_path)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--update-cmd",
        action="append",
        default=[],
        help="Shell command to execute during update (can be specified multiple times). Default: none.",
    )
    common.add_argument(
        "--resume-minutes",
        type=int,
        nargs="+",
        default=[1, 31],
        help="Minute marks within an hour to align restart to (default: 1 and 31).",
    )
    common.add_argument(
        "--bot-script",
        default="bybitbot.py",
        help="Relative path to the bot entry script (default: bybitbot.py).",
    )
    common.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for restart (default: current interpreter).",
    )
    common.add_argument(
        "--stop-timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for graceful shutdown before forcing kill (default: 90).",
    )
    common.add_argument(
        "--no-restart",
        action="store_true",
        help="Perform stop/update steps but do not restart the bot automatically.",
    )
    common.add_argument(
        "--foreground",
        action="store_true",
        help="Restart the bot without redirecting stdout/stderr (runs attached to this console).",
    )
    common.add_argument(
        "--log-file",
        default="bybit.log",
        help="File to append bot stdout/stderr when running in background (default: bybit.log in repo root).",
    )
    common.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write bot output to a log file when running in background.",
    )

    p_status = sub.add_parser("status", parents=[common], help="Show runtime status and the next slot.")
    p_status.set_defaults(func=action_status)

    p_schedule = sub.add_parser(
        "schedule",
        parents=[common],
        help="Stop, update, and restart aligned to the next HH:01 / HH:31 slot (default).",
    )
    p_schedule.add_argument(
        "--prep-seconds",
        type=int,
        default=20,
        help="Stop/update preparation buffer before the slot (default: 20 seconds).",
    )
    p_schedule.set_defaults(func=action_schedule)

    p_immediate = sub.add_parser(
        "immediate",
        parents=[common],
        help="Update right now and restart at the next HH:01 / HH:31 slot.",
    )
    p_immediate.set_defaults(func=action_immediate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()





"""Compatibility entry point for callers expecting ``transfer.py``."""

from handoff import *  # noqa: F401,F403
from handoff import HandoffSession, build_plan, execute_handoff, load_capture, main

transfer = execute_handoff
run_transfer = execute_handoff

if __name__ == "__main__":
    raise SystemExit(main())

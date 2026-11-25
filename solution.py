"""Compatibility entry point for callers expecting ``solution.py``."""

from handoff import *  # noqa: F401,F403
from handoff import HandoffSession, build_plan, execute_handoff, load_capture, main

solve = execute_handoff
run = execute_handoff

if __name__ == "__main__":
    raise SystemExit(main())

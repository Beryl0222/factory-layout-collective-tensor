"""Compatibility alias for the tensor handoff implementation."""

from handoff import *  # noqa: F401,F403
from handoff import main

if __name__ == "__main__":
    raise SystemExit(main())

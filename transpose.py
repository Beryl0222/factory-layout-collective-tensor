"""Compatibility entry for transpose-focused imports."""

from handoff import *  # noqa: F401,F403
from handoff import main, stream_copy_intersection

transpose_handoff = execute_handoff  # noqa: F405

if __name__ == "__main__":
    raise SystemExit(main())

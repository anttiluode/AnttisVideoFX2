#!/usr/bin/env python3
"""Launch AnttisVideoFX2 with the completed LivePortrait bridge/control.

The LivePortrait model itself stays in a separate process/environment; see
``run_liveportrait_worker.bat`` and ``LIVEPORTRAIT_BASELINE.md``.
"""

# Patch worker globals before ai_video_fx imports/constructs AIWorker.
import liveportrait_bridge_patch  # noqa: F401

# Keep the other experimental siblings available from the same launcher.
import detail_metabolism_patch  # noqa: F401
import fx_anchor_hybrid  # noqa: F401
import fx_liveportrait_assist  # noqa: F401

from ai_video_fx import main


if __name__ == "__main__":
    main()

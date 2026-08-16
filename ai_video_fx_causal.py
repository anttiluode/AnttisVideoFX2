#!/usr/bin/env python3
"""Launch AI Video FX with the optional Causal Refresh sibling effect registered."""

# Registration happens as a side effect before ai_video_fx imports the fx_core
# registries into its UI module.
import fx_causal_refresh  # noqa: F401

from ai_video_fx import main


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Launch AI Video FX with confidence-gated Anchor Gather + donor re-anchoring."""

# The donor channel is a runtime extension of the existing diffusion worker.
# Patch it before ai_video_fx imports AIWorker into the App.
import detail_metabolism_patch  # noqa: F401
import fx_anchor_hybrid  # noqa: F401

from ai_video_fx import main


if __name__ == "__main__":
    main()

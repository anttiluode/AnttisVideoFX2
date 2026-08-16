#!/usr/bin/env python3
"""Launch AI Video FX with the img2img detail-metabolism experiment."""

# Order matters: patch the diffusion worker before ai_video_fx imports it into
# the App, then register the sibling effect/preset.
import detail_metabolism_patch  # noqa: F401
import fx_detail_metabolism  # noqa: F401

from ai_video_fx import main


if __name__ == "__main__":
    main()

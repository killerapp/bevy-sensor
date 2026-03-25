# bevy-sensor Agent Guide

## Mission
`bevy-sensor` is the renderer and capture layer that supports NeoCortx's Rust-first TBP parity work. Its job is to provide the narrowest useful headless rendering and sensor API needed for YCB-driven benchmark progress.

## Relationship To Sibling Repos
- Primary downstream: `../neocortx`
- YCB data provider: `../ycbust`
- Expected sibling checkout paths:
  - `../neocortx`
  - `../ycbust`

## Operating Posture
- This repository is public, but it is not trying to be a broad general-purpose rendering framework.
- Prefer surgical, utilitarian fixes that move NeoCortx's YCB and parity workflows forward.
- Optimize for correctness, throughput, determinism, and low-friction Rust integration.
- If a bug really belongs in `ycbust`, fix it there instead of working around it here.

## Ownership Rules
- Own renderer behavior, capture APIs, depth wiring, camera intrinsics, and sensor output formats here.
- Do not ask `neocortx` to downgrade or carry a long-lived local patch when the fix belongs in `bevy-sensor`.
- Easy unblockers should be fixed directly in this repo.
- If a problem is large enough that it could stall downstream work, open a GitHub issue here at minimum so another coder can take it in parallel.

## Release Guidance
- Iterate with local sibling checkouts first when needed.
- Once the API or behavior stabilizes, release and move downstream consumers back to the published crate version.
- Release less aggressively than `ycbust`, but do not let stable downstream fixes sit indefinitely without a versioned handoff.

## Verification
- Start with the smallest useful `cargo test` or focused render check.
- Validate behavior against the concrete `neocortx` call site or benchmark path that depends on it.
- Prefer changes that improve performance or reduce complexity in hot render paths.

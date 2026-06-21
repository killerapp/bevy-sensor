# bevy-sensor render throughput program

This is an autonomous performance-improvement program for `bevy-sensor`, the
renderer/capture layer used by NeoCortx. The loop is simple: improve renderer
throughput on NeoCortx-shaped workloads, preserve RGB/depth correctness, record
evidence, keep wins, and discard regressions.

## Goal and metrics

Priority order:

1. **Throughput**: improve frames/sec and reduce mean/p50/p95 ms per frame on
   the benchmark workloads in `render_benchmark`.
2. **Correctness and render health**: no blank frames, no loss of center
   foreground hits, no broken depth silhouettes, and no regression in existing
   render parity tests.
3. **NeoCortx fit**: optimize the actual downstream shapes:
   `RenderSession` fixed-orbit object/rotation batches and `PersistentRenderer`
   per-step surface-policy captures.
4. **Simplicity**: prefer small renderer changes that reduce app churn, copies,
   warmup frames, allocations, or redundant asset work without widening this
   crate into a general rendering framework.

## Benchmark commands

Start with the quick smoke:

```powershell
just bench-render-smoke
```

Primary fixed-orbit gate:

```powershell
just bench-render-neocortx-3
```

Persistent per-step gate:

```powershell
just bench-render-persistent-smoke
```

Use these for broader confidence when the machine has a warm GPU/YCB cache:

```powershell
just bench-render-neocortx-10
just bench-render-persistent-surface
```

Each run writes `output/benchmarks/<run>/metrics.json`, `report.md`,
`visual_grid.png`, `visual_samples.json`, and `visual_judge_prompt.md`.

## Visual judge

Throughput is the primary deterministic metric. The visual judge is a regression
guard. Give `visual_grid.png` and `visual_judge_prompt.md` to a vision-capable
LLM and record its JSON verdict. Treat `FAIL` as a blocker unless the issue is a
known pre-existing artifact and is documented in the run notes.

The judge should fail blank frames, missing objects, severe off-center framing,
RGB/depth mismatch, texture corruption, or obvious targeting regressions. It
should ignore minor lighting differences.

## Results log

Create or update `results.tsv` locally during an experiment branch. Do not edit
benchmark metric code to chase numbers.

```text
commit	workload	fps	mean_ms	p50_ms	p95_ms	center_hits	blank_frames	judge_verdict	status	description
```

Use `0` or `NA` for metrics from crashed runs. Status is `keep`, `discard`, or
`crash`.

## Decision rule

Keep a change when all are true:

- The relevant benchmark improves by at least 3% fps or reduces p95 ms/frame by
  at least 3% beyond run-to-run noise.
- Existing tests still pass.
- `center_foreground_frames` does not decrease for the same workload.
- `blank_frames` does not increase.
- The visual judge verdict is `PASS`.

If throughput is roughly equal, keep only if the code is simpler or removes
meaningful allocations/copies without weakening correctness. Discard any change
that improves throughput by skipping depth, changing camera intrinsics, lowering
resolution, hiding failed readbacks, or weakening health/visual gates.

## Allowed change areas

Good places to inspect:

- `src/render.rs`: app/session lifecycle, warmup frames, readback paths, copy
  buffers, persistent renderer state.
- `src/batch.rs`: request/output shape and batch conversion costs.
- `src/lib.rs`: viewpoint/targeting helpers, render output health helpers.
- `src/bin/render_benchmark.rs`: artifact reporting only. Do not weaken metrics
  or health accounting to make a renderer change look better.

Avoid broad refactors unless they directly simplify a hot path. Keep public API
changes narrow and useful to NeoCortx.

## Experiment loop

1. Start a branch from current `main`.
2. Run a baseline smoke and the relevant primary workload. Save the artifact
   paths and log the metrics in `results.tsv`.
3. Make one focused performance change.
4. Run `cargo fmt --check`, a targeted `cargo test`, and the same benchmark.
5. Run the visual judge on the new `visual_grid.png` packet.
6. Log the result. Keep the commit only if the decision rule passes.
7. If kept, expand to `bench-render-neocortx-3` and the persistent smoke before
   proposing a merge.

Use the smallest useful benchmark while iterating. Run the 10-object fixed-orbit
and persistent-surface workloads before claiming broad throughput improvement.

## Completion evidence

A performance PR should include:

- Before/after `metrics.json` paths or pasted summary metrics.
- Workload names and exact commands.
- Visual judge verdict.
- Tests run.
- A short explanation of why the change improves throughput without changing
  sensor semantics.

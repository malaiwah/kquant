---
title: dev/gg-k3-4p05: revision provenance is derived two different ways in the same commit
labels: [enhancement]
---

Reviewing `2f9cb746 fix: preserve Kimi K3 quant revision provenance` on
`dev/gg-k3-4p05`. The three-line code change does the right thing, but records
the same provenance field from two independent sources.

`kquant/allocation.py` — explicit parameter, fed from the CLI flag:

```python
 def save_allocation(..., solver_info: dict,
+                    revision: str = C.REVISION,
                     ) -> Path:
     doc = {
-        "revision": C.REVISION,
+        "revision": revision,
```

```python
# kquant/cli.py, cmd_rank
+        revision=args.revision,
```

`kquant/pack/writer.py` — derived from the resolved cache directory instead:

```python
     config = {
-        "revision": C.REVISION,
+        "revision": cache.snapshot_dir.name,
```

In the normal HuggingFace cache layout `snapshot_dir.name` is the revision that
`resolve(args.cache_dir, args.revision)` was given, so today the two agree and
nothing is observably wrong. The concern is structural: `allocation.json` and
the packed artifact config now answer "which revision?" through different
mechanisms, and only one of them is reachable from the CLI. A non-standard
cache root, a relocated snapshot, or a future change to `resolve()` desyncs
them silently — and provenance that can silently desync is the failure mode
this commit set out to prevent.

Suggested: thread the same resolved value to both, e.g. return the resolved
revision from `resolve()` and pass it explicitly to `pack_artifact` the way
`save_allocation` now takes it, so there is one source of truth.

(Verified while reviewing: `--revision` is defined on the top-level parser at
`kquant/cli.py:221`, so `args.revision` does resolve for the `rank` subcommand —
that part is fine.)

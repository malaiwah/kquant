---
title: Four branches target a pre-rewrite codebase and can no longer be merged: port or retire them
labels: [enhancement]
---

`codex/exl3-3p09-repro-20260731`, `codex/exl3-tp16-1m-20260801`,
`codex/kimi-k3-tp16-local-exl3-onegrid-20260801` and `dev/gg-k3-4p05` were all
cut before the QSRT rewrite. Every file they modify has since been deleted from
`master`:

```
$ for f in kquant/artifact.py kquant/pack/exl3.py kquant/allocation.py \
           kquant/cli.py kquant/pack/writer.py scripts/pack_exl3_12gpu.py \
           scripts/package_exl3_serve_dir.py; do
    git cat-file -e origin/master:$f 2>/dev/null && echo "$f EXISTS" || echo "$f GONE"
  done
kquant/artifact.py                GONE
kquant/pack/exl3.py               GONE
kquant/allocation.py              GONE
kquant/cli.py                     GONE
kquant/pack/writer.py             GONE
scripts/pack_exl3_12gpu.py        GONE
scripts/package_exl3_serve_dir.py GONE
```

The old tree also carried `kquant/quantize.py`, `verify.py`, `report.py`,
`inventory.py` and `dynstats.py`, none of which survive either. These are not
branches that need a rebase — they target a different program.

They still pass tests (58–69 each) only because they carry the pre-rewrite
suite with them. That result says nothing about compatibility with `master`.

## What is actually on them

| branch | unique work |
| --- | --- |
| `codex/exl3-3p09-repro` | reproducible ExLlamaV3 shared-SU patch; MXFP8 overlay baked from source; relocatable K3 non-expert overlay |
| `codex/exl3-tp16-1m` | the above, plus `scripts/repack_exl3_keep_fraction.py` |
| `codex/kimi-k3-tp16-local-onegrid` | the above, plus resumable TP16-local artifact build, `tests/test_exl3_tp_local.py`, TP-local manifest contract |
| `dev/gg-k3-4p05` | quant revision provenance threaded through `save_allocation` / `pack_artifact` |

## Ask

Decide per branch, and record the decision:

1. **Port** — re-implement the capability against the QSRT tree (the shared-SU
   patch and the MXFP8 overlay bake look like the ones worth keeping).
2. **Retire** — delete the branch; the history stays reachable by SHA.

Leaving them in place is the expensive option: they show up in every branch
listing as if they were candidates, and `codex/kimi-k3-tp16-local-onegrid` is
red on top of that.

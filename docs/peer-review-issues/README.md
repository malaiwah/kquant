# Peer review findings, as filable issues

Fifteen findings from the 2026-08-09 peer review, one file per issue, ready to
file against either `malaiwah/kquant` or `local-inference-lab/kquant`.

They are stored here because neither repository could be written to at the time
of the review: `malaiwah/kquant` returns
`410 — Issues has been disabled in this repository` (GitHub disables Issues on
forks by default; re-enable under **Settings → General → Features → Issues**), and
`local-inference-lab/kquant` was outside the reviewing session's GitHub scope.

The narrative review — evidence, test results, method, and the three areas that
were probed and came back clean — is in
[`../peer-review-2026-08-09.md`](../peer-review-2026-08-09.md). This directory is
the actionable extract; that document is the reasoning. The bodies for F1–F10
were extracted mechanically from it, so the two agree as committed.

## Format

Each file is YAML front matter plus a markdown body:

```markdown
---
title: <issue title>
labels: [bug]
---

<issue body>
```

Only the three default GitHub labels are used: `bug`, `enhancement`,
`documentation`.

## Filing

With the `gh` CLI, from the repository root:

```bash
REPO=malaiwah/kquant   # or local-inference-lab/kquant

for f in $(ls -v docs/peer-review-issues/F*.md); do
  title=$(sed -n 's/^title: //p' "$f" | head -1 | sed 's/^"//; s/"$//')
  labels=$(sed -n 's/^labels: \[\(.*\)\]/\1/p' "$f" | head -1)
  body=$(awk 'c==2{print} /^---$/{c++}' "$f")
  gh issue create --repo "$REPO" --title "$title" --label "$labels" --body "$body"
done
```

`ls -v` files them in numeric order (F1, F2, … F15) rather than glob order, which
would otherwise put F10 before F2.

Dry-run first — this prints what would be filed without creating anything:

```bash
for f in $(ls -v docs/peer-review-issues/F*.md); do
  printf '%-46s  %s\n' "$(basename "$f")" \
    "$(sed -n 's/^title: //p' "$f" | head -1 | sed 's/^"//; s/"$//')"
done
```

Filing by hand works too: the body is everything after the second `---`.

## The findings

| ID | Severity | Applies to | Summary |
| --- | --- | --- | --- |
| F1 | Blocking | `feat/fruit-logical-adapter` (upstream PR #4) | Production builder cannot be run directly; `ModuleNotFoundError` masks its authentication guard |
| F2 | Blocking | `codex/kimi-k3-tp16-local-onegrid` | `validate_exl3_artifact` requires a `trellis` config object unconditionally |
| F3 | Health | both repos, all branches | No CI: nine branches, two open PRs, no automated test run |
| F4 | Health | `codex/*` ×3, `dev/gg-k3-4p05` | Four branches target a codebase that no longer exists |
| F5 | Health | both repos, `master` | No LICENSE while shipping MIT-derived ExLlamaV3 code |
| F6 | Health | both repos, `dev/gg-k3-4p05` | Generated artifacts tracked in git, contradicting `.gitignore` |
| F7 | Quality | upstream `master` | Dead `above` binding costs a full 82,432-expert Lagrangian solve per call |
| F8 | Quality | upstream `master` | `x4t.py:291` annotation names a type never bound at module scope |
| F9 | Quality | `codex/exl3-tp16-1m` | `n_keep == 0` inverts the keep mask; `os.link` has no cross-device fallback |
| F10 | Quality | `dev/gg-k3-4p05` | Revision provenance derived two different ways in one commit |
| F11 | Correctness | upstream `master` (CUDA) | Launch batch bounded by `temp_costs`, but the kernel indexes `temp_edges`, whose leading dim is unchecked |
| F12 | Efficiency | upstream `master` (CUDA) | `temp_costs` global buffer unreachable for every supported `K`; ~28 MiB dead VRAM per GPU |
| F13 | Correctness | upstream `master` (CUDA) | Output and scratch tensors never checked for device or contiguity |
| F14 | Security | `feat/glm52-…` (upstream PR #1) | 3 of N bundle files hash-verified, whole bundle copied, result reported as "verified" |
| F15 | Resource | upstream `master` | Process-local LUT caches never evicted |

F3–F13 and F15 apply to upstream `master` at `104dd92` as well as to the fork,
because the two share the same tree for every file involved.

## Caveats worth carrying into the issues

- **F11 is latent, not live.** No in-tree caller triggers it —
  `_sqg_temp_buffers` sizes both scratch buffers with the same `max_batch`. It is
  a hole in an exported pybind11 API, and the issue body says so.
- **F11–F13 are static reads.** The CUDA extension could not be compiled during
  review (no nvcc, no GPU), so they are argued from the source and the exported
  ABI rather than from an observed crash.
- **F1 and F2 were reproduced**, and F2 was bisected to the commit that
  introduced it.

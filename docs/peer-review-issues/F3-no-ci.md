---
title: Add CI: nine branches and two open PRs, with no automated test run anywhere
labels: [enhancement]
---

No branch in this repository contains any file under `.github/`:

```
$ for b in $(git branch -r | grep -v HEAD); do
    echo "$b: $(git ls-tree -r --name-only $b | grep -c '^\.github/')"
  done
origin/master:                                    0
origin/agent/qsrt-profile5-pipeline:              0
origin/codex/exl3-3p09-repro-20260731:            0
origin/codex/exl3-tp16-1m-20260801:               0
origin/codex/kimi-k3-tp16-local-exl3-onegrid-…:   0
origin/dev/gg-k3-4p05:                            0
origin/feat/fruit-logical-adapter:                0
origin/feat/fruit-qsrt:                           0
origin/feat/glm52-exl3-shared-h-recipe-20260801:  0
```

The test suites are substantial and cheap. Measured on Python 3.12 with CPU
torch 2.13, no GPU and no model weights:

| branch | result | wall time |
| --- | --- | --- |
| `master` (+ upstream's 7 commits) | 341 passed, 1 skipped | 73 s |
| `feat/fruit-qsrt` | 421 passed, 1 skipped | 41 s |
| `feat/fruit-logical-adapter` | **1 failed**, 468 passed | 78 s |
| `feat/glm52-exl3-shared-h-recipe` | 58 passed, 2 skipped | 4 s |
| `codex/kimi-k3-tp16-local-onegrid` | **1 failed**, 68 passed | 4 s |
| `dev/gg-k3-4p05` | 14 passed | 3 s |

Two branch heads are red right now, one of them the head of an open PR. Both
failures are deterministic and reproduce on a clean checkout, so a
push-triggered workflow would have caught them immediately.

## Suggested minimum

A single `.github/workflows/ci.yml` on `push` and `pull_request`:

- `uv sync` (or `pip install -e . --group dev`) with CPU-only torch;
- `pytest tests -q`;
- `ruff check` — see the companion findings; the tree is already close to clean.

Both jobs run on a standard `ubuntu-latest` runner in well under five minutes.
Worth adding a `pytest-timeout` dev dependency at the same time: `--timeout` is
referenced in dev workflows but the plugin is not declared, so
`pytest --timeout=600` currently fails with `unrecognized arguments`.

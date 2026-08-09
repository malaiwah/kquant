# kquant peer review — 2026-08-09

Peer review of every non-default branch of `malaiwah/kquant` and of the upstream
default branch, `local-inference-lab/kquant@master`.

Each branch was checked out into its own worktree and run against its own test
suite on Python 3.12 with CPU torch 2.13, safetensors 0.8.0 and numpy 2.x — no
GPU, no model weights, no CUDA extension build. Ruff was run across `F`, `E9`,
`B`, `PLE`, `PLW`, `RUF` and the `S` security rules on every tree.

Reviewed at fork `79461d3`, upstream `104dd92`. Nine worktrees, nine suites,
ten findings.

- **Two branch heads are red**: `feat/fruit-logical-adapter` (F1, head of open
  upstream PR #4) and `codex/kimi-k3-tp16-local-exl3-onegrid-20260801` (F2).
- Upstream `master` is otherwise in good shape: 341 tests green, invariants
  asserted at import time, and the TP-independence rewrite deleted seven
  scripts without leaving a single dangling reference in code or docs.

Each finding below carries a ready-to-file issue title, label and body, so it
can be pasted into either repository verbatim.

---

## Filing status

Neither repository could be written to from the session that produced this
review:

- `malaiwah/kquant` returns `410 — Issues has been disabled in this repository`.
  GitHub turns Issues off by default on forks; re-enable under
  **Settings → General → Features → Issues**.
- `local-inference-lab/kquant` was outside that session's GitHub scope, and
  cross-owner repository adds are not supported. Filing there needs a session
  whose initial source is that repository.

F3–F10 all apply to upstream `master` at `104dd92` as well as to the fork,
because the fork's `master` and upstream's `master` share the same tree for
every file involved.

---

## Branch and test status

Fork `master` is `79461d3`. Upstream `master` is `104dd92` — the same history
plus the seven TP-independence commits merged as upstream PR #2.

| Branch | Head | Ahead | PR | Test result |
| --- | --- | --- | --- | --- |
| `upstream/master` | `104dd92` | +7 | #2 merged | 341 passed, 1 skipped |
| `agent/qsrt-profile5-pipeline` | `5428f34` | +2 | in #2 | 343 passed, 1 skipped |
| `feat/fruit-qsrt` | `08a35a9` | +24 | — | 421 passed, 1 skipped |
| `feat/fruit-logical-adapter` | `ff8e782` | +38 | #4 open | **1 FAILED**, 468 passed |
| `feat/glm52-exl3-shared-h-recipe-20260801` | `bd0e95c` | +1 | #1 open | 58 passed, 2 skipped |
| `codex/exl3-3p09-repro-20260731` | `5d20d85` | +5 | — | 58 passed |
| `codex/exl3-tp16-1m-20260801` | `e61adc8` | +6 | — | 58 passed |
| `codex/kimi-k3-tp16-local-exl3-onegrid-20260801` | `f45264c` | +7 | — | **1 FAILED**, 68 passed |
| `dev/gg-k3-4p05` | `2f9cb74` | +1 | — | 14 passed |

The three `codex/*` branches and `dev/gg-k3-4p05` branch off a pre-rewrite tree;
their suites are small because they predate the QSRT test expansion. Their green
results say nothing about compatibility with current `master` — see F4.

---

## Findings index

| ID | Severity | Target | Summary |
| --- | --- | --- | --- |
| F1 | Blocking | `feat/fruit-logical-adapter` (PR #4) | Production builder cannot be run directly; `ModuleNotFoundError` masks its authentication guard |
| F2 | Blocking | `codex/kimi-k3-tp16-local-onegrid` | `validate_exl3_artifact` requires a `trellis` config object unconditionally |
| F3 | Health | both repos, all branches | No CI: nine branches, two open PRs, no automated test run |
| F4 | Health | `codex/*` ×3, `dev/gg-k3-4p05` | Four branches target a codebase that no longer exists |
| F5 | Health | both repos, `master` | No LICENSE while shipping MIT-derived ExLlamaV3 code |
| F6 | Health | both repos, `dev/gg-k3-4p05` | Generated artifacts tracked in git, contradicting `.gitignore` |
| F7 | Quality | upstream `master` | Dead `above` binding costs a full 82,432-expert Lagrangian solve per call |
| F8 | Quality | upstream `master` | `x4t.py:291` annotation names a type never bound at module scope |
| F9 | Quality | `codex/exl3-tp16-1m` | `n_keep == 0` inverts the keep mask; `os.link` has no cross-device fallback |
| F10 | Quality | `dev/gg-k3-4p05` | Revision provenance derived two different ways in one commit |

---

## F1 — Blocking — `feat/fruit-logical-adapter`, head of upstream PR #4

### The production builder cannot be run directly; a `ModuleNotFoundError` masks its own authentication guard

`scripts/build_fruit_qsrt_model.py:21` imports its guard as
`from scripts.kquant_import_guard import …` but never puts the repository root
on `sys.path`, and there is no `scripts/__init__.py`. Running a script directly
puts *the script's own directory* on `sys.path[0]`, so the import cannot
resolve. Sibling scripts — `closure_exl3_layer.py`, `stream_k3_pytorch.py`,
`fruit_builder_bootstrap.py` — all bootstrap the path themselves; this one does
not.

```console
$ python scripts/build_fruit_qsrt_model.py /tmp/base /tmp/out
ModuleNotFoundError: No module named 'scripts'

$ PYTHONPATH=$PWD python scripts/build_fruit_qsrt_model.py /tmp/base /tmp/out
RuntimeError: direct production Fruit builder execution is forbidden; invoke an
externally authenticated standalone bootstrap with trusted Python -I -S
```

The guard is correct and does fire — it just never gets the chance unless the
caller already happens to have fixed `sys.path`. That makes guard behaviour
depend on ambient `PYTHONPATH`, and it means the `already_imported`
pre-bootstrap check at `kquant_import_guard.py:146-153` never executes on the
direct-invocation path. The branch's own test catches it.

<details>
<summary>Ready-to-file issue</summary>

**Title**

```
feat/fruit-logical-adapter: build_fruit_qsrt_model.py cannot be run directly; ModuleNotFoundError masks the production-builder guard
```

**Label:** `bug`

**Body**

````markdown
Peer review of `feat/fruit-logical-adapter` (`ff8e782`), the head of upstream PR
local-inference-lab/kquant#4. This is the one red test on the branch.

## Symptom

`tests/test_fruit_builder_bootstrap.py::test_live_production_builder_without_bootstrap_context_fails_before_output`
fails:

```
468 passed, 1 skipped, 1 failed in 78.03s

E   assert 'direct production Fruit builder execution is forbidden' in
E     'Traceback (most recent call last):\n  File ".../scripts/build_fruit_qsrt_model.py",
E      line 21, in <module>\n    from scripts.kquant_import_guard import (  # isort: skip
E      \nModuleNotFoundError: No module named 'scripts'\n'
```

Repro: clean checkout of the branch, `python -m pytest tests -q`.
Python 3.12, torch 2.13 CPU, safetensors 0.8.0.

## Root cause

`scripts/build_fruit_qsrt_model.py:21` imports its own authentication guard by
absolute package path:

```python
if __name__ == "__main__":
    from scripts.kquant_import_guard import (  # isort: skip
        authenticate_production_builder,
    )
```

but the module never puts the repository root on `sys.path`, and there is no
`scripts/__init__.py`. When a script is run directly, Python puts the *script's
own directory* (`scripts/`) on `sys.path[0]`, not the repository root — so
`import scripts.…` cannot resolve. Other scripts in the tree do bootstrap
`sys.path` themselves (`closure_exl3_layer.py`, `stream_k3_pytorch.py`,
`capture_fruit_qsrt_calibration.py`, `fruit_builder_bootstrap.py`, …); this one
does not.

The happy path works only because `scripts/fruit_builder_bootstrap.py:59` does
`sys.path.insert(0, root)` before exec'ing the builder, which incidentally
satisfies the import.

## Reproduction

```console
$ python scripts/build_fruit_qsrt_model.py /tmp/base /tmp/out
Traceback (most recent call last):
  File ".../scripts/build_fruit_qsrt_model.py", line 21, in <module>
    from scripts.kquant_import_guard import (  # isort: skip
ModuleNotFoundError: No module named 'scripts'

$ PYTHONPATH=$PWD python scripts/build_fruit_qsrt_model.py /tmp/base /tmp/out
  File ".../scripts/kquant_import_guard.py", line 156, in authenticate_production_builder
    raise RuntimeError(
RuntimeError: direct production Fruit builder execution is forbidden; invoke an
externally authenticated standalone bootstrap with trusted Python -I -S
```

The guard is correct and does fire — it just never gets the chance unless the
caller happens to have already fixed `sys.path`.

## Why this matters beyond the red test

The branch's stated goal across `b47cc6b fix(fruit): authenticate builder
execution`, `5cdf270 fix(fruit): seal producer and qualification trust chains`
and `b619442 fix(fruit): authenticate QSRT production builds` is that an
unauthenticated direct invocation is refused *with a specific, diagnosable
message*. As shipped, a direct invocation dies on an unrelated import error.
Nothing gets built either way, but it is the wrong failure:

- the operator sees `ModuleNotFoundError` and will reasonably conclude the
  checkout is broken, not that they skipped the bootstrap;
- the guard's `already_imported` check for pre-bootstrap `kquant` imports
  (`kquant_import_guard.py:146-153`) never executes on this path;
- behaviour depends on ambient `PYTHONPATH` — a dev shell with the repo root on
  the path gets the intended `RuntimeError`, a clean shell gets the import
  error. Guard behaviour should not depend on the caller's environment.

## Suggested fix

Bootstrap `sys.path` before the guard import, matching the other scripts:

```python
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.kquant_import_guard import (  # isort: skip
        authenticate_production_builder,
    )
```

This must not import `kquant` as a side effect, or it trips the guard's own
`already_imported` check. Adding `scripts/__init__.py` alone would not fix it —
the repo root still would not be on `sys.path`.

Worth adding a second test that asserts the guard message appears with a
*cleared* `PYTHONPATH`, so the ambient-path dependence cannot regress silently.
````

</details>

---

## F2 — Blocking — `codex/kimi-k3-tp16-local-exl3-onegrid-20260801`

### Artifact validation now demands a `trellis` config object from every artifact, including non-TP-local ones

Head commit `f45264c` added an *unconditional* requirement to
`validate_exl3_artifact`:

```python
trellis = quant.get("trellis")
if not isinstance(trellis, dict):
    issues.add("quantization config has no trellis object")
    trellis = {}
```

Everything that follows it is correctly gated on `tp_local` — the presence check
is not. Any serve directory packaged before this commit, and every non-TP-local
artifact, now fails validation.

Bisected: the branch's own
`tests/test_artifact.py::test_validate_exl3_artifact_accepts_complete_contract`
passes at the merge base `c82a896` and at `e61adc8`, and fails at `f45264c` with
exactly one issue — `"quantization config has no trellis object"`.

<details>
<summary>Ready-to-file issue</summary>

**Title**

```
codex/kimi-k3-tp16-local-exl3-onegrid: validate_exl3_artifact requires a `trellis` config object unconditionally, rejecting every non-TP-local artifact
```

**Label:** `bug`

**Body**

````markdown
Peer review of `codex/kimi-k3-tp16-local-exl3-onegrid-20260801` (`f45264c`).
The branch head is red.

## Symptom

```
$ python -m pytest tests -q
FAILED tests/test_artifact.py::test_validate_exl3_artifact_accepts_complete_contract
1 failed, 68 passed in 3.76s
```

The report returned by `validate_exl3_artifact` for the branch's own
`_mini_artifact` fixture:

```json
{
  "expected_expert_tensors": 15,
  "scanned_expert_tensors": 15,
  "issue_count": 1,
  "issue_samples": ["quantization config has no trellis object"],
  "pass": false
}
```

## Bisect

| commit | `tests/test_artifact.py` |
| --- | --- |
| `c82a896` (merge base with master) | 2 passed |
| `e61adc8` (previous branch commit) | 2 passed |
| `f45264c` (branch head) | **1 failed**, 1 passed |

The regression is introduced by the branch's own head commit,
`quant: build resumable TP16-local K3 EXL3 artifacts`.

## Root cause

`f45264c` adds to `validate_exl3_artifact` in `kquant/artifact.py`:

```python
trellis = quant.get("trellis")
if not isinstance(trellis, dict):
    issues.add("quantization config has no trellis object")
    trellis = {}
tp_keys = (
    "compatible_tp_sizes",
    "tp_local_quantization",
    "tp_local_intermediate_size",
    "intermediate_hadamard_blocks",
)
if tp_local:
    for key in tp_keys:
        if trellis.get(key) != manifest.get(key):
            issues.add(f"trellis {key} differs from the TP-local manifest")
elif trellis.get("tp_local_quantization"):
    issues.add("serve config is TP-local but the artifact manifest is not")
```

The comparisons are correctly gated on `tp_local`. The *presence* check is not.
The new mandatory field is a TP-local concept applied to every artifact, so:

- every serve `config.json` packaged before this commit now fails validation,
  even though nothing about those artifacts changed;
- the branch's own test fixture (`tests/test_artifact.py:88-93`) writes a
  `quantization_config` with `kept_format`, `demoted_format` and
  `hybrid_bit_map` and no `trellis` object, and was not updated.

## Suggested fix

Either gate the presence check the same way the comparisons are gated:

```python
trellis = quant.get("trellis")
if tp_local and not isinstance(trellis, dict):
    issues.add("TP-local quantization config has no trellis object")
if not isinstance(trellis, dict):
    trellis = {}
```

or keep it mandatory and bump the artifact schema version, update
`scripts/package_exl3_serve_dir.py` to always emit `trellis`, and update the
test fixture — so the incompatibility is explicit rather than silent.
````

</details>

---

## F3 — Health — both repos, all branches

### No CI: nine branches, two open PRs, and not one automated test run

No branch in either repository contains a single file under `.github/`. The
suites are substantial — 341 tests on upstream `master`, 469 on
`feat/fruit-logical-adapter` — and they run in about 75 seconds on CPU with no
GPU and no model weights. Both currently-red heads (F1, F2) would have been
caught the moment they were pushed. PR #4's only automated feedback today is a
docstring-coverage warning and a review bot.

<details>
<summary>Ready-to-file issue</summary>

**Title**

```
Add CI: nine branches and two open PRs, with no automated test run anywhere
```

**Label:** `enhancement`

**Body**

````markdown
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
````

</details>

---

## F4 — Health — `codex/*` ×3 and `dev/gg-k3-4p05`

### Four branches target a codebase that no longer exists — every file they touch has been deleted from master

The three `codex/*` branches and `dev/gg-k3-4p05` were cut before the QSRT
rewrite. Between them they modify `kquant/artifact.py`, `kquant/pack/exl3.py`,
`kquant/allocation.py`, `kquant/cli.py`, `kquant/pack/writer.py`,
`scripts/pack_exl3_12gpu.py` and `scripts/package_exl3_serve_dir.py`. None of
those paths exists on `master` or upstream `master`. The branches still pass
their own tests because they carry the old suite with them — that result says
nothing about mergeability.

These carry real work that is worth a decision rather than quiet decay: the EXL3
keep-fraction repacker, the resumable TP16-local build path, the MXFP8 overlay
bake, and the reproducible ExLlamaV3 shared-SU patch.

<details>
<summary>Ready-to-file issue</summary>

**Title**

```
Four branches target a pre-rewrite codebase and can no longer be merged: port or retire them
```

**Label:** `enhancement`

**Body**

````markdown
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
````

</details>

---

## F5 — Health — both repos, `master`

### No LICENSE, while the tree ships code derived from MIT-licensed ExLlamaV3

`THIRD_PARTY_NOTICES.md` correctly reproduces the MIT grant covering
`kquant/exl3_encoder_backend.py` and
`kquant/csrc/qsrt_quantize_tiles_kernel.cuh`. But no branch of either repository
carries a `LICENSE` for kquant's own code, so it defaults to all-rights-reserved:
no fork, contributor or downstream consumer has a grant.

This is a live question, not a hypothetical — the fork's own head branch commit
`ff8e782 fix(license): remove unowned repository grant` deleted the Apache-2.0
file it had added, correctly recognising a fork cannot license the upstream
project. That leaves the decision squarely with the upstream owner.

<details>
<summary>Ready-to-file issue</summary>

**Title**

```
No LICENSE: the project ships MIT-derived ExLlamaV3 code but grants no rights to its own
```

**Label:** `documentation`

**Body**

````markdown
Neither `master` nor any other branch carries a `LICENSE` file:

```
$ for b in $(git branch -r | grep -v HEAD); do
    git cat-file -e $b:LICENSE 2>/dev/null && echo "$b: yes" || echo "$b: none"
  done
# none, on every branch
```

`THIRD_PARTY_NOTICES.md` correctly reproduces the MIT grant that covers
`kquant/exl3_encoder_backend.py` and `kquant/csrc/qsrt_quantize_tiles_kernel.cuh`
(© 2025 Turboderp). That handles the inbound obligation. There is no outbound
grant for kquant's own ~24k lines, so the default applies: all rights reserved.

Practical consequences today:

- forks and contributors have no license to use, modify or redistribute the
  work, including the cross-fork PRs currently open against this repository;
- anyone packaging an artifact built with this pipeline has no stated terms;
- `pyproject.toml` declares no `license` field and no license classifier, so a
  built wheel carries nothing either.

## Ask

Pick a license and add `LICENSE` plus `license = { text = "…" }` in
`pyproject.toml`. MIT or Apache-2.0 both compose cleanly with the inbound MIT
code; Apache-2.0 additionally carries an explicit patent grant, which is worth
considering for a codec.

Context: a contributor's fork added Apache-2.0 and then reverted it in
`ff8e782 fix(license): remove unowned repository grant` — correctly, since a
fork cannot license the upstream project. The choice belongs to this
repository's owner, and until it is made the reversion leaves everyone
without terms.
````

</details>

---

## F6 — Health — both repos and `dev/gg-k3-4p05`

### Generated research artifacts are committed, and `.gitignore` disagrees with what is tracked

`.gitignore:9` ignores `out/` with the comment that these outputs "can be many
GiB" and "live outside the repository" — yet ten files under `out/` are tracked,
including two ~490 KB allocation JSONs and two `.safetensors` blobs. Because they
are already tracked, the ignore rule does not apply to them; it only hides *new*
files there, so the tracked copies quietly go stale.

`dev/gg-k3-4p05` compounds this: it adds `out-tp16-4p05/`, outside the ignore
pattern, carrying an 82,843-line generated `allocation.json` and three committed
symlinks pointing back into `out/`. That single commit is 82,868 insertions
against three lines of actual code change.

<details>
<summary>Ready-to-file issue</summary>

**Title**

```
Generated artifacts are tracked in git and contradict .gitignore
```

**Label:** `enhancement`

**Body**

````markdown
## `out/` is ignored but tracked

`.gitignore:9` ignores `out/`, with the comment that these outputs "can be many
GiB" and that "durable checkpoint payloads live outside the repository". Ten
files under `out/` are nevertheless tracked:

```
$ git ls-files out
out/allocation-hybrid-3p20.json          (486 KiB)
out/allocation.json                      (486 KiB)
out/distortion.kqstats/arrays.safetensors
out/distortion.kqstats/manifest.json
out/inventory.json
out/refit.json
out/report-hybrid-3p20.md
out/report.md
out/static.kqstats/arrays.safetensors
out/static.kqstats/manifest.json

$ git check-ignore -v --no-index out/allocation.json
.gitignore:9:out/    out/allocation.json
```

Because they are already tracked, the ignore rule has no effect on them — it
only suppresses *new* files in that directory. The practical result is that
regenerating these outputs produces working-tree changes that `git status`
still shows for the ten tracked paths but hides for everything else, and the
committed copies drift out of sync with the code that produced them.

Note `scripts/repack_exl3_keep_fraction.py:36` reads `out/static.kqstats`
through a hard-coded repo-relative path, so at least one script depends on
these files being present. That dependency should be an explicit CLI argument
rather than a checked-in artifact.

## `dev/gg-k3-4p05` adds more, outside the ignore rule

```
$ git show --stat 2f9cb746
 kquant/allocation.py             |     3 +-
 kquant/cli.py                    |     1 +
 kquant/pack/writer.py            |     2 +-
 out-tp16-4p05/allocation.json    | 82843 +++++++++++++++++++++++++++++
 out-tp16-4p05/distortion.kqstats |     1 +
 out-tp16-4p05/inventory.json     |     1 +
 out-tp16-4p05/static.kqstats     |     1 +
 out-tp16-4p05/verify.json        |    18 +
 8 files changed, 82868 insertions(+), 2 deletions(-)

$ git check-ignore -v --no-index out-tp16-4p05/allocation.json
# no match — not ignored
```

Three of those entries are committed symlinks into `out/`
(`distortion.kqstats -> ../out/distortion.kqstats`, and likewise for
`inventory.json` and `static.kqstats`), which only resolve for someone who has
the tracked `out/` contents. The commit's actual change is three lines.

## Ask

1. `git rm --cached` the tracked `out/` files, or drop the `out/` ignore rule
   and keep them deliberately — either is defensible, the contradiction is not.
2. Broaden the ignore to `out*/` so sibling run directories are covered.
3. Give `repack_exl3_keep_fraction.py` a `--stats` argument instead of the
   hard-coded `out/static.kqstats` path.
4. Rework `dev/gg-k3-4p05` down to its three-line code change.
````

</details>

---

## F7 — Quality — upstream `master`, `kquant/pack/qsrt_allocation.py`

### A dead binding costs one full 82,432-expert Lagrangian solve on every target search

In `choose_qsrt_target`, `above` is assigned at line 194 and reassigned inside
the bisection loop, and never read. The line-194 assignment is a complete
`choose_qsrt_lagrangian` evaluation — 92 layers × 896 experts, each layer an
argsort plus a 897-step prefix scan — thrown away.

```python
above = choose_qsrt_lagrangian(          # line 194 — never read
    damage, costs,
    lagrange_lambda=low,
    target_container_bytes=target_container_bytes,
)
```

Either delete it, or use it as it appears to have been intended — to assert the
bracket is valid and to report the frontier point immediately above the cap,
which the docstring already talks about.

<details>
<summary>Ready-to-file issue</summary>

**Title**

```
choose_qsrt_target: dead `above` binding wastes a full 82,432-expert Lagrangian solve per call
```

**Label:** `bug`

**Body**

````markdown
`kquant/pack/qsrt_allocation.py:194` in `choose_qsrt_target`:

```python
low = 0.0
high = max(float(np.max(damage / costs)), np.finfo(np.float64).tiny)
above = choose_qsrt_lagrangian(          # <- never read
    damage,
    costs,
    lagrange_lambda=low,
    target_container_bytes=target_container_bytes,
)
below = choose_qsrt_lagrangian(
    damage, costs, lagrange_lambda=high,
    target_container_bytes=target_container_bytes,
)
...
    if candidate.container_bytes > target_container_bytes:
        low = middle
        above = candidate                # <- also never read
    else:
        high = middle
        below = candidate
...
return below
```

`above` is written twice and read never. `ruff check --select F` reports it:

```
kquant/pack/qsrt_allocation.py:224:13: F841 Local variable `above` is assigned to but never used
```

The line-194 write is not free. `choose_qsrt_lagrangian` runs
`_choose_layer_lagrangian` for all 92 MoE layers, and each of those does an
`argsort` over 896 experts plus a 897-step prefix scan — so a full solve over
all 82,432 layer/expert assignments is computed and discarded on every
`choose_qsrt_target` call, on top of the 1 + `iterations` (default 72) solves
the bisection genuinely needs.

## Suggested fix

Either drop the binding entirely, or give it the job it looks like it was
written for. The docstring already says:

> A hard byte cap can lie between two supported rate-distortion points, so the
> returned allocation may leave less than one frontier jump of slack

so returning or reporting the bracketing point above the cap would make that
statement checkable, and would justify keeping the variable:

```python
if above.container_bytes <= target_container_bytes:
    raise AssertionError("lambda=0 point already fits; caller should have early-returned")
```

Either way the dead write should not survive. Adding `ruff check --select F` to
CI (see the CI finding) would keep this class of defect out.
````

</details>

---

## F8 — Quality — upstream `master`, `kquant/x4t.py:291`

### An annotation names a type that is never in scope, so runtime type introspection raises

`partition_x4t_components` is annotated `raw: "PackedMXFP4Matrix | X4TMatrix"`,
but `PackedMXFP4Matrix` is imported *inside* the function body — it is never
bound at module scope. The call works; anything that resolves annotations does
not.

```pycon
>>> typing.get_type_hints(kquant.x4t.partition_x4t_components)
NameError: name 'PackedMXFP4Matrix' is not defined
```

The local import exists to break a cycle with `kquant.source_weights`, so the
fix is a `TYPE_CHECKING` import rather than moving it to module scope.

<details>
<summary>Ready-to-file issue</summary>

**Title**

```
x4t.partition_x4t_components: annotation references PackedMXFP4Matrix, which is never bound at module scope
```

**Label:** `bug`

**Body**

````markdown
`kquant/x4t.py:291`:

```python
def partition_x4t_components(
    raw: "PackedMXFP4Matrix | X4TMatrix",
    matrix: str,
    ...
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    from kquant.source_weights import PackedMXFP4Matrix   # line 306, function-local
```

`PackedMXFP4Matrix` is imported inside the function body and never bound at
module scope, so the string annotation cannot be resolved:

```
$ ruff check --select F kquant/x4t.py
kquant/x4t.py:291:11: F821 Undefined name `PackedMXFP4Matrix`

>>> import typing, kquant.x4t
>>> typing.get_type_hints(kquant.x4t.partition_x4t_components)
NameError: name 'PackedMXFP4Matrix' is not defined
```

Calls work — `from __future__ import annotations` means the annotation is never
evaluated at call time — but anything that resolves hints breaks:
`get_type_hints`, `dataclasses` interop, `pydantic`-style validation,
`typing.get_overloads`, and API-doc generators. A type checker also cannot see
the intended signature, so the union is unverified.

## Suggested fix

The function-local import is there to break an import cycle with
`kquant.source_weights`, so keep it and add a `TYPE_CHECKING` binding:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kquant.source_weights import PackedMXFP4Matrix
```

That resolves the annotation for checkers and `get_type_hints` under
`from __future__ import annotations`, with no runtime import cost and no cycle.

The same sweep would catch the dead imports in the tree:

```
kquant/qsrt.py:25-28   CODEBOOK_SQG_CHEB_NORMAL_E4M3, CODEBOOK_SQG_CHEB,
                       CODEBOOK_SQG_NORMAL_E4M3, QSRT_CODEBOOKS  — unused
kquant/qsrt_storage.py:30   RECORDS_PER_EXPERT  — unused
kquant/correctness.py:12    os  — unused
```

Note `kquant/qsrt.py` re-exports those four codebook names in practice, so if
that is deliberate they belong in an `__all__` rather than a bare import.
````

</details>

---

## F9 — Quality — `codex/exl3-tp16-1m`, `scripts/repack_exl3_keep_fraction.py`

### A tiny keep-fraction inverts the mask, and shard linking has no cross-filesystem fallback

Two independent defects in the keep-fraction repacker. First, when `n_keep`
rounds to zero the negative slice `[-0:]` is `[0:]` — the whole index array — so
`keep_mask` becomes all-`True` and the script produces the *largest* possible
artifact instead of the smallest. Second, `link_existing_keeps` calls `os.link`
with no fallback, which fails outright when source and destination are on
different filesystems — a normal setup when repacking a multi-terabyte artifact
onto a scratch volume.

<details>
<summary>Ready-to-file issue</summary>

**Title**

```
repack_exl3_keep_fraction: n_keep==0 inverts the keep mask, and os.link has no cross-device fallback
```

**Label:** `bug`

**Body**

````markdown
Two defects in `scripts/repack_exl3_keep_fraction.py`
(branches `codex/exl3-tp16-1m-20260801`,
`codex/kimi-k3-tp16-local-exl3-onegrid-20260801`).

## 1. `n_keep == 0` keeps everything

```python
n_keep = int(round(keep_frac * flat.size))
keep_idx = np.argpartition(flat, -n_keep)[-n_keep:]
keep_mask = np.zeros(flat.size, dtype=bool)
keep_mask[keep_idx] = True
```

`keep_frac` is validated as `0.0 < keep_frac <= 1.0`, so a very small value is
accepted. With `flat.size == 82_432`, any `keep_frac < 6.1e-6` rounds `n_keep`
to 0 — and `[-0:]` is `[0:]`, i.e. the *entire* index array:

```python
>>> np.arange(5)[-0:]
array([0, 1, 2, 3, 4])
```

So `keep_mask` becomes all-`True`: every expert is kept at MXFP4 and nothing is
demoted to EXL3. The script silently produces the largest possible artifact
when asked for the smallest. It is caught downstream only by the
`old_keep <= new_keep` subset assertion in `main()`, which happens to pass
because a superset always satisfies it.

Fix — guard the empty case explicitly:

```python
if n_keep == 0:
    keep_mask = np.zeros(flat.size, dtype=bool)
else:
    keep_idx = np.argpartition(flat, -n_keep)[-n_keep:]
    ...
```

## 2. `os.link` fails across filesystems

```python
def link_existing_keeps(base: Path, destination: Path) -> None:
    for source in sorted(base.glob("keep-mxfp4-*.safetensors")):
        os.link(source, destination / source.name)
```

`os.link` raises `OSError: [Errno 18] Invalid cross-device link` when `base` and
`destination` are on different filesystems. Repacking a multi-terabyte artifact
onto a scratch volume is exactly the case the script is for, and the failure
lands after `repack_exl3_layers` has already written every layer shard — so a
long run dies near the end with a low-level errno and no cleanup.

Fix — fall back, and say which path was taken:

```python
try:
    os.link(source, target)
except OSError as exc:
    if exc.errno != errno.EXDEV:
        raise
    shutil.copy2(source, target)
```

A `--link-mode {hardlink,symlink,copy}` flag would be better still, since a
symlink is usually the right answer for an immutable base artifact.

## Also worth a look

`compute_allocation` reads
`Path(__file__).resolve().parents[1] / "out/static.kqstats"`, a hard-coded
repo-relative path with no CLI override, and the emitted metadata claims
`"traffic": "bias-proxy (L0); same ranking as base artifact"` without verifying
that the stats bundle still matches the one the base artifact was built from.
````

</details>

---

## F10 — Quality — `dev/gg-k3-4p05`

### Revision provenance is threaded from two different sources in one commit

The commit's stated goal is "preserve Kimi K3 quant revision provenance", but it
records that provenance two different ways: `save_allocation` gains an explicit
`revision` parameter fed from `--revision`, while `pack_artifact` instead
switches from `C.REVISION` to `cache.snapshot_dir.name`. In the normal HF-cache
path the two agree, so the inconsistency is latent rather than currently wrong —
but a single provenance field with two independent derivations is exactly the
thing that drifts.

<details>
<summary>Ready-to-file issue</summary>

**Title**

```
dev/gg-k3-4p05: revision provenance is derived two different ways in the same commit
```

**Label:** `enhancement`

**Body**

````markdown
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
````

</details>

---

## Method: what was and wasn't checked

Each branch was checked out into its own worktree and run against its own suite
with CPU torch — no GPU, no model weights, no CUDA extension build. Ruff was run
across `F`, `E9`, `B`, `PLE`, `PLW`, `RUF` and the `S` security rules on every
tree.

**Not covered**, and worth knowing:

- the CUDA kernels under `kquant/csrc/` were never compiled or executed;
- nothing that requires the Kimi-K3 checkpoint, the EXL3 teacher, or a real
  capture was exercised;
- the numerical claims in `docs/qsrt-technical-brief.md` — reconstruction-law
  quality, rate-shift laws, dense-H validity — were read for internal
  consistency but not independently reproduced.

**Checked and dismissed** rather than filed:

- the `sha1` calls in `scripts/fruit_builder_bootstrap.py:163` and
  `scripts/tracked_worktree.py:250` are Git object-ID verification and correct;
- `atom_rates()` in `kquant/qsrt_storage.py` looks inconsistent with
  `ModeSpec.context_bits`, but is right once `record_contexts()`' low/high
  pairing is applied — for R1 the pairing yields `(K2, K4)` at pair 0, matching
  the `record_pair < spec.mode_id` test;
- `args.revision` on the `rank` subcommand resolves via the top-level parser at
  `kquant/cli.py:221`, so `dev/gg-k3-4p05` does not crash there.

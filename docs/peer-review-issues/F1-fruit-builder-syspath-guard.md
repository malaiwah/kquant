---
title: feat/fruit-logical-adapter: build_fruit_qsrt_model.py cannot be run directly; ModuleNotFoundError masks the production-builder guard
labels: [bug]
---

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

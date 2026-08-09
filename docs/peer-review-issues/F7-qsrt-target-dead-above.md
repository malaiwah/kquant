---
title: choose_qsrt_target: dead `above` binding wastes a full 82,432-expert Lagrangian solve per call
labels: [bug]
---

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

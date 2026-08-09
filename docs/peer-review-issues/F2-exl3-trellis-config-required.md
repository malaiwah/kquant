---
title: codex/kimi-k3-tp16-local-exl3-onegrid: validate_exl3_artifact requires a `trellis` config object unconditionally, rejecting every non-TP-local artifact
labels: [bug]
---

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

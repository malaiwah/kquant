---
title: x4t.partition_x4t_components: annotation references PackedMXFP4Matrix, which is never bound at module scope
labels: [bug]
---

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

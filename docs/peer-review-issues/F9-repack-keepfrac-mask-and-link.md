---
title: repack_exl3_keep_fraction: n_keep==0 inverts the keep mask, and os.link has no cross-device fallback
labels: [bug]
---

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

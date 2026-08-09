---
title: "GLM shared-H recipe: only three bundle files are hash-verified, but the whole bundle is copied and the result is reported as verified"
labels: [bug]
---

`recipes/glm52_exl3_shared_h/prepare_shared_h_encoder.py` is built around hash
pinning, and most of it is right: it verifies the declared inputs, applies the
patches, verifies the patched **outputs** against `OUTPUT_SHA256`, and
`py_compile`s them. Verifying the post-patch outputs is the step most recipes
omit, and this one does it.

The gap is what happens between those two checks.

## Mechanism

`verify_files` only walks the keys it is handed:

```python
def verify_files(root: Path, expected: dict[str, str], label: str) -> None:
    for relative, digest in expected.items():
        path = root / relative
        ...
```

`INPUT_SHA256` declares three entries — `encode_tr3_v31.py`, `encode_b300.py`,
and `calibration/reap_recall_calib.jsonl`. But `prepare` copies the bundle
wholesale:

```python
verify_files(bundle, INPUT_SHA256, "input bundle")     # 3 files
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise RuntimeError(f"output must be absent or empty: {output}")
output.mkdir(parents=True, exist_ok=True)
for source in bundle.iterdir():                        # everything
    destination = output / source.name
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
```

Every other file in the published `calibration_encoder` bundle — any helper
module the two encoders import, anything else under `calibration/` — is copied
into the output having never been hashed. `OUTPUT_SHA256` then re-checks only the
two patched `.py` files, so those passengers are never verified on the way out
either.

The script finishes with:

```python
print(f"Prepared verified shared-H encoder: {args.output.resolve()}")
```

So "verified" covers 3 of N inputs and 2 of N outputs. The two pinned encoders
are `py_compile`d but not executed in isolation, so a substituted sibling module
that they import at run time would flow straight into a production encode with
the preparation step reporting success.

For a recipe whose stated value is byte-reproducibility from a pinned upstream
bundle — the README leads with a SHA-256 table — that is the wrong guarantee to
print.

## Test coverage

`tests/test_glm52_exl3_shared_h_recipe.py` has two tests,
`test_recipe_patches_reproduce_pinned_encoder` and
`test_shared_h_algebra_seed_and_artifact_contract`. Neither constructs a bundle
containing an unpinned extra file, so nothing currently fails when one passes
through.

## Suggested fix

Make the printed claim equal the checked claim. Either:

1. pin a complete manifest of every file in the bundle, and have `verify_files`
   additionally assert that the set of files found equals the set pinned — so a
   file that is *added* upstream is a hard error rather than a silent passenger; or
2. compute one digest over the full recursive tree (sorted relative paths, each
   path plus its content) and pin that single value, which keeps the README table
   short and still covers everything.

Either way, add a test that drops an extra file into a fixture bundle and asserts
`prepare` refuses it.

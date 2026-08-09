---
title: Generated artifacts are tracked in git and contradict .gitignore
labels: [enhancement]
---

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

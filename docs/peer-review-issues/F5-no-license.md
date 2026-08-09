---
title: No LICENSE: the project ships MIT-derived ExLlamaV3 code but grants no rights to its own
labels: [documentation]
---

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

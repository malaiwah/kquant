#!/usr/bin/env python3
"""Encode and assemble a complete exact-rate Fruit QSRT Hugging Face model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.kquant_import_guard import (  # isort: skip
    KQUANT_IMPORT_IDENTITY as _KQUANT_IMPORT_IDENTITY,
    KQUANT_IMPORT_SOURCE_ROOT as _KQUANT_IMPORT_SOURCE_ROOT,
)
from scripts.tracked_worktree import tracked_worktree_sha256  # isort: skip

from kquant.exl3_loader import exllamav3_source_identity, load_qsrt_encoder
from kquant.fruit_calibration import (
    FruitCalibrationStore,
    fruit_calibration_authority,
)
from kquant.fruit_qsrt import (
    FRUIT_QSRT_ARTIFACT_TENSORS,
    FRUIT_QSRT_ATOM_BUNDLE_BYTES,
    FRUIT_QSRT_ATOM_CHANNELS,
    FRUIT_QSRT_ATOM_SCHEMA,
    FRUIT_QSRT_ATOM_STORAGE,
    FRUIT_QSRT_ATOM_TENSOR,
    FRUIT_QSRT_ATOM_TENSORS,
    FRUIT_QSRT_ATOMS_PER_EXPERT,
    FRUIT_QSRT_CODEBOOK,
    FRUIT_QSRT_FORMAT_SECTION_BYTES,
    FRUIT_QSRT_FORMAT_TENSOR,
    FRUIT_QSRT_PAIR_COUNT,
    FRUIT_QSRT_PAIR_WORDS,
    FRUIT_QSRT_PROFILE_ID,
    FRUIT_QSRT_SCHEMA,
    FRUIT_QSRT_SHARED_SCALE_TENSOR,
    FRUIT_QSRT_STORAGE_ALIGNMENT,
    FruitMatrixStore,
    encode_fruit_expert,
    pack_fruit_atom_layer,
)
from kquant.fruit_source import (
    FRUIT_INSTRUCT_SPEC,
    FruitModelSpec,
    FruitSafetensorsStore,
)
from kquant.sqg_quantizer import install_sqg_quantizer

LAYERS = (*FRUIT_INSTRUCT_SPEC.layers, FRUIT_INSTRUCT_SPEC.mtp_layer)
EXPERTS = FRUIT_INSTRUCT_SPEC.num_experts
HIDDEN_SIZE = FRUIT_INSTRUCT_SPEC.hidden_size
INTERMEDIATE_SIZE = FRUIT_INSTRUCT_SPEC.intermediate_size
PAIR_COUNT = FRUIT_QSRT_PAIR_COUNT
PAIR_WORDS = FRUIT_QSRT_PAIR_WORDS


@dataclass(frozen=True)
class FruitPublicationSpec:
    """Human-facing publication identity for one Fruit checkpoint variant."""

    repository: str
    title: str
    introduction: str
    quality_limitations: str
    fruit_audit_rows: str


FRUIT_PUBLICATIONS = {
    "instruct": FruitPublicationSpec(
        repository="malaiwah/GLM-5.2-QSRT-Fruit-Instruct",
        title="GLM-5.2 QSRT Fruit Instruct",
        introduction=(
            "This is the assistant-masked SFT form of a **5.04B-parameter "
            "GLM-5.2 serving proxy**, not the 754B GLM-5.2 model. It is encoded "
            "in KQuant's canonical QSRT atom format and served without "
            "reconstructing dense expert weights.\n\n"
            "The artifact packages the codec/storage/runtime integration for "
            "the instruction-tuned Fruit checkpoint. Packaging, provenance, "
            "exact state decoding, and the canonical single-GPU runtime path "
            "are implemented. No downstream task quality is claimed; see "
            "[Known limitations](#known-limitations)."
        ),
        quality_limitations=(
            "- **Not assistant-quality.** The required runtime qualification "
            "receipt is a narrow protocol record, not a representative downstream "
            "instruction/chat task-quality benchmark."
        ),
        fruit_audit_rows=(
            "| [Fruit Instruct BF16](https://huggingface.co/malaiwah/"
            "GLM-5.2-SIQ-Fruit-Instruct-bf16/tree/"
            "678954f65e056a0f508e21eeb9251c655bb9463f) | `678954f6` | "
            "10,102,017,674 | 10,081,800,232 |\n"
            "| [Fruit Instruct prior mixed SIQ](https://huggingface.co/"
            "malaiwah/GLM-5.2-SIQ-Fruit-Instruct/tree/"
            "48452ef397d8b4a4d6d0c00ea376a2abb3ef6314) | `48452ef3` | "
            "3,122,333,594 | 3,102,116,152 |"
        ),
    ),
}


def fruit_publication_spec(variant: str) -> FruitPublicationSpec:
    try:
        return FRUIT_PUBLICATIONS[variant]
    except KeyError as exc:
        raise ValueError(f"unsupported Fruit publication variant {variant!r}") from exc


EXLLAMAV3_REVISION = "791c83073f7f90c44f765a0ceeab7a05fa15b96b"
_COMPLETE_MARKER_NAME = "QSRT_COMPLETE.json"
_CANDIDATE_MARKER_NAME = "QSRT_CANDIDATE.json"
_KQUANT_SNAPSHOT_CLEAN_VERIFIED = False
MODEL_CARD_TEMPLATE = r"""---
license: mit
library_name: vllm
pipeline_tag: text-generation
tags:
- glm
- mixture-of-experts
- kquant
- qsrt
- vllm
- b12x
- experimental
---

# __MODEL_TITLE__

__MODEL_INTRODUCTION__

## What is included

- 13 transformer layers: 3 dense and 10 MoE layers, plus the packaged MTP
  expert layer.
- Hidden size 1,024; MoE intermediate size 512; 256 routed experts per MoE/MTP
  layer.
- 2,816 QSRT experts in 11 canonical atom files.
- SQG-XOR-Cheb-T12 E4M3 codebook, three-bit trellis payload, fixed P24/P33 pair
  records, and physical atom rotation.
- Canonical `qsrt_atoms_v1` storage with complete per-file SHA-256 manifests
  and a fail-closed `QSRT_COMPLETE.json` marker.
- W4A16 prefill/reference execution and W4A8 decode execution through B12X.

The expert allocation selected by the frozen calibration evidence is recorded
in each `qsrt-layer-*.json` sidecar. Aggregate allocation counts are:

| Allocation code | Experts |
|---|---:|
__ALLOCATION_ROWS__

## Size and memory

The apples-to-apples baseline is the complete BF16 tensor set. All three rows
below cover the same 5,040,368,896 logical parameters and count only
Safetensors files; `effective bpw` is stored bytes times eight divided by that
parameter count, so it includes container and quantization metadata.

| Tensor payload | Bytes | GiB | Effective bpw | Relative to BF16 |
|---|---:|---:|---:|---:|
| BF16 source | 10,081,800,232 | 9.3894 | 16.0017 | baseline |
| Prior SIQ mixed | 3,102,116,152 | 2.8891 | 4.9236 | 69.23% smaller |
| This QSRT model | 2,909,352,104 | 2.7095 | 4.6177 | 71.14% smaller |

The whole-model rates include 611,183,872 non-routed parameters retained in
BF16. Isolating the 4,429,185,024 routed-expert weights gives:

| Routed-expert format | Stored bytes | Nominal path bpw | Effective stored bpw |
|---|---:|---:|---:|
| BF16 | 8,858,370,048 | 16.0000 | 16.0000 |
| Prior SIQ mixed (1,856 K3 / 960 K4 experts) | 1,879,717,272 | 3.3409 | 3.3951 |
| QSRT P24/P33 atoms | 1,686,953,224 | 3.0000 | 3.0470 |

QSRT is therefore 10.25% smaller than SIQ on the routed-expert component and
6.21% smaller on the compared tensor files. Package-level totals are not used
for the comparison because tokenizer, card, and evaluation evidence files are
not model weights. The previous card's 7,593,020,594-byte BF16 row was not the
complete BF16 tensor set and has been removed.

W4A8 and W4A16 use the same stored weights, so their loader weight storage is
identical; W4A8 changes the decode execution path, not the checkpoint size.

### Hugging Face repository-size audit

The Hugging Face model API with `blobs=true` reported the following immutable
snapshot on 2026-08-08. `Repository bytes` sums every sibling's reported size;
`Safetensors bytes` sums only `*.safetensors`. These are observed repository
payloads, not parameter-count estimates.

| Artifact | Revision | Repository bytes | Safetensors bytes |
|---|---|---:|---:|
__FRUIT_AUDIT_ROWS__
| [Full GLM-5.2 BF16](https://huggingface.co/zai-org/GLM-5.2/tree/b4734de4facf877f85769a911abafc5283eab3d9) | `b4734de4` | 1,506,693,036,946 | 1,506,667,387,408 |
| [Full GLM-5.2 FP8](https://huggingface.co/zai-org/GLM-5.2-FP8/tree/ba978f7d347eaf65d22f1a86833408afdb953541) | `ba978f7d` | 755,663,676,164 | 755,632,050,320 |
| [Full GLM-5.2 NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4/tree/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa) | `aec724e8` | 464,874,323,992 | 464,823,042,096 |

The three full-model rows ground real download/storage scale only. They are not
used for Fruit percentage claims because Fruit has 5.04B parameters while the
production model has roughly 754B. The apples-to-apples Fruit tensor
comparison above remains the codec-size result.

__RATE_SWEEP_SECTION__
__RUNTIME_QUALIFICATION_SECTION__


## Evidence boundary

The completion seal covers every top-level package file and every regular file
under `evaluation/`. The sealed adjacent-rate report measures local routed
expert reconstruction on authenticated, document-disjoint calibration rows.
It does not establish chat quality, broad downstream task quality, or general
serving throughput.

## Reproducible runtime

The runtime is pinned to the reviewed commits below:

- KQuant encoder: [`local-inference-lab/kquant#4`](https://github.com/local-inference-lab/kquant/pull/4),
  encoded with KQuant revision `__KQUANT_REVISION__`.
- B12X kernels: [`local-inference-lab/b12x#129`](https://github.com/local-inference-lab/b12x/pull/129),
  packaged producer revision `__B12X_REVISION__`.
- vLLM loader: [`local-inference-lab/vllm#269`](https://github.com/local-inference-lab/vllm/pull/269),
  packaged producer revision `__VLLM_REVISION__`.

The derived image starts from the content-addressed public base
`docker.io/voipmonitor/vllm@sha256:3230c25ff95f8678a8eeb52a463f0d3b9f96f6ad550418cc51ea12177a55b41c`
hard-coded by `Dockerfile.fruit-qsrt`. It installs the exact B12X checkout,
copies the base's compiled vLLM extensions into the reviewed source tree, and
seals the exact runtime package bytes. `MODEL_REVISION` resolves the Hub branch
once; `hf download` then uses the resulting immutable commit SHA.

```bash
git clone https://github.com/malaiwah/vllm-voipmonitor.git vllm-fruit
git -C vllm-fruit checkout --detach __VLLM_REVISION__

docker build \
  --file vllm-fruit/Dockerfile.fruit-qsrt \
  --build-arg VLLM_REVISION=__VLLM_REVISION__ \
  --build-arg B12X_REVISION=__B12X_REVISION__ \
  --tag fruit-qsrt:__VLLM_REVISION__ \
  vllm-fruit

# This digest is an operator-supplied trust root obtained independently of the
# package being authenticated. Never derive it from MODEL_DIR.
test -n "${FRUIT_QSRT_EXPECTED_COMPLETE_SHA256:?set an independently supplied completion digest}"
test "${#FRUIT_QSRT_EXPECTED_COMPLETE_SHA256}" -eq 64

MODEL_REVISION="$(
  curl -fsSL https://huggingface.co/api/models/__MODEL_REPOSITORY__ \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])'
)"
MODEL_DIR="Fruit-QSRT-${MODEL_REVISION}"
test ! -e "${MODEL_DIR}"
hf download __MODEL_REPOSITORY__ \
  --revision "${MODEL_REVISION}" \
  --local-dir "${MODEL_DIR}"

docker run --rm --gpus '"device=0"' --shm-size=16g \
  --read-only \
  --tmpfs /tmp:rw,exec,nosuid,size=8g \
  --tmpfs /cache:rw,exec,nosuid,size=16g \
  --tmpfs /root/.cache:rw,nosuid,size=1g \
  --publish 8000:8000 \
  --volume "$PWD/${MODEL_DIR}:/model:ro" \
  --env MODEL=/model \
  --env FRUIT_QSRT_EXPECTED_COMPLETE_SHA256="${FRUIT_QSRT_EXPECTED_COMPLETE_SHA256}" \
  fruit-qsrt:__VLLM_REVISION__
```

The container configuration targets SM120 with CUDA 13.2.1 and PyTorch
2.12.0+cu132 in the content-addressed base, plus
`nvidia-cutlass-dsl == 4.6.0` in the derived image.
The launcher rejects extra vLLM arguments and any value other than TP1,
`max_num_seqs=1`, `max_model_len=4096`, and
`max_num_batched_tokens=4096` before importing the GPU runtime. The current
B12X sparse-prefill backend requires single-request prefill chunks.

The runtime manifest is an integrity check rooted in the trusted immutable
image, not an independent signature. The build host and container operator
remain trusted. Run the container read-only, keep writable tmpfs mounts outside
`/opt/vllm-fruit`, `/opt/b12x-fruit`, and `/opt/fruit-runtime`, and mount the
authenticated model read-only as shown above.

W4A16 is used for prefill and any row count above the W4A8 decode ceiling. W4A8
is selected for decode-sized batches of at most 16 rows. Unsupported shapes,
activation modes, metadata, or incomplete manifests fail closed.

## Provenance and integrity

- Authenticated BF16 source: [`__SOURCE_REPOSITORY__`](https://huggingface.co/__SOURCE_REPOSITORY__)
  at immutable revision `__SOURCE_REVISION__`.
- Authenticated source manifest (`__SOURCE_KIND__`) SHA-256:
  `__SOURCE_SHA256__`.
- Calibration capture ID:
  `__CALIBRATION_CAPTURE_ID__`.
- Calibration manifest SHA-256:
  `__CALIBRATION_MANIFEST_SHA256__`.
- The encoder authenticated __CALIBRATION_DOCUMENTS__ documents /
  __CALIBRATION_TOKENS__ tokens from disjoint fit, confirmation, and validation
  splits.
- Full encoding: __ENCODED_EXPERTS__ experts, __ENCODE_SECONDS__ GPU-seconds,
  __PEAK_GIB__ GiB peak CUDA allocation.
- `MANIFEST.sha256`, `qsrt-manifest.json`, `.qsrt-source-evidence.json`,
  `qsrt-calibration-evidence.json`,
  `evaluation/fruit-runtime-qualification.json`, and `QSRT_COMPLETE.json` bind
  the published package to the source, producer, and evaluation receipts.

## Known limitations

__QUALITY_LIMITATIONS__
- The packaged launcher permits TP1. TP2 atom ownership is unit-tested, but no
  package-specific TP2 serving benchmark is claimed.
- The current sparse-attention prefill backend requires `max_num_seqs=1`.
- This release implements the QSRT codec, storage, loader, and kernels. It does
  not establish broad downstream task quality.

## License

The packaged model files are MIT, matching the authenticated Fruit BF16 source
license. B12X and vLLM are Apache-2.0. KQuant is not redistributed in this
model repository and remains subject to its upstream repository licensing.
"""
_SOURCE_EVIDENCE_NAME = ".qsrt-source-evidence.json"
_SOURCE_EVIDENCE_SHA_NAME = ".qsrt-source-evidence.sha256"
_CALIBRATION_EVIDENCE_NAME = "qsrt-calibration-evidence.json"

_RATE_SWEEP_SCHEMA = "kquant_fruit_uniform_rate_sweep_v1"
_RATE_SWEEP_NAME = "evaluation/fruit-uniform-rate-sweep.json"
_RUNTIME_QUALIFICATION_SCHEMA = "kquant_fruit_runtime_qualification_v1"
_RUNTIME_QUALIFICATION_NAME = "evaluation/fruit-runtime-qualification.json"
_RUNTIME_ARMS = ("bf16", "siq", "qsrt")
_RUNTIME_PATHS_SCHEMA = "kquant_fruit_runtime_paths_v1"
_INSTRUCT_COMPARATOR_MODELS = {
    "bf16": {
        "repository": "malaiwah/GLM-5.2-SIQ-Fruit-Instruct-bf16",
        "revision": "678954f65e056a0f508e21eeb9251c655bb9463f",
        "manifest_sha256": "8f23aed5e9b12000ed103a76da772a20730ca53ab7e352d6cb94da2709165245",
        "config_sha256": "1b1ea852c2bea8644774ec795025df2d0247b67131bccc8bf7e1137699518d55",
        "model_index_sha256": "86e6cc1d8548c7bdbbc117e93b85b8ae249f446de9b48d2195e51f358674ba56",
        "safetensors_bytes": 10_081_800_232,
        "safetensors_sha256": "01fb6ad26356fc22f07f2598385b132db59df4eddd92bc005dfc0622284ee12b",
    },
    "siq": {
        "repository": "malaiwah/GLM-5.2-SIQ-Fruit-Instruct",
        "revision": "48452ef397d8b4a4d6d0c00ea376a2abb3ef6314",
        "manifest_sha256": "ac5485e2552f54850eebfecf11e23f3f640c391ed335d06562f91eb34f613639",
        "config_sha256": "9d137e2b59fff529eb122581b0bce6eb7ace458a0785368d2ba587b4a5c2aa6f",
        "model_index_sha256": "5808a4b3e75c4a949a1ede42e6c6fb2576089ec1544038b77de24076e99bf3da",
        "safetensors_bytes": 3_102_116_152,
        "safetensors_sha256": "9c6c5c2c07eeb3aed026db4f6c5fc208dc04272304ba4f39ea9d23a31f9012b5",
    },
}

_FIXED_CUDAGRAPH_CAPTURE_SIZES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    20,
    24,
    32,
    48,
    64,
]
_FIXED_COMPILATION_CONFIG = {
    "backend": "inductor",
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "custom_ops": ["all"],
    "cudagraph_capture_sizes": _FIXED_CUDAGRAPH_CAPTURE_SIZES,
}
_FIXED_RUNTIME_OPTIONS = {
    "--attention-backend": "B12X_MLA_SPARSE",
    "--moe-backend": "b12x",
    "--kv-cache-dtype": "nvfp4_ds_mla",
}
_FIXED_RUNTIME_SWITCHES = (
    "--enable-chunked-prefill",
    "--enable-prefix-caching",
)
_ENCODER_FINGERPRINT_SCHEMA = "kquant_fruit_qsrt_encoder_source_v3"
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot resolve source revision: {root}") from exc
    revision = result.stdout.strip()
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"source revision is not a commit digest: {root}")
    return revision


def _encoder_fingerprint_payload(encoder: dict[str, object]) -> dict[str, object]:
    return {
        "schema": _ENCODER_FINGERPRINT_SCHEMA,
        "kquant_revision": encoder["kquant_revision"],
        "kquant_source_sha256": encoder["kquant_source_sha256"],
        "exllamav3_source_sha256": encoder["exllamav3_source_sha256"],
        "exllamav3_revision": encoder["exllamav3_revision"],
        "calibration_fingerprint": encoder["calibration_fingerprint"],
        "calibration_capture_id": encoder["calibration_capture_id"],
        "calibration_manifest_sha256": encoder["calibration_manifest_sha256"],
    }


def current_encoder_provenance(
    *,
    exllamav3_root: Path,
    calibration: FruitCalibrationStore,
    kquant_root: Path | None = None,
) -> dict[str, object]:
    global _KQUANT_SNAPSHOT_CLEAN_VERIFIED
    if kquant_root is None:
        if _KQUANT_IMPORT_IDENTITY is None:
            raise RuntimeError("KQuant source snapshot identity is unavailable")
        if not _KQUANT_SNAPSHOT_CLEAN_VERIFIED:
            clean_identity = (
                _git_revision(_KQUANT_IMPORT_SOURCE_ROOT),
                tracked_worktree_sha256(_KQUANT_IMPORT_SOURCE_ROOT),
            )
            if clean_identity != _KQUANT_IMPORT_IDENTITY:
                raise ValueError(
                    "KQuant source snapshot does not match the clean committed tree"
                )
            _KQUANT_SNAPSHOT_CLEAN_VERIFIED = True
        kquant_revision, kquant_source_sha256 = _KQUANT_IMPORT_IDENTITY
    else:
        kquant_checkout = kquant_root.resolve(strict=True)
        kquant_revision = _git_revision(kquant_checkout)
        kquant_source_sha256 = tracked_worktree_sha256(kquant_checkout)
    exllamav3_revision, exllamav3_source_sha256 = exllamav3_source_identity(
        exllamav3_root
    )
    if exllamav3_revision != EXLLAMAV3_REVISION:
        raise ValueError("ExLlamaV3 source revision does not match the pinned encoder")
    encoder: dict[str, object] = {
        "kquant_revision": kquant_revision,
        "kquant_source_sha256": kquant_source_sha256,
        "exllamav3_revision": exllamav3_revision,
        "exllamav3_source_sha256": exllamav3_source_sha256,
        "calibration_fingerprint": calibration.fingerprint,
        "calibration_capture_id": calibration.capture_id,
        "calibration_manifest_sha256": calibration.manifest_sha256,
        "fingerprint_schema": _ENCODER_FINGERPRINT_SCHEMA,
    }
    encoder["fingerprint"] = hashlib.sha256(
        _canonical_json(_encoder_fingerprint_payload(encoder)).encode("utf-8")
    ).hexdigest()
    return encoder


def _producer_provenance(
    *,
    exllamav3_root: Path,
    b12x_root: Path,
    vllm_root: Path,
    calibration: FruitCalibrationStore,
) -> dict[str, object]:
    encoder = current_encoder_provenance(
        exllamav3_root=exllamav3_root,
        calibration=calibration,
    )
    runtime = {
        "b12x_revision": _git_revision(b12x_root),
        "b12x_source_sha256": tracked_worktree_sha256(b12x_root),
        "vllm_revision": _git_revision(vllm_root),
        "vllm_source_sha256": tracked_worktree_sha256(vllm_root),
    }
    provenance: dict[str, object] = {
        "schema": "kquant_fruit_qsrt_producer_v1",
        "encoder": encoder,
        "runtime": runtime,
    }
    provenance["fingerprint"] = hashlib.sha256(
        _canonical_json(provenance).encode("utf-8")
    ).hexdigest()
    return provenance


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _rate_number(value: object, *, name: str, positive: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
        or (positive and float(value) <= 0)
    ):
        raise ValueError(f"Fruit rate-sweep {name} is invalid")
    return float(value)


def _validate_rate_sweep(
    path: Path,
    *,
    source_evidence: dict[str, object],
    calibration: FruitCalibrationStore,
    producer: dict[str, object],
) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Fruit rate sweep: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != _RATE_SWEEP_SCHEMA
        or payload.get("complete") is not True
    ):
        raise ValueError("Fruit rate sweep is incomplete or has the wrong schema")
    signature = payload.get("signature")
    results = payload.get("results")
    if not isinstance(signature, dict) or not isinstance(results, dict):
        raise TypeError("Fruit rate sweep omits its signature or results")
    expected_calibration = {
        "capture_id": calibration.capture_id,
        "fingerprint": calibration.fingerprint,
        "manifest_sha256": calibration.manifest_sha256,
    }
    measured_encoder = signature.get("encoder")
    expected_encoder = producer.get("encoder")
    if not isinstance(measured_encoder, dict) or not isinstance(expected_encoder, dict):
        raise TypeError("Fruit rate sweep omits its encoder identity")
    if (
        signature.get("schema") != _RATE_SWEEP_SCHEMA
        or signature.get("source") != source_evidence
        or signature.get("calibration") != expected_calibration
        or measured_encoder != expected_encoder
        or signature.get("rates") != [2, 3, 4]
    ):
        raise ValueError("Fruit rate-sweep provenance does not match this build")
    assignments = signature.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("Fruit rate sweep has no assignments")
    assignment_keys: list[str] = []
    for assignment in assignments:
        if (
            not isinstance(assignment, list)
            or len(assignment) != 2
            or any(type(value) is not int for value in assignment)
        ):
            raise TypeError("Fruit rate-sweep assignment is malformed")
        layer, expert = assignment
        if layer not in LAYERS or not 0 <= expert < EXPERTS:
            raise ValueError("Fruit rate-sweep assignment is outside model geometry")
        assignment_keys.append(f"{layer}:{expert}")
    if len(set(assignment_keys)) != len(assignment_keys):
        raise ValueError("Fruit rate-sweep assignments are not unique")
    if set(results) != set(assignment_keys):
        raise ValueError("Fruit rate-sweep results do not cover every assignment")
    measured = 0
    for key in assignment_keys:
        result = results[key]
        if not isinstance(result, dict):
            raise TypeError(f"Fruit rate-sweep result {key} is malformed")
        status = result.get("status")
        if status == "skipped":
            if not isinstance(result.get("reason"), str) or not isinstance(
                result.get("support"), dict
            ):
                raise ValueError(f"Fruit rate-sweep skip {key} lacks evidence")
            continue
        if status != "measured":
            raise ValueError(f"Fruit rate-sweep result {key} has invalid status")
        measured += 1
        rates = result.get("rates")
        if not isinstance(rates, dict) or set(rates) != {"K2", "K3", "K4"}:
            raise ValueError(f"Fruit rate-sweep result {key} omits rate endpoints")
    if measured == 0:
        raise ValueError("Fruit rate sweep contains no measured assignments")
    return payload


def _rate_sweep_section(payload: dict[str, object]) -> str:
    results = payload.get("results")
    signature = payload.get("signature")
    if not isinstance(results, dict) or not isinstance(signature, dict):
        raise TypeError("Fruit rate-sweep summary input is malformed")
    measured = [
        value
        for value in results.values()
        if isinstance(value, dict) and value.get("status") == "measured"
    ]
    assignments = signature.get("assignments")
    if not isinstance(assignments, list) or not measured:
        raise ValueError("Fruit rate-sweep summary has no measured assignments")

    summaries: dict[str, dict[str, float]] = {}
    for rate_name, bits in (("K2", 2), ("K3", 3), ("K4", 4)):
        weight_error = 0.0
        weight_reference = 0.0
        h_error = 0.0
        h_reference = 0.0
        validation_error = 0.0
        validation_reference = 0.0
        endpoint_bytes = 0.0
        endpoint_bpw = 0.0
        for result in measured:
            rates = result.get("rates")
            if not isinstance(rates, dict):
                raise TypeError("Fruit measured rate result is malformed")
            endpoint = rates.get(rate_name)
            if not isinstance(endpoint, dict) or endpoint.get("bits") != bits:
                raise ValueError(f"Fruit {rate_name} endpoint is malformed")
            aggregate = endpoint.get("aggregate")
            matrices = endpoint.get("matrices")
            routed = endpoint.get("routed_function")
            if (
                not isinstance(aggregate, dict)
                or not isinstance(matrices, list)
                or len(matrices) != 3
                or not isinstance(routed, dict)
            ):
                raise TypeError(f"Fruit {rate_name} metric evidence is malformed")
            matrix_names = {
                matrix.get("matrix") for matrix in matrices if isinstance(matrix, dict)
            }
            if matrix_names != {"w1", "w3", "w2"}:
                raise ValueError(f"Fruit {rate_name} matrix evidence is incomplete")
            for matrix in matrices:
                assert isinstance(matrix, dict)
                weight_error += _rate_number(
                    matrix.get("weight_squared_error"),
                    name=f"{rate_name}.weight_squared_error",
                )
                weight_reference += _rate_number(
                    matrix.get("weight_reference_energy"),
                    name=f"{rate_name}.weight_reference_energy",
                    positive=True,
                )
                h_error += _rate_number(
                    matrix.get("captured_dense_h_numerator"),
                    name=f"{rate_name}.captured_dense_h_numerator",
                )
                h_reference += _rate_number(
                    matrix.get("captured_dense_h_denominator"),
                    name=f"{rate_name}.captured_dense_h_denominator",
                    positive=True,
                )
            validation = routed.get("validation")
            if not isinstance(validation, dict):
                raise TypeError(f"Fruit {rate_name} validation evidence is malformed")
            validation_error += _rate_number(
                validation.get("routed_sse"),
                name=f"{rate_name}.validation.routed_sse",
            )
            validation_reference += _rate_number(
                validation.get("reference_energy"),
                name=f"{rate_name}.validation.reference_energy",
                positive=True,
            )
            endpoint_bytes += _rate_number(
                aggregate.get("bytes_before_layer_deduplication"),
                name=f"{rate_name}.bytes",
                positive=True,
            )
            endpoint_bpw += _rate_number(
                aggregate.get("bpw_before_layer_deduplication"),
                name=f"{rate_name}.bpw",
                positive=True,
            )
        summaries[rate_name] = {
            "weight_nmse": weight_error / weight_reference,
            "captured_h_nmse": h_error / h_reference,
            "validation_nmse": validation_error / validation_reference,
            "mean_bytes": endpoint_bytes / len(measured),
            "mean_bpw": endpoint_bpw / len(measured),
        }

    k2 = summaries["K2"]
    k3 = summaries["K3"]
    k4 = summaries["K4"]
    rows = "\n".join(
        "| {rate} | {mean_bpw:.4f} | {mean_bytes:,.0f} | {weight_nmse:.6f} | "
        "{captured_h_nmse:.6f} | {validation_nmse:.6f} |".format(
            rate=rate,
            **summaries[rate],
        )
        for rate in ("K2", "K3", "K4")
    )
    skipped = len(assignments) - len(measured)
    return f"""## Adjacent-rate evidence

`{_RATE_SWEEP_NAME}` re-encodes the same authenticated expert sample at uniform
K2, K3, and K4, with fresh per-endpoint FP16 scales. It measured
{len(measured)} of {len(assignments)} predeclared assignments; {skipped} lacked
the minimum routed calibration support and were skipped rather than imputed.
These are pre-layer-deduplication expert-local endpoints, not package sizes.

| Endpoint | Mean bpw | Mean bytes/expert | Weight NMSE | Captured-H NMSE | Validation routed NMSE |
|---|---:|---:|---:|---:|---:|
{rows}

Relative to K3, K2 is {k2["weight_nmse"] / k3["weight_nmse"]:.3f}x /
{k2["captured_h_nmse"] / k3["captured_h_nmse"]:.3f}x /
{k2["validation_nmse"] / k3["validation_nmse"]:.3f}x on weight,
captured-H, and validation-routed NMSE. K4 is
{k4["weight_nmse"] / k3["weight_nmse"]:.3f}x /
{k4["captured_h_nmse"] / k3["captured_h_nmse"]:.3f}x /
{k4["validation_nmse"] / k3["validation_nmse"]:.3f}x on the same metrics.
"""


def _qualification_object(
    value: object, *, name: str, keys: set[str]
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"Fruit runtime qualification {name} must be an object")
    if set(value) != keys:
        raise ValueError(
            f"Fruit runtime qualification {name} keys mismatch; "
            f"missing={sorted(keys - set(value))}, "
            f"unknown={sorted(set(value) - keys)}"
        )
    return value


def _qualification_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"Fruit runtime qualification {name} must be a nonempty string")
    return value


def _qualification_number(
    value: object,
    *,
    name: str,
    positive: bool = False,
    maximum: float | None = None,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
        or (positive and float(value) <= 0)
        or (maximum is not None and float(value) > maximum)
    ):
        raise ValueError(f"Fruit runtime qualification {name} is invalid")
    return float(value)


def _qualification_digest(value: object, *, name: str) -> str:
    digest = _qualification_string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"Fruit runtime qualification {name} is not a SHA-256 digest")
    return digest


def _qualification_revision(value: object, *, name: str) -> str:
    revision = _qualification_string(value, name=name)
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"Fruit runtime qualification {name} is not a Git revision")
    return revision


def _runtime_argv_option(argv: list[str], flag: str) -> str:
    values: list[str] = []
    prefix = f"{flag}="
    for index, argument in enumerate(argv):
        if argument.startswith(prefix):
            values.append(argument[len(prefix) :])
        elif argument == flag:
            if index + 1 >= len(argv) or argv[index + 1].startswith("-"):
                raise ValueError(
                    f"Fruit runtime qualification argv option {flag} has no value"
                )
            values.append(argv[index + 1])
    if len(values) != 1 or not values[0]:
        raise ValueError(
            f"Fruit runtime qualification argv must contain exactly one {flag}"
        )
    return values[0]


def _runtime_argv_json(argv: list[str], flag: str) -> dict[str, object]:
    raw_value = _runtime_argv_option(argv, flag)
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Fruit runtime qualification argv option {flag} is not JSON"
        ) from exc
    if not isinstance(value, dict):
        raise TypeError(
            f"Fruit runtime qualification argv option {flag} must be an object"
        )
    return value


def _normalized_runtime_argv(argv: list[str]) -> tuple[str, ...]:
    if len(argv) < 3 or argv[:2] != ["vllm", "serve"] or argv[2].startswith("-"):
        raise ValueError("Fruit runtime qualification argv is not vllm serve MODEL")
    normalized = list(argv)
    normalized[2] = "<MODEL>"
    variable_options = {
        "--model": "<MODEL>",
        "--served-model-name": "<MODEL>",
        "--port": "<PORT>",
        "--download-dir": "<PATH>",
    }
    for index, argument in enumerate(normalized):
        for flag, replacement in variable_options.items():
            if argument == flag:
                if index + 1 >= len(normalized):
                    raise ValueError(
                        f"Fruit runtime qualification argv option {flag} has no value"
                    )
                normalized[index + 1] = replacement
            elif argument.startswith(f"{flag}="):
                normalized[index] = f"{flag}={replacement}"
    return tuple(normalized)


def _validate_runtime_argv(
    argv: list[str],
    *,
    arm: str,
    protocol: dict[str, object],
    compilation_backend: str,
    cudagraph_mode: object,
) -> tuple[str, ...]:
    if "--enforce-eager" in argv:
        raise ValueError("Fruit runtime qualification argv may not force eager mode")
    expected_integers = {
        "--tensor-parallel-size": int(protocol["tensor_parallel_size"]),
        "--pipeline-parallel-size": 1,
        "--max-num-seqs": int(protocol["max_num_seqs"]),
        "--max-model-len": 4096,
        "--max-num-batched-tokens": 4096,
    }
    for flag, expected in expected_integers.items():
        raw_value = _runtime_argv_option(argv, flag)
        try:
            measured = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Fruit runtime qualification argv option {flag} is not an integer"
            ) from exc
        if str(measured) != raw_value or measured != expected:
            raise ValueError(
                f"Fruit runtime qualification argv option {flag} is not {expected}"
            )

    if compilation_backend != "inductor" or cudagraph_mode != "FULL_AND_PIECEWISE":
        raise ValueError(
            f"Fruit runtime qualification loaders.{arm} must use non-eager "
            "inductor FULL_AND_PIECEWISE"
        )
    compilation = _runtime_argv_json(argv, "--compilation-config")
    if compilation != _FIXED_COMPILATION_CONFIG:
        raise ValueError(
            f"Fruit runtime qualification loaders.{arm} compilation config "
            "is not the fixed deployment contract"
        )
    speculative = _runtime_argv_json(argv, "--speculative-config")
    if speculative != {"method": "mtp", "num_speculative_tokens": 1}:
        raise ValueError(
            f"Fruit runtime qualification loaders.{arm} MTP argv is not qualified"
        )
    for flag, expected in _FIXED_RUNTIME_OPTIONS.items():
        if _runtime_argv_option(argv, flag) != expected:
            raise ValueError(
                f"Fruit runtime qualification loaders.{arm} argv option {flag} "
                f"is not {expected}"
            )
    for flag in _FIXED_RUNTIME_SWITCHES:
        if argv.count(flag) != 1 or any(
            argument.startswith(f"{flag}=") for argument in argv
        ):
            raise ValueError(
                f"Fruit runtime qualification loaders.{arm} argv must contain "
                f"exactly one {flag}"
            )

    for flag, expected in {
        "--quantization": "kquant_hybrid",
        "--load-format": "fastsafetensors",
    }.items():
        if _runtime_argv_option(argv, flag) != expected:
            raise ValueError(
                f"Fruit runtime qualification loaders.{arm} argv option {flag} "
                f"is not {expected}"
            )
    return _normalized_runtime_argv(argv)


def _positive_runtime_count(value: object, *, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"Fruit runtime path evidence {name} must be positive")


def _validate_runtime_paths(value: object) -> None:
    runtime_paths = _qualification_object(
        value,
        name="runtime_paths",
        keys={"schema", "version", "layers", "cudagraph", "speculative"},
    )
    if (
        runtime_paths["schema"] != _RUNTIME_PATHS_SCHEMA
        or type(runtime_paths["version"]) is not int
        or runtime_paths["version"] != 1
    ):
        raise ValueError("Fruit runtime path evidence has the wrong schema")

    layers = _qualification_object(
        runtime_paths["layers"],
        name="runtime_paths.layers",
        keys={str(layer) for layer in range(3, 14)},
    )
    for layer in range(3, 13):
        layer_paths = _qualification_object(
            layers[str(layer)],
            name=f"runtime_paths.layers.{layer}",
            keys={"prefill", "decode"},
        )
        prefill = _qualification_object(
            layer_paths["prefill"],
            name=f"runtime_paths.layers.{layer}.prefill",
            keys={"mode", "calls"},
        )
        if prefill["mode"] != "w4a16":
            raise ValueError(
                f"Fruit runtime path layer {layer} prefill did not use W4A16"
            )
        _positive_runtime_count(prefill["calls"], name=f"layers.{layer}.prefill.calls")
        decode = _qualification_object(
            layer_paths["decode"],
            name=f"runtime_paths.layers.{layer}.decode",
            keys={
                "mode",
                "calls",
                "part_count",
                "capture_calls",
                "replay_calls",
            },
        )
        if (
            decode["mode"] != "w4a8"
            or type(decode["part_count"]) is not int
            or decode["part_count"] != 2
        ):
            raise ValueError(
                f"Fruit runtime path layer {layer} decode is not two-part W4A8"
            )
        for key in ("calls", "capture_calls", "replay_calls"):
            _positive_runtime_count(decode[key], name=f"layers.{layer}.decode.{key}")

    mtp_layer = _qualification_object(
        layers["13"],
        name="runtime_paths.layers.13",
        keys={"mtp_decode"},
    )
    mtp_decode = _qualification_object(
        mtp_layer["mtp_decode"],
        name="runtime_paths.layers.13.mtp_decode",
        keys={"mode", "calls", "part_count", "capture_calls", "replay_calls"},
    )
    if (
        mtp_decode["mode"] != "w4a8"
        or type(mtp_decode["part_count"]) is not int
        or mtp_decode["part_count"] != 2
    ):
        raise ValueError("Fruit runtime path MTP decode is not two-part W4A8")
    for key in ("calls", "capture_calls", "replay_calls"):
        _positive_runtime_count(mtp_decode[key], name=f"layers.13.mtp_decode.{key}")

    cudagraph = _qualification_object(
        runtime_paths["cudagraph"],
        name="runtime_paths.cudagraph",
        keys={"mode", "capture_count", "replay_count"},
    )
    if cudagraph["mode"] != "FULL_AND_PIECEWISE":
        raise ValueError("Fruit runtime path CUDA graph mode is not qualified")
    _positive_runtime_count(cudagraph["capture_count"], name="cudagraph.capture_count")
    _positive_runtime_count(cudagraph["replay_count"], name="cudagraph.replay_count")

    speculative = _qualification_object(
        runtime_paths["speculative"],
        name="runtime_paths.speculative",
        keys={"method", "num_speculative_tokens", "draft_tokens"},
    )
    if (
        speculative["method"] != "mtp"
        or type(speculative["num_speculative_tokens"]) is not int
        or speculative["num_speculative_tokens"] != 1
    ):
        raise ValueError("Fruit runtime path speculative configuration is not MTP")
    _positive_runtime_count(
        speculative["draft_tokens"], name="speculative.draft_tokens"
    )


def _candidate_safetensors_sha256(output: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in output.iterdir():
        if path.suffix != ".safetensors":
            continue
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError(f"Fruit candidate tensor is not a regular file: {path}")
        result[path.name] = _sha256(path)
    if not result:
        raise ValueError("Fruit runtime qualification candidate has no Safetensors")
    return dict(sorted(result.items()))


def _validate_sealed_runtime_qualification(
    output: Path, runtime_qualification: dict[str, object]
) -> None:
    qualification_path = output / _RUNTIME_QUALIFICATION_NAME
    if qualification_path.read_bytes() != _canonical_json(runtime_qualification).encode(
        "utf-8"
    ):
        raise ValueError("Fruit QSRT sealed runtime qualification changed")
    candidate = runtime_qualification["candidate"]
    if not isinstance(candidate, dict) or (
        candidate.get("model_index_sha256")
        != _sha256(output / "model.safetensors.index.json")
        or candidate.get("safetensors_sha256") != _candidate_safetensors_sha256(output)
    ):
        raise ValueError("Fruit QSRT sealed runtime qualification candidate changed")


def _validate_runtime_qualification(
    path: Path,
    *,
    output: Path,
    variant: str,
    publication: FruitPublicationSpec,
    producer: dict[str, object],
    source_evidence: dict[str, object],
) -> dict[str, object]:
    try:
        receipt_bytes = path.read_bytes()
        payload = json.loads(receipt_bytes)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read Fruit runtime qualification: {path}") from exc
    if receipt_bytes != _canonical_json(payload).encode("utf-8"):
        raise ValueError("Fruit runtime qualification receipt is not canonical JSON")
    payload = _qualification_object(
        payload,
        name="root",
        keys={
            "schema",
            "version",
            "complete",
            "publication",
            "producer",
            "source",
            "candidate",
            "environment",
            "protocol",
            "loaders",
            "models",
            "decode",
            "generation",
            "fidelity",
            "runtime_paths",
        },
    )
    if (
        payload["schema"] != _RUNTIME_QUALIFICATION_SCHEMA
        or type(payload["version"]) is not int
        or payload["version"] != 1
        or payload["complete"] is not True
    ):
        raise ValueError(
            "Fruit runtime qualification is incomplete or has the wrong schema"
        )
    measured_publication = _qualification_object(
        payload["publication"],
        name="publication",
        keys={"variant", "repository"},
    )
    if measured_publication != {
        "variant": "instruct",
        "repository": publication.repository,
    }:
        raise ValueError(
            "Fruit runtime qualification publication does not match this build"
        )
    if variant != "instruct":
        raise ValueError("only the Instruct Fruit publication may be qualified")
    if payload["producer"] != producer:
        raise ValueError(
            "Fruit runtime qualification producer does not match this build"
        )
    if payload["source"] != source_evidence:
        raise ValueError("Fruit runtime qualification source does not match this build")

    candidate = _qualification_object(
        payload["candidate"],
        name="candidate",
        keys={
            "marker_sha256",
            "model_index_sha256",
            "safetensors_sha256",
        },
    )
    measured_candidate_marker = _qualification_digest(
        candidate["marker_sha256"], name="candidate.marker_sha256"
    )
    candidate_marker_path = output / _CANDIDATE_MARKER_NAME
    if not candidate_marker_path.is_file() or measured_candidate_marker != _sha256(
        candidate_marker_path
    ):
        raise ValueError(
            "Fruit runtime qualification candidate marker does not match output"
        )
    measured_index = _qualification_digest(
        candidate["model_index_sha256"], name="candidate.model_index_sha256"
    )
    if measured_index != _sha256(output / "model.safetensors.index.json"):
        raise ValueError(
            "Fruit runtime qualification model index does not match output"
        )
    measured_tensors = candidate["safetensors_sha256"]
    if not isinstance(measured_tensors, dict) or not measured_tensors:
        raise TypeError("Fruit runtime qualification Safetensors map must be nonempty")
    for filename, digest in measured_tensors.items():
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".safetensors")
        ):
            raise ValueError(
                "Fruit runtime qualification has an invalid tensor filename"
            )
        _qualification_digest(digest, name=f"candidate.safetensors_sha256.{filename}")
    if measured_tensors != _candidate_safetensors_sha256(output):
        raise ValueError(
            "Fruit runtime qualification Safetensors map does not exactly match output"
        )
    models = _qualification_object(
        payload["models"], name="models", keys=set(_RUNTIME_ARMS)
    )
    actual_tensor_bytes = sum(
        (output / filename).stat().st_size for filename in measured_tensors
    )
    actual_tensor_set_sha256 = hashlib.sha256(
        _canonical_json(measured_tensors).encode("utf-8")
    ).hexdigest()
    for arm in _RUNTIME_ARMS:
        model = _qualification_object(
            models[arm],
            name=f"models.{arm}",
            keys={
                "repository",
                "revision",
                "manifest_sha256",
                "config_sha256",
                "model_index_sha256",
                "safetensors_bytes",
                "safetensors_sha256",
            },
        )
        _qualification_string(model["repository"], name=f"models.{arm}.repository")
        _qualification_revision(model["revision"], name=f"models.{arm}.revision")
        for key in ("manifest_sha256", "config_sha256", "model_index_sha256"):
            _qualification_digest(model[key], name=f"models.{arm}.{key}")
        if (
            type(model["safetensors_bytes"]) is not int
            or model["safetensors_bytes"] <= 0
        ):
            raise ValueError(
                f"Fruit runtime qualification models.{arm}.safetensors_bytes is invalid"
            )
        _qualification_digest(
            model["safetensors_sha256"],
            name=f"models.{arm}.safetensors_sha256",
        )
    if variant == "instruct":
        for arm, expected_identity in _INSTRUCT_COMPARATOR_MODELS.items():
            if models[arm] != expected_identity:
                raise ValueError(
                    f"Fruit runtime qualification {arm.upper()} comparator "
                    "identity is not the pinned Instruct checkpoint"
                )
    qsrt_model = models["qsrt"]
    if (
        qsrt_model["repository"] != publication.repository
        or qsrt_model["model_index_sha256"] != measured_index
        or qsrt_model["config_sha256"] != _sha256(output / "config.json")
        or qsrt_model["safetensors_bytes"] != actual_tensor_bytes
        or qsrt_model["safetensors_sha256"] != actual_tensor_set_sha256
    ):
        raise ValueError(
            "Fruit runtime qualification QSRT model identity does not match output"
        )

    environment = _qualification_object(
        payload["environment"],
        name="environment",
        keys={"gpu_model", "gpu_driver", "host"},
    )
    for key in ("gpu_model", "gpu_driver", "host"):
        _qualification_string(environment[key], name=f"environment.{key}")

    protocol = _qualification_object(
        payload["protocol"],
        name="protocol",
        keys={
            "tensor_parallel_size",
            "max_num_seqs",
            "max_tokens",
            "temperature",
            "repetitions",
            "prompt_id",
            "prompt",
            "prompt_token_ids",
            "launch_order",
        },
    )
    launch_order = protocol["launch_order"]
    if (
        type(protocol["tensor_parallel_size"]) is not int
        or protocol["tensor_parallel_size"] != 1
        or type(protocol["max_num_seqs"]) is not int
        or protocol["max_num_seqs"] != 1
        or type(protocol["max_tokens"]) is not int
        or int(protocol["max_tokens"]) <= 0
        or type(protocol["repetitions"]) is not int
        or int(protocol["repetitions"]) < 3
        or not isinstance(launch_order, list)
        or len(launch_order) != len(_RUNTIME_ARMS)
        or set(launch_order) != set(_RUNTIME_ARMS)
    ):
        raise ValueError("Fruit runtime qualification is not a matched TP1 protocol")
    _qualification_number(protocol["temperature"], name="protocol.temperature")
    decode_prompt_id = _qualification_string(
        protocol["prompt_id"], name="protocol.prompt_id"
    )
    _qualification_string(protocol["prompt"], name="protocol.prompt")
    prompt_token_ids = protocol["prompt_token_ids"]
    if (
        not isinstance(prompt_token_ids, list)
        or not prompt_token_ids
        or any(type(token) is not int or token < 0 for token in prompt_token_ids)
    ):
        raise ValueError("Fruit runtime qualification prompt tokens are invalid")

    loaders = _qualification_object(
        payload["loaders"], name="loaders", keys=set(_RUNTIME_ARMS)
    )
    producer_runtime = producer.get("runtime")
    producer_encoder = producer.get("encoder")
    if not isinstance(producer_runtime, dict) or not isinstance(producer_encoder, dict):
        raise TypeError("Fruit runtime qualification producer identity is malformed")
    normalized_runtime_argv: tuple[str, ...] | None = None
    for arm in _RUNTIME_ARMS:
        loader = _qualification_object(
            loaders[arm],
            name=f"loaders.{arm}",
            keys={
                "runtime",
                "log_line",
                "weight_bytes",
                "peak_activation_bytes",
                "non_torch_bytes",
                "cudagraph_bytes",
                "kv_cache_bytes",
                "load_seconds",
                "torch_allocated_bytes",
                "torch_reserved_bytes",
                "nvml_used_bytes",
            },
        )
        runtime = _qualification_object(
            loader["runtime"],
            name=f"loaders.{arm}.runtime",
            keys={
                "image",
                "vllm_revision",
                "b12x_revision",
                "kquant_revision",
                "argv",
                "environment",
                "software",
                "compilation_backend",
                "cudagraph_mode",
            },
        )
        image = _qualification_string(
            runtime["image"], name=f"loaders.{arm}.runtime.image"
        )
        image_name, separator, image_digest = image.rpartition("@sha256:")
        if not image_name or separator != "@sha256:":
            raise ValueError(
                f"Fruit runtime qualification loaders.{arm} image is not immutable"
            )
        _qualification_digest(image_digest, name=f"loaders.{arm}.runtime.image_digest")
        for key in ("vllm_revision", "b12x_revision", "kquant_revision"):
            _qualification_revision(runtime[key], name=f"loaders.{arm}.runtime.{key}")
        if arm == "qsrt" and (
            runtime["vllm_revision"] != producer_runtime.get("vllm_revision")
            or runtime["b12x_revision"] != producer_runtime.get("b12x_revision")
            or runtime["kquant_revision"] != producer_encoder.get("kquant_revision")
        ):
            raise ValueError(
                "Fruit runtime qualification QSRT runtime does not match producer"
            )
        argv = runtime["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(argument, str) or not argument for argument in argv)
        ):
            raise TypeError(
                f"Fruit runtime qualification loaders.{arm} argv is invalid"
            )
        runtime_environment = runtime["environment"]
        if (
            not isinstance(runtime_environment, dict)
            or not runtime_environment
            or any(
                not isinstance(name, str) or not name or not isinstance(value, str)
                for name, value in runtime_environment.items()
            )
        ):
            raise TypeError(
                f"Fruit runtime qualification loaders.{arm} environment is invalid"
            )
        software = runtime["software"]
        if (
            not isinstance(software, dict)
            or not software
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(version, str)
                or not version
                for name, version in software.items()
            )
        ):
            raise TypeError(
                f"Fruit runtime qualification loaders.{arm} software is invalid"
            )
        compilation_backend = _qualification_string(
            runtime["compilation_backend"],
            name=f"loaders.{arm}.runtime.compilation_backend",
        )
        if (
            compilation_backend != "inductor"
            or runtime["cudagraph_mode"] != "FULL_AND_PIECEWISE"
        ):
            raise ValueError(
                f"Fruit runtime qualification loaders.{arm} must use "
                "inductor FULL_AND_PIECEWISE"
            )
        measured_runtime_argv = _validate_runtime_argv(
            argv,
            arm=arm,
            protocol=protocol,
            compilation_backend=compilation_backend,
            cudagraph_mode=runtime["cudagraph_mode"],
        )
        if normalized_runtime_argv is None:
            normalized_runtime_argv = measured_runtime_argv
        elif measured_runtime_argv != normalized_runtime_argv:
            raise ValueError(
                "Fruit runtime qualification arms do not use the same fixed "
                "launcher contract"
            )
        _qualification_string(loader["log_line"], name=f"loaders.{arm}.log_line")
        for key in (
            "weight_bytes",
            "peak_activation_bytes",
            "non_torch_bytes",
            "cudagraph_bytes",
            "kv_cache_bytes",
            "torch_allocated_bytes",
            "torch_reserved_bytes",
            "nvml_used_bytes",
        ):
            if type(loader[key]) is not int or loader[key] < 0:
                raise ValueError(
                    f"Fruit runtime qualification loaders.{arm}.{key} is invalid"
                )
        for key in ("weight_bytes", "cudagraph_bytes", "kv_cache_bytes"):
            if loader[key] <= 0:
                raise ValueError(
                    f"Fruit runtime qualification loaders.{arm}.{key} must be positive"
                )
        _qualification_number(
            loader["load_seconds"],
            name=f"loaders.{arm}.load_seconds",
            positive=True,
        )
    _validate_runtime_paths(payload["runtime_paths"])

    decode = _qualification_object(
        payload["decode"], name="decode", keys=set(_RUNTIME_ARMS)
    )
    expected_repetitions = set(range(1, int(protocol["repetitions"]) + 1))
    matched_completion_tokens: dict[int, int] = {}
    for arm in _RUNTIME_ARMS:
        runs = decode[arm]
        if not isinstance(runs, list) or len(runs) != len(expected_repetitions):
            raise ValueError(
                f"Fruit runtime qualification decode.{arm} repetition count mismatch"
            )
        measured_repetitions: set[int] = set()
        for row_index, raw_run in enumerate(runs):
            run = _qualification_object(
                raw_run,
                name=f"decode.{arm}[{row_index}]",
                keys={
                    "prompt_id",
                    "repetition",
                    "http_status",
                    "elapsed_seconds",
                    "completion_tokens",
                    "tokens_per_second",
                    "finish_reason",
                    "content",
                },
            )
            if run["prompt_id"] != decode_prompt_id:
                raise ValueError("Fruit runtime qualification decode prompt mismatch")
            repetition = run["repetition"]
            if type(repetition) is not int or repetition not in expected_repetitions:
                raise ValueError("Fruit runtime qualification repetition is invalid")
            measured_repetitions.add(repetition)
            status = run["http_status"]
            if type(status) is not int or not 200 <= status < 300:
                raise ValueError(
                    "Fruit runtime qualification HTTP response was unsuccessful"
                )
            elapsed = _qualification_number(
                run["elapsed_seconds"],
                name=f"decode.{arm}.elapsed_seconds",
                positive=True,
            )
            tokens = run["completion_tokens"]
            if (
                type(tokens) is not int
                or tokens <= 0
                or tokens > int(protocol["max_tokens"])
            ):
                raise ValueError(
                    "Fruit runtime qualification completion token count is invalid"
                )
            rate = _qualification_number(
                run["tokens_per_second"],
                name=f"decode.{arm}.tokens_per_second",
                positive=True,
            )
            if arm == "bf16":
                matched_completion_tokens[repetition] = tokens
            elif matched_completion_tokens.get(repetition) != tokens:
                raise ValueError(
                    "Fruit runtime qualification matched completion counts differ"
                )
            if not math.isclose(rate, tokens / elapsed, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Fruit runtime qualification token-rate math mismatch")
            finish_reason = _qualification_string(
                run["finish_reason"], name=f"decode.{arm}.finish_reason"
            )
            if finish_reason == "length" and tokens != int(protocol["max_tokens"]):
                raise ValueError(
                    "Fruit runtime qualification length finish disagrees with token cap"
                )
            if not isinstance(run["content"], str):
                raise TypeError(
                    "Fruit runtime qualification decode content must be a string"
                )
        if measured_repetitions != expected_repetitions:
            raise ValueError(
                f"Fruit runtime qualification decode.{arm} repetitions are not unique"
            )

    generation = _qualification_object(
        payload["generation"],
        name="generation",
        keys={"prompts", "results"},
    )
    prompts = generation["prompts"]
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Fruit runtime qualification generation prompts are empty")
    prompt_ids: list[str] = []
    for row_index, raw_prompt in enumerate(prompts):
        prompt = _qualification_object(
            raw_prompt,
            name=f"generation.prompts[{row_index}]",
            keys={"id", "prompt", "prompt_token_ids"},
        )
        prompt_id = _qualification_string(
            prompt["id"], name=f"generation.prompts[{row_index}].id"
        )
        _qualification_string(
            prompt["prompt"], name=f"generation.prompts[{row_index}].prompt"
        )
        generation_tokens = prompt["prompt_token_ids"]
        if (
            not isinstance(generation_tokens, list)
            or not generation_tokens
            or any(type(token) is not int or token < 0 for token in generation_tokens)
        ):
            raise ValueError(
                "Fruit runtime qualification generation tokens are invalid"
            )
        prompt_ids.append(prompt_id)
    if len(set(prompt_ids)) != len(prompt_ids) or decode_prompt_id in prompt_ids:
        raise ValueError("Fruit runtime qualification prompt IDs are not unique")
    results = _qualification_object(
        generation["results"],
        name="generation.results",
        keys=set(_RUNTIME_ARMS),
    )
    for arm in _RUNTIME_ARMS:
        rows = results[arm]
        if not isinstance(rows, list) or len(rows) != len(prompt_ids):
            raise ValueError(
                f"Fruit runtime qualification generation.{arm} coverage mismatch"
            )
        covered: list[str] = []
        for row_index, raw_result in enumerate(rows):
            result = _qualification_object(
                raw_result,
                name=f"generation.results.{arm}[{row_index}]",
                keys={"prompt_id", "content"},
            )
            covered.append(
                _qualification_string(
                    result["prompt_id"],
                    name=f"generation.results.{arm}[{row_index}].prompt_id",
                )
            )
            if not isinstance(result["content"], str):
                raise TypeError(
                    "Fruit runtime qualification generation content must be a string"
                )
        if len(set(covered)) != len(covered) or set(covered) != set(prompt_ids):
            raise ValueError(
                f"Fruit runtime qualification generation.{arm} coverage mismatch"
            )

    fidelity = _qualification_object(
        payload["fidelity"],
        name="fidelity",
        keys={"full_vocabulary", "positions", "vocab_size", "candidates"},
    )
    positions = fidelity["positions"]
    if (
        fidelity["full_vocabulary"] is not True
        or not isinstance(positions, list)
        or not positions
        or any(type(position) is not int or position < 0 for position in positions)
        or len(set(positions)) != len(positions)
        or type(fidelity["vocab_size"]) is not int
        or int(fidelity["vocab_size"]) <= 10
    ):
        raise ValueError("Fruit runtime qualification fidelity geometry is invalid")
    candidates = _qualification_object(
        fidelity["candidates"],
        name="fidelity.candidates",
        keys={"siq", "qsrt"},
    )
    for candidate_name in ("siq", "qsrt"):
        candidate_result = _qualification_object(
            candidates[candidate_name],
            name=f"fidelity.candidates.{candidate_name}",
            keys={
                "mean_forward_kl",
                "max_forward_kl",
                "top1_agreement",
                "top10_agreement",
                "per_position",
            },
        )
        rows = candidate_result["per_position"]
        if not isinstance(rows, list) or len(rows) != len(positions):
            raise ValueError("Fruit runtime qualification fidelity coverage mismatch")
        measured_positions: list[int] = []
        divergences: list[float] = []
        top1_matches = 0
        top10_matches = 0
        for row_index, raw_row in enumerate(rows):
            row = _qualification_object(
                raw_row,
                name=f"fidelity.candidates.{candidate_name}.per_position[{row_index}]",
                keys={"position", "forward_kl", "top1_agreement", "top10_agreement"},
            )
            if type(row["position"]) is not int:
                raise TypeError(
                    "Fruit runtime qualification fidelity position is invalid"
                )
            measured_positions.append(int(row["position"]))
            divergences.append(
                _qualification_number(
                    row["forward_kl"],
                    name=f"fidelity.{candidate_name}.forward_kl",
                )
            )
            if (
                type(row["top1_agreement"]) is not bool
                or type(row["top10_agreement"]) is not bool
            ):
                raise TypeError(
                    "Fruit runtime qualification fidelity agreement is invalid"
                )
            if row["top1_agreement"] and not row["top10_agreement"]:
                raise ValueError(
                    "Fruit runtime qualification top-1 match is absent from top-10"
                )
            top1_matches += int(row["top1_agreement"])
            top10_matches += int(row["top10_agreement"])
        if measured_positions != positions:
            raise ValueError("Fruit runtime qualification fidelity positions mismatch")
        aggregates = {
            "mean_forward_kl": sum(divergences) / len(divergences),
            "max_forward_kl": max(divergences),
            "top1_agreement": top1_matches / len(divergences),
            "top10_agreement": top10_matches / len(divergences),
        }
        for name, computed in aggregates.items():
            measured = _qualification_number(
                candidate_result[name],
                name=f"fidelity.{candidate_name}.{name}",
                maximum=1.0 if "agreement" in name else None,
            )
            if not math.isclose(measured, computed, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    "Fruit runtime qualification fidelity aggregate mismatch"
                )
    return payload


def _runtime_qualification_section(payload: dict[str, object]) -> str:
    protocol = payload["protocol"]
    loaders = payload["loaders"]
    decode = payload["decode"]
    fidelity = payload["fidelity"]
    medians: dict[str, float] = {}
    for arm in _RUNTIME_ARMS:
        rates = sorted(float(run["tokens_per_second"]) for run in decode[arm])
        middle = len(rates) // 2
        medians[arm] = (
            rates[middle] if len(rates) % 2 else (rates[middle - 1] + rates[middle]) / 2
        )
    runtime_rows = "\n".join(
        "| {arm} | {rate:.2f} | {ratio:.3f}x | {weight:.3f} | `{backend}` / `{graphs}` |".format(
            arm=arm.upper(),
            rate=medians[arm],
            ratio=medians[arm] / medians["bf16"],
            weight=int(loaders[arm]["weight_bytes"]) / (1 << 30),
            backend=loaders[arm]["runtime"]["compilation_backend"],
            graphs=loaders[arm]["runtime"]["cudagraph_mode"],
        )
        for arm in _RUNTIME_ARMS
    )
    fidelity_rows = "\n".join(
        "| {name} | {mean:.6g} | {maximum:.6g} | {top1:.2%} | {top10:.2%} |".format(
            name=name.upper(),
            mean=float(fidelity["candidates"][name]["mean_forward_kl"]),
            maximum=float(fidelity["candidates"][name]["max_forward_kl"]),
            top1=float(fidelity["candidates"][name]["top1_agreement"]),
            top10=float(fidelity["candidates"][name]["top10_agreement"]),
        )
        for name in ("siq", "qsrt")
    )
    return f"""## Runtime and quality qualification

The sealed receipt [`{_RUNTIME_QUALIFICATION_NAME}`]({_RUNTIME_QUALIFICATION_NAME})
records the exact BF16, SIQ, and QSRT model identities, candidate tensors,
producer, GPU and driver, per-arm immutable runtime images, exact launch argv,
environment and software revisions, parsed loader memory fields, decode runs,
generation outputs, and full-vocabulary fidelity rows. Under its fixed
hardware, prompt tokens, generation settings, recorded launch order, TP1, and
`max_num_seqs=1`, {protocol["repetitions"]}-repetition same-prompt protocol:

| Arm | Median client-observed end-to-end generated-token rate (tokens/s) | Rate / BF16 | Loader weight GiB | Backend / CUDA graph mode |
|---|---:|---:|---:|---|
{runtime_rows}

The legacy SIQ comparator uses its recorded compatible runtime rather than the
QSRT clean-cutover runtime. The derived ratios therefore compare observed
end-to-end runs on identical hardware and inputs, not identical software.
These rates include request and serving overhead; they are not decode-only
kernel rates, an apples-to-apples software comparison, or a general throughput
benchmark.

| Candidate relative to BF16 | Mean forward KL | Max forward KL | Top-1 agreement | Top-10 agreement |
|---|---:|---:|---:|---:|
{fidelity_rows}

The raw generation section covers {len(payload["generation"]["prompts"])} matched
prompts across BF16, SIQ, and QSRT. It is explicitly non-representative and does
not establish downstream instruction/chat task quality."""


def _render_model_card(
    *,
    source_evidence: dict[str, object],
    calibration: FruitCalibrationStore,
    producer: dict[str, object],
    rate_sweep: dict[str, object],
    layers: dict[str, dict[str, object]],
    publication: FruitPublicationSpec,
    runtime_qualification: dict[str, object],
) -> str:
    format_counts: Counter[str] = Counter()
    elapsed_seconds = 0.0
    peak_cuda_bytes = 0
    encoded_experts = 0
    for layer in layers.values():
        counts = layer.get("format_counts")
        experts = layer.get("experts")
        if not isinstance(counts, dict) or not isinstance(experts, list):
            raise TypeError("Fruit layer evidence cannot render the model card")
        for name, count in counts.items():
            if not isinstance(name, str) or type(count) is not int or count < 0:
                raise TypeError("Fruit allocation evidence is malformed")
            format_counts[name] += count
        for expert in experts:
            if not isinstance(expert, dict):
                raise TypeError("Fruit expert evidence is malformed")
            elapsed = expert.get("encode_elapsed_seconds")
            peak = expert.get("peak_cuda_bytes")
            if (
                not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0
                or type(peak) is not int
                or peak < 0
            ):
                raise ValueError("Fruit encoding telemetry is malformed")
            elapsed_seconds += float(elapsed)
            peak_cuda_bytes = max(peak_cuda_bytes, peak)
            encoded_experts += 1
    expected_experts = len(LAYERS) * EXPERTS
    if (
        encoded_experts != expected_experts
        or sum(format_counts.values()) != expected_experts
    ):
        raise ValueError("Fruit model-card expert evidence is incomplete")

    documents = calibration.manifest.get("documents")
    if not isinstance(documents, list) or not all(
        isinstance(document, dict)
        and type(document.get("tokens")) is int
        and int(document["tokens"]) > 0
        for document in documents
    ):
        raise TypeError("Fruit calibration document evidence is malformed")
    calibration_tokens = sum(int(document["tokens"]) for document in documents)
    encoder = producer.get("encoder")
    runtime = producer.get("runtime")
    if not isinstance(encoder, dict) or not isinstance(runtime, dict):
        raise TypeError("Fruit producer evidence is malformed")
    replacements = {
        "__MODEL_TITLE__": publication.title,
        "__MODEL_INTRODUCTION__": publication.introduction,
        "__MODEL_REPOSITORY__": publication.repository,
        "__QUALITY_LIMITATIONS__": publication.quality_limitations,
        "__FRUIT_AUDIT_ROWS__": publication.fruit_audit_rows,
        "__ALLOCATION_ROWS__": "\n".join(
            f"| `{name}` | {count:,} |" for name, count in sorted(format_counts.items())
        ),
        "__RATE_SWEEP_SECTION__": _rate_sweep_section(rate_sweep),
        "__RUNTIME_QUALIFICATION_SECTION__": _runtime_qualification_section(
            runtime_qualification
        ),
        "__KQUANT_REVISION__": str(encoder["kquant_revision"]),
        "__B12X_REVISION__": str(runtime["b12x_revision"]),
        "__VLLM_REVISION__": str(runtime["vllm_revision"]),
        "__SOURCE_REPOSITORY__": str(source_evidence["source_repository"]),
        "__SOURCE_REVISION__": str(source_evidence["source_revision"]),
        "__SOURCE_KIND__": str(source_evidence["source_kind"]),
        "__SOURCE_SHA256__": str(source_evidence["source_sha256"]),
        "__CALIBRATION_CAPTURE_ID__": calibration.capture_id,
        "__CALIBRATION_MANIFEST_SHA256__": calibration.manifest_sha256,
        "__CALIBRATION_DOCUMENTS__": f"{len(documents):,}",
        "__CALIBRATION_TOKENS__": f"{calibration_tokens:,}",
        "__ENCODED_EXPERTS__": f"{encoded_experts:,}",
        "__ENCODE_SECONDS__": f"{elapsed_seconds:,.2f}",
        "__PEAK_GIB__": f"{peak_cuda_bytes / (1 << 30):.3f}",
    }
    card = MODEL_CARD_TEMPLATE
    for marker, value in replacements.items():
        if marker not in card:
            raise ValueError(f"Fruit model-card marker is missing: {marker}")
        card = card.replace(marker, value)
    return card


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_safetensors(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str] | None,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_file(tensors, temporary, metadata=metadata)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _is_encoder_temporary(path: Path) -> bool:
    name = path.name
    if not (
        name.startswith(".tmp")
        or (name.startswith((".expert-", ".run-")) and ".tmp-" in name)
    ):
        return False
    try:
        identity = path.lstat()
    except FileNotFoundError:
        return True
    return stat.S_ISREG(identity.st_mode) and identity.st_nlink == 1


def _validate_part_cache_root(
    root: Path,
    *,
    create: bool,
    allow_run_manifests: bool = False,
) -> None:
    if root.is_symlink():
        raise ValueError(f"Fruit QSRT part cache must not be symbolic: {root}")
    if not root.exists():
        if create:
            root.mkdir(parents=True)
        return
    if not root.is_dir():
        raise ValueError(f"Fruit QSRT part cache must be a directory: {root}")
    expected_layers = {f"layer-{layer:03d}" for layer in LAYERS}
    for directory in root.iterdir():
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"unexpected Fruit QSRT cache path: {directory}")
        if allow_run_manifests and directory.name == "run-manifests":
            for path in directory.iterdir():
                if _is_encoder_temporary(path):
                    continue
                if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
                    raise ValueError(f"unexpected Fruit QSRT run manifest: {path}")
                if path.suffix != ".json":
                    raise ValueError(f"unexpected Fruit QSRT run manifest: {path}")
            continue
        if directory.name not in expected_layers:
            raise ValueError(f"unexpected Fruit QSRT cache directory: {directory}")
        for path in directory.iterdir():
            if allow_run_manifests and _is_encoder_temporary(path):
                continue
            if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
                raise ValueError(f"unexpected Fruit QSRT part path: {path}")
            stem, suffix = path.name.rsplit(".", 1)
            expert_text = stem.removeprefix("expert-")
            if (
                not stem.startswith("expert-")
                or suffix not in {"json", "safetensors"}
                or len(expert_text) != 3
                or not expert_text.isdigit()
                or not 0 <= int(expert_text) < EXPERTS
            ):
                raise ValueError(f"unexpected Fruit QSRT part filename: {path}")


def _prepare_part_layer(output: Path, layer: int) -> None:
    root = output / ".qsrt-parts"
    _validate_part_cache_root(root, create=True)
    directory = root / f"layer-{layer:03d}"
    if directory.is_symlink():
        raise ValueError(f"Fruit QSRT part layer must not be symbolic: {directory}")
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Fruit QSRT part layer must be a directory: {directory}")
    directory.mkdir(exist_ok=True)


def _part_paths(output: Path, layer: int, expert: int) -> tuple[Path, Path]:
    root = output / ".qsrt-parts" / f"layer-{layer:03d}"
    return root / f"expert-{expert:03d}.safetensors", root / f"expert-{expert:03d}.json"


def _validate_qsrt_tensor_contract(
    tensor_path: Path,
    *,
    expected_expert_ids: torch.Tensor,
    expected_formats: torch.Tensor,
) -> dict[str, str]:
    expected_count = int(expected_expert_ids.numel())
    expected_shapes = {
        "expert_ids": (expected_count,),
        "formats": (expected_count, 2),
        "permutations": (expected_count, INTERMEDIATE_SIZE),
        "w13_trellis": (2, expected_count, PAIR_COUNT, PAIR_WORDS),
        "w2_trellis": (expected_count, PAIR_COUNT, PAIR_WORDS),
        "fc1_pair_modes": (expected_count, PAIR_COUNT),
        "fc2_pair_modes": (expected_count, PAIR_COUNT),
        "gate_suh": (expected_count, HIDDEN_SIZE),
        "up_suh": (expected_count, HIDDEN_SIZE),
        "intermediate_rotations": (expected_count, 3 * INTERMEDIATE_SIZE),
        "down_svh": (expected_count, HIDDEN_SIZE),
    }
    expected_dtypes = {
        "expert_ids": torch.int32,
        "formats": torch.int8,
        "permutations": torch.int16,
        "w13_trellis": torch.int16,
        "w2_trellis": torch.int16,
        "fc1_pair_modes": torch.int32,
        "fc2_pair_modes": torch.int32,
        "gate_suh": torch.float16,
        "up_suh": torch.float16,
        "intermediate_rotations": torch.float16,
        "down_svh": torch.float16,
    }
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
        if keys != set(FRUIT_QSRT_ARTIFACT_TENSORS):
            raise ValueError(f"Fruit QSRT tensor inventory mismatch: {tensor_path}")
        tensors = {
            name: handle.get_tensor(name) for name in FRUIT_QSRT_ARTIFACT_TENSORS
        }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(
                f"Fruit QSRT {name} shape mismatch in {tensor_path}: "
                f"{tuple(tensor.shape)} != {expected_shapes[name]}"
            )
        if tensor.dtype != expected_dtypes[name]:
            raise TypeError(
                f"Fruit QSRT {name} dtype mismatch in {tensor_path}: "
                f"{tensor.dtype} != {expected_dtypes[name]}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"Fruit QSRT {name} is noncontiguous in {tensor_path}")
    if not torch.equal(tensors["expert_ids"], expected_expert_ids):
        raise ValueError(f"Fruit QSRT expert IDs mismatch in {tensor_path}")
    formats = tensors["formats"]
    if bool(((formats < 0) | (formats > 2)).any().item()):
        raise ValueError(f"Fruit QSRT formats are outside R0/R1/R2 in {tensor_path}")
    if not torch.equal(formats, expected_formats):
        raise ValueError(
            f"Fruit QSRT tensor formats disagree with manifest in {tensor_path}"
        )
    mode_table = torch.tensor(((0, 0), (1, 0), (1, 1)), dtype=torch.int32)
    if not torch.equal(
        tensors["fc1_pair_modes"], mode_table.index_select(0, formats[:, 0].long())
    ) or not torch.equal(
        tensors["fc2_pair_modes"], mode_table.index_select(0, formats[:, 1].long())
    ):
        raise ValueError(
            f"Fruit QSRT pair modes disagree with formats in {tensor_path}"
        )
    expected_permutation = torch.arange(INTERMEDIATE_SIZE, dtype=torch.int16)
    if not bool(
        torch.all(
            torch.sort(tensors["permutations"], dim=1).values == expected_permutation
        ).item()
    ):
        raise ValueError(f"Fruit QSRT permutations are not bijections in {tensor_path}")
    return metadata


def _align_storage(value: int) -> int:
    return (
        (value + FRUIT_QSRT_STORAGE_ALIGNMENT - 1)
        // FRUIT_QSRT_STORAGE_ALIGNMENT
        * FRUIT_QSRT_STORAGE_ALIGNMENT
    )


def _fruit_atom_metadata(
    *, layer: int, source_sha256: str, encoder_fingerprint: str
) -> dict[str, str]:
    atom_payload_bytes = EXPERTS * FRUIT_QSRT_ATOM_BUNDLE_BYTES
    shared_scale_bytes = 3 * EXPERTS * HIDDEN_SIZE * torch.float16.itemsize
    return {
        "schema": FRUIT_QSRT_ATOM_SCHEMA,
        "version": "1",
        "encoding": "qsrt_sqg_e4m3",
        "profile_id": str(FRUIT_QSRT_PROFILE_ID),
        "codebook": FRUIT_QSRT_CODEBOOK,
        "layer": str(layer),
        "experts": str(EXPERTS),
        "compressed_experts": str(EXPERTS),
        "x4t_experts": "0",
        "intermediate_channels": str(INTERMEDIATE_SIZE),
        "latent_channels": str(HIDDEN_SIZE),
        "record_channels": "128",
        "pair_count": str(PAIR_COUNT),
        "atom_channels": str(FRUIT_QSRT_ATOM_CHANNELS),
        "atom_slots": str(FRUIT_QSRT_ATOMS_PER_EXPERT),
        "atom_bundle_bytes": str(FRUIT_QSRT_ATOM_BUNDLE_BYTES),
        "atom_slot_payload_bytes": str(atom_payload_bytes),
        "atom_slot_stride_bytes": str(_align_storage(atom_payload_bytes)),
        "format_section_bytes": str(FRUIT_QSRT_FORMAT_SECTION_BYTES),
        "shared_scale_rows": str(EXPERTS),
        "shared_scale_section_bytes": str(_align_storage(shared_scale_bytes)),
        "alignment_bytes": str(FRUIT_QSRT_STORAGE_ALIGNMENT),
        "rotation_multiplier": "5",
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
    }


def _validate_fruit_atom_contract(
    tensor_path: Path,
    *,
    layer: int,
    source_sha256: str,
    encoder_fingerprint: str,
) -> dict[str, str]:
    expected_metadata = {
        "format": "pt",
        **_fruit_atom_metadata(
            layer=layer,
            source_sha256=source_sha256,
            encoder_fingerprint=encoder_fingerprint,
        ),
    }
    shared_scale_bytes = 3 * EXPERTS * HIDDEN_SIZE * torch.float16.itemsize
    atom_payload_bytes = EXPERTS * FRUIT_QSRT_ATOM_BUNDLE_BYTES
    atom_stride_bytes = _align_storage(atom_payload_bytes)
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if set(handle.keys()) != set(FRUIT_QSRT_ATOM_TENSORS):
            raise ValueError(
                f"Fruit QSRT atom tensor inventory mismatch: {tensor_path}"
            )
        for name, expected_value in expected_metadata.items():
            if metadata.get(name) != expected_value:
                raise ValueError(
                    f"Fruit QSRT atom metadata {name} mismatch in {tensor_path}"
                )
        format_section = handle.get_tensor(FRUIT_QSRT_FORMAT_TENSOR)
        shared_section = handle.get_tensor(FRUIT_QSRT_SHARED_SCALE_TENSOR)
        atom_shape = handle.get_slice(FRUIT_QSRT_ATOM_TENSOR).get_shape()
    if (
        format_section.dtype != torch.uint8
        or tuple(format_section.shape) != (FRUIT_QSRT_FORMAT_SECTION_BYTES,)
        or bool(torch.any(format_section[EXPERTS:] != 0))
    ):
        raise ValueError(f"Fruit QSRT format section is malformed: {tensor_path}")
    codes = format_section[:EXPERTS]
    r13 = codes >> 4
    r2 = codes & 0xF
    if bool(torch.any((r13 > 2) | (r2 > 2))):
        raise ValueError(f"Fruit QSRT format codes are invalid: {tensor_path}")
    expected_shared_section = _align_storage(shared_scale_bytes)
    if (
        shared_section.dtype != torch.uint8
        or tuple(shared_section.shape) != (expected_shared_section,)
        or bool(torch.any(shared_section[shared_scale_bytes:] != 0))
    ):
        raise ValueError(f"Fruit QSRT shared-scale section is malformed: {tensor_path}")
    shared = (
        shared_section[:shared_scale_bytes]
        .view(torch.float16)
        .reshape(3, EXPERTS, HIDDEN_SIZE)
    )
    if not bool(torch.all(torch.isfinite(shared))):
        raise ValueError(f"Fruit QSRT shared scales are non-finite: {tensor_path}")
    expected_atom_shape = [
        FRUIT_QSRT_ATOMS_PER_EXPERT,
        atom_stride_bytes,
    ]
    if atom_shape != expected_atom_shape:
        raise ValueError(
            f"Fruit QSRT atom slab shape {atom_shape} != {expected_atom_shape}"
        )
    return metadata


def _validate_part(
    tensor_path: Path,
    manifest_path: Path,
    *,
    layer: int,
    expert: int,
    source_sha256: str,
    encoder_fingerprint: str,
    repair: bool = True,
) -> dict[str, object] | None:
    for path in (tensor_path, manifest_path):
        if path.is_symlink():
            raise ValueError(f"Fruit QSRT part must not be symbolic: {path}")
        if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
            raise ValueError(f"Fruit QSRT part must be a private regular file: {path}")
    if not tensor_path.exists() and not manifest_path.exists():
        return None
    if not tensor_path.is_file() or not manifest_path.is_file():
        if repair:
            for path in (tensor_path, manifest_path):
                if path.is_file():
                    path.unlink()
                elif path.exists():
                    raise ValueError(f"unexpected Fruit QSRT part path: {path}")
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"malformed Fruit QSRT part manifest: {manifest_path}"
        ) from exc
    if not isinstance(value, dict):
        raise TypeError(f"malformed Fruit QSRT part manifest: {manifest_path}")
    manifest_format = value.get("format")
    if (
        not isinstance(manifest_format, dict)
        or set(manifest_format) != {"r13", "r2"}
        or any(
            isinstance(manifest_format[name], bool)
            or not isinstance(manifest_format[name], int)
            or manifest_format[name] not in (0, 1, 2)
            for name in ("r13", "r2")
        )
    ):
        raise ValueError(f"malformed Fruit QSRT part format: {manifest_path}")
    expected = {
        "schema": FRUIT_QSRT_SCHEMA,
        "version": 1,
        "profile_id": FRUIT_QSRT_PROFILE_ID,
        "codebook": FRUIT_QSRT_CODEBOOK,
        "layer": layer,
        "expert": expert,
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
    }
    mismatches = {
        name
        for name, expected_value in expected.items()
        if value.get(name) != expected_value
    }
    if mismatches:
        if mismatches == {"encoder_fingerprint"} and repair:
            tensor_path.unlink()
            manifest_path.unlink()
            return None
        if not repair:
            return None
        raise ValueError(f"Fruit QSRT part identity mismatch: {manifest_path}")
    if value.get("safetensors_bytes") != tensor_path.stat().st_size:
        raise ValueError(f"Fruit QSRT part size mismatch: {tensor_path}")
    if value.get("safetensors_sha256") != _sha256(tensor_path):
        raise ValueError(f"Fruit QSRT part hash mismatch: {tensor_path}")
    metadata = _validate_qsrt_tensor_contract(
        tensor_path,
        expected_expert_ids=torch.tensor([expert], dtype=torch.int32),
        expected_formats=torch.tensor(
            [[manifest_format["r13"], manifest_format["r2"]]],
            dtype=torch.int8,
        ),
    )
    for name, expected_value in (
        ("schema", FRUIT_QSRT_SCHEMA),
        ("version", "1"),
        ("profile_id", str(FRUIT_QSRT_PROFILE_ID)),
        ("codebook", FRUIT_QSRT_CODEBOOK),
        ("layer", str(layer)),
        ("expert", str(expert)),
        ("source_sha256", source_sha256),
    ):
        if metadata.get(name) != expected_value:
            raise ValueError(f"Fruit QSRT part metadata mismatch: {tensor_path}")
    if metadata.get("encoder_fingerprint") != encoder_fingerprint:
        if not repair:
            return None
        raise ValueError(f"Fruit QSRT part producer mismatch: {tensor_path}")
    return value


def _write_part(
    output: Path,
    encoding,
    *,
    source_sha256: str,
    encoder_fingerprint: str,
    elapsed_seconds: float,
    peak_cuda_bytes: int,
) -> dict[str, object]:
    tensor_path, manifest_path = _part_paths(output, encoding.layer, encoding.expert)
    _prepare_part_layer(output, encoding.layer)
    _atomic_safetensors(
        tensor_path,
        encoding.artifact_tensors(),
        {
            "schema": FRUIT_QSRT_SCHEMA,
            "version": "1",
            "profile_id": str(FRUIT_QSRT_PROFILE_ID),
            "codebook": FRUIT_QSRT_CODEBOOK,
            "layer": str(encoding.layer),
            "expert": str(encoding.expert),
            "source_sha256": source_sha256,
            "encoder_fingerprint": encoder_fingerprint,
        },
    )
    manifest = {
        **encoding.manifest(),
        "version": 1,
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
        "safetensors_file": tensor_path.name,
        "safetensors_bytes": tensor_path.stat().st_size,
        "safetensors_sha256": _sha256(tensor_path),
        "encode_elapsed_seconds": elapsed_seconds,
        "peak_cuda_bytes": peak_cuda_bytes,
    }
    _atomic_text(manifest_path, _canonical_json(manifest))
    return manifest


def _seed_parts(
    output: Path,
    seed: Path | None,
    source_sha256: str,
    encoder_fingerprint: str,
) -> int:
    if seed is None:
        return 0
    if seed.is_symlink() or not seed.is_dir():
        raise ValueError(f"Fruit QSRT seed cache must be a real directory: {seed}")
    _validate_part_cache_root(
        seed,
        create=False,
        allow_run_manifests=True,
    )
    _validate_part_cache_root(output / ".qsrt-parts", create=True)
    copied = 0
    for layer in LAYERS:
        for expert in range(EXPERTS):
            source_tensor = (
                seed / f"layer-{layer:03d}" / f"expert-{expert:03d}.safetensors"
            )
            source_manifest = seed / f"layer-{layer:03d}" / f"expert-{expert:03d}.json"
            if not source_tensor.is_file() and not source_manifest.is_file():
                continue
            target_tensor, target_manifest = _part_paths(output, layer, expert)
            if target_tensor.exists() or target_manifest.exists():
                continue
            if not source_tensor.is_file() or not source_manifest.is_file():
                continue
            try:
                valid = _validate_part(
                    source_tensor,
                    source_manifest,
                    layer=layer,
                    expert=expert,
                    source_sha256=source_sha256,
                    encoder_fingerprint=encoder_fingerprint,
                    repair=False,
                )
            except (OSError, TypeError, ValueError):
                valid = None
            if valid is None:
                continue
            _prepare_part_layer(output, layer)
            try:
                tensor_sha256 = valid.get("safetensors_sha256")
                if not isinstance(tensor_sha256, str):
                    raise TypeError("Fruit QSRT part digest must be a string")
                _copy_authenticated(source_tensor, target_tensor, tensor_sha256)
                _copy_authenticated(
                    source_manifest,
                    target_manifest,
                    _sha256(source_manifest),
                )
                if (
                    _validate_part(
                        target_tensor,
                        target_manifest,
                        layer=layer,
                        expert=expert,
                        source_sha256=source_sha256,
                        encoder_fingerprint=encoder_fingerprint,
                        repair=False,
                    )
                    is None
                ):
                    raise ValueError("copied Fruit QSRT part failed validation")
            except (OSError, TypeError, ValueError):
                for target in (target_tensor, target_manifest):
                    if target.is_file() and not target.is_symlink():
                        target.unlink()
                continue
            copied += 1
    return copied


def _parts_need_encoder(
    output: Path,
    *,
    source_sha256: str,
    encoder_fingerprint: str,
) -> bool:
    _validate_part_cache_root(output / ".qsrt-parts", create=True)
    expected = {
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
    }
    for layer in LAYERS:
        for expert in range(EXPERTS):
            tensor_path, manifest_path = _part_paths(output, layer, expert)
            if not tensor_path.is_file() or not manifest_path.is_file():
                return True
            try:
                value = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return True
            if not isinstance(value, dict) or any(
                value.get(name) != expected_value
                for name, expected_value in expected.items()
            ):
                return True
    return False


def _validate_source_evidence(
    value: object,
    *,
    spec: FruitModelSpec = FRUIT_INSTRUCT_SPEC,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("sealed Fruit source evidence must be a JSON object")
    expected: dict[str, object] = {
        "kind": "kquant-fruit-source-preflight",
        "version": 1,
        "status": "pass",
        "model_id": spec.model_id,
    }
    if value.get("source_container") == "hf_bf16_safetensors":
        if spec.safetensors_manifest_sha256 is None:
            raise ValueError("Fruit model spec has no Safetensors source identity")
        expected.update(
            {
                "checkpoint_sha256": spec.safetensors_manifest_sha256,
                "checkpoint_sha256_provenance": ("safetensors_manifest_authenticated"),
                "source_sha256": spec.safetensors_manifest_sha256,
                "source_kind": "safetensors_manifest",
                "expected_checkpoint_sha256": spec.checkpoint_sha256,
                "safetensors_manifest_sha256": spec.safetensors_manifest_sha256,
                "source_repository": spec.safetensors_repository,
                "source_revision": spec.safetensors_revision,
            }
        )
    elif value.get("source_container") in ("model", "state_dict"):
        expected.update(
            {
                "checkpoint_sha256": spec.checkpoint_sha256,
                "checkpoint_sha256_provenance": "checkpoint_file_authenticated",
                "source_sha256": spec.checkpoint_sha256,
                "source_kind": "torch_checkpoint",
            }
        )
    else:
        raise ValueError("sealed Fruit source evidence has an unsupported container")
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(
                f"sealed Fruit source evidence {name!r} mismatch: "
                f"{value.get(name)!r} != {expected_value!r}"
            )
    normalized = dict(value)
    normalized.pop("path", None)
    return normalized


def _write_source_evidence_seal(
    root: Path,
    source_evidence: dict[str, object],
    producer: dict[str, object],
    *,
    spec: FruitModelSpec = FRUIT_INSTRUCT_SPEC,
) -> None:
    source_evidence = _validate_source_evidence(source_evidence, spec=spec)
    encoder = producer.get("encoder")
    if not isinstance(encoder, dict) or not isinstance(encoder.get("fingerprint"), str):
        raise TypeError("Fruit QSRT producer has no encoder fingerprint")
    if not isinstance(producer.get("fingerprint"), str):
        raise TypeError("Fruit QSRT producer has no package fingerprint")
    envelope = {
        "schema": "kquant_fruit_source_seal_v1",
        "producer_fingerprint": producer["fingerprint"],
        "encoder_fingerprint": encoder["fingerprint"],
        "source": source_evidence,
    }
    content = _canonical_json(envelope)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    _atomic_text(root / _SOURCE_EVIDENCE_NAME, content)
    _atomic_text(
        root / _SOURCE_EVIDENCE_SHA_NAME,
        f"{digest}  {_SOURCE_EVIDENCE_NAME}\n",
    )


def _read_source_evidence_seal(
    root: Path | None,
    producer: dict[str, object],
    *,
    spec: FruitModelSpec = FRUIT_INSTRUCT_SPEC,
) -> dict[str, object] | None:
    if root is None:
        return None
    evidence_path = root / _SOURCE_EVIDENCE_NAME
    seal_path = root / _SOURCE_EVIDENCE_SHA_NAME
    if not evidence_path.exists() and not seal_path.exists():
        return None
    if not evidence_path.is_file() or not seal_path.is_file():
        raise ValueError(f"incomplete Fruit source evidence seal in {root}")
    fields = seal_path.read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[1] != _SOURCE_EVIDENCE_NAME:
        raise ValueError(f"malformed Fruit source evidence seal {seal_path}")
    if fields[0] != _sha256(evidence_path):
        raise ValueError(f"Fruit source evidence seal mismatch for {evidence_path}")
    envelope = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict):
        raise TypeError("Fruit source evidence envelope must be a JSON object")
    encoder = producer["encoder"]
    if (
        envelope.get("schema") != "kquant_fruit_source_seal_v1"
        or envelope.get("encoder_fingerprint") != encoder["fingerprint"]
        or not isinstance(envelope.get("producer_fingerprint"), str)
    ):
        raise ValueError(f"Fruit source evidence producer mismatch in {evidence_path}")
    return _validate_source_evidence(envelope.get("source"), spec=spec)


def _write_calibration_evidence(
    output: Path,
    calibration: FruitCalibrationStore,
    producer: dict[str, object],
) -> None:
    encoder = producer.get("encoder")
    if not isinstance(encoder, dict):
        raise TypeError("Fruit QSRT producer encoder evidence is invalid")
    expected = {
        "calibration_fingerprint": calibration.fingerprint,
        "calibration_capture_id": calibration.capture_id,
        "calibration_manifest_sha256": calibration.manifest_sha256,
    }
    if any(encoder.get(name) != value for name, value in expected.items()):
        raise ValueError("Fruit calibration evidence disagrees with the producer")
    _atomic_text(
        output / _CALIBRATION_EVIDENCE_NAME,
        _canonical_json(
            {
                "schema": "kquant_fruit_qsrt_calibration_reference_v1",
                "producer_fingerprint": producer["fingerprint"],
                "encoder_fingerprint": encoder["fingerprint"],
                **expected,
                "raw_activation_layers_included": False,
                "capture_manifest": calibration.manifest,
            }
        ),
    )


def _encode_parts(
    output: Path,
    store: FruitMatrixStore | None,
    quantizer_module,
    calibration_store: FruitCalibrationStore,
    *,
    device: torch.device,
    source_sha256: str,
    encoder_fingerprint: str,
) -> None:
    total = len(LAYERS) * EXPERTS
    ordinal = 0
    started = time.perf_counter()
    for layer in LAYERS:
        calibration_layer = calibration_store.load_layer(layer)
        for expert in range(EXPERTS):
            ordinal += 1
            tensor_path, manifest_path = _part_paths(output, layer, expert)
            previous = _validate_part(
                tensor_path,
                manifest_path,
                layer=layer,
                expert=expert,
                source_sha256=source_sha256,
                encoder_fingerprint=encoder_fingerprint,
            )
            if previous is not None:
                print(
                    f"[{ordinal}/{total}] layer {layer} expert {expert}: resume",
                    flush=True,
                )
                continue
            if quantizer_module is None or store is None:
                raise RuntimeError(
                    "authenticated Fruit source was not loaded for a missing part"
                )
            torch.cuda.reset_peak_memory_stats(device)
            item_started = time.perf_counter()
            encoding = encode_fruit_expert(
                store,
                layer=layer,
                expert=expert,
                device=device,
                calibration=calibration_layer.expert_rows(expert),
                quantizer_module=quantizer_module,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - item_started
            peak = torch.cuda.max_memory_allocated(device)
            _write_part(
                output,
                encoding,
                source_sha256=source_sha256,
                encoder_fingerprint=encoder_fingerprint,
                elapsed_seconds=elapsed,
                peak_cuda_bytes=peak,
            )
            print(
                f"[{ordinal}/{total}] layer {layer} expert {expert}: "
                f"R13={encoding.r13} R2={encoding.r2}, {elapsed:.3f}s, "
                f"peak={peak / (1 << 20):.1f} MiB, "
                f"run={time.perf_counter() - started:.1f}s",
                flush=True,
            )
            del encoding
        del calibration_layer


def _layer_paths(output: Path, layer: int) -> tuple[Path, Path]:
    return (
        output / f"qsrt-layer-{layer:03d}.safetensors",
        output / f"qsrt-layer-{layer:03d}.json",
    )


def _validate_layer(
    tensor_path: Path,
    manifest_path: Path,
    *,
    layer: int,
    source_sha256: str,
    encoder_fingerprint: str,
) -> dict[str, object] | None:
    if not tensor_path.exists() and not manifest_path.exists():
        return None
    if not tensor_path.is_file() or not manifest_path.is_file():
        for path in (tensor_path, manifest_path):
            if path.is_file():
                path.unlink()
            elif path.exists():
                raise ValueError(f"unexpected assembled Fruit QSRT path: {path}")
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"malformed assembled Fruit QSRT manifest: {manifest_path}"
        ) from exc
    expected = {
        "schema": FRUIT_QSRT_ATOM_SCHEMA,
        "version": 1,
        "profile_id": FRUIT_QSRT_PROFILE_ID,
        "codebook": FRUIT_QSRT_CODEBOOK,
        "layer": layer,
        "expert_ids": list(range(EXPERTS)),
        "expert_count": EXPERTS,
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
    }
    mismatches = {
        name
        for name, expected_value in expected.items()
        if value.get(name) != expected_value
    }
    if mismatches:
        if mismatches == {"encoder_fingerprint"} or value.get("schema") == (
            FRUIT_QSRT_SCHEMA
        ):
            tensor_path.unlink()
            manifest_path.unlink()
            return None
        raise ValueError(
            f"assembled Fruit QSRT layer identity mismatch: {manifest_path}"
        )
    if value.get("safetensors_bytes") != tensor_path.stat().st_size:
        raise ValueError(f"assembled Fruit QSRT layer size mismatch: {tensor_path}")
    if value.get("safetensors_sha256") != _sha256(tensor_path):
        raise ValueError(f"assembled Fruit QSRT layer hash mismatch: {tensor_path}")
    _validate_fruit_atom_contract(
        tensor_path,
        layer=layer,
        source_sha256=source_sha256,
        encoder_fingerprint=encoder_fingerprint,
    )
    experts = value.get("experts")
    if not isinstance(experts, list) or len(experts) != EXPERTS:
        raise ValueError(
            f"assembled Fruit QSRT expert evidence mismatch: {manifest_path}"
        )
    for expert, evidence in enumerate(experts):
        if not isinstance(evidence, dict) or any(
            evidence.get(name) != expected_value
            for name, expected_value in (
                ("layer", layer),
                ("expert", expert),
                ("source_sha256", source_sha256),
                ("encoder_fingerprint", encoder_fingerprint),
            )
        ):
            raise ValueError(
                f"assembled Fruit QSRT expert {expert} identity mismatch: "
                f"{manifest_path}"
            )
    return value


def _assemble_layers(
    output: Path,
    *,
    source_sha256: str,
    encoder_fingerprint: str,
) -> dict[str, dict[str, object]]:
    _validate_part_cache_root(output / ".qsrt-parts", create=False)
    results: dict[str, dict[str, object]] = {}
    for layer in LAYERS:
        tensor_path, manifest_path = _layer_paths(output, layer)
        previous = _validate_layer(
            tensor_path,
            manifest_path,
            layer=layer,
            source_sha256=source_sha256,
            encoder_fingerprint=encoder_fingerprint,
        )
        if previous is not None:
            results[str(layer)] = previous
            print(f"layer {layer}: assembled artifact resume", flush=True)
            continue
        parts: dict[str, list[torch.Tensor]] = defaultdict(list)
        experts: list[dict[str, object]] = []
        for expert in range(EXPERTS):
            part_tensor, part_manifest = _part_paths(output, layer, expert)
            evidence = _validate_part(
                part_tensor,
                part_manifest,
                layer=layer,
                expert=expert,
                source_sha256=source_sha256,
                encoder_fingerprint=encoder_fingerprint,
            )
            if evidence is None:
                raise ValueError(f"Fruit QSRT layer {layer} is missing expert {expert}")
            experts.append(evidence)
            with safe_open(part_tensor, framework="pt", device="cpu") as handle:
                for name in FRUIT_QSRT_ARTIFACT_TENSORS:
                    parts[name].append(handle.get_tensor(name))
        pair_tensors = {
            name: torch.cat(values, dim=1 if name == "w13_trellis" else 0).contiguous()
            for name, values in parts.items()
        }
        tensors = pack_fruit_atom_layer(pair_tensors, layer=layer)
        _atomic_safetensors(
            tensor_path,
            tensors,
            {
                "format": "pt",
                **_fruit_atom_metadata(
                    layer=layer,
                    source_sha256=source_sha256,
                    encoder_fingerprint=encoder_fingerprint,
                ),
            },
        )
        format_counts = Counter(
            (int(expert["format"]["r13"]), int(expert["format"]["r2"]))
            for expert in experts
        )
        manifest: dict[str, object] = {
            "schema": FRUIT_QSRT_ATOM_SCHEMA,
            "version": 1,
            "profile_id": FRUIT_QSRT_PROFILE_ID,
            "codebook": FRUIT_QSRT_CODEBOOK,
            "layer": layer,
            "expert_ids": list(range(EXPERTS)),
            "expert_count": EXPERTS,
            "source_sha256": source_sha256,
            "encoder_fingerprint": encoder_fingerprint,
            "tensor_shapes": {
                name: list(tensor.shape) for name, tensor in tensors.items()
            },
            "format_counts": {
                f"R13={r13},R2={r2}": count
                for (r13, r2), count in sorted(format_counts.items())
            },
            "safetensors_file": tensor_path.name,
            "safetensors_bytes": tensor_path.stat().st_size,
            "safetensors_sha256": _sha256(tensor_path),
            "experts": experts,
        }
        _atomic_text(manifest_path, _canonical_json(manifest))
        results[str(layer)] = manifest
        print(
            f"layer {layer}: assembled {tensor_path.stat().st_size / (1 << 20):.2f} MiB",
            flush=True,
        )
    return results


def _authenticate_base_model(
    base_model: Path,
    *,
    spec: FruitModelSpec = FRUIT_INSTRUCT_SPEC,
) -> dict[str, object]:
    manifest_path = base_model / "MANIFEST.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != spec.safetensors_manifest_sha256:
        raise ValueError(
            "Fruit BF16 base manifest identity mismatch: "
            f"{manifest_sha256} != {spec.safetensors_manifest_sha256}"
        )
    entries: dict[str, str] = {}
    try:
        manifest_text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Fruit BF16 base manifest is not UTF-8") from exc
    for line in manifest_text.splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"malformed Fruit BF16 base manifest line: {line!r}")
        digest, filename = fields
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or Path(filename).name != filename
            or filename in entries
        ):
            raise ValueError(f"invalid Fruit BF16 base manifest line: {line!r}")
        entries[filename] = digest
    required_static = {
        "LICENSE",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if not required_static.issubset(entries):
        raise ValueError("Fruit BF16 base manifest omits required static files")
    for filename, expected_sha256 in entries.items():
        path = base_model / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Fruit BF16 base hash mismatch for {filename}: "
                f"{actual_sha256} != {expected_sha256}"
            )
    index = json.loads(
        _authenticated_bytes(
            base_model / "model.safetensors.index.json",
            entries["model.safetensors.index.json"],
        ).decode("utf-8")
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("Fruit BF16 base index has no weight_map")
    shard_names = set(weight_map.values())
    if not shard_names.issubset(entries):
        raise ValueError("Fruit BF16 base manifest omits indexed weight shards")
    config = json.loads(
        _authenticated_bytes(
            base_model / "config.json",
            entries["config.json"],
        ).decode("utf-8")
    )
    expected_config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "dtype": "bfloat16",
        "hidden_size": spec.hidden_size,
        "moe_intermediate_size": spec.intermediate_size,
        "n_routed_experts": spec.num_experts,
        "num_experts_per_tok": 8,
        "num_hidden_layers": spec.mtp_layer,
        "vocab_size": 154_880,
    }
    if any(config.get(name) != value for name, value in expected_config.items()):
        raise ValueError("Fruit BF16 base model geometry mismatch")
    return {
        "schema": "kquant_fruit_bf16_base_v1",
        "manifest_file": "MANIFEST.sha256",
        "manifest_sha256": manifest_sha256,
        "file_count": len(entries),
        "files": entries,
        "config_identity": expected_config,
    }


def _authenticated_bytes(path: Path, expected_sha256: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"authenticated Fruit source file is not regular: {path}")
    content = path.read_bytes()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Fruit BF16 base hash mismatch for {path.name}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return content


def _assert_authenticated_file(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"authenticated Fruit source file is not regular: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Fruit BF16 base hash mismatch for {path.name}: "
            f"{actual_sha256} != {expected_sha256}"
        )


def _copy_authenticated(
    source: Path,
    target: Path,
    expected_sha256: str,
) -> None:
    _assert_authenticated_file(source, expected_sha256)
    if target.is_symlink():
        raise ValueError(f"Fruit package target must not be symbolic: {target}")
    if target.exists():
        if not target.is_file():
            raise ValueError(f"Fruit package target must be regular: {target}")
        if not target.samefile(source) and target.stat().st_nlink == 1:
            if _sha256(target) != expected_sha256:
                raise ValueError(
                    f"existing static model file differs from source: {target}"
                )
            _assert_authenticated_file(source, expected_sha256)
            return
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=8 << 20)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        if _sha256(temporary) != expected_sha256:
            raise ValueError(f"copied Fruit model file changed: {target}")
        _assert_authenticated_file(source, expected_sha256)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _tensor_nbytes(handle, name: str) -> int:
    tensor_slice = handle.get_slice(name)
    dtype = tensor_slice.get_dtype()
    try:
        itemsize = _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}") from exc
    elements = 1
    for extent in tensor_slice.get_shape():
        elements *= int(extent)
    return elements * itemsize


def _strip_routed_experts(
    source: Path,
    target: Path,
    expected_sha256: str,
) -> tuple[set[str], int]:
    _assert_authenticated_file(source, expected_sha256)
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        names = handle.keys()
        expert_names = {name for name in names if ".mlp.experts." in name}
        if len(expert_names) != 3 * EXPERTS:
            raise ValueError(
                f"expected {3 * EXPERTS} routed-expert tensors in {source}, "
                f"got {len(expert_names)}"
            )
        removed_bytes = sum(_tensor_nbytes(handle, name) for name in expert_names)
        kept = {
            name: handle.get_tensor(name) for name in names if name not in expert_names
        }
    _assert_authenticated_file(source, expected_sha256)
    _atomic_safetensors(target, kept, metadata)
    _assert_authenticated_file(source, expected_sha256)
    return expert_names, removed_bytes


def _materialize_base_model(
    base_model: Path,
    output: Path,
    base_provenance: dict[str, object],
) -> None:
    files = base_provenance.get("files")
    if not isinstance(files, dict) or any(
        not isinstance(name, str) or not isinstance(digest, str)
        for name, digest in files.items()
    ):
        raise TypeError("Fruit BF16 base provenance has no authenticated file map")
    index_path = base_model / "model.safetensors.index.json"
    index = json.loads(
        _authenticated_bytes(
            index_path,
            files["model.safetensors.index.json"],
        ).decode("utf-8")
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("base model safetensors index has no weight_map")
    removed_names: set[str] = set()
    removed_bytes = 0
    source_files = sorted(set(weight_map.values()))
    if any(not isinstance(filename, str) for filename in source_files):
        raise TypeError("base model safetensors index has invalid shard filenames")
    if not set(source_files).issubset(files):
        raise ValueError("Fruit BF16 base provenance omits indexed weight shards")
    for filename in source_files:
        source = base_model / filename
        target = output / filename
        if filename.startswith("model-layer-"):
            layer_text = filename.removeprefix("model-layer-").removesuffix(
                ".safetensors"
            )
            try:
                layer = int(layer_text)
            except ValueError as exc:
                raise ValueError(
                    f"invalid Fruit BF16 model layer shard filename: {filename}"
                ) from exc
        else:
            layer = -1
        if layer in LAYERS:
            names, byte_count = _strip_routed_experts(
                source,
                target,
                files[filename],
            )
            removed_names.update(names)
            removed_bytes += byte_count
        else:
            _copy_authenticated(source, target, files[filename])
    expected_removed = 3 * EXPERTS * len(LAYERS)
    if len(removed_names) != expected_removed:
        raise ValueError(
            f"expected {expected_removed} routed-expert weights, got {len(removed_names)}"
        )
    filtered_map = {
        name: filename
        for name, filename in weight_map.items()
        if name not in removed_names
    }
    if len(weight_map) - len(filtered_map) != expected_removed:
        raise ValueError("base index routed-expert inventory disagrees with shards")
    metadata = dict(index.get("metadata") or {})
    total_size = int(metadata.get("total_size", 0))
    if total_size <= removed_bytes:
        raise ValueError("base index total_size is invalid")
    metadata["total_size"] = total_size - removed_bytes
    _atomic_text(
        output / "model.safetensors.index.json",
        _canonical_json({"metadata": metadata, "weight_map": filtered_map}),
    )
    for filename in (
        "LICENSE",
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        _copy_authenticated(
            base_model / filename,
            output / filename,
            files[filename],
        )


def _write_config(
    base_model: Path,
    output: Path,
    producer: dict[str, object],
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
) -> None:
    files = base_provenance.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("config.json"), str):
        raise TypeError("Fruit BF16 base provenance has no config digest")
    config = json.loads(
        _authenticated_bytes(
            base_model / "config.json",
            files["config.json"],
        ).decode("utf-8")
    )
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "hybrid_bit_map": {str(layer): [3] * EXPERTS for layer in LAYERS},
        "kept_format": "mxfp4_e8m0k32",
        "demoted_format": "qsrt_sqg_e4m3",
        "qsrt": {
            "schema": FRUIT_QSRT_ATOM_SCHEMA,
            "storage_format": FRUIT_QSRT_ATOM_STORAGE,
            "encoding": "qsrt_sqg_e4m3",
            "codebook": FRUIT_QSRT_CODEBOOK,
            "profile_id": FRUIT_QSRT_PROFILE_ID,
            "artifact_manifest": "qsrt-manifest.json",
            "producer_fingerprint": producer["fingerprint"],
            "encoder_fingerprint": producer["encoder"]["fingerprint"],
            "source_kind": source_evidence["source_kind"],
            "source_sha256": source_evidence["source_sha256"],
            "runtime": "w4a8",
        },
    }
    _atomic_text(output / "config.json", _canonical_json(config))


def _prepare_output_root(output: Path) -> None:
    if output.is_symlink():
        raise ValueError(f"Fruit package root must not be a symbolic link: {output}")
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Fruit package root must be a directory: {output}")
    else:
        output.mkdir(parents=True)
    _recover_staged_part_cache(output)


def _remove_part_cache(output: Path) -> None:
    parts = output / ".qsrt-parts"
    _validate_part_cache_root(parts, create=False)
    if parts.exists():
        shutil.rmtree(parts)


def _staged_part_cache_path(output: Path) -> Path:
    return output.with_name(f".{output.name}.qsrt-parts-finalizing")


def _recover_staged_part_cache(output: Path) -> None:
    staged = _staged_part_cache_path(output)
    if staged.is_symlink():
        raise ValueError(f"staged Fruit QSRT cache must not be symbolic: {staged}")
    if not staged.exists():
        return
    _validate_part_cache_root(staged, create=False)
    parts = output / ".qsrt-parts"
    if parts.exists() or parts.is_symlink():
        raise ValueError("both active and staged Fruit QSRT caches exist")
    os.replace(staged, parts)


def _stage_part_cache(output: Path) -> Path | None:
    parts = output / ".qsrt-parts"
    _validate_part_cache_root(parts, create=False)
    if not parts.exists():
        return None
    staged = _staged_part_cache_path(output)
    if staged.exists() or staged.is_symlink():
        raise ValueError(f"staged Fruit QSRT cache already exists: {staged}")
    os.replace(parts, staged)
    return staged


def _restore_staged_part_cache(output: Path, staged: Path | None) -> None:
    if staged is None:
        return
    _validate_part_cache_root(staged, create=False)
    parts = output / ".qsrt-parts"
    if parts.exists() or parts.is_symlink():
        raise ValueError("cannot restore Fruit QSRT cache over an existing path")
    os.replace(staged, parts)


def _discard_staged_part_cache(staged: Path | None) -> None:
    if staged is None:
        return
    _validate_part_cache_root(staged, create=False)
    shutil.rmtree(staged)


def _package_files(
    output: Path,
    *,
    allow_part_cache: bool = False,
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in output.iterdir():
        if path.is_symlink():
            raise ValueError(f"Fruit package path must not be a symbolic link: {path}")
        if path.is_dir():
            if allow_part_cache and path.name == ".qsrt-parts":
                continue
            if path.name != "evaluation":
                raise ValueError(f"unexpected Fruit package directory: {path}")
            continue
        if not path.is_file():
            raise ValueError(f"unexpected Fruit package path: {path}")
        if path.stat().st_nlink != 1:
            raise ValueError(f"Fruit package file must not be hard-linked: {path}")
        if path.name not in {
            "MANIFEST.sha256",
            _COMPLETE_MARKER_NAME,
            _CANDIDATE_MARKER_NAME,
        }:
            files[path.name] = path
    evaluation = output / "evaluation"
    if evaluation.is_dir():
        for path in evaluation.rglob("*"):
            if path.is_symlink():
                raise ValueError(
                    f"Fruit evaluation path must not be a symbolic link: {path}"
                )
            if path.is_dir():
                raise ValueError(f"unexpected Fruit evaluation directory: {path}")
            if not path.is_file():
                raise ValueError(f"unexpected Fruit evaluation path: {path}")
            if path.stat().st_nlink != 1:
                raise ValueError(
                    f"Fruit evaluation file must not be hard-linked: {path}"
                )
            relative = path.relative_to(output).as_posix()
            files[relative] = path
    return dict(sorted(files.items()))


def _expected_package_inventory(
    base_provenance: dict[str, object],
    *,
    include_runtime_qualification: bool = True,
) -> set[str]:
    base_files = base_provenance.get("files")
    if not isinstance(base_files, dict) or any(
        not isinstance(name, str) or not isinstance(digest, str)
        for name, digest in base_files.items()
    ):
        raise TypeError("Fruit BF16 base provenance has no authenticated file map")
    expected = set(base_files)
    expected.update(
        {
            "README.md",
            "qsrt-manifest.json",
            _SOURCE_EVIDENCE_NAME,
            _SOURCE_EVIDENCE_SHA_NAME,
            _CALIBRATION_EVIDENCE_NAME,
            _RATE_SWEEP_NAME,
            *((_RUNTIME_QUALIFICATION_NAME,) if include_runtime_qualification else ()),
        }
    )
    for layer in LAYERS:
        expected.add(f"qsrt-layer-{layer:03d}.json")
        expected.add(f"qsrt-layer-{layer:03d}.safetensors")
    return expected


def _validated_package_files(
    output: Path,
    base_provenance: dict[str, object],
    allow_part_cache: bool = False,
    *,
    include_runtime_qualification: bool = True,
) -> dict[str, Path]:
    files = _package_files(output, allow_part_cache=allow_part_cache)
    expected = _expected_package_inventory(
        base_provenance,
        include_runtime_qualification=include_runtime_qualification,
    )
    if set(files) != expected:
        raise ValueError(
            "Fruit package inventory mismatch; "
            f"missing={sorted(expected - set(files))}, "
            f"unexpected={sorted(set(files) - expected)}"
        )
    return files


def _manifest_record(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
    layers: dict[str, dict[str, object]],
    publication: FruitPublicationSpec,
    runtime_qualification: dict[str, object] | None,
) -> dict[str, object]:
    manifest_layers = {
        layer: {
            "qsrt_atoms": value["safetensors_file"],
            "bytes": value["safetensors_bytes"],
            "sha256": value["safetensors_sha256"],
            "expert_count": value["expert_count"],
            "evidence": f"qsrt-layer-{int(layer):03d}.json",
        }
        for layer, value in sorted(layers.items(), key=lambda item: int(item[0]))
    }
    evaluation = {
        "uniform_rate_sweep": {
            "file": _RATE_SWEEP_NAME,
            "sha256": _sha256(output / _RATE_SWEEP_NAME),
        }
    }
    if runtime_qualification is not None:
        evaluation["runtime_qualification"] = {
            "file": _RUNTIME_QUALIFICATION_NAME,
            "sha256": _sha256(output / _RUNTIME_QUALIFICATION_NAME),
        }
    return {
        "schema": "kquant_qsrt_model_manifest_v1",
        "version": 1,
        "publication": {
            "variant": "instruct",
            "repository": publication.repository,
        },
        "codec": "QSRT",
        "storage_schema": FRUIT_QSRT_ATOM_SCHEMA,
        "encoding": "qsrt_sqg_e4m3",
        "codebook": FRUIT_QSRT_CODEBOOK,
        "profile_id": FRUIT_QSRT_PROFILE_ID,
        "geometry": {
            "layers": list(LAYERS),
            "experts_per_layer": EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "topk": 8,
        },
        "runtime": {
            "tensor_parallel": "whole_atom_partition",
            "validated_tensor_parallel_sizes": [1],
            "decode": "trellis_w4a8",
            "decode_max_tokens": 16,
            "fallback": "trellis_w4a16",
            "prefill": "trellis_w4a16",
        },
        "base_model": base_provenance,
        "producer": producer,
        "source": source_evidence,
        "evaluation": evaluation,
        "layers": manifest_layers,
        "complete": runtime_qualification is not None,
    }


def _write_checksum_manifest(
    output: Path,
    base_provenance: dict[str, object],
    *,
    include_runtime_qualification: bool,
) -> None:
    files = _validated_package_files(
        output,
        base_provenance,
        allow_part_cache=True,
        include_runtime_qualification=include_runtime_qualification,
    )
    entries = [f"{_sha256(path)}  {relative}" for relative, path in files.items()]
    _atomic_text(output / "MANIFEST.sha256", "\n".join(entries) + "\n")


def _write_package_manifests(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
    calibration: FruitCalibrationStore,
    rate_sweep: dict[str, object],
    layers: dict[str, dict[str, object]],
    publication: FruitPublicationSpec,
    runtime_qualification: dict[str, object],
) -> None:
    evaluation = output / "evaluation"
    if evaluation.exists():
        shutil.rmtree(evaluation)
    evaluation.mkdir()
    rate_sweep_path = output / _RATE_SWEEP_NAME
    _atomic_text(rate_sweep_path, _canonical_json(rate_sweep))
    runtime_qualification_path = output / _RUNTIME_QUALIFICATION_NAME
    _atomic_text(
        runtime_qualification_path,
        _canonical_json(runtime_qualification),
    )
    manifest = _manifest_record(
        output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
        layers=layers,
        publication=publication,
        runtime_qualification=runtime_qualification,
    )
    _atomic_text(output / "qsrt-manifest.json", _canonical_json(manifest))
    _atomic_text(
        output / "README.md",
        _render_model_card(
            source_evidence=source_evidence,
            calibration=calibration,
            producer=producer,
            rate_sweep=rate_sweep,
            layers=layers,
            publication=publication,
            runtime_qualification=runtime_qualification,
        ),
    )
    _write_checksum_manifest(
        output,
        base_provenance,
        include_runtime_qualification=True,
    )


def _candidate_record(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
    publication: FruitPublicationSpec,
) -> dict[str, object]:
    return {
        "schema": "kquant_qsrt_candidate_v1",
        "publication": {
            "variant": "instruct",
            "repository": publication.repository,
        },
        "package_manifest_sha256": _sha256(output / "qsrt-manifest.json"),
        "checksum_manifest_sha256": _sha256(output / "MANIFEST.sha256"),
        "model_index_sha256": _sha256(output / "model.safetensors.index.json"),
        "source": {
            "kind": source_evidence["source_kind"],
            "sha256": source_evidence["source_sha256"],
        },
        "base_manifest_sha256": base_provenance["manifest_sha256"],
        "producer_fingerprint": producer["fingerprint"],
        "encoder_fingerprint": producer["encoder"]["fingerprint"],
    }


def _write_candidate_package(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
    rate_sweep: dict[str, object],
    layers: dict[str, dict[str, object]],
    publication: FruitPublicationSpec,
) -> None:
    (output / _COMPLETE_MARKER_NAME).unlink(missing_ok=True)
    evaluation = output / "evaluation"
    if evaluation.exists():
        shutil.rmtree(evaluation)
    evaluation.mkdir()
    _atomic_text(output / _RATE_SWEEP_NAME, _canonical_json(rate_sweep))
    manifest = _manifest_record(
        output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
        layers=layers,
        publication=publication,
        runtime_qualification=None,
    )
    _atomic_text(output / "qsrt-manifest.json", _canonical_json(manifest))
    _atomic_text(
        output / "README.md",
        "# Fruit QSRT qualification candidate\n\n"
        "This package is not a production publication. It may be launched only "
        "by the qualification workflow with an independently supplied "
        "`QSRT_CANDIDATE.json` digest.\n",
    )
    _write_checksum_manifest(
        output,
        base_provenance,
        include_runtime_qualification=False,
    )
    _atomic_text(
        output / _CANDIDATE_MARKER_NAME,
        _canonical_json(
            _candidate_record(
                output,
                source_evidence=source_evidence,
                base_provenance=base_provenance,
                producer=producer,
                publication=publication,
            )
        ),
    )


def _validate_candidate_package(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
    rate_sweep: dict[str, object],
    layers: dict[str, dict[str, object]],
    publication: FruitPublicationSpec,
    allow_part_cache: bool,
) -> None:
    marker_path = output / _CANDIDATE_MARKER_NAME
    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    if marker_bytes != _canonical_json(marker).encode("utf-8"):
        raise ValueError("Fruit QSRT candidate marker is not canonical JSON")
    if marker != _candidate_record(
        output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
        publication=publication,
    ):
        raise ValueError("Fruit QSRT candidate marker mismatch")
    manifest = json.loads((output / "qsrt-manifest.json").read_text(encoding="utf-8"))
    expected_manifest = _manifest_record(
        output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
        layers=layers,
        publication=publication,
        runtime_qualification=None,
    )
    if manifest != expected_manifest:
        raise ValueError("Fruit QSRT candidate package identity mismatch")
    if (output / _COMPLETE_MARKER_NAME).exists():
        raise ValueError("Fruit QSRT candidate contains a production marker")
    sealed_rate_sweep = json.loads(
        (output / _RATE_SWEEP_NAME).read_text(encoding="utf-8")
    )
    if sealed_rate_sweep != rate_sweep:
        raise ValueError("Fruit QSRT candidate and sealed rate sweep disagree")
    _validate_checksum_manifest(
        output,
        base_provenance,
        allow_part_cache=allow_part_cache,
        include_runtime_qualification=False,
    )


def _completion_record(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
    runtime_qualification: dict[str, object],
) -> dict[str, object]:
    candidate = runtime_qualification["candidate"]
    if not isinstance(candidate, dict):
        raise TypeError("Fruit runtime qualification candidate is malformed")
    return {
        "schema": "kquant_qsrt_complete_v3",
        "publication": {
            "variant": "instruct",
            "repository": fruit_publication_spec("instruct").repository,
        },
        "qualified_candidate_sha256": candidate["marker_sha256"],
        "runtime_qualification_sha256": _sha256(output / _RUNTIME_QUALIFICATION_NAME),
        "package_manifest_sha256": _sha256(output / "qsrt-manifest.json"),
        "checksum_manifest_sha256": _sha256(output / "MANIFEST.sha256"),
        "model_index_sha256": _sha256(output / "model.safetensors.index.json"),
        "source": {
            "kind": source_evidence["source_kind"],
            "sha256": source_evidence["source_sha256"],
        },
        "base_manifest_sha256": base_provenance["manifest_sha256"],
        "producer_fingerprint": producer["fingerprint"],
        "encoder_fingerprint": producer["encoder"]["fingerprint"],
    }


def _validate_checksum_manifest(
    output: Path,
    base_provenance: dict[str, object],
    *,
    allow_part_cache: bool,
    include_runtime_qualification: bool = True,
) -> None:
    manifest_path = output / "MANIFEST.sha256"
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, separator, filename = line.partition("  ")
        relative = Path(filename)
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not filename
            or relative.is_absolute()
            or relative.as_posix() != filename
            or ".." in relative.parts
            or filename in entries
        ):
            raise ValueError(f"invalid package checksum line: {line!r}")
        entries[filename] = digest
    expected_files = set(
        _validated_package_files(
            output,
            base_provenance,
            allow_part_cache=allow_part_cache,
            include_runtime_qualification=include_runtime_qualification,
        )
    )
    if set(entries) != expected_files:
        raise ValueError("package checksum inventory mismatch")
    for filename, expected_sha256 in entries.items():
        actual_sha256 = _sha256(output / filename)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"package hash mismatch for {filename}: "
                f"{actual_sha256} != {expected_sha256}"
            )


def _validate_output_package(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
    rate_sweep: dict[str, object],
    require_complete: bool,
    runtime_qualification: dict[str, object],
    spec: FruitModelSpec = FRUIT_INSTRUCT_SPEC,
) -> None:
    manifest_path = output / "qsrt-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Fruit QSRT package manifest must be a JSON object")
    expected_identity = {
        "schema": "kquant_qsrt_model_manifest_v1",
        "version": 1,
        "publication": {
            "variant": "instruct",
            "repository": fruit_publication_spec("instruct").repository,
        },
        "codec": "QSRT",
        "storage_schema": FRUIT_QSRT_ATOM_SCHEMA,
        "encoding": "qsrt_sqg_e4m3",
        "codebook": FRUIT_QSRT_CODEBOOK,
        "profile_id": FRUIT_QSRT_PROFILE_ID,
        "geometry": {
            "layers": list(LAYERS),
            "experts_per_layer": EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "topk": 8,
        },
        "runtime": {
            "tensor_parallel": "whole_atom_partition",
            "validated_tensor_parallel_sizes": [1],
            "decode": "trellis_w4a8",
            "decode_max_tokens": 16,
            "fallback": "trellis_w4a16",
            "prefill": "trellis_w4a16",
        },
        "base_model": base_provenance,
        "producer": producer,
        "source": source_evidence,
        "evaluation": {
            "uniform_rate_sweep": {
                "file": _RATE_SWEEP_NAME,
                "sha256": _sha256(output / _RATE_SWEEP_NAME),
            },
            "runtime_qualification": {
                "file": _RUNTIME_QUALIFICATION_NAME,
                "sha256": hashlib.sha256(
                    _canonical_json(runtime_qualification).encode("utf-8")
                ).hexdigest(),
            },
        },
        "complete": True,
    }
    if any(manifest.get(name) != value for name, value in expected_identity.items()):
        raise ValueError("Fruit QSRT package identity mismatch")
    sealed_source = _read_source_evidence_seal(output, producer, spec=spec)
    if sealed_source != source_evidence:
        raise ValueError("Fruit QSRT package and sealed source evidence disagree")
    sealed_rate_sweep = json.loads(
        (output / _RATE_SWEEP_NAME).read_text(encoding="utf-8")
    )
    if sealed_rate_sweep != rate_sweep:
        raise ValueError("Fruit QSRT package and sealed rate sweep disagree")
    _validate_sealed_runtime_qualification(output, runtime_qualification)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise TypeError("Fruit QSRT output config has no quantization_config")
    qsrt = quantization.get("qsrt")
    if not isinstance(qsrt, dict) or any(
        qsrt.get(name) != value
        for name, value in (
            ("schema", FRUIT_QSRT_ATOM_SCHEMA),
            ("storage_format", FRUIT_QSRT_ATOM_STORAGE),
            ("encoding", "qsrt_sqg_e4m3"),
            ("codebook", FRUIT_QSRT_CODEBOOK),
            ("profile_id", FRUIT_QSRT_PROFILE_ID),
            ("runtime", "w4a8"),
            ("producer_fingerprint", producer["fingerprint"]),
            ("encoder_fingerprint", producer["encoder"]["fingerprint"]),
            ("source_kind", source_evidence["source_kind"]),
            ("source_sha256", source_evidence["source_sha256"]),
        )
    ):
        raise ValueError("Fruit QSRT output config identity mismatch")
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("Fruit QSRT output index has no weight_map")
    if any(".mlp.experts." in name for name in weight_map):
        raise ValueError("Fruit QSRT output index still references routed experts")
    for filename in set(weight_map.values()):
        path = output / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if filename.startswith("model-layer-"):
            with safe_open(path, framework="pt", device="cpu") as handle:
                names = handle.keys()
                if any(".mlp.experts." in name for name in names):
                    raise ValueError(
                        f"Fruit QSRT output shard still contains routed experts: {path}"
                    )
    encoder_fingerprint = producer["encoder"]["fingerprint"]
    validated_layers: dict[str, dict[str, object]] = {}
    for layer in LAYERS:
        tensor_path, layer_manifest_path = _layer_paths(output, layer)
        value = _validate_layer(
            tensor_path,
            layer_manifest_path,
            layer=layer,
            source_sha256=str(source_evidence["source_sha256"]),
            encoder_fingerprint=encoder_fingerprint,
        )
        if value is None:
            raise ValueError(f"Fruit QSRT output is missing assembled layer {layer}")
        validated_layers[str(layer)] = value
    expected_layers = {
        layer: {
            "qsrt_atoms": value["safetensors_file"],
            "bytes": value["safetensors_bytes"],
            "sha256": value["safetensors_sha256"],
            "expert_count": value["expert_count"],
            "evidence": f"qsrt-layer-{int(layer):03d}.json",
        }
        for layer, value in validated_layers.items()
    }
    if manifest.get("layers") != expected_layers:
        raise ValueError("Fruit QSRT package layer ledger mismatch")
    _validate_checksum_manifest(
        output,
        base_provenance,
        allow_part_cache=not require_complete,
    )
    marker_path = output / _COMPLETE_MARKER_NAME
    if require_complete:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker != _completion_record(
            output,
            source_evidence=source_evidence,
            base_provenance=base_provenance,
            producer=producer,
            runtime_qualification=runtime_qualification,
        ):
            raise ValueError("Fruit QSRT completion marker mismatch")
        if (output / _CANDIDATE_MARKER_NAME).exists():
            raise ValueError(
                "Fruit QSRT production package contains a candidate marker"
            )
    elif marker_path.exists():
        raise ValueError("Fruit QSRT completion marker exists before final validation")


def _write_complete_marker(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
    runtime_qualification: dict[str, object],
) -> None:
    _validate_sealed_runtime_qualification(output, runtime_qualification)
    _atomic_text(
        output / _COMPLETE_MARKER_NAME,
        _canonical_json(
            _completion_record(
                output,
                source_evidence=source_evidence,
                base_provenance=base_provenance,
                producer=producer,
                runtime_qualification=runtime_qualification,
            )
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("instruct",), default="instruct")
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--candidate-only", action="store_true")
    stage.add_argument("--runtime-qualification", type=Path)
    parser.add_argument("--exllamav3-root", required=True, type=Path)
    parser.add_argument("--b12x-root", required=True, type=Path)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--rate-sweep", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed-cache", type=Path)
    return parser.parse_args()


def _assert_producer_unchanged(
    producer: dict[str, object],
    *,
    exllamav3_root: Path,
    b12x_root: Path,
    vllm_root: Path,
    calibration: FruitCalibrationStore,
) -> None:
    current = _producer_provenance(
        exllamav3_root=exllamav3_root,
        b12x_root=b12x_root,
        vllm_root=vllm_root,
        calibration=calibration,
    )
    if current != producer:
        raise ValueError("Fruit QSRT producer sources changed during the build")


def main() -> None:
    args = parse_args()
    authority = fruit_calibration_authority(args.variant)
    spec = authority.spec
    publication = fruit_publication_spec(args.variant)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Fruit QSRT encoding requires a CUDA device")
    if not args.base_model.is_dir():
        raise FileNotFoundError(args.base_model)
    if args.output.resolve() == args.base_model.resolve():
        raise ValueError("output must not alias base_model")
    calibration = FruitCalibrationStore(
        args.calibration,
        authority=authority,
    )
    producer = _producer_provenance(
        exllamav3_root=args.exllamav3_root,
        b12x_root=args.b12x_root,
        vllm_root=args.vllm_root,
        calibration=calibration,
    )
    encoder_fingerprint = producer["encoder"]["fingerprint"]
    if not isinstance(encoder_fingerprint, str):
        raise TypeError("Fruit QSRT encoder fingerprint must be a string")
    print("authenticating pinned Fruit BF16 base model", flush=True)
    if spec.safetensors_manifest_sha256 is None:
        raise ValueError("Fruit variant has no pinned Safetensors source")
    store = FruitSafetensorsStore(
        args.base_model,
        spec=spec,
        expected_manifest_sha256=spec.safetensors_manifest_sha256,
    )
    try:
        _build_from_snapshot(
            args=args,
            spec=spec,
            publication=publication,
            device=device,
            calibration=calibration,
            producer=producer,
            encoder_fingerprint=encoder_fingerprint,
            store=store,
        )
    finally:
        store.close()


def _build_from_snapshot(
    *,
    args: argparse.Namespace,
    spec: FruitModelSpec,
    publication: FruitPublicationSpec,
    device: torch.device,
    calibration: FruitCalibrationStore,
    producer: dict[str, object],
    encoder_fingerprint: str,
    store: FruitSafetensorsStore,
) -> None:
    base_provenance = _authenticate_base_model(store.path, spec=spec)
    source_evidence = _validate_source_evidence(store.evidence, spec=spec)
    if source_evidence.get("source_kind") != "safetensors_manifest":
        raise ValueError(
            "Fruit QSRT publication requires the authenticated BF16 safetensors source"
        )
    source_sha256 = source_evidence.get("source_sha256")
    if not isinstance(source_sha256, str):
        raise TypeError("Fruit source evidence has no authenticated source digest")
    rate_sweep = _validate_rate_sweep(
        args.rate_sweep,
        source_evidence=source_evidence,
        calibration=calibration,
        producer=producer,
    )
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    _prepare_output_root(args.output)
    (args.output / _COMPLETE_MARKER_NAME).unlink(missing_ok=True)
    _write_source_evidence_seal(
        args.output,
        source_evidence,
        producer,
        spec=spec,
    )
    seeded = _seed_parts(
        args.output,
        args.seed_cache,
        source_sha256,
        encoder_fingerprint,
    )
    if seeded:
        print(f"seeded {seeded} authenticated expert parts", flush=True)
    needs_encoder = _parts_need_encoder(
        args.output,
        source_sha256=source_sha256,
        encoder_fingerprint=encoder_fingerprint,
    )
    quantizer_module = None
    if needs_encoder:
        quantizer_module = load_qsrt_encoder(args.exllamav3_root)
        install_sqg_quantizer(quantizer_module)
    _encode_parts(
        args.output,
        store,
        quantizer_module,
        calibration,
        device=device,
        source_sha256=source_sha256,
        encoder_fingerprint=encoder_fingerprint,
    )
    layers = _assemble_layers(
        args.output,
        source_sha256=source_sha256,
        encoder_fingerprint=encoder_fingerprint,
    )
    _materialize_base_model(store.path, args.output, base_provenance)
    _write_config(
        store.path,
        args.output,
        producer,
        source_evidence,
        base_provenance,
    )
    _write_source_evidence_seal(
        args.output,
        source_evidence,
        producer,
        spec=spec,
    )
    _write_calibration_evidence(args.output, calibration, producer)
    _assert_producer_unchanged(
        producer,
        exllamav3_root=args.exllamav3_root,
        b12x_root=args.b12x_root,
        vllm_root=args.vllm_root,
        calibration=calibration,
    )
    if args.candidate_only:
        _write_candidate_package(
            args.output,
            source_evidence=source_evidence,
            base_provenance=base_provenance,
            producer=producer,
            rate_sweep=rate_sweep,
            layers=layers,
            publication=publication,
        )
        staged_cache = _stage_part_cache(args.output)
        try:
            _validate_candidate_package(
                args.output,
                source_evidence=source_evidence,
                base_provenance=base_provenance,
                producer=producer,
                rate_sweep=rate_sweep,
                layers=layers,
                publication=publication,
                allow_part_cache=False,
            )
        except BaseException:
            (args.output / _CANDIDATE_MARKER_NAME).unlink(missing_ok=True)
            _restore_staged_part_cache(args.output, staged_cache)
            raise
        print(
            f"sealed Fruit QSRT qualification candidate: {args.output}",
            flush=True,
        )
        return

    _validate_candidate_package(
        args.output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
        rate_sweep=rate_sweep,
        layers=layers,
        publication=publication,
        allow_part_cache=True,
    )
    runtime_qualification = _validate_runtime_qualification(
        args.runtime_qualification,
        output=args.output,
        variant=args.variant,
        publication=publication,
        producer=producer,
        source_evidence=source_evidence,
    )
    _write_package_manifests(
        args.output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
        calibration=calibration,
        rate_sweep=rate_sweep,
        layers=layers,
        publication=publication,
        runtime_qualification=runtime_qualification,
    )
    _validate_output_package(
        args.output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
        rate_sweep=rate_sweep,
        require_complete=False,
        runtime_qualification=runtime_qualification,
        spec=spec,
    )
    staged_cache = _stage_part_cache(args.output)
    candidate_marker_path = args.output / _CANDIDATE_MARKER_NAME
    candidate_marker_bytes = candidate_marker_path.read_bytes()
    candidate_marker_path.unlink()
    try:
        _write_complete_marker(
            args.output,
            source_evidence=source_evidence,
            base_provenance=base_provenance,
            producer=producer,
            runtime_qualification=runtime_qualification,
        )
        _validate_output_package(
            args.output,
            source_evidence=source_evidence,
            base_provenance=base_provenance,
            producer=producer,
            rate_sweep=rate_sweep,
            require_complete=True,
            runtime_qualification=runtime_qualification,
            spec=spec,
        )
    except BaseException:
        (args.output / _COMPLETE_MARKER_NAME).unlink(missing_ok=True)
        _atomic_text(candidate_marker_path, candidate_marker_bytes.decode("utf-8"))
        _restore_staged_part_cache(args.output, staged_cache)
        raise
    _discard_staged_part_cache(staged_cache)
    print(f"complete Fruit QSRT model: {args.output}", flush=True)


if __name__ == "__main__":
    main()

"""Revision-two TP-independent QSRT atom storage.

Atoms-v2 retains the 96-row atom-major container used by atoms-v1.  Its
fixed H308 profile gives two physical record pairs a seven-bit P43 payload
and the other ten pairs a six-bit P33 payload.  Each atom row therefore stores
two compact, fixed-width expert groups instead of padding every expert to the
larger P43 bundle.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from functools import lru_cache

from kquant.qsrt import (
    EXPERTS_PER_LAYER,
    FIXED_HIGH_RATE_RECORD_BITS,
    FORMAT_SECTION_BYTES,
    INTERMEDIATE_CHANNELS,
    LATENT_CHANNELS,
    LAYER_HEADER_BYTES,
    PAIRS_PER_EXPERT,
    RECORDS_PER_EXPERT,
    SCALE_BYTES,
    SHARED_SCALE_SECTION_BYTES,
    STORAGE_ALIGNMENT,
    align_up,
    pair_rotation,
)
from kquant.qsrt_storage import (
    ATOM_CHANNELS,
    ATOM_SCALE_BYTES,
    ATOM_SIDE_CHANNELS,
    ATOM_SLAB_OFFSET,
    ATOMS_PER_EXPERT,
    ATOMS_PER_RECORD_PAIR,
    EQUAL_SHARD_COUNTS,
    MATRIX_ATOM_SCALE_BYTES,
    QSRTShardExtent,
    balanced_atom_partition,
)


SCHEMA = "kquant_kimi_k3_qsrt_atoms_v2"
VERSION = 2
ENCODING = "qsrt_sqg_e4m3"
CODEBOOK = "sqg_xor_cheb_t12"
PROFILE = "k3x22_k4x2"
PROFILE_ID = 2

K3_RECORDS = FIXED_HIGH_RATE_RECORD_BITS.count(3)
K4_RECORDS = FIXED_HIGH_RATE_RECORD_BITS.count(4)
K4_LOGICAL_RECORDS = tuple(
    index for index, bits in enumerate(FIXED_HIGH_RATE_RECORD_BITS) if bits == 4
)

P33_MATRIX_TRELLIS_BYTES = ATOM_CHANNELS * LATENT_CHANNELS * 3 // 8
P43_MATRIX_TRELLIS_BYTES = (
    ATOM_SIDE_CHANNELS * LATENT_CHANNELS * (4 + 3) // 8
)
P33_ATOM_BUNDLE_BYTES = 3 * P33_MATRIX_TRELLIS_BYTES + ATOM_SCALE_BYTES
P43_ATOM_BUNDLE_BYTES = 3 * P43_MATRIX_TRELLIS_BYTES + ATOM_SCALE_BYTES

_SAFETENSORS_HEADER_LENGTH = struct.Struct("<Q")
FORMAT_TENSOR = "_qsrt_format_section"
SHARED_SCALE_TENSOR = "_qsrt_shared_scale_section"
ATOM_TENSOR = "qsrt_atoms"


def _validate_layer(layer: int) -> None:
    if isinstance(layer, bool) or not isinstance(layer, int) or not 1 <= layer <= 92:
        raise ValueError("Kimi-K3 MoE layer must lie in 1..92")


def _validate_physical_pair(physical_pair: int) -> None:
    if (
        isinstance(physical_pair, bool)
        or not isinstance(physical_pair, int)
        or not 0 <= physical_pair < PAIRS_PER_EXPERT
    ):
        raise ValueError(f"physical_pair must lie in 0..{PAIRS_PER_EXPERT - 1}")


def expert_pair_is_p43(layer: int, expert: int, physical_pair: int) -> bool:
    """Return whether one physical pair contains K4-low/K3-high records."""

    _validate_layer(layer)
    _validate_physical_pair(physical_pair)
    if not 0 <= expert < EXPERTS_PER_LAYER:
        raise ValueError(f"expert must lie in 0..{EXPERTS_PER_LAYER - 1}")
    base_pair = (physical_pair - pair_rotation(layer, expert)) % PAIRS_PER_EXPERT
    return base_pair in (0, PAIRS_PER_EXPERT // 2)


def _base_physical_to_logical_record() -> tuple[int, ...]:
    """Place the two logical K4 records in distinct low-record positions."""

    result: list[int | None] = [None] * RECORDS_PER_EXPERT
    result[0] = K4_LOGICAL_RECORDS[0]
    result[RECORDS_PER_EXPERT // 2] = K4_LOGICAL_RECORDS[1]
    ordinary = iter(range(K3_RECORDS))
    for physical in range(RECORDS_PER_EXPERT):
        if result[physical] is None:
            result[physical] = next(ordinary)
    return tuple(int(value) for value in result)


BASE_PHYSICAL_TO_LOGICAL_RECORD = _base_physical_to_logical_record()


@lru_cache(maxsize=None)
def physical_to_logical_records(layer: int, expert: int) -> tuple[int, ...]:
    """Return the shared record placement for one atoms-v2 expert."""

    _validate_layer(layer)
    if not 0 <= expert < EXPERTS_PER_LAYER:
        raise ValueError(f"expert must lie in 0..{EXPERTS_PER_LAYER - 1}")
    rotation = pair_rotation(layer, expert)
    result = []
    for physical in range(RECORDS_PER_EXPERT):
        physical_pair, within_pair = divmod(physical, 2)
        base_pair = (physical_pair - rotation) % PAIRS_PER_EXPERT
        result.append(BASE_PHYSICAL_TO_LOGICAL_RECORD[2 * base_pair + within_pair])
    return tuple(result)


def logical_rate_record_index(logical_record: int) -> tuple[int, int]:
    """Return ``(rate, index within that rate)`` for one H308 record."""

    if not 0 <= logical_record < RECORDS_PER_EXPERT:
        raise ValueError(
            f"logical_record must lie in 0..{RECORDS_PER_EXPERT - 1}"
        )
    bits = FIXED_HIGH_RATE_RECORD_BITS[logical_record]
    return bits, logical_record if bits == 3 else logical_record - K3_RECORDS


@lru_cache(maxsize=None)
def pair_experts(layer: int, physical_pair: int, *, p43: bool) -> tuple[int, ...]:
    _validate_layer(layer)
    _validate_physical_pair(physical_pair)
    return tuple(
        expert
        for expert in range(EXPERTS_PER_LAYER)
        if expert_pair_is_p43(layer, expert, physical_pair) is p43
    )


def atom_pair(physical_atom: int) -> int:
    if not 0 <= physical_atom < ATOMS_PER_EXPERT:
        raise ValueError(f"physical_atom must lie in 0..{ATOMS_PER_EXPERT - 1}")
    return physical_atom // ATOMS_PER_RECORD_PAIR


@dataclass(frozen=True)
class QSRTAtomsV2Layout:
    layer: int

    def __post_init__(self) -> None:
        _validate_layer(self.layer)

    def group_experts(self, physical_atom: int, *, p43: bool) -> tuple[int, ...]:
        return pair_experts(self.layer, atom_pair(physical_atom), p43=p43)

    def group_bundle_bytes(self, *, p43: bool) -> int:
        return P43_ATOM_BUNDLE_BYTES if p43 else P33_ATOM_BUNDLE_BYTES

    def group_payload_bytes(self, physical_atom: int, *, p43: bool) -> int:
        return len(self.group_experts(physical_atom, p43=p43)) * self.group_bundle_bytes(
            p43=p43
        )

    def atom_slot_payload_bytes(self, physical_atom: int) -> int:
        return self.group_payload_bytes(
            physical_atom, p43=False
        ) + self.group_payload_bytes(physical_atom, p43=True)

    @property
    def atom_slot_stride_bytes(self) -> int:
        return align_up(
            max(self.atom_slot_payload_bytes(slot) for slot in range(ATOMS_PER_EXPERT)),
            STORAGE_ALIGNMENT,
        )

    @property
    def compressed_payload_bytes(self) -> int:
        return sum(
            self.atom_slot_payload_bytes(slot) for slot in range(ATOMS_PER_EXPERT)
        )

    @property
    def disk_bytes(self) -> int:
        return ATOM_SLAB_OFFSET + ATOMS_PER_EXPERT * self.atom_slot_stride_bytes

    def group_offset(self, physical_atom: int, *, p43: bool) -> int:
        atom_pair(physical_atom)
        within = self.group_payload_bytes(physical_atom, p43=False) if p43 else 0
        return (
            ATOM_SLAB_OFFSET
            + physical_atom * self.atom_slot_stride_bytes
            + within
        )

    def bundle_offset(self, physical_atom: int, expert: int) -> int:
        p43 = expert_pair_is_p43(self.layer, expert, atom_pair(physical_atom))
        experts = self.group_experts(physical_atom, p43=p43)
        try:
            slot = experts.index(expert)
        except ValueError as exc:
            raise AssertionError("atoms-v2 expert grouping is incomplete") from exc
        return self.group_offset(physical_atom, p43=p43) + slot * self.group_bundle_bytes(
            p43=p43
        )

    def shard_extent(
        self, shard_count: int, shard_index: int, *, require_equal: bool = False
    ) -> QSRTShardExtent:
        first, atom_slots = balanced_atom_partition(shard_count, shard_index)
        if require_equal and shard_count not in EQUAL_SHARD_COUNTS:
            raise ValueError(
                f"{shard_count} shards do not divide the QSRT atom axis; "
                f"equal partitions are {EQUAL_SHARD_COUNTS}"
            )
        extent = atom_slots * self.atom_slot_stride_bytes
        payload = sum(
            self.atom_slot_payload_bytes(slot)
            for slot in range(first, first + atom_slots)
        )
        return QSRTShardExtent(
            shard_count=shard_count,
            shard_index=shard_index,
            first_atom_slot=first,
            atom_slots=atom_slots,
            intermediate_channels=atom_slots * ATOM_CHANNELS,
            offset_bytes=ATOM_SLAB_OFFSET + first * self.atom_slot_stride_bytes,
            extent_bytes=extent,
            payload_bytes=payload,
            padding_bytes=extent - payload,
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "encoding": ENCODING,
            "codebook": CODEBOOK,
            "profile": PROFILE,
            "atom_channels": ATOM_CHANNELS,
            "atom_slots": ATOMS_PER_EXPERT,
            "p33_atom_bundle_bytes": P33_ATOM_BUNDLE_BYTES,
            "p43_atom_bundle_bytes": P43_ATOM_BUNDLE_BYTES,
            "atom_slot_stride_bytes": self.atom_slot_stride_bytes,
            "compressed_payload_bytes": self.compressed_payload_bytes,
            "disk_bytes": self.disk_bytes,
            "equal_shard_counts": list(EQUAL_SHARD_COUNTS),
        }


@dataclass(frozen=True)
class QSRTAtomsV2Header:
    layer: int
    layout: QSRTAtomsV2Layout

    def __post_init__(self) -> None:
        _validate_layer(self.layer)
        if self.layout.layer != self.layer:
            raise ValueError("atoms-v2 header and layout layer disagree")

    def _document(self) -> dict[str, object]:
        atom_data_bytes = ATOMS_PER_EXPERT * self.layout.atom_slot_stride_bytes
        metadata = {
            "format": "pt",
            "schema": SCHEMA,
            "version": str(VERSION),
            "encoding": ENCODING,
            "codebook": CODEBOOK,
            "profile": PROFILE,
            "profile_id": str(PROFILE_ID),
            "layer": str(self.layer),
            "experts": str(EXPERTS_PER_LAYER),
            "intermediate_channels": str(INTERMEDIATE_CHANNELS),
            "latent_channels": str(LATENT_CHANNELS),
            "atom_channels": str(ATOM_CHANNELS),
            "atom_slots": str(ATOMS_PER_EXPERT),
            "p33_atom_bundle_bytes": str(P33_ATOM_BUNDLE_BYTES),
            "p43_atom_bundle_bytes": str(P43_ATOM_BUNDLE_BYTES),
            "atom_slot_stride_bytes": str(self.layout.atom_slot_stride_bytes),
            "alignment_bytes": str(STORAGE_ALIGNMENT),
        }
        return {
            "__metadata__": metadata,
            FORMAT_TENSOR: {
                "dtype": "U8",
                "shape": [FORMAT_SECTION_BYTES],
                "data_offsets": [0, FORMAT_SECTION_BYTES],
            },
            SHARED_SCALE_TENSOR: {
                "dtype": "U8",
                "shape": [SHARED_SCALE_SECTION_BYTES],
                "data_offsets": [
                    FORMAT_SECTION_BYTES,
                    FORMAT_SECTION_BYTES + SHARED_SCALE_SECTION_BYTES,
                ],
            },
            ATOM_TENSOR: {
                "dtype": "U8",
                "shape": [ATOMS_PER_EXPERT, self.layout.atom_slot_stride_bytes],
                "data_offsets": [
                    FORMAT_SECTION_BYTES + SHARED_SCALE_SECTION_BYTES,
                    FORMAT_SECTION_BYTES
                    + SHARED_SCALE_SECTION_BYTES
                    + atom_data_bytes,
                ],
            },
        }

    def to_bytes(self) -> bytes:
        capacity = LAYER_HEADER_BYTES - _SAFETENSORS_HEADER_LENGTH.size
        document = json.dumps(
            self._document(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(document) > capacity:
            raise AssertionError("QSRT atoms-v2 header exceeds its 4 KiB budget")
        return (
            _SAFETENSORS_HEADER_LENGTH.pack(capacity)
            + document
            + b" " * (capacity - len(document))
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> "QSRTAtomsV2Header":
        if len(payload) != LAYER_HEADER_BYTES:
            raise ValueError("QSRT atoms-v2 header must contain exactly 4 KiB")
        header_length = _SAFETENSORS_HEADER_LENGTH.unpack_from(payload)[0]
        if header_length != LAYER_HEADER_BYTES - _SAFETENSORS_HEADER_LENGTH.size:
            raise ValueError("QSRT atoms-v2 safetensors header is not canonical")
        try:
            document = json.loads(payload[8:].decode("utf-8"))
            metadata = document["__metadata__"]
            layer = int(metadata["layer"])
            result = cls(layer, QSRTAtomsV2Layout(layer))
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("QSRT atoms-v2 header is malformed") from exc
        if document != result._document():
            raise ValueError("QSRT atoms-v2 header is noncanonical")
        return result


if P33_ATOM_BUNDLE_BYTES != 129216:
    raise AssertionError("atoms-v2 P33 bundles must close atoms-v1 geometry")
if P43_ATOM_BUNDLE_BYTES != 150720:
    raise AssertionError("atoms-v2 P43 bundle accounting drifted")


__all__ = [
    "ATOM_TENSOR",
    "BASE_PHYSICAL_TO_LOGICAL_RECORD",
    "CODEBOOK",
    "ENCODING",
    "FORMAT_TENSOR",
    "K3_RECORDS",
    "K4_LOGICAL_RECORDS",
    "K4_RECORDS",
    "P33_ATOM_BUNDLE_BYTES",
    "P33_MATRIX_TRELLIS_BYTES",
    "P43_ATOM_BUNDLE_BYTES",
    "P43_MATRIX_TRELLIS_BYTES",
    "PROFILE",
    "PROFILE_ID",
    "QSRTAtomsV2Header",
    "QSRTAtomsV2Layout",
    "SCHEMA",
    "SHARED_SCALE_TENSOR",
    "VERSION",
    "atom_pair",
    "expert_pair_is_p43",
    "logical_rate_record_index",
    "pair_experts",
    "physical_to_logical_records",
]

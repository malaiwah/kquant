from __future__ import annotations

import torch

from kquant.pack.qsrt_atoms_v2 import (
    assemble_candidate_records,
    disassemble_candidate_records,
    pack_local_scale_rate_records,
    pack_matrix_rate_records,
    unpack_local_scale_rate_records,
    unpack_matrix_rate_records,
)
from kquant.qsrt import (
    FIXED_HIGH_RATE_RECORD_BITS,
    H308,
    RECORDS_PER_EXPERT,
    PackedQSRTTrellis,
    QSRTTrellisDescriptor,
)
from kquant.qsrt_atoms_v2 import (
    BASE_PHYSICAL_TO_LOGICAL_RECORD,
    K3_RECORDS,
    K4_RECORDS,
    logical_rate_record_index,
    physical_to_logical_records,
    P33_ATOM_BUNDLE_BYTES,
    P43_ATOM_BUNDLE_BYTES,
    QSRTAtomsV2Header,
    QSRTAtomsV2Layout,
    expert_pair_is_p43,
)


def _descriptor(rate_axis: str) -> QSRTTrellisDescriptor:
    return QSRTTrellisDescriptor(
        mode_id=H308.mode_id,
        rate_axis=rate_axis,  # type: ignore[arg-type]
        k_tiles=192 if rate_axis == "k" else 224,
        n_tiles=224 if rate_axis == "k" else 192,
    )


def test_high_rate_record_placement_preserves_funding_and_balances_tp12() -> None:
    assert sorted(BASE_PHYSICAL_TO_LOGICAL_RECORD) == list(range(RECORDS_PER_EXPERT))
    k4_per_rank = [0] * 12
    for expert in range(896):
        mapping = physical_to_logical_records(24, expert)
        rates = tuple(FIXED_HIGH_RATE_RECORD_BITS[logical] for logical in mapping)
        assert sorted(mapping) == list(range(RECORDS_PER_EXPERT))
        assert rates.count(3) == K3_RECORDS
        assert rates.count(4) == K4_RECORDS
        high = [index for index, bits in enumerate(rates) if bits == 4]
        assert all(index % 2 == 0 for index in high)
        assert high[1] - high[0] in (12, -12)
        for record in high:
            k4_per_rank[record // 2] += 1
    assert max(k4_per_rank) - min(k4_per_rank) <= 2

def test_high_rate_physical_placement_preserves_each_funded_record() -> None:
    """Physical balancing must never substitute another same-rate record."""

    logical_bundles = {
        3: tuple((3, index) for index in range(K3_RECORDS)),
        4: tuple((4, index) for index in range(K4_RECORDS)),
    }
    for layer, expert in ((1, 0), (24, 37), (92, 895)):
        physical = []
        for logical_record in physical_to_logical_records(layer, expert):
            bits, rate_index = logical_rate_record_index(logical_record)
            physical.append(logical_bundles[bits][rate_index])
        recovered = [None] * RECORDS_PER_EXPERT
        for physical_record, logical_record in enumerate(
            physical_to_logical_records(layer, expert)
        ):
            recovered[logical_record] = physical[physical_record]
        assert recovered == [
            logical_bundles[3][index] for index in range(K3_RECORDS)
        ] + [logical_bundles[4][index] for index in range(K4_RECORDS)]


def test_high_rate_matrix_records_round_trip_without_decoding() -> None:
    generator = torch.Generator().manual_seed(9127)
    for rate_axis in ("k", "n"):
        descriptor = _descriptor(rate_axis)
        payload = torch.randint(
            -32768,
            32768,
            (descriptor.payload_words,),
            dtype=torch.int16,
            generator=generator,
        )
        packed = PackedQSRTTrellis(descriptor, payload)
        records = pack_matrix_rate_records(packed)
        actual = unpack_matrix_rate_records(records, descriptor)
        assert torch.equal(actual.payload, payload)


def test_high_rate_scale_records_round_trip() -> None:
    scale = torch.arange(3072, dtype=torch.float32).to(torch.float16)
    records = pack_local_scale_rate_records(scale)
    actual = unpack_local_scale_rate_records(records)
    assert records[3].shape == (22, 128)
    assert records[4].shape == (2, 128)
    assert torch.equal(actual, scale)


def test_high_rate_candidate_bundle_round_trip_preserves_funding_order() -> None:
    generator = torch.Generator().manual_seed(10931)
    tensors = {}
    for matrix, rate_axis in (("w1", "k"), ("w3", "k"), ("w2", "n")):
        descriptor = _descriptor(rate_axis)
        shared_part = "svh" if matrix == "w2" else "suh"
        local_part = "suh" if matrix == "w2" else "svh"
        tensors[matrix] = {
            "trellis": torch.randint(
                -32768,
                32768,
                (descriptor.payload_words,),
                dtype=torch.int16,
                generator=generator,
            ),
            shared_part: torch.randn(3584, generator=generator).to(torch.float16),
            local_part: torch.randn(3072, generator=generator).to(torch.float16),
        }
    bundles, shared = assemble_candidate_records(tensors=tensors)
    actual = disassemble_candidate_records(bundles=bundles, shared=shared)
    for matrix in tensors:
        for part in tensors[matrix]:
            assert torch.equal(actual[matrix][part], tensors[matrix][part])


def test_high_rate_physical_record_permutation_preserves_expert_function() -> None:
    generator = torch.Generator().manual_seed(3107)
    channels = 24
    inputs = 7
    outputs = 5
    w1 = torch.randn(channels, inputs, generator=generator)
    w3 = torch.randn(channels, inputs, generator=generator)
    w2 = torch.randn(outputs, channels, generator=generator)
    x = torch.randn(inputs, generator=generator)
    logical = torch.tensor(physical_to_logical_records(24, 37), dtype=torch.long)
    w1_physical = w1.index_select(0, logical)
    w3_physical = w3.index_select(0, logical)
    w2_physical = w2.index_select(1, logical)

    def activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * torch.tanh(up)

    expected = w2 @ activation(w1 @ x, w3 @ x)
    actual = w2_physical @ activation(w1_physical @ x, w3_physical @ x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0e-6)


def test_high_rate_record_permutation_commutes_with_block_transform() -> None:
    """The storage permutation moves transform blocks, never lanes within one."""

    generator = torch.Generator().manual_seed(7021)
    values = torch.randn(24, 4, generator=generator)
    hadamard4 = torch.tensor(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
        dtype=torch.float32,
    ) / 2
    logical = torch.tensor(physical_to_logical_records(24, 37), dtype=torch.long)
    transform_then_place = (values @ hadamard4).index_select(0, logical)
    place_then_transform = values.index_select(0, logical) @ hadamard4
    assert torch.equal(place_then_transform, transform_then_place)


def test_atoms_v2_preserves_atom_major_balanced_high_rate_layout() -> None:
    layout = QSRTAtomsV2Layout(24)
    assert P33_ATOM_BUNDLE_BYTES == 129216
    assert P43_ATOM_BUNDLE_BYTES == 150720
    assert layout.disk_bytes == 11_424_530_432
    for pair in range(12):
        p33 = layout.group_experts(8 * pair, p43=False)
        p43 = layout.group_experts(8 * pair, p43=True)
        assert sorted(p33 + p43) == list(range(896))
        assert not set(p33).intersection(p43)
        assert len(p43) in (149, 150)
        for expert in p43:
            assert expert_pair_is_p43(24, expert, pair)
        for stripe in range(8):
            slot = 8 * pair + stripe
            assert layout.atom_slot_payload_bytes(slot) <= (
                layout.atom_slot_stride_bytes
            )


def test_atoms_v2_header_is_canonical_safetensors_metadata() -> None:
    layout = QSRTAtomsV2Layout(24)
    header = QSRTAtomsV2Header(24, layout)
    assert QSRTAtomsV2Header.from_bytes(header.to_bytes()) == header
    assert len(header.to_bytes()) == 4096

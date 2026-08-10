from __future__ import annotations

from kquant import constants as C
from kquant.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from kquant.pack.qsrt_allocation import (
    QSRT_ALLOCATION_KIND,
    QSRT_ALLOCATION_SCHEMA_VERSION,
)
from kquant.pack.qsrt_model_view import (
    qsrt_atoms_v2_quantization_config,
    qsrt_hybrid_bit_map,
    qsrt_quantization_config,
)
from kquant.qsrt import FORMAT_X4T, PHASE1_MODE_IDS


def _allocation() -> dict:
    layers = {}
    for layer in C.MOE_LAYERS:
        codes = [0] * C.NUM_EXPERTS
        codes[layer % C.NUM_EXPERTS] = FORMAT_X4T
        x4t = [layer % C.NUM_EXPERTS]
        layers[str(layer)] = {
            "format_codes": codes,
            "x4t": x4t,
            "compressed": [
                expert for expert in range(C.NUM_EXPERTS) if expert not in x4t
            ],
        }
    return {
        "kind": QSRT_ALLOCATION_KIND,
        "schema_version": QSRT_ALLOCATION_SCHEMA_VERSION,
        "meta": {
            "codec": "QSRT",
            "high_tier_storage": "x4t",
            "candidate_codebook": CODEBOOK_SQG_XOR_CHEB_T12,
            "candidate_mode_ids": list(PHASE1_MODE_IDS),
        },
        "layers": layers,
    }


def test_qsrt_model_view_config_is_tp_independent() -> None:
    allocation = _allocation()
    bit_map = qsrt_hybrid_bit_map(allocation)
    config = qsrt_quantization_config(allocation)

    assert bit_map["1"].count(4) == 1
    assert bit_map["1"].count(3) == C.NUM_EXPERTS - 1
    assert config["demoted_format"] == "qsrt_sqg_e4m3"
    assert config["kept_storage"] == "x4t"
    assert config["qsrt"]["codebook"] == "sqg_xor_cheb_t12"
    assert "vision_tower" not in config["ignored_layers"]
    assert "mm_projector" not in config["ignored_layers"]
    assert "tp_size" not in config


def test_qsrt_atoms_v2_model_view_is_all_qsrt_and_tp_independent() -> None:
    config = qsrt_atoms_v2_quantization_config()
    assert set(config["hybrid_bit_map"]) == {
        str(layer) for layer in C.MOE_LAYERS
    }
    assert all(
        bits == [3] * C.NUM_EXPERTS
        for bits in config["hybrid_bit_map"].values()
    )
    assert config["qsrt"] == {
        "schema": "kquant_kimi_k3_qsrt_atoms_v2",
        "storage_format": "qsrt_atoms_v2",
        "encoding": "qsrt_sqg_e4m3",
        "codebook": "sqg_xor_cheb_t12",
        "artifact_manifest": "qsrt-manifest.json",
        "profile": "k3x22_k4x2",
    }
    assert "vision_tower" not in config["ignored_layers"]
    assert "mm_projector" not in config["ignored_layers"]
    assert "tp_size" not in config


def test_qsrt_atoms_v2_pure_k2_model_view_contract() -> None:
    config = qsrt_atoms_v2_quantization_config("k2_coupled_h512_h128")
    assert all(
        bits == [2] * C.NUM_EXPERTS
        for bits in config["hybrid_bit_map"].values()
    )
    assert config["qsrt"]["profile"] == "k2_coupled_h512_h128"

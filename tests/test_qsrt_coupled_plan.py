from __future__ import annotations

import pytest

from kquant import constants as C
from kquant.qsrt_coupled_plan import K2CoupledRotationPlan
from kquant.qsrt_coupled_plan import select_k2_coupled_draw


def _plan() -> K2CoupledRotationPlan:
    return K2CoupledRotationPlan(
        {
            layer: tuple((layer + expert) % 8 for expert in range(C.NUM_EXPERTS))
            for layer in C.MOE_LAYERS
        },
        "test_fit_policy",
    )


def test_coupled_rotation_plan_roundtrips_and_slices_experts() -> None:
    plan = _plan()
    restored = K2CoupledRotationPlan.from_json(plan.to_json())

    assert restored == plan
    assert restored.for_experts(24, [0, 17, 895]) == {
        expert: (24 + expert) % 8 for expert in (0, 17, 895)
    }


def test_coupled_rotation_plan_requires_complete_model_coverage() -> None:
    with pytest.raises(ValueError, match="every MoE layer"):
        K2CoupledRotationPlan({1: (0,) * C.NUM_EXPERTS}, "test_fit_policy")


def test_coupled_draw_selection_uses_fit_to_propose_and_confirmation_to_accept() -> None:
    accepted = select_k2_coupled_draw(
        (0, 6),
        {0: 10.0, 6: 9.0},
        {0: 8.0, 6: 7.5},
        fit_documents=8,
        confirmation_documents=6,
        min_fit_documents=6,
        min_confirmation_documents=4,
        minimum_improvement=0.0,
    )
    assert accepted.proposed_draw == accepted.selected_draw == 6
    assert accepted.accepted
    assert accepted.confirmation_relative_improvement == 0.0625

    rejected = select_k2_coupled_draw(
        (0, 6),
        {0: 10.0, 6: 9.0},
        {0: 8.0, 6: 8.5},
        fit_documents=8,
        confirmation_documents=6,
        min_fit_documents=6,
        min_confirmation_documents=4,
        minimum_improvement=0.0,
    )
    assert rejected.proposed_draw == 6
    assert rejected.selected_draw == 0
    assert not rejected.accepted


def test_coupled_draw_selection_falls_back_without_disjoint_support() -> None:
    decision = select_k2_coupled_draw(
        (0, 6),
        {0: 10.0, 6: 9.0},
        {0: 8.0, 6: 7.0},
        fit_documents=8,
        confirmation_documents=3,
        min_fit_documents=6,
        min_confirmation_documents=4,
        minimum_improvement=0.0,
    )
    assert decision.proposed_draw == 6
    assert decision.selected_draw == 0
    assert decision.reason == "insufficient_confirmation_document_support"

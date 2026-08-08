from __future__ import annotations

from dataclasses import dataclass


FAST_CHAIN_IK_STAGES = ("pregrasp", "grasp", "transport_hover", "release")


@dataclass(frozen=True)
class FastChainConfig:
    relation_slots: int = 16
    ik_seeds: int = 32
    cuda_graph_fixed_batch_size: int = 16
    top_pairs: int = 3


def full_intersection_expression() -> str:
    return " && ".join(f"{name}_ok[relation]" for name in FAST_CHAIN_IK_STAGES)

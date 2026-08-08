from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PickPlaceLayer:
    key: str
    responsibility: str
    modules: tuple[str, ...]


PICKPLACE_LAYERS = (
    PickPlaceLayer(
        key="task",
        responsibility="任务对象、目标和运行模式定义",
        modules=("rm75_app.tasks.pickplace", "rm75_app.pickplace.config"),
    ),
    PickPlaceLayer(
        key="perception",
        responsibility="FoundationPose、SAM3/SAM6D 和腕带关系估计",
        modules=("rm75_app.runtime.foundationpose_scene", "rm75_app.perception.sam6d_pose_provider"),
    ),
    PickPlaceLayer(
        key="placement",
        responsibility="目标对象、放置规则和候选目标姿态",
        modules=("rm75_app.placement.place_rules", "rm75_app.runtime.targeted_place"),
    ),
    PickPlaceLayer(
        key="planning",
        responsibility="IK、候选配对、cuRobo 和短直线约束段",
        modules=("rm75_app.runtime.targeted_curobo", "rm75_app.planning.curobo_planner"),
    ),
    PickPlaceLayer(
        key="execution",
        responsibility="仿真/真机运动、夹爪动作和结果校验",
        modules=("rm75_app.execution.rm75_bridge", "rm75_app.execution.pick_move_v10"),
    ),
    PickPlaceLayer(
        key="orchestration",
        responsibility="按 cycle 组织抓取、搬运、放置、返回和 profile",
        modules=("rm75_app.runtime.direct_pre_place", "rm75_app.runtime.sam6d_pick_place"),
    ),
)


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
        modules=(
            "rm75_app.tasks.pickplace",
            "rm75_app.tasks.manipulation_plan",
            "rm75_app.pickplace.config",
        ),
    ),
    PickPlaceLayer(
        key="perception",
        responsibility="FoundationPose、SAM3/SAM6D、RRTrack 和未见物体动态几何",
        modules=(
            "rm75_app.perception.sam6d_pose_provider",
            "rm75_app.perception.rrtrack.tracker",
            "rm75_app.perception.openworld_geometry.session",
            "rm75_app.pickplace.cached_scene",
        ),
    ),
    PickPlaceLayer(
        key="placement",
        responsibility="目标对象、放置规则和候选目标姿态",
        modules=(
            "rm75_app.placement.place_rules",
            "rm75_app.pickplace.atom_task_builder",
            "rm75_app.pickplace.coordinator",
        ),
    ),
    PickPlaceLayer(
        key="planning",
        responsibility="IK、候选配对、cuRobo 和短直线约束段",
        modules=(
            "rm75_app.planning.contracts",
            "rm75_app.planning.interfaces",
            "rm75_app.planning.backends.curobo2",
            "rm75_app.planning.dynamic_geometry_world",
        ),
    ),
    PickPlaceLayer(
        key="execution",
        responsibility="仿真/真机运动、夹爪动作和结果校验",
        modules=(
            "rm75_app.execution.trajectory_executor",
            "rm75_app.execution.maniskill_task_bridge",
        ),
    ),
    PickPlaceLayer(
        key="orchestration",
        responsibility="显式组织任务依赖、三级验证、抓取、附着、搬运、释放和撤退状态边界",
        modules=(
            "rm75_app.orchestration.multi_object_executor",
            "rm75_app.validation.three_gate",
            "rm75_app.pickplace.multi_object_adapter",
            "rm75_app.pickplace.coordinator",
            "rm75_app.runtime.curobo2_pick_place",
        ),
    ),
)

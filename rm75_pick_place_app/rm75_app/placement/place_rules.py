from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from rm75_app.assets.object_specs import normalize_object_name

VIRTUAL_PLACE_TARGET_NAMES = {"clean_table"}


def is_virtual_place_target(name: str | None) -> bool:
    normalized = normalize_object_name(name)
    return normalized in VIRTUAL_PLACE_TARGET_NAMES


@dataclass(frozen=True)
class LocalPoseSpec:
    position: tuple[float, float, float]
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PlaceSlotSpec:
    name: str
    object_pose_local: LocalPoseSpec


@dataclass(frozen=True)
class GraspBiasVariant:
    axis_shift_m: float = 0.0
    tilt_toward_robot_deg: float = 0.0
    tilt_direction: str = "toward_robot"
    tilt_shift_m: float = 0.0
    z_lift_m: float = 0.0
    label: str | None = None


@dataclass(frozen=True)
class PlaceRule:
    source_object_name: str
    target_object_name: str
    primitive: str
    hover_height: float = 0.06
    release_retreat_height: float = 0.08
    preserve_long_axis_vertical: bool = False
    orientation_invariant: bool = False
    allow_tabletop_yaw_variants: bool = False
    world_tabletop_yaw_variants: bool = False
    allow_long_axis_flip: bool = False
    face_robot_axis_local: tuple[float, float, float] | None = None
    tabletop_axial_spin_deg: tuple[float, ...] = ()
    tabletop_place_tcp_verticality_target: float | None = None
    tabletop_place_tcp_axis_vertical: str | None = None
    insert_axis_local: tuple[float, float, float] | None = None
    insert_spin_axis_local: tuple[float, float, float] | None = None
    object_pose_local: LocalPoseSpec | None = None
    slots: tuple[PlaceSlotSpec, ...] = ()
    grasp_bias_variants: tuple[GraspBiasVariant, ...] = ()


def _normalize_vec(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        return None
    return (arr / norm).astype(np.float32)


DESK_SLOT_LAYOUT_XZ: tuple[tuple[str, tuple[float, float]], ...] = (
    ("slot_1", (0.14, 0.07)),
    ("slot_2", (0.00, 0.07)),
    ("slot_3", (-0.14, 0.07)),
    ("slot_4", (0.14, -0.07)),
    ("slot_5", (0.00, -0.07)),
    ("slot_6", (-0.14, -0.07)),
)


def get_runtime_slot_specs(
    target_object_name: str | None,
    slots: Iterable[PlaceSlotSpec],
    T_world_target: np.ndarray | None,
    robot_base_p: np.ndarray | None,
) -> list[PlaceSlotSpec]:
    slots = list(slots or [])
    if normalize_object_name(target_object_name) != "desk" or len(slots) != 6:
        return slots
    if T_world_target is None or robot_base_p is None:
        return slots

    T_world_target = np.asarray(T_world_target, dtype=np.float32).reshape(4, 4)
    robot_base_p = np.asarray(robot_base_p, dtype=np.float32).reshape(3)
    target_up_axis = _normalize_vec(T_world_target[:3, 1])
    if target_up_axis is None:
        return slots

    annotated = []
    for slot in slots:
        local_p = np.asarray(slot.object_pose_local.position, dtype=np.float32).reshape(3)
        world_p = (T_world_target[:3, :3] @ local_p) + T_world_target[:3, 3]
        distance_xy = float(np.linalg.norm(world_p[:2] - robot_base_p[:2]))
        annotated.append((slot, world_p.astype(np.float32), distance_xy))

    if len(annotated) != 6:
        return slots

    annotated.sort(key=lambda item: item[2], reverse=True)
    far_row = annotated[:3]
    near_row = annotated[3:]
    far_center = np.mean([item[1] for item in far_row], axis=0).astype(np.float32)
    near_center = np.mean([item[1] for item in near_row], axis=0).astype(np.float32)

    front_axis = _normalize_vec(far_center - near_center)
    if front_axis is None:
        target_row_axis = T_world_target[:3, 2] - float(np.dot(T_world_target[:3, 2], target_up_axis)) * target_up_axis
        front_axis = _normalize_vec(target_row_axis)
    if front_axis is None:
        return slots

    right_axis = _normalize_vec(np.cross(target_up_axis, front_axis))
    if right_axis is None:
        target_col_axis = T_world_target[:3, 0] - float(np.dot(T_world_target[:3, 0], target_up_axis)) * target_up_axis
        right_axis = _normalize_vec(target_col_axis)
    if right_axis is None:
        return slots

    def _sort_row(row_items, row_center):
        return sorted(
            row_items,
            key=lambda item: -float(np.dot(item[1] - row_center, right_axis)),
        )

    far_sorted = _sort_row(far_row, far_center)
    near_sorted = _sort_row(near_row, near_center)
    runtime_order = far_sorted + near_sorted

    remapped = []
    for idx, (slot, _, _) in enumerate(runtime_order, start=1):
        remapped.append(
            PlaceSlotSpec(
                name=f"slot_{idx}",
                object_pose_local=slot.object_pose_local,
            )
        )
    return remapped


def make_slot_grid_rule(
    source_object_name: str,
    target_object_name: str,
    slot_specs: Iterable[tuple[str, tuple[float, float, float], tuple[float, float, float] | None]],
    *,
    hover_height: float = 0.06,
    release_retreat_height: float = 0.08,
    preserve_long_axis_vertical: bool = False,
    orientation_invariant: bool = False,
) -> PlaceRule:
    slots = []
    for slot_name, position, rpy_deg in slot_specs:
        slots.append(
            PlaceSlotSpec(
                name=str(slot_name),
                object_pose_local=LocalPoseSpec(
                    position=tuple(float(v) for v in position),
                    rpy_deg=tuple(float(v) for v in (rpy_deg or (0.0, 0.0, 0.0))),
                ),
            )
        )
    return PlaceRule(
        source_object_name=source_object_name,
        target_object_name=target_object_name,
        primitive="place_on_slots",
        hover_height=float(hover_height),
        release_retreat_height=float(release_retreat_height),
        preserve_long_axis_vertical=bool(preserve_long_axis_vertical),
        orientation_invariant=bool(orientation_invariant),
        slots=tuple(slots),
    )


def make_tabletop_slot_rule(
    source_object_name: str,
    *,
    target_object_name: str = "desk",
    center_y: float,
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    hover_height: float = 0.08,
    release_retreat_height: float = 0.10,
    preserve_long_axis_vertical: bool = False,
    orientation_invariant: bool = False,
) -> PlaceRule:
    slot_specs = [
        (slot_name, (float(x), float(center_y), float(z)), rpy_deg)
        for slot_name, (x, z) in DESK_SLOT_LAYOUT_XZ
    ]
    return make_slot_grid_rule(
        source_object_name=source_object_name,
        target_object_name=target_object_name,
        slot_specs=slot_specs,
        hover_height=hover_height,
        release_retreat_height=release_retreat_height,
        preserve_long_axis_vertical=preserve_long_axis_vertical,
        orientation_invariant=orientation_invariant,
    )


def make_vertical_long_axis_grasp_bias_variants() -> tuple[GraspBiasVariant, ...]:
    # The direct cuRobo script treats these signed values as legacy magnitudes
    # for top-biased grasps and remaps their final sign from the target vertical
    # placement pose, so "top" follows the end that will point upward.
    return (
        GraspBiasVariant(axis_shift_m=0.000, tilt_toward_robot_deg=0.0, z_lift_m=0.000, label="top_bias_center_vertical"),
        GraspBiasVariant(axis_shift_m=-0.002, tilt_toward_robot_deg=0.0, z_lift_m=0.000, label="top_bias_neg2_vertical"),
        GraspBiasVariant(axis_shift_m=-0.006, tilt_toward_robot_deg=0.0, z_lift_m=0.000, label="top_bias_neg6_vertical"),
        GraspBiasVariant(axis_shift_m=-0.010, tilt_toward_robot_deg=0.0, z_lift_m=0.000, label="top_bias_neg10_vertical"),
        GraspBiasVariant(axis_shift_m=-0.014, tilt_toward_robot_deg=0.0, z_lift_m=0.000, label="top_bias_neg14_vertical"),
        GraspBiasVariant(axis_shift_m=-0.014, tilt_toward_robot_deg=15.0, tilt_direction="away_robot", tilt_shift_m=0.0, label="top_bias_neg14_tilt15_away"),
        GraspBiasVariant(axis_shift_m=-0.010, tilt_toward_robot_deg=15.0, tilt_direction="away_robot", tilt_shift_m=0.0, label="top_bias_neg10_tilt15_away"),
        GraspBiasVariant(axis_shift_m=-0.006, tilt_toward_robot_deg=15.0, tilt_direction="away_robot", tilt_shift_m=0.0, label="top_bias_neg6_tilt15_away"),
        GraspBiasVariant(axis_shift_m=-0.002, tilt_toward_robot_deg=15.0, tilt_direction="away_robot", tilt_shift_m=0.0, label="top_bias_neg2_tilt15_away"),
        GraspBiasVariant(axis_shift_m=-0.014, tilt_toward_robot_deg=25.0, tilt_direction="away_robot", tilt_shift_m=0.0, label="top_bias_neg14_tilt25_away"),
        GraspBiasVariant(axis_shift_m=-0.010, tilt_toward_robot_deg=25.0, tilt_direction="away_robot", tilt_shift_m=0.0, label="top_bias_neg10_tilt25_away"),
        GraspBiasVariant(axis_shift_m=-0.006, tilt_toward_robot_deg=25.0, tilt_direction="away_robot", tilt_shift_m=0.0, label="top_bias_neg6_tilt25_away"),
        GraspBiasVariant(axis_shift_m=-0.002, tilt_toward_robot_deg=25.0, tilt_direction="away_robot", tilt_shift_m=0.0, label="top_bias_neg2_tilt25_away"),
    )


def make_pen_insert_grasp_bias_variants() -> tuple[GraspBiasVariant, ...]:
    # Keep the pen grasp centered on the raw target pose.  Axis-shifted picks
    # changed the TCP<->pen frame and made downstream insertion expensive to
    # validate.  Small tilt-only variants preserve the target point while giving
    # cuRobo extra IK branches around the same physical grasp.
    return (
        GraspBiasVariant(axis_shift_m=0.000, tilt_toward_robot_deg=0.0, z_lift_m=0.000, label="pen_raw"),
        GraspBiasVariant(axis_shift_m=0.000, tilt_toward_robot_deg=12.0, tilt_direction="toward_robot", z_lift_m=0.000, label="pen_tilt_toward_12deg"),
        GraspBiasVariant(axis_shift_m=0.000, tilt_toward_robot_deg=12.0, tilt_direction="away_robot", z_lift_m=0.000, label="pen_tilt_away_12deg"),
    )


def make_tingzi_pillar_insert_rule(
    source_object_name: str,
    *,
    x_m: float,
    y_m: float,
) -> PlaceRule:
    return PlaceRule(
        source_object_name=source_object_name,
        target_object_name="tingzi_base",
        primitive="insert_vertical",
        hover_height=0.12,
        release_retreat_height=0.12,
        preserve_long_axis_vertical=True,
        insert_axis_local=(0.0, 0.0, 1.0),
        insert_spin_axis_local=(0.0, 0.0, 1.0),
        object_pose_local=LocalPoseSpec(
            # The base OBJ uses local Z as the physical vertical axis.  The
            # pillar OBJ origin is near the lower plug, so placing the origin
            # at the base top surface seats the plug into the corner socket.
            position=(float(x_m), float(y_m), 0.014),
            rpy_deg=(0.0, 0.0, 0.0),
        ),
    )


def apply_vertical_long_axis_rule_overrides(
    rule: PlaceRule,
    *,
    face_robot_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> PlaceRule:
    return PlaceRule(
        **{
            **rule.__dict__,
            "allow_tabletop_yaw_variants": True,
            "face_robot_axis_local": face_robot_axis_local,
            "grasp_bias_variants": make_vertical_long_axis_grasp_bias_variants(),
        }
    )


PLACE_RULES: Dict[str, PlaceRule] = {
    "bi": PlaceRule(
        source_object_name="bi",
        target_object_name="bitong",
        primitive="insert_vertical",
        hover_height=0.15,
        release_retreat_height=0.15,
        allow_long_axis_flip=True,
        grasp_bias_variants=make_pen_insert_grasp_bias_variants(),
        # Desired source-object pose in the target object's local frame.
        # Both pen.glb and holder.glb currently use their longest local axis as +Y,
        # so identity orientation is a reasonable first rule for vertical insertion.
        object_pose_local=LocalPoseSpec(
            position=(0.0, 0.09, 0.0),
            rpy_deg=(0.0, 0.0, 0.0),
        ),
    ),
    "shuazi": make_tabletop_slot_rule(
        "shuazi",
        center_y=0.074,
        rpy_deg=(-90.0, 90.0, 0.0),
        hover_height=0.08,
        release_retreat_height=0.10,
    ),
    "hongshupian": make_tabletop_slot_rule(
        "hongshupian",
        center_y=0.127,
        rpy_deg=(0.0, 0.0, 0.0),
        hover_height=0.12,
        release_retreat_height=0.12,
        preserve_long_axis_vertical=True,
    ),
    "lvmukuai": make_tabletop_slot_rule(
        "lvmukuai",
        center_y=0.07,
        rpy_deg=(90.0, 90.0, 0.0),
        hover_height=0.08,
        release_retreat_height=0.10,
    ),
    "carriot": make_tabletop_slot_rule(
        "carriot",
        center_y=0.070,
        # The carrot mesh is flat on the table with local +Y as the table
        # normal in the captured scenes. Keep the long axis at 45deg in world
        # XY, but do not swap the down-facing mesh axis; otherwise the shared
        # TCP-object relation forces a horizontal/upside-down gripper at one
        # end of the grasp-place chain.
        rpy_deg=(180.0, 0.0, 90.0),
        hover_height=0.08,
        release_retreat_height=0.10,
    ),
    "gluestick": make_tabletop_slot_rule(
        "gluestick",
        center_y=0.10,
        rpy_deg=(0.0, 0.0, 0.0),
        hover_height=0.12,
        release_retreat_height=0.12,
        preserve_long_axis_vertical=True,
    ),
    "tennis": make_tabletop_slot_rule(
        "tennis",
        center_y=0.086,
        rpy_deg=(0.0, 0.0, 0.0),
        hover_height=0.10,
        release_retreat_height=0.10,
        orientation_invariant=True,
    ),
    # ---------------  roof assembly  ---------------
    "red_triangle_front": PlaceRule(
        source_object_name="red_triangle_front",
        target_object_name="red_bricks_cube",
        primitive="roof_assembly",
    ),
    "red_triangle_back": PlaceRule(
        source_object_name="red_triangle_back",
        target_object_name="red_bricks_cube",
        primitive="roof_assembly",
    ),
    "red_triangle_left": PlaceRule(
        source_object_name="red_triangle_left",
        target_object_name="red_bricks_cube",
        primitive="roof_assembly",
    ),
    "red_triangle_right": PlaceRule(
        source_object_name="red_triangle_right",
        target_object_name="red_bricks_cube",
        primitive="roof_assembly",
    ),
    # ---------------  pavilion base + pillars  ---------------
    "tingzi_base": PlaceRule(
        source_object_name="tingzi_base",
        target_object_name="clean_table",
        primitive="place_on_slots",
        hover_height=0.08,
        release_retreat_height=0.10,
        allow_tabletop_yaw_variants=False,
        world_tabletop_yaw_variants=True,
        slots=(
            PlaceSlotSpec(
                name="tingzi_base_center",
                object_pose_local=LocalPoseSpec(
                    position=(0.0, 0.006, 0.0),
                    # Fusion OBJ uses +Z as up.  Table placement rules use the
                    # target local +Y as up, so rotate OBJ +Z onto target +Y.
                    rpy_deg=(-90.0, 0.0, 0.0),
                ),
            ),
        ),
    ),
    "tingzi_pillar_front_left": make_tingzi_pillar_insert_rule(
        "tingzi_pillar_front_left",
        x_m=-0.0248,
        y_m=-0.0248,
    ),
    "tingzi_pillar_front_right": make_tingzi_pillar_insert_rule(
        "tingzi_pillar_front_right",
        x_m=0.0248,
        y_m=-0.0248,
    ),
    "tingzi_pillar_back_left": make_tingzi_pillar_insert_rule(
        "tingzi_pillar_back_left",
        x_m=-0.0248,
        y_m=0.0248,
    ),
    "tingzi_pillar_back_right": make_tingzi_pillar_insert_rule(
        "tingzi_pillar_back_right",
        x_m=0.0248,
        y_m=0.0248,
    ),
}

PLACE_RULES["gluestick"] = apply_vertical_long_axis_rule_overrides(
    PLACE_RULES["gluestick"],
    face_robot_axis_local=(0.0, 0.0, -1.0),
)
PLACE_RULES["hongshupian"] = apply_vertical_long_axis_rule_overrides(
    PLACE_RULES["hongshupian"],
    face_robot_axis_local=(0.0, 0.0, 1.0),
)
PLACE_RULES["lvmukuai"] = PlaceRule(
    **{
        **PLACE_RULES["lvmukuai"].__dict__,
        "allow_tabletop_yaw_variants": False,
        "tabletop_axial_spin_deg": (),
    }
)
PLACE_RULES["carriot"] = PlaceRule(
    **{
        **PLACE_RULES["carriot"].__dict__,
        "allow_tabletop_yaw_variants": False,
    }
)
PLACE_RULES["shuazi"] = PlaceRule(
    **{
        **PLACE_RULES["shuazi"].__dict__,
        "allow_tabletop_yaw_variants": False,
    }
)
PLACE_RULES["tennis"] = PlaceRule(
    **{
        **PLACE_RULES["tennis"].__dict__,
        "allow_tabletop_yaw_variants": True,
    }
)


def get_place_rule(source_object_name: str | None) -> PlaceRule | None:
    normalized = normalize_object_name(source_object_name)
    if normalized is None:
        return None
    return PLACE_RULES.get(normalized)


def _local_pose_spec_from_data(data) -> LocalPoseSpec:
    if isinstance(data, LocalPoseSpec):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"LocalPoseSpec must be a dict, got {type(data).__name__}")
    position = data.get("position")
    if position is None:
        raise ValueError("LocalPoseSpec is missing required field: position")
    rpy_deg = data.get("rpy_deg", (0.0, 0.0, 0.0))
    return LocalPoseSpec(
        position=tuple(float(v) for v in position),
        rpy_deg=tuple(float(v) for v in rpy_deg),
    )


def _slot_spec_from_data(data) -> PlaceSlotSpec:
    if isinstance(data, PlaceSlotSpec):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"PlaceSlotSpec must be a dict, got {type(data).__name__}")
    name = str(data.get("name") or "llm_goal")
    return PlaceSlotSpec(name=name, object_pose_local=_local_pose_spec_from_data(data.get("object_pose_local")))


def _float_tuple_or_none(value):
    if value is None:
        return None
    return tuple(float(v) for v in value)


def place_rule_from_data(data: dict, *, base_rule: PlaceRule | None = None) -> PlaceRule:
    if not isinstance(data, dict):
        raise ValueError(f"dynamic place rule must be a dict, got {type(data).__name__}")
    source = normalize_object_name(data.get("source_object_name"))
    target = normalize_object_name(data.get("target_object_name"))
    primitive = str(data.get("primitive") or (base_rule.primitive if base_rule else ""))
    if source is None:
        raise ValueError("dynamic place rule is missing source_object_name")
    if target is None:
        raise ValueError("dynamic place rule is missing target_object_name")
    if not primitive:
        raise ValueError("dynamic place rule is missing primitive")

    values = dict(base_rule.__dict__) if base_rule is not None else {}
    values.update(
        {
            "source_object_name": source,
            "target_object_name": target,
            "primitive": primitive,
        }
    )
    for key in (
        "hover_height",
        "release_retreat_height",
        "preserve_long_axis_vertical",
        "orientation_invariant",
        "allow_tabletop_yaw_variants",
        "world_tabletop_yaw_variants",
        "allow_long_axis_flip",
        "tabletop_place_tcp_verticality_target",
        "tabletop_place_tcp_axis_vertical",
        "insert_axis_local",
        "insert_spin_axis_local",
    ):
        if key in data:
            values[key] = data[key]
    if "insert_axis_local" in data:
        values["insert_axis_local"] = _float_tuple_or_none(data.get("insert_axis_local"))
    if "insert_spin_axis_local" in data:
        values["insert_spin_axis_local"] = _float_tuple_or_none(data.get("insert_spin_axis_local"))
    if "face_robot_axis_local" in data:
        values["face_robot_axis_local"] = _float_tuple_or_none(data.get("face_robot_axis_local"))
    if "tabletop_axial_spin_deg" in data:
        values["tabletop_axial_spin_deg"] = tuple(float(v) for v in data.get("tabletop_axial_spin_deg") or [])
    if "object_pose_local" in data:
        values["object_pose_local"] = _local_pose_spec_from_data(data.get("object_pose_local"))
    if "slots" in data:
        values["slots"] = tuple(_slot_spec_from_data(item) for item in data.get("slots") or [])
    return PlaceRule(**values)


def install_dynamic_place_rules_from_file(path: str | Path | None) -> list[str]:
    if not path:
        return []
    rule_path = Path(path).expanduser().resolve()
    data = json.loads(rule_path.read_text(encoding="utf-8"))
    raw_rules = data.get("rules") if isinstance(data, dict) and "rules" in data else data
    if isinstance(raw_rules, dict):
        raw_rules = [raw_rules]
    if not isinstance(raw_rules, list):
        raise ValueError(f"dynamic place rule file must contain a rule dict or rules list: {rule_path}")
    installed: list[str] = []
    for raw in raw_rules:
        source = normalize_object_name(raw.get("source_object_name")) if isinstance(raw, dict) else None
        base_rule = PLACE_RULES.get(source) if source is not None else None
        rule = place_rule_from_data(raw, base_rule=base_rule)
        PLACE_RULES[rule.source_object_name] = rule
        installed.append(rule.source_object_name)
    return installed


def list_place_rule_sources() -> list[str]:
    return sorted(PLACE_RULES.keys())


def get_required_place_target_names(source_names) -> list[str]:
    target_names: list[str] = []
    for source_name in list(source_names or []):
        rule = get_place_rule(source_name)
        target_name = normalize_object_name(rule.target_object_name) if rule is not None else None
        if is_virtual_place_target(target_name):
            continue
        if target_name is not None and target_name not in target_names:
            target_names.append(target_name)
    return target_names


def describe_place_rules() -> str:
    lines = []
    for source_name in list_place_rule_sources():
        rule = PLACE_RULES[source_name]
        slot_names = [slot.name for slot in rule.slots]
        lines.append(
            f"- {source_name}: primitive={rule.primitive}, "
            f"target={rule.target_object_name}, "
            f"hover_height={rule.hover_height}, "
            f"release_retreat_height={rule.release_retreat_height}, "
            f"slots={slot_names}"
        )
    return "\n".join(lines)

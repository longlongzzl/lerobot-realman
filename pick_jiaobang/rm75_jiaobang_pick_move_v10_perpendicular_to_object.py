#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mani_skill  # noqa: F401
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
import mplib
import numpy as np
import sapien
from transforms3d import quaternions

ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
GRIPPER_ACTIVE_NAMES = ["gripper_Left_1_Joint", "gripper_Right_1_Joint"]
GRIPPER_PASSIVE_NAMES = [
    "gripper_Left_2_Joint",
    "gripper_Right_2_Joint",
    "gripper_Left_Support_Joint",
    "gripper_Right_Support_Joint",
]
ALL_GRIPPER_JOINT_NAMES = set(GRIPPER_ACTIVE_NAMES + GRIPPER_PASSIVE_NAMES)


def flatten(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def find_existing_urdf(default_path=None):
    if default_path and os.path.exists(default_path):
        return default_path
    cands = [
        os.path.join(os.getcwd(), "RM75-B.urdf"),
        os.path.expanduser("~/.maniskill/data/robots/RM75_gripper/RM75-B/urdf/RM75-B.urdf"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot locate RM75-B.urdf. Pass --urdf-path explicitly.")


def tiny_box_size_for_link(link_name: str) -> np.ndarray:
    lname = link_name.lower()
    if lname == "gripper_tcp":
        return np.array([0.001, 0.001, 0.001], dtype=np.float64)
    if "pad" in lname:
        return np.array([0.005, 0.005, 0.005], dtype=np.float64)
    if "support" in lname or "gripper" in lname or "flange" in lname:
        return np.array([0.005, 0.005, 0.005], dtype=np.float64)
    if lname in {"link_6", "link_7"}:
        return np.array([0.010, 0.010, 0.010], dtype=np.float64)
    return np.array([0.008, 0.008, 0.008], dtype=np.float64)


def generate_near_collision_free_planning_urdf(source_urdf_path: str, planning_urdf_path: str) -> None:
    src_root = ET.parse(source_urdf_path).getroot()
    dst_root = ET.Element("robot", {"name": src_root.attrib.get("name", "RM75_planning_tiny")})
    for link in src_root.findall("link"):
        lname = link.attrib["name"]
        new_link = ET.SubElement(dst_root, "link", {"name": lname})
        has_geom = (link.find("visual") is not None) or (link.find("collision") is not None)
        has_inertial = link.find("inertial") is not None
        if (not has_geom and not has_inertial) or lname == "gripper_tcp":
            continue
        inertial = ET.SubElement(new_link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": "0.001"})
        ET.SubElement(
            inertial,
            "inertia",
            {"ixx": "1e-7", "ixy": "0", "ixz": "0", "iyy": "1e-7", "iyz": "0", "izz": "1e-7"},
        )
        size = tiny_box_size_for_link(lname)
        size_str = f"{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}"
        visual = ET.SubElement(new_link, "visual")
        ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        vgeom = ET.SubElement(visual, "geometry")
        ET.SubElement(vgeom, "box", {"size": size_str})
        collision = ET.SubElement(new_link, "collision")
        ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        cgeom = ET.SubElement(collision, "geometry")
        ET.SubElement(cgeom, "box", {"size": size_str})

    for joint in src_root.findall("joint"):
        attrs = dict(joint.attrib)
        if attrs.get("name") in ALL_GRIPPER_JOINT_NAMES:
            attrs["type"] = "fixed"
        new_joint = ET.SubElement(dst_root, "joint", attrs)
        for child in list(joint):
            if attrs.get("type") == "fixed" and child.tag in {"axis", "limit", "dynamics", "mimic", "calibration", "safety_controller"}:
                continue
            new_joint.append(child)

    ET.indent(dst_root, space="  ")
    ET.ElementTree(dst_root).write(planning_urdf_path, encoding="utf-8", xml_declaration=True)


def resolve_planning_artifact_paths(sim_urdf_path: str, args):
    sim_urdf_path = os.path.abspath(sim_urdf_path)
    sim_urdf_dir = os.path.dirname(sim_urdf_path)
    sim_urdf_stem = Path(sim_urdf_path).stem

    planning_urdf_path = os.path.join(sim_urdf_dir, f"{sim_urdf_stem}.planning.tiny.urdf")
    if args.srdf_path is not None:
        srdf_path = os.path.abspath(args.srdf_path)
    else:
        srdf_path = os.path.join(sim_urdf_dir, f"{sim_urdf_stem}.permissive.srdf")

    return planning_urdf_path, srdf_path


def parse_link_names(urdf_path: str):
    root = ET.parse(urdf_path).getroot()
    return [x.attrib["name"] for x in root.findall("link")]


def write_permissive_srdf(urdf_path: str, srdf_path: str):
    root = ET.parse(urdf_path).getroot()
    link_names = [x.attrib["name"] for x in root.findall("link")]
    adjacent = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            a = parent.attrib["link"]
            b = child.attrib["link"]
            adjacent.add(tuple(sorted((a, b))))

    lines = ['<?xml version="1.0" ?>', '<robot name="RM75-B">', '  <group name="rm75_arm">']
    for j in ARM_JOINT_NAMES:
        lines.append(f'    <joint name="{j}"/>')
    lines += [
        '  </group>',
        '  <group name="tool_end">',
        '    <joint name="gripper_tcp_joint"/>',
        '  </group>',
        '  <end_effector group="tool_end" name="end_effector" parent_link="gripper_base_link"/>',
    ]
    for a, b in sorted(adjacent):
        lines.append(f'  <disable_collisions link1="{a}" link2="{b}" reason="Adjacent"/>')
    hand_like = [n for n in link_names if any(k in n.lower() for k in ["gripper", "pad", "flange"])]
    arm_like = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6", "link_7"]
    for i in range(len(hand_like)):
        for j in range(i + 1, len(hand_like)):
            a, b = sorted((hand_like[i], hand_like[j]))
            lines.append(f'  <disable_collisions link1="{a}" link2="{b}" reason="DebugPermissive"/>')
    for a in arm_like:
        for b in hand_like:
            x, y = sorted((a, b))
            lines.append(f'  <disable_collisions link1="{x}" link2="{y}" reason="DebugPermissive"/>')
    lines.append('</robot>\n')
    Path(srdf_path).write_text("\n".join(lines), encoding="utf-8")


def build_official_style_grasp_pose_visual(scene):
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05
    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[0.03 - grasp_pose_visual_width * 3, grasp_width + grasp_pose_visual_width, 0.03 - 0.05],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[0.03 - grasp_pose_visual_width * 3, -grasp_width - grasp_pose_visual_width, 0.03 - 0.05],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    return builder.build_kinematic(name="grasp_pose_visual")


def quat_angle_deg(q1, q2):
    q1 = flatten(q1)[:4]
    q2 = flatten(q2)[:4]
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    dot = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
    angle = 2.0 * np.arccos(dot)
    return float(np.degrees(angle))


def normalize(v, eps=1e-8):
    v = flatten(v)
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def quat2mat_np(q):
    return quaternions.quat2mat(flatten(q)[:4])


def mat2quat_np(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    t = np.trace(m)

    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    return q


class RM75JiaobangPickMove:
    def __init__(self, env, urdf_path: str, srdf_path: str, args):
        self.env = env
        self.base_env = env.unwrapped
        self.robot = self.base_env.agent.robot
        self.tcp = self.base_env.agent.tcp
        self.tip_link_name = "gripper_tcp"
        self.action_dim = int(np.prod(env.action_space.shape))
        self.control_timestep = 1.0 / 20.0
        self.args = args

        self.link_names = parse_link_names(urdf_path)
        self.active_joint_names = [j.get_name() for j in self.robot.get_active_joints()]
        self.arm_indices = [self.active_joint_names.index(n) for n in ARM_JOINT_NAMES]

        self.planner = mplib.Planner(
            urdf=urdf_path,
            srdf=srdf_path,
            user_link_names=self.link_names,
            user_joint_names=ARM_JOINT_NAMES,
            move_group=self.tip_link_name,
            joint_vel_limits=np.ones(7, dtype=np.float64),
            joint_acc_limits=np.ones(7, dtype=np.float64),
        )
        self.grasp_pose_visual = None

        self.step_counter = 0
        self.left_finger_link = None
        self.right_finger_link = None
        self.last_obs = None
        self.last_reward = None
        self.last_terminated = None
        self.last_truncated = None
        self.last_info = {}
        self.refresh_runtime_handles(rebuild_visual=True)


    def refresh_runtime_handles(self, rebuild_visual=True):
        self.base_env = self.env.unwrapped
        self.robot = self.base_env.agent.robot
        self.tcp = self.base_env.agent.tcp
        self.active_joint_names = [j.get_name() for j in self.robot.get_active_joints()]
        self.arm_indices = [self.active_joint_names.index(n) for n in ARM_JOINT_NAMES]

        base_pose = self.robot.pose
        self.planner.set_base_pose(np.hstack([flatten(base_pose.p)[:3], flatten(base_pose.q)[:4]]))

        if rebuild_visual or self.grasp_pose_visual is None:
            self.grasp_pose_visual = build_official_style_grasp_pose_visual(self.base_env.scene)
        try:
            self.grasp_pose_visual.set_pose(self.tcp.pose)
        except Exception:
            self.grasp_pose_visual = build_official_style_grasp_pose_visual(self.base_env.scene)
            self.grasp_pose_visual.set_pose(self.tcp.pose)

        self._init_contact_logging_targets()

    def reset_episode(self, seed):
        obs, info = self.env.reset(seed=seed)
        self.step_counter = 0
        self.last_obs = obs
        self.last_reward = None
        self.last_terminated = False
        self.last_truncated = False
        self.last_info = info if isinstance(info, dict) else {}
        self.refresh_runtime_handles(rebuild_visual=True)
        return obs, info
    def _get_robot_links(self):
        try:
            return list(self.robot.get_links())
        except Exception:
            return []

    def _safe_entity_name(self, entity):
        if entity is None:
            return None
        for attr in ["name", "get_name"]:
            try:
                value = getattr(entity, attr)
                if callable(value):
                    value = value()
                if isinstance(value, str) and len(value) > 0:
                    return value
            except Exception:
                pass
        return None

    def _link_name_matches_side(self, name, side):
        lname = (name or "").lower()
        if side == "left" and "left" not in lname:
            return False
        if side == "right" and "right" not in lname:
            return False
        return any(k in lname for k in ["gripper", "finger", "pad"])

    def _pick_finger_link(self, side):
        links = self._get_robot_links()
        candidates = []
        for link in links:
            name = self._safe_entity_name(link)
            if self._link_name_matches_side(name, side):
                score = 0
                lname = name.lower()
                for k, w in [("pad", 4), ("finger", 3), ("gripper", 2), ("_1_", 1), ("_2_", 1)]:
                    if k in lname:
                        score += w
                candidates.append((score, len(lname), name, link))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        return candidates[0][3]

    def _init_contact_logging_targets(self):
        self.left_finger_link = self._pick_finger_link("left")
        self.right_finger_link = self._pick_finger_link("right")
        if getattr(self.args, "log_contact_forces", False):
            left_name = self._safe_entity_name(self.left_finger_link)
            right_name = self._safe_entity_name(self.right_finger_link)
            print("[force-log] left finger link :", left_name)
            print("[force-log] right finger link:", right_name)

    def _get_scene_contacts(self):
        scene = getattr(self.base_env, "scene", None)
        if scene is None:
            return []
        for fn_name in ["get_contacts", "get_contacts_with_impulse"]:
            fn = getattr(scene, fn_name, None)
            if callable(fn):
                try:
                    return list(fn())
                except Exception:
                    pass
        return []

    def _unwrap_scene_entity(self, entity):
        if entity is None:
            return None
        for attr in ["_objs", "_obj", "entity", "_body", "body", "_actor", "actor"]:
            if hasattr(entity, attr):
                try:
                    value = getattr(entity, attr)
                    if value is not None:
                        return value
                except Exception:
                    pass
        return entity

    def _get_pairwise_force_vector(self, a, b):
        scene = getattr(self.base_env, "scene", None)
        if scene is None:
            return None
        fn = getattr(scene, "get_pairwise_contact_forces", None)
        if not callable(fn):
            return None

        candidates_a = [a, self._unwrap_scene_entity(a)]
        candidates_b = [b, self._unwrap_scene_entity(b)]
        tried = set()
        for aa in candidates_a:
            for bb in candidates_b:
                if aa is None or bb is None:
                    continue
                key = (id(aa), id(bb))
                if key in tried:
                    continue
                tried.add(key)
                try:
                    force = fn(aa, bb)
                    if force is None:
                        continue
                    force = np.asarray(force, dtype=np.float64).reshape(-1)[:3]
                    return force
                except Exception:
                    continue
        return None

    def _get_contact_actors(self, contact):
        actor0 = getattr(contact, "actor0", None)
        actor1 = getattr(contact, "actor1", None)
        if actor0 is None:
            actor0 = getattr(contact, "body0", None)
        if actor1 is None:
            actor1 = getattr(contact, "body1", None)
        return actor0, actor1

    def _same_entity(self, a, b):
        if a is None or b is None:
            return False
        if a is b:
            return True
        try:
            if hasattr(a, "entity") and a.entity is b:
                return True
            if hasattr(b, "entity") and b.entity is a:
                return True
            if hasattr(a, "entity") and hasattr(b, "entity") and a.entity is b.entity:
                return True
        except Exception:
            pass
        return False

    def _extract_force_from_contact(self, contact, finger_link, object_actor):
        actor0, actor1 = self._get_contact_actors(contact)
        if not ((self._same_entity(actor0, finger_link) and self._same_entity(actor1, object_actor)) or
                (self._same_entity(actor1, finger_link) and self._same_entity(actor0, object_actor))):
            return None

        pts = getattr(contact, "points", None)
        if pts is None:
            pts = getattr(contact, "contacts", None)
        if pts is None:
            return None

        total_force_obj = np.zeros(3, dtype=np.float64)
        total_normal_obj = np.zeros(3, dtype=np.float64)
        total_tangent_obj = np.zeros(3, dtype=np.float64)
        point_count = 0

        finger_is_actor0 = self._same_entity(actor0, finger_link)
        obj_is_actor1 = self._same_entity(actor1, object_actor)
        sign_to_object = 1.0 if (finger_is_actor0 and obj_is_actor1) else -1.0

        for pt in pts:
            impulse = getattr(pt, "impulse", None)
            if impulse is None:
                continue
            impulse = np.asarray(impulse, dtype=np.float64).reshape(-1)[:3]
            force_obj = sign_to_object * impulse / max(self.control_timestep, 1e-8)

            normal = getattr(pt, "normal", None)
            if normal is not None:
                normal = normalize(normal)
            if normal is None:
                normal_force_obj = np.zeros(3, dtype=np.float64)
                tangent_force_obj = force_obj.copy()
            else:
                normal_force_mag = np.dot(force_obj, normal)
                normal_force_obj = normal_force_mag * normal
                tangent_force_obj = force_obj - normal_force_obj

            total_force_obj += force_obj
            total_normal_obj += normal_force_obj
            total_tangent_obj += tangent_force_obj
            point_count += 1

        if point_count == 0:
            return None

        return {
            "force_obj": total_force_obj,
            "normal_obj": total_normal_obj,
            "tangent_obj": total_tangent_obj,
            "point_count": point_count,
        }

    def _project_force_components(self, force_vec, closing_dir, vertical_dir):
        force_vec = np.asarray(force_vec, dtype=np.float64).reshape(-1)[:3]
        closing_scalar = float(np.dot(force_vec, closing_dir))
        vertical_scalar = float(np.dot(force_vec, vertical_dir))
        return closing_scalar, vertical_scalar

    def log_grasp_contact_forces(self, tag=""):
        obj = getattr(self.base_env, "obj", None)
        if obj is None:
            print(f"[force-log][{tag}] object not found")
            return

        tcp_q = flatten(self.base_env.agent.tcp.pose.q)[:4]
        R_tcp = quat2mat_np(tcp_q)
        ortho_dir = normalize(R_tcp[:, 0])
        closing_dir = normalize(R_tcp[:, 1])
        approaching_dir = normalize(R_tcp[:, 2])
        vertical_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        contacts = self._get_scene_contacts()
        pairwise_fn = getattr(getattr(self.base_env, "scene", None), "get_pairwise_contact_forces", None)
        can_use_pairwise = callable(pairwise_fn)

        if (not can_use_pairwise) and len(contacts) == 0:
            print(f"[force-log][{tag}] no contacts returned by scene")
            return

        sides = [("left", self.left_finger_link), ("right", self.right_finger_link)]
        total_force_obj = np.zeros(3, dtype=np.float64)
        total_tangent_obj = np.zeros(3, dtype=np.float64)
        total_normal_obj = np.zeros(3, dtype=np.float64)
        any_hit = False

        print(f"[force-log][{tag}] tcp axes | ortho={np.round(ortho_dir, 4)} closing={np.round(closing_dir, 4)} approaching={np.round(approaching_dir, 4)}")
        print(f"[force-log][{tag}] force source = {'scene.get_pairwise_contact_forces' if can_use_pairwise else 'scene contacts fallback'}")
        for side_name, finger_link in sides:
            finger_name = self._safe_entity_name(finger_link)
            if finger_link is None:
                print(f"[force-log][{tag}] {side_name}: finger link not found")
                continue

            pair_force = None
            pair_tangent = np.zeros(3, dtype=np.float64)
            pair_normal = np.zeros(3, dtype=np.float64)
            pair_points = 0

            if can_use_pairwise:
                pair_force = self._get_pairwise_force_vector(obj, finger_link)
                if pair_force is not None:
                    pair_force = np.asarray(pair_force, dtype=np.float64).reshape(-1)[:3]
                    pair_tangent = pair_force.copy()
                    pair_points = -1

            if pair_force is None:
                pair_force = np.zeros(3, dtype=np.float64)
                for contact in contacts:
                    info = self._extract_force_from_contact(contact, finger_link, obj)
                    if info is None:
                        continue
                    pair_force += info["force_obj"]
                    pair_tangent += info["tangent_obj"]
                    pair_normal += info["normal_obj"]
                    pair_points += info["point_count"]

            if np.linalg.norm(pair_force) < 1e-9 and pair_points == 0:
                print(f"[force-log][{tag}] {side_name} ({finger_name}): no finger-object contact")
                continue

            any_hit = True
            total_force_obj += pair_force
            total_tangent_obj += pair_tangent
            total_normal_obj += pair_normal

            close_total, vert_total = self._project_force_components(pair_force, closing_dir, vertical_dir)
            close_normal, vert_normal = self._project_force_components(pair_normal, closing_dir, vertical_dir)
            close_tan, vert_tan = self._project_force_components(pair_tangent, closing_dir, vertical_dir)
            pts_text = "pairwise" if pair_points < 0 else str(pair_points)

            print(
                f"[force-log][{tag}] {side_name} ({finger_name}) | pts={pts_text} "
                f"F_obj={np.round(pair_force, 4)}N | "
                f"closing_total={close_total:.4f}N closing_normal={close_normal:.4f}N closing_tangent={close_tan:.4f}N | "
                f"vertical_total={vert_total:.4f}N vertical_normal={vert_normal:.4f}N vertical_tangent={vert_tan:.4f}N"
            )

        if not any_hit:
            print(f"[force-log][{tag}] no object contact from left/right fingers")
            return

        close_total, vert_total = self._project_force_components(total_force_obj, closing_dir, vertical_dir)
        close_normal, vert_normal = self._project_force_components(total_normal_obj, closing_dir, vertical_dir)
        close_tan, vert_tan = self._project_force_components(total_tangent_obj, closing_dir, vertical_dir)
        print(
            f"[force-log][{tag}] total_on_object | F={np.round(total_force_obj, 4)}N | "
            f"closing_total={close_total:.4f}N closing_normal={close_normal:.4f}N closing_tangent={close_tan:.4f}N | "
            f"vertical_total={vert_total:.4f}N vertical_normal={vert_normal:.4f}N vertical_tangent={vert_tan:.4f}N"
        )

    def _scalarize_bool(self, x):
        if x is None:
            return False
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        try:
            arr = np.asarray(x)
            if arr.size == 0:
                return False
            return bool(arr.reshape(-1)[0])
        except Exception:
            pass
        try:
            if hasattr(x, "item"):
                return bool(x.item())
        except Exception:
            pass
        return bool(x)

    def _extract_step_flags(self, info=None, terminated=None, truncated=None):
        if info is None:
            info = self.last_info
        if terminated is None:
            terminated = self.last_terminated
        if truncated is None:
            truncated = self.last_truncated

        success = False
        is_obj_placed = False
        is_grasped = False
        if isinstance(info, dict):
            success = self._scalarize_bool(info.get("success", False))
            is_obj_placed = self._scalarize_bool(info.get("is_obj_placed", False))
            is_grasped = self._scalarize_bool(info.get("is_grasped", False))

        terminated_flag = self._scalarize_bool(terminated)
        truncated_flag = self._scalarize_bool(truncated)
        done = terminated_flag or truncated_flag
        return {
            "success": success,
            "is_obj_placed": is_obj_placed,
            "is_grasped": is_grasped,
            "terminated": terminated_flag,
            "truncated": truncated_flag,
            "done": done,
        }

    def print_step_flags(self, prefix=""):
        flags = self._extract_step_flags()
        print(
            f"{prefix}success={flags['success']} is_obj_placed={flags['is_obj_placed']} "
            f"is_grasped={flags['is_grasped']} terminated={flags['terminated']} "
            f"truncated={flags['truncated']} done={flags['done']}"
        )
        return flags

    def current_arm_qpos(self):
        q = flatten(self.robot.get_qpos())
        return q[self.arm_indices]

    def compose_action(self, arm_target_q, gripper_value):
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[:7] = flatten(arm_target_q)[:7].astype(np.float32)
        if self.action_dim > 7:
            action[7] = np.float32(gripper_value)
        return action

    def step_and_render(self, action, tag=""):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        self.last_reward = reward
        self.last_terminated = terminated
        self.last_truncated = truncated
        self.last_info = info if isinstance(info, dict) else {}
        self.step_counter += 1
        if getattr(self.args, "log_contact_forces", False) and getattr(self.args, "log_force_every", 0) > 0:
            should_log = (self.step_counter % self.args.log_force_every == 0)
            if should_log and (tag.startswith("close") or tag.startswith("goal") or tag.startswith("lift")):
                self.log_grasp_contact_forces(tag=f"{tag}@step{self.step_counter}")
        if getattr(self.args, "render_mode", None) == "human":
            try:
                self.env.render()
            except Exception:
                pass
        render_step_sleep = getattr(self.args, "render_step_sleep", None)
        if render_step_sleep is None:
            render_step_sleep = getattr(self.args, "sim_render_sleep", 0.0)
        render_step_sleep = float(render_step_sleep or 0.0)
        if getattr(self.args, "render_mode", None) == "human" and render_step_sleep > 0.0:
            time.sleep(render_step_sleep)

    def hold_current_and_set_gripper(self, gripper_value, steps=20):
        q_hold = self.current_arm_qpos()
        action = self.compose_action(q_hold, gripper_value)
        for _ in range(steps):
            self.step_and_render(action, tag="close_gripper" if gripper_value > 0 else "open_gripper")

    def preview_target_pose(self, pose):
        self.grasp_pose_visual.set_pose(sapien.Pose(flatten(pose.p)[:3], flatten(pose.q)[:4]))

    def make_target(self, pose, variant_name):
        p = flatten(pose.p)[:3]
        q_wxyz = flatten(pose.q)[:4]
        q_xyzw = q_wxyz[[1, 2, 3, 0]]
        pymp = getattr(mplib, "pymp", None)
        PoseCls = getattr(pymp, "Pose", None) if pymp is not None else None
        if variant_name == "array_wxyz":
            return np.concatenate([p, q_wxyz])
        if variant_name == "array_xyzw":
            return np.concatenate([p, q_xyzw])
        if variant_name == "pymp_pose_wxyz" and PoseCls is not None:
            return PoseCls(p, q_wxyz)
        if variant_name == "pymp_pose_xyzw" and PoseCls is not None:
            return PoseCls(p, q_xyzw)
        raise ValueError(f"Unsupported or unavailable variant: {variant_name}")

    def plan_terminal_q(self, pose, variant_name="array_wxyz"):
        start = self.current_arm_qpos()
        target = self.make_target(pose, variant_name)
        planner_name = str(getattr(self.args, "planner_name", "RRTConnect") or "RRTConnect").strip()
        planner_name_l = planner_name.lower().replace(" ", "")
        if planner_name_l in {"rrtconnect", "rrt_connect"}:
            planner_name = "RRTConnect"
        else:
            planner_name = "RRTstar"
        print("[planner] current arm q:", np.round(start, 6))
        print("[planner] target pose p:", np.round(flatten(pose.p)[:3], 6))
        print("[planner] target pose q:", np.round(flatten(pose.q)[:4], 6))
        result = self.planner.plan_qpos_to_pose(
            target,
            start,
            time_step=self.control_timestep,
            use_point_cloud=False,
            planner_name=planner_name,
        )
        print("[planner] status:", result.get("status", result))
        if result.get("status", "") != "Success":
            return None
        if "position" not in result or len(result["position"]) == 0:
            print("[planner] planner returned no waypoints")
            return None
        last_q = flatten(result["position"][-1])[:7]
        print("[planner] last waypoint:", np.round(last_q, 6))
        return last_q

    def execute_linear(self, q_target, gripper_value, max_delta_per_step=0.05, hold_steps=20, tag=""):
        q_start = self.current_arm_qpos()
        q_target = flatten(q_target)[:7]
        q_delta = (q_target - q_start + np.pi) % (2 * np.pi) - np.pi
        q_target_short = q_start + q_delta
        num_steps = int(np.ceil(np.max(np.abs(q_delta)) / max_delta_per_step))
        num_steps = max(1, num_steps)
        print(f"[{tag}] execute_linear num_steps={num_steps}")
        for alpha in np.linspace(0.0, 1.0, num_steps):
            q = (1.0 - alpha) * q_start + alpha * q_target_short
            action = self.compose_action(q, gripper_value)
            self.step_and_render(action, tag=tag)
        final_action = self.compose_action(q_target_short, gripper_value)
        for _ in range(hold_steps):
            self.step_and_render(final_action, tag=f"{tag}_hold")

    def get_obj_pose(self):
        obj_pose = self.base_env.obj.pose
        p = flatten(obj_pose.p)[:3]
        q = flatten(obj_pose.q)[:4]
        return p, q

    def get_goal_pose(self):
        goal_pose = self.base_env.goal_jiaobang.pose
        p = flatten(goal_pose.p)[:3]
        q = flatten(goal_pose.q)[:4]
        return p, q

    def get_object_world_aabb_corners(self):
        obj_p, obj_q = self.get_obj_pose()
        lo = np.asarray(self.base_env.obj_local_aabb_min[0], dtype=np.float64)
        hi = np.asarray(self.base_env.obj_local_aabb_max[0], dtype=np.float64)
        local_corners = np.asarray(
            [[sx, sy, sz] for sx in (lo[0], hi[0]) for sy in (lo[1], hi[1]) for sz in (lo[2], hi[2])],
            dtype=np.float64,
        )
        R_obj = quaternions.quat2mat(flatten(obj_q)[:4])
        obj_p = np.asarray(flatten(obj_p)[:3], dtype=np.float64)
        return local_corners @ R_obj.T + obj_p.reshape(1, 3)

    def get_object_world_aabb_center(self):
        corners = self.get_object_world_aabb_corners()
        return np.asarray(0.5 * (corners.min(axis=0) + corners.max(axis=0)), dtype=np.float64)

    def get_object_aligned_axes_from_quat(self, quat_wxyz):
        quat_wxyz = flatten(quat_wxyz)[:4]

        lo = np.asarray(self.base_env.obj_local_aabb_min[0], dtype=np.float64)
        hi = np.asarray(self.base_env.obj_local_aabb_max[0], dtype=np.float64)
        size_local = hi - lo
        sx = float(size_local[0])
        sy = float(size_local[1])

        R_obj = quaternions.quat2mat(quat_wxyz)
        axis_x_world = normalize(R_obj[:, 0])
        axis_y_world = normalize(R_obj[:, 1])
        axis_z_world = normalize(R_obj[:, 2])

        if axis_x_world is None or axis_y_world is None or axis_z_world is None:
            raise RuntimeError("Failed to compute object local axes from quaternion.")

        # Grasp approach is a table/world constraint, not an object-local-Z
        # constraint.  Always approach from above along world -Z; object axes
        # are only used to choose the horizontal pad/opening direction.
        approaching = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        # 闭合方向取物体局部 xy 平面中较短的边，但只使用其世界 XY 投影。
        closing_seed = axis_x_world if sx <= sy else axis_y_world
        closing = closing_seed - np.dot(closing_seed, approaching) * approaching
        closing = normalize(closing)
        if closing is None:
            fallback = axis_y_world if sx <= sy else axis_x_world
            closing = fallback - np.dot(fallback, approaching) * approaching
            closing = normalize(closing)
        if closing is None:
            raise RuntimeError("Failed to construct closing axis in object plane.")

        ortho = np.cross(closing, approaching)
        ortho = normalize(ortho)
        if ortho is None:
            raise RuntimeError("Failed to construct orthogonal grasp basis.")

        closing = np.cross(approaching, ortho)
        closing = normalize(closing)
        if closing is None:
            raise RuntimeError("Failed to re-orthogonalize closing axis.")

        return ortho, closing, approaching

    def get_topdown_long_axis_axes_from_quat(self, quat_wxyz):
        quat_wxyz = flatten(quat_wxyz)[:4]
        lo = np.asarray(self.base_env.obj_local_aabb_min[0], dtype=np.float64)
        hi = np.asarray(self.base_env.obj_local_aabb_max[0], dtype=np.float64)
        size_local = hi - lo
        longest_axis_idx = int(np.argmax(size_local))

        R_obj = quaternions.quat2mat(quat_wxyz)
        long_axis_world = normalize(R_obj[:, longest_axis_idx])
        if long_axis_world is None:
            raise RuntimeError("Failed to compute object long axis from quaternion.")

        long_axis_xy = np.asarray(long_axis_world, dtype=np.float64).copy()
        long_axis_xy[2] = 0.0
        long_axis_xy = normalize(long_axis_xy)
        if long_axis_xy is None:
            for alt_idx in np.argsort(size_local)[::-1]:
                alt_axis = np.asarray(R_obj[:, int(alt_idx)], dtype=np.float64).copy()
                alt_axis[2] = 0.0
                long_axis_xy = normalize(alt_axis)
                if long_axis_xy is not None:
                    break
        if long_axis_xy is None:
            long_axis_xy = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        approaching = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        closing = np.array([-long_axis_xy[1], long_axis_xy[0], 0.0], dtype=np.float64)
        closing = normalize(closing)
        if closing is None:
            closing = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        ortho = np.cross(closing, approaching)
        ortho = normalize(ortho)
        if ortho is None:
            raise RuntimeError("Failed to construct top-down orthogonal grasp basis.")

        closing = np.cross(approaching, ortho)
        closing = normalize(closing)
        if closing is None:
            raise RuntimeError("Failed to re-orthogonalize top-down closing axis.")

        return ortho, closing, approaching

    def get_long_axis_adaptive_axes_from_quat(self, quat_wxyz):
        # Adaptive grasping used to tilt the approach to be perpendicular to
        # the object long axis.  For this tabletop pipeline that produces
        # non-vertical grasps on objects such as carriot.  Keep the long-axis
        # horizontal alignment, but force the approach to world -Z.
        return self.get_topdown_long_axis_axes_from_quat(quat_wxyz)

    def get_grasp_axes_from_quat(self, quat_wxyz):
        grasp_mode = str(getattr(self.args, "grasp_mode", "object_normal") or "object_normal").strip().lower()
        if grasp_mode == "object_normal":
            return self.get_object_aligned_axes_from_quat(quat_wxyz)
        if grasp_mode == "topdown_long_axis":
            return self.get_topdown_long_axis_axes_from_quat(quat_wxyz)
        if grasp_mode == "pen_topdown_insert_ready":
            return self.get_topdown_long_axis_axes_from_quat(quat_wxyz)
        if grasp_mode == "long_axis_adaptive":
            return self.get_long_axis_adaptive_axes_from_quat(quat_wxyz)
        raise ValueError(f"Unsupported grasp_mode: {grasp_mode}")

    def build_parallel_topdown_pose(self, center_p, quat_wxyz, z_offset=0.0):
        center_p = np.asarray(flatten(center_p)[:3], dtype=np.float64).copy()
        quat_wxyz = flatten(quat_wxyz)[:4]

        ortho, closing, approaching = self.get_grasp_axes_from_quat(quat_wxyz)
        R_tcp = np.stack([ortho, closing, approaching], axis=1)
        q = mat2quat_np(R_tcp)

        p = center_p.copy()
        p = p - approaching * float(z_offset)
        return sapien.Pose(p, q)

    def build_topdown_grasp_pose(self):
        _, obj_q = self.get_obj_pose()
        bbox_center = self.get_object_world_aabb_center()
        return self.build_parallel_topdown_pose(bbox_center, obj_q, z_offset=self.args.grasp_z_offset)

    def build_pregrasp_pose(self, grasp_pose):
        p = flatten(grasp_pose.p)[:3].copy()
        q = flatten(grasp_pose.q)[:4].copy()
        R_tcp = quat2mat_np(q)
        approach_axis = np.asarray(R_tcp[:, 2], dtype=np.float64).reshape(3)
        retreat = -approach_axis * float(self.args.pregrasp_height)
        # For nearly vertical top-down grasps, keeping the historical world +Z
        # retreat remains the most stable behavior. Once the chosen grasp is
        # meaningfully tilted, however, the pregrasp must be backed off along
        # the grasp's local approach axis so the final segment stays a clean
        # straight-line approach in the goal frame.
        if float(np.linalg.norm(retreat[:2])) <= 0.002:
            p[2] = p[2] + float(self.args.pregrasp_height)
        else:
            p = p + retreat
        return sapien.Pose(p, q)

    def build_goal_tcp_pose(self, grasp_pose=None):
        goal_p, goal_q = self.get_goal_pose()
        return self.build_parallel_topdown_pose(goal_p, goal_q, z_offset=self.args.goal_z_offset)

    def is_grasped(self):
        try:
            g = self.base_env.agent.is_grasping(self.base_env.obj)
            if hasattr(g, "item"):
                return bool(g.item())
            return bool(g)
        except Exception:
            return False

    def refresh_goal_and_plan(self, announce=True):
        goal_tcp_pose = self.build_goal_tcp_pose()
        if announce:
            print("goal tcp p:", np.round(flatten(goal_tcp_pose.p)[:3], 6), "q:", np.round(flatten(goal_tcp_pose.q)[:4], 6))
        self.preview_target_pose(goal_tcp_pose)
        q_goal = self.plan_terminal_q(goal_tcp_pose, variant_name=self.args.variant)
        return goal_tcp_pose, q_goal

    def chase_goals_until_done(self):
        goal_move_idx = 0
        while True:
            print(f"\n[move to goal #{goal_move_idx}]")
            goal_tcp_pose, q_goal = self.refresh_goal_and_plan(announce=True)
            if q_goal is None:
                print("[FAIL] goal planning failed")
                return False, False

            self.execute_linear(
                q_goal,
                gripper_value=self.args.gripper_close,
                max_delta_per_step=self.args.max_delta_per_step,
                hold_steps=self.args.hold_steps,
                tag=f"goal_{goal_move_idx}",
            )
            if getattr(self.args, "log_contact_forces", False):
                self.log_grasp_contact_forces(tag=f"after_goal_{goal_move_idx}")
            self.report_final_error(goal_tcp_pose)
            flags = self.print_step_flags(prefix=f"[goal #{goal_move_idx}] ")

            if flags["done"]:
                print("[goal loop] episode done, waiting for outer reset")
                return True, True

            if flags["success"]:
                print("[goal loop] success detected, environment should resample a new goal; refresh visualization and continue")
                refreshed_goal_pose = self.build_goal_tcp_pose()
                self.preview_target_pose(refreshed_goal_pose)
                print(
                    "[goal loop] refreshed goal tcp p:",
                    np.round(flatten(refreshed_goal_pose.p)[:3], 6),
                    "q:",
                    np.round(flatten(refreshed_goal_pose.q)[:4], 6),
                )
                goal_move_idx += 1
                continue

            print("[goal loop] current motion finished without success/done, hold current pose and keep monitoring")
            hold_q = self.current_arm_qpos()
            monitor_steps = max(1, self.args.goal_monitor_steps)
            for monitor_idx in range(monitor_steps):
                action = self.compose_action(hold_q, self.args.gripper_close)
                self.step_and_render(action, tag="goal_monitor")
                flags = self._extract_step_flags()
                if flags["done"]:
                    self.print_step_flags(prefix=f"[goal monitor {monitor_idx}] ")
                    print("[goal loop] episode done during monitor")
                    return True, True
                if flags["success"]:
                    self.print_step_flags(prefix=f"[goal monitor {monitor_idx}] ")
                    refreshed_goal_pose = self.build_goal_tcp_pose()
                    self.preview_target_pose(refreshed_goal_pose)
                    print(
                        "[goal loop] refreshed goal tcp p:",
                        np.round(flatten(refreshed_goal_pose.p)[:3], 6),
                        "q:",
                        np.round(flatten(refreshed_goal_pose.q)[:4], 6),
                    )
                    goal_move_idx += 1
                    break
            else:
                print("[goal loop] monitor window ended without success/done, replan to current goal and continue")
                goal_move_idx += 1
                continue

    def report_final_error(self, target_pose):
        tcp_pose = self.base_env.agent.tcp.pose
        p_cur = flatten(tcp_pose.p)[:3]
        q_cur = flatten(tcp_pose.q)[:4]
        p_tgt = flatten(target_pose.p)[:3]
        q_tgt = flatten(target_pose.q)[:4]
        pos_err = float(np.linalg.norm(p_cur - p_tgt))
        ang_err = quat_angle_deg(q_cur, q_tgt)
        print("\n[final] tcp p:", np.round(p_cur, 6))
        print("[final] tcp q:", np.round(q_cur, 6))
        print("[final] target p:", np.round(p_tgt, 6))
        print("[final] target q:", np.round(q_tgt, 6))
        print(f"[final] position error: {pos_err:.6f} m")
        print(f"[final] orientation error: {ang_err:.3f} deg")
        print(f"[final] is_grasped = {self.is_grasped()}")

    def run_demo(self):
        completed_episodes = 0
        while completed_episodes < self.args.num_episodes:
            print(f"\n[reset] starting episode {completed_episodes}")
            self.reset_episode(seed=self.args.seed + completed_episodes)
            print(f"\n[episode {completed_episodes}] begin planning")

            # 0. 一开始完全张开夹爪
            print("\n[open gripper]")
            self.hold_current_and_set_gripper(self.args.gripper_open, steps=self.args.open_steps)

            grasp_pose = self.build_topdown_grasp_pose()
            pregrasp_pose = self.build_pregrasp_pose(grasp_pose)

            print("\n[poses]")
            print("grasp p:", np.round(flatten(grasp_pose.p)[:3], 6), "q:", np.round(flatten(grasp_pose.q)[:4], 6))
            print("pregrasp p:", np.round(flatten(pregrasp_pose.p)[:3], 6), "q:", np.round(flatten(pregrasp_pose.q)[:4], 6))

            # 1. 到预抓取位置
            print("\n[move to pregrasp]")
            self.preview_target_pose(pregrasp_pose)
            q_pre = self.plan_terminal_q(pregrasp_pose, variant_name=self.args.variant)
            if q_pre is None:
                print("[FAIL] pregrasp planning failed")
                return False
            self.execute_linear(
                q_pre,
                gripper_value=self.args.gripper_open,
                max_delta_per_step=self.args.max_delta_per_step,
                hold_steps=0,
                tag="pregrasp",
            )

            # 2. 到抓取位置
            print("\n[move to grasp]")
            self.preview_target_pose(grasp_pose)
            q_grasp = self.plan_terminal_q(grasp_pose, variant_name=self.args.variant)
            if q_grasp is None:
                print("[FAIL] grasp planning failed")
                return False
            self.execute_linear(
                q_grasp,
                gripper_value=self.args.gripper_open,
                max_delta_per_step=self.args.max_delta_per_step,
                hold_steps=self.args.hold_steps,
                tag="grasp",
            )

            # 3. 闭合夹爪抓住 obj
            print("\n[close gripper]")
            self.hold_current_and_set_gripper(self.args.gripper_close, steps=self.args.close_steps)
            print("[close gripper] is_grasped =", self.is_grasped())
            self.print_step_flags(prefix="[after close] ")
            if getattr(self.args, "log_contact_forces", False):
                self.log_grasp_contact_forces(tag="after_close")

            ok, episode_finished = self.chase_goals_until_done()
            if not ok:
                return False

            if episode_finished:
                completed_episodes += 1
                print(f"[episode {completed_episodes - 1}] finished by reset/done; moving to next episode")
            else:
                print(f"[episode {completed_episodes}] not done yet; restarting planning loop in same episode")

        print(f"\n[demo] completed {completed_episodes} episodes")
        return True

    def run_tests(self):
        print("\n[test_env_creation]")
        assert self.robot is not None
        assert self.tcp is not None
        print("pass")

        print("\n[test_planner_creation]")
        assert self.planner is not None
        print("pass")

        grasp_pose = self.build_topdown_grasp_pose()
        pregrasp_pose = self.build_pregrasp_pose(grasp_pose)
        goal_tcp_pose = self.build_goal_tcp_pose()

        print("\n[test_pose_build]")
        assert len(flatten(grasp_pose.p)) == 3
        assert len(flatten(grasp_pose.q)) == 4
        pregrasp_delta = flatten(pregrasp_pose.p)[:3] - flatten(grasp_pose.p)[:3]
        assert abs(np.linalg.norm(pregrasp_delta) - self.args.pregrasp_height) < 1e-8
        assert len(flatten(goal_tcp_pose.p)) == 3
        print("pass")

        print("\n[test_pregrasp_plan]")
        assert self.plan_terminal_q(pregrasp_pose, variant_name=self.args.variant) is not None
        print("pass")

        print("\n[test_grasp_plan]")
        assert self.plan_terminal_q(grasp_pose, variant_name=self.args.variant) is not None
        print("pass")

        print("\n[test_goal_plan]")
        assert self.plan_terminal_q(goal_tcp_pose, variant_name=self.args.variant) is not None
        print("pass")

        print("\nAll tests passed.")
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="demo", choices=["demo", "test"])
    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--srdf-path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="array_wxyz", choices=["array_wxyz", "array_xyzw", "pymp_pose_wxyz", "pymp_pose_xyzw"])
    parser.add_argument("--pregrasp-height", type=float, default=0.07)
    parser.add_argument("--grasp-z-offset", type=float, default=0.0)
    parser.add_argument("--goal-z-offset", type=float, default=0.0)
    parser.add_argument("--yaw-offset-deg", type=float, default=180.0)
    parser.add_argument("--max-delta-per-step", type=float, default=0.05)
    parser.add_argument("--hold-steps", type=int, default=15)
    parser.add_argument("--open-steps", type=int, default=0)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--gripper-open", type=float, default=-1.0)
    parser.add_argument("--gripper-close", type=float, default=1.0)
    parser.add_argument("--video-dir", default="./rm75_jiaobang_pick_move_v5")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--goal-monitor-steps", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--log-contact-forces", action="store_true")
    parser.add_argument("--log-force-every", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.video_dir, exist_ok=True)

    sim_urdf_path = find_existing_urdf(args.urdf_path)
    env = gym.make(
        "Two_finger_PickJiaobang-v1",
        robot_uids="RM75",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    env = RecordEpisode(
        env,
        output_dir=args.video_dir,
        save_trajectory=False,
        save_video=True,
        source_type="motionplanning",
        source_desc="RM75 jiaobang pick move v5",
        video_fps=args.video_fps,
    )
    initial_obs, initial_info = env.reset(seed=args.seed)

    planning_urdf_path, srdf_path = resolve_planning_artifact_paths(sim_urdf_path, args)
    os.makedirs(os.path.dirname(planning_urdf_path), exist_ok=True)
    os.makedirs(os.path.dirname(srdf_path), exist_ok=True)

    generate_near_collision_free_planning_urdf(sim_urdf_path, planning_urdf_path)
    if args.srdf_path is None:
        write_permissive_srdf(planning_urdf_path, srdf_path)

    print("Sim URDF     :", sim_urdf_path)
    print("Planning URDF:", planning_urdf_path)
    print("SRDF         :", srdf_path)
    print("Video Dir    :", os.path.abspath(args.video_dir))

    demo = RM75JiaobangPickMove(env, planning_urdf_path, srdf_path, args)
    demo.last_obs = initial_obs
    demo.last_info = initial_info if isinstance(initial_info, dict) else {}
    demo.last_terminated = False
    demo.last_truncated = False
    ok = demo.run_tests() if args.mode == "test" else demo.run_demo()
    print("\nfinal success =", ok)

    try:
        env.flush_video()
    except Exception:
        pass
    try:
        env.flush()
    except Exception:
        pass
    try:
        env.close()
    except Exception as e:
        print("close warning:", e)


if __name__ == "__main__":
    main()

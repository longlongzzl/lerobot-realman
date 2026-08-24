from __future__ import annotations

import numpy as np

from rm75_app.execution.maniskill_task_bridge import ManiSkillTaskBridge
from rm75_app.orchestration.multi_object_executor import AtomExecution, SceneObjectState, TaskSceneState
from rm75_app.tasks.manipulation_plan import ManipulationAtom, ManipulationPrimitive


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakePose:
    def __init__(self, matrix):
        self.matrix = matrix

    def to_transformation_matrix(self):
        return FakeTensor(self.matrix[None])


class FakeActor:
    def __init__(self, matrix):
        self.pose = FakePose(matrix)


class FakeRobot:
    def get_qpos(self):
        return FakeTensor([0.1, 0.2, 0.3])


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.agent = type("Agent", (), {"robot": FakeRobot()})()
        self.action_space = type("Space", (), {"shape": (4,)})()
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action).copy())


def test_maniskill_bridge_observes_settles_and_validates() -> None:
    target = np.eye(4)
    target[0, 3] = 0.2
    env = FakeEnv()
    bridge = ManiSkillTaskBridge(
        env,
        {"carrot_1": FakeActor(target)},
        settle_steps=3,
        hold_action=lambda: np.asarray([0.1, 0.2, 0.3, 0.0]),
    )
    atom = ManipulationAtom(
        "atom_01",
        ManipulationPrimitive.PICK_PLACE,
        "carrot_1",
        "carriot",
        target,
    )
    scene = TaskSceneState({"carrot_1": SceneObjectState("carrot_1", "carriot", np.eye(4))})
    validation = bridge.validate_atom(atom, AtomExecution(True, target), scene)
    assert validation.success
    assert len(env.actions) == 3
    np.testing.assert_allclose(bridge.observe_object_pose("carrot_1"), target)

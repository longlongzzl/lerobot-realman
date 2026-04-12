import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type


@dataclass
class RobotConfig:
    id: str = "default"
    calibration_dir: Optional[str] = None  # placeholder to match LeRobot style


class Robot(abc.ABC):
    """Minimal LeRobot-style abstract base."""

    config_class: Type[RobotConfig]
    name: str

    def __init__(self, config: RobotConfig):
        self.robot_type = self.name
        self.id = config.id

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    @property
    @abc.abstractmethod
    def observation_features(self) -> Dict[str, Any]:
        pass

    @property
    @abc.abstractmethod
    def action_features(self) -> Dict[str, Any]:
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        # RealMan generally doesn't need LeRobot motor calibration
        return True

    def calibrate(self) -> None:
        return None

    @abc.abstractmethod
    def configure(self) -> None:
        pass

    @abc.abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass
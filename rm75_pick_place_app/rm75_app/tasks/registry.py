from __future__ import annotations

from dataclasses import dataclass

from rm75_app.tasks.base import TaskAdapterBase, normalize_token
from rm75_app.tasks.jimu import JimuTask
from rm75_app.tasks.lego import LegoTask
from rm75_app.tasks.pickplace import PickPlaceTask


@dataclass
class TaskRegistry:
    _adapters: dict[str, TaskAdapterBase]
    _aliases: dict[str, str]

    @classmethod
    def with_builtin_tasks(cls) -> "TaskRegistry":
        registry = cls({}, {})
        for adapter in (PickPlaceTask(), JimuTask(), LegoTask()):
            registry.register(adapter)
        return registry

    def register(self, adapter: TaskAdapterBase) -> None:
        key = normalize_token(adapter.definition.key)
        if key in self._adapters:
            raise ValueError(f"task already registered: {key}")
        self._adapters[key] = adapter
        for alias in (adapter.definition.key, *adapter.definition.aliases):
            alias_key = normalize_token(alias)
            owner = self._aliases.get(alias_key)
            if owner is not None and owner != key:
                raise ValueError(f"task alias collision: {alias_key} ({owner}, {key})")
            self._aliases[alias_key] = key

    def get(self, name: str) -> TaskAdapterBase:
        normalized = normalize_token(name)
        try:
            return self._adapters[self._aliases[normalized]]
        except KeyError as exc:
            valid = ", ".join(sorted(self._adapters))
            raise KeyError(f"unknown task {name!r}; available tasks: {valid}") from exc

    def adapters(self) -> tuple[TaskAdapterBase, ...]:
        return tuple(self._adapters[key] for key in sorted(self._adapters))


TASK_REGISTRY = TaskRegistry.with_builtin_tasks()


def get_task_adapter(name: str) -> TaskAdapterBase:
    return TASK_REGISTRY.get(name)


def list_task_adapters() -> tuple[TaskAdapterBase, ...]:
    return TASK_REGISTRY.adapters()


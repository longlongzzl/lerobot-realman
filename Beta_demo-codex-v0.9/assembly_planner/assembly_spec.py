from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PanelSpec:
    role: str
    shape: str
    stage_group: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "shape": self.shape,
            "stage_group": self.stage_group,
        }


@dataclass(frozen=True)
class ConnectionSpec:
    child: str
    child_edge: str
    parent: str
    parent_edge: str
    relation: str
    order: int

    def to_json(self) -> dict[str, Any]:
        return {
            "child": self.child,
            "child_edge": self.child_edge,
            "parent": self.parent,
            "parent_edge": self.parent_edge,
            "relation": self.relation,
            "order": int(self.order),
        }


@dataclass(frozen=True)
class AssemblySpec:
    name: str
    panels: list[PanelSpec] = field(default_factory=list)
    connections: list[ConnectionSpec] = field(default_factory=list)

    def ordered_connections(self) -> list[ConnectionSpec]:
        return sorted(self.connections, key=lambda item: (int(item.order), item.child))

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "panels": [panel.to_json() for panel in self.panels],
            "connections": [connection.to_json() for connection in self.ordered_connections()],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


def standard_two_layer_wall_spec() -> AssemblySpec:
    panels = [
        PanelSpec("floor", "square"),
        PanelSpec("right_wall", "square", "left_square_row"),
        PanelSpec("back_wall", "square", "left_square_row"),
        PanelSpec("left_wall", "square", "left_square_row"),
        PanelSpec("front_wall", "square", "left_square_row"),
        PanelSpec("front_second_triangle", "triangle", "right_triangle_row"),
        PanelSpec("right_second_triangle", "triangle", "right_triangle_row"),
        PanelSpec("left_second_triangle", "triangle", "right_triangle_row"),
        PanelSpec("back_second_triangle", "triangle", "right_triangle_row"),
    ]
    connections = [
        ConnectionSpec("right_wall", "bottom_edge", "floor", "right_edge", "square_wall_on_floor", 1),
        ConnectionSpec("back_wall", "bottom_edge", "floor", "back_edge", "square_wall_on_floor", 2),
        ConnectionSpec("left_wall", "bottom_edge", "floor", "left_edge", "square_wall_on_floor", 3),
        ConnectionSpec("front_wall", "bottom_edge", "floor", "front_edge", "square_wall_on_floor", 4),
        ConnectionSpec("front_second_triangle", "base_edge", "front_wall", "top_edge", "triangle_wall_on_square", 5),
        ConnectionSpec("right_second_triangle", "base_edge", "right_wall", "top_edge", "triangle_wall_on_square", 6),
        ConnectionSpec("left_second_triangle", "base_edge", "left_wall", "top_edge", "triangle_wall_on_square", 7),
        ConnectionSpec("back_second_triangle", "base_edge", "back_wall", "top_edge", "triangle_wall_on_square", 8),
    ]
    return AssemblySpec("standard_two_layer_wall", panels=panels, connections=connections)

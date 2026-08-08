# RM75 LLM Pick-Place Interface

LLM 只输出一个 JSON object，不要输出 markdown 或解释文字。

`schema_version`: `rm75_pick_place_plan_v1`

顶层格式：

```json
{
  "schema_version": "rm75_pick_place_plan_v1",
  "user_command": "原始用户指令",
  "assumptions": ["可选，简短写不确定假设"],
  "steps": [
    {
      "action": "pick_place",
      "object": "物体引用",
      "goal": {
        "type": "inside | on_top | beside | lean_against | between | slot | nearest_empty | upright_in_place | rotate_in_place"
      }
    }
  ]
}
```

原则：LLM 输出目标约束，不输出最终 `xyz/quaternion`。物体引用可以用场景里的 canonical id，也可以用中文名。

## Goal Types

`inside`

```json
{"type": "inside", "target": "笔筒", "release": "drop"}
```

`on_top`

```json
{
  "type": "on_top",
  "target": "绿木块",
  "stability_required": true,
  "fallback": {
    "type": "beside",
    "target": "绿木块",
    "side": "right",
    "clearance_m": 0.015,
    "long_axis": "parallel_table_edge"
  }
}
```

`beside`

```json
{
  "type": "beside",
  "target": "胶棒",
  "side": "left",
  "clearance_m": 0.02,
  "long_axis": "preserve"
}
```

`lean_against`

```json
{
  "type": "lean_against",
  "target": "笔筒",
  "side": "right",
  "bottom_on": "table",
  "lean_angle_deg": 30,
  "long_axis": "toward_target"
}
```

`between`

```json
{"type": "between", "object_a": "绿木块", "object_b": "笔筒", "face": "笔筒"}
```

`slot`

```json
{"type": "slot", "surface": "small_desk", "slot": "slot_5"}
```

`nearest_empty`

```json
{"type": "nearest_empty", "around": "小桌子", "surface": "small_desk"}
```

`upright_in_place`

```json
{"type": "upright_in_place"}
```

`rotate_in_place`

```json
{"type": "rotate_in_place", "yaw_deg": 45}
```

## Non Pick-Place Actions

`exchange`

```json
{"action": "exchange", "object_a": "绿木块", "object_b": "网球"}
```

`collection`

```json
{
  "action": "collection",
  "objects": "all_movable",
  "goal": {"type": "slots", "surface": "small_desk", "order": "left_to_right"}
}
```

## Examples

```json
{
  "schema_version": "rm75_pick_place_plan_v1",
  "user_command": "把网球扔进笔筒，然后把笔靠在笔筒右侧",
  "steps": [
    {
      "action": "pick_place",
      "object": "网球",
      "goal": {"type": "inside", "target": "笔筒", "release": "drop"}
    },
    {
      "action": "pick_place",
      "object": "笔",
      "goal": {
        "type": "lean_against",
        "target": "笔筒",
        "side": "right",
        "bottom_on": "table",
        "lean_angle_deg": 30,
        "long_axis": "toward_target"
      }
    }
  ]
}
```

```json
{
  "schema_version": "rm75_pick_place_plan_v1",
  "user_command": "把胶棒叠到绿木块上面，如果支撑不稳就放到绿木块旁边",
  "steps": [
    {
      "action": "pick_place",
      "object": "胶棒",
      "goal": {
        "type": "on_top",
        "target": "绿木块",
        "stability_required": true,
        "fallback": {
          "type": "beside",
          "target": "绿木块",
          "side": "right",
          "clearance_m": 0.015,
          "long_axis": "parallel_table_edge"
        }
      }
    }
  ]
}
```

校验接口：

```bash
python -m rm75_app.llm.orchestrator \
  --fixed-scene-pose-file assets/test_scenes/current_table.json \
  --llm-plan-json-file /path/to/llm_output.json \
  --validate-llm-plan-only
```

打印完整接口提示：

```bash
python -m rm75_app.llm.orchestrator --print-llm-interface --fixed-scene-pose-file assets/test_scenes/current_table.json
```

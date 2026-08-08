# Lego task assets

这里先放主线任务契约使用的 Lego task JSON。固定连接头模型、完整 brick mesh 和真机执行器仍在 `rm75_lego_snap_place_app` 兼容目录，迁移执行器时保持 JSON 字段不变。

任务文件可以是步骤数组，也可以是带 `name`、`version`、`steps` 的对象。每一步至少包含：

```json
{
  "action": "assemble",
  "brick": "b2x4",
  "grid_x": 14,
  "grid_y": 14,
  "layer": 1,
  "orientation": 0
}
```

`action` 支持 `assemble` 和 `disassemble`；`grid_x/grid_y` 是底板网格坐标，`layer` 从 1 开始。

# Fast-Chain 四组 IK Batch 取交集流程总结

本文记录当前 `rm75_app/runtime/direct_pre_place.py` 里的 fast-chain 处理逻辑。核心目标是：在真正调用较重的 MotionGen 之前，先用固定 batch size 的 cuRobo CUDA Graph IK 快速筛掉不可能成功的抓取/放置关系，只让少量完整链路进入运输规划。

## 核心目标

原来的宽搜索方式容易把大量候选送进 MotionGen，慢且失败分支代价大。现在的思路是：

1. 先生成一组有优先级的抓取/放置关系候选。
2. 用 cuRobo batch IK 检查四个关键端点：
   - `pregrasp`
   - `grasp`
   - `transport hover`
   - `release/final contact endpoint`
3. 对四组 IK 结果按同一个关系 key 取交集。
4. 只把交集里的 top pair 交给后续 MotionGen 和短直线段验证。

因此 fast-chain 不是直接证明整条轨迹一定成功，而是先证明一条链路的关键端点都可达，再用后面的规划阶段验证运输路径和最终接触段。

## 相关代码位置

主要实现集中在：

- `rm75_app/runtime/direct_pre_place.py`
  - `_fast_chain_preselect_grasp_place_pair`
  - `_fast_chain_rank_paired_relation_candidates`
  - `_fast_chain_evaluate_paired_relation_records`
  - `_make_ik_preselected_grasp_success`
  - `plan_transport_to_hover`
- `rm75_app/planning/curobo_planner.py`
  - `solve_batch_start_goal_ik`
  - `_solve_batch_start_goal_ik_cuda_graph`
  - `_get_cuda_graph_batch_ik_solver`

SAM6D 版本的脚本主要是接入感知和参数裁剪，fast-chain 主逻辑仍复用上面的 direct 脚本。

## 关键默认参数

当前 fast-chain 默认是开启的：

```bash
--fast-chain-screening
--fast-chain-relation-ik-slots 16
--fast-chain-ik-seeds 32
--fast-chain-cuda-graph-ik
--fast-chain-cuda-graph-ik-fixed-batch-size 16
--fast-chain-cuda-graph-ik-max-batch-size 128
--fast-chain-top-pairs 3
--fast-chain-place-rank-grasp-limit 16
--fixed-tabletop-fast-chain-place-rank-grasp-limit 16
--no-fast-chain-allow-legacy-fallback
--transport-use-prefilter-q-goal
--transport-prefilter-q-goal-max-trials 1
--transport-prefilter-q-goal-timeout 2.0
--transport-prefilter-q-goal-num-trajopt-seeds 1
```

`--fast-chain-cuda-graph-ik-fixed-batch-size 16` 是速度稳定性的关键。少于 16 的请求会 padding 到 16；多于 16 的请求会拆成多个 16-size chunk。这样 cuRobo 的 CUDAGraph 不需要因为动态 shape 反复重建。

## 总流程

```text
场景/目标确定
  ↓
生成抓取候选 grasp_candidates
  ↓
按物体类别排序，截取最多 fast_chain_relation_ik_slots 个 relation
  ↓
Batch 1: current_q -> pregrasp IK
  ↓
Batch 2: q_pregrasp -> grasp IK + 直线 approach 验证
  ↓
为每个 relation 绑定一个最高优先级 place candidate
  ↓
Batch 3/4: q_grasp -> hover IK 与 q_grasp -> release IK
  ↓
按 relation key 做四组交集
  ↓
交集为空时，对特定物体做第二阶段 yaw/spin/release 扩展 IK
  ↓
按 score 排序，选 top pair
  ↓
跳过候选阶段 grasp MotionGen，使用 IK 结果进入运输链路
  ↓
短 lift/transport MotionGen/final contact/release/return
```

## 候选关系生成

### 固定桌面姿态类

这类主要包括：

- `lvmukuai`
- `carriot`
- `shuazi`

这类物体的目标物体姿态本身通常是固定的，fast-chain 不应该随意改变最终 object pose。代码采用 place-first 方式：

1. 先从 place rule 得到最终目标物体位姿。
2. 用 world Z 构造一个稳定的 tabletop relation frame，不直接信任物体 mesh 的本地 Z。
3. 生成一组 TCP 与物体之间的 relation tilt。
4. 把 release relation 映射回当前物体姿态下的 grasp relation。

`lvmukuai` 当前使用一组更密的 16 槽 tilt ladder：

```text
0, -10, +10, -20, +20, -30, +30, -40,
+40, -50, +50, -60, +60, -70, +70, -80 deg
```

木块的对称性在匹配阶段处理：`+60` 和 `-60` 可以按 `abs(tilt)` 视为同一类关系。

其他固定桌面物体默认 tilt 是：

```text
0, -15, +15, -30, +30, -45, +45, -60, +60 deg
```

### 竖直长轴类

典型是：

- `gluestick`
- `hongshupian`

这类物体会生成 top-bias / axis shift / tilt 相关的抓取关系。第一阶段先只用较少的高优先级 relation 做四图交集。如果 `pregrasp/grasp` 有解，但 `hover/release` 全 0，会进入第二阶段 yaw 扩展。

### 笔 `bi`

笔属于插入/放置关系更敏感的细长物体。第一阶段会优先尝试插入 proven 的 roll/tilt relation。若第一阶段 hover/release 全 0，会进入 spin 扩展。

### 网球 `tennis`

网球是球体，抓取 yaw/放置朝向理论上不敏感。但 release endpoint 仍然要可达。第一阶段失败时，会做 sphere release tilt/roll 扩展，默认最多 `--sphere-fast-chain-place-expansion-candidates 16` 个候选。

## 四组 IK Batch

当前日志里显示为：

```text
pregrasp图
grasp图
hover图
release图
四图交集
```

### Batch 1: pregrasp IK

输入：

- start q: 当前机械臂关节
- goal pose: 每个候选的 `pregrasp_pose`

碰撞世界：

- 包含当前活动物体
- 不包含桌子

输出：

- `_winner_preselect_pregrasp_success`
- `_winner_preselect_q_pregrasp`
- `_winner_preselect_pregrasp_ik_score`

只有 pregrasp IK 成功的候选才可能继续进入完整链路。

### Batch 2: grasp IK

输入：

- start q: 对应候选的 `q_pregrasp`
- goal pose: 对应候选的 `grasp_pose`

碰撞世界：

- 当前活动物体从 IK collision world 里移除，因为这一段是接触抓取。
- world 改变后会 invalidate CUDA graph IK solver，避免旧 graph 使用旧 collision world。

额外验证：

即使 grasp IK 成功，还要用 `_build_validated_linear_path_to_q` 检查 `pregrasp -> grasp` 的短直线 approach 是否合理。通过后才记录：

- `_winner_preselect_grasp_success`
- `_winner_preselect_q_grasp`
- `_winner_preselect_grasp_approach_q_path`
- `_winner_preselect_grasp_ik_score`

这里的 `grasp图` 实际上不是单纯 endpoint IK，还包含短 approach 的基础验证。

### Batch 3: hover IK

输入：

- start q: 对应候选的 `q_grasp`
- goal pose: 该 relation 绑定的 place hover pose

输出：

- `fast_chain_hover_q`
- `q_hover`
- hover IK 误差

这个结果后面用于运输阶段：优先尝试 `plan_to_joint_state(start_q, q_hover)`，避免 MotionGen 再自己搜索 IK 分支。

### Batch 4: release IK

输入：

- start q: 仍然是对应候选的 `q_grasp`
- goal pose: release/final contact endpoint

注意：release IK 这里是 endpoint 可达性检查，不是证明 `hover -> release` 这条短直线已经成功。真正的最终下降段在后面 final contact 阶段执行时再验证。

输出：

- `fast_chain_release_q`
- `q_release`
- release IK 误差

后续 final contact 优先使用这个 cached `q_release` 做直线接触段：

1. 先尝试 cached `q_release` 直线段。
2. 不行再尝试 endpoint IK straight final contact。
3. 再不行才走 constrained MotionGen / segmented fallback。

### 代码层面的合并

概念上是四组 IK batch，但代码里 `hover` 和 `release` 是一次合并调用：

```python
start_qs = q_grasps + q_grasps
goal_poses = hover_poses + release_poses
```

然后把返回结果切成两半：

```python
hover_results = combined_results[:len(records)]
release_results = combined_results[len(records):]
```

如果总数超过固定 batch size 16，底层 wrapper 会拆成多个 16-size CUDA Graph chunk。也就是说，日志里是 `hover图/release图` 两组结果，底层可以是一次或多次固定 shape cuRobo batch IK 调用。

## 四图交集规则

对每个候选 relation，必须满足：

```text
pregrasp_ok[relation]
&& grasp_ok[relation]
&& grasp_approach_path_ok[relation]
&& hover_ok[relation]
&& release_ok[relation]
```

匹配时不是简单按 label 字符串全部相等，而是通过 relation match key：

- `lvmukuai`: 使用 `abs(tilt)`，所以 `+60` 和 `-60` 可以配对。
- 其他 fixed-tabletop: 使用 signed tilt，`+60` 和 `-60` 不混用。
- 普通物体: 默认按 label 匹配。

这样可以避免抓取 relation 和放置 relation 不一致。例如“抓取是 +30 tilt，但 release 是 -45 yaw/tilt”的组合不会被误认为同一条链路，除非物体类别明确允许这种对称匹配。

## 第二阶段扩展

第一阶段交集为空时，不立刻扩大所有搜索。只有满足以下条件才扩展：

```text
pregrasp/grasp 有成功 relation
hover/release 成功数都为 0
物体属于支持扩展的类别
```

当前扩展类别：

- `gluestick`
- `hongshupian`
- `bi`
- `tennis`

### gluestick / hongshupian yaw 扩展

使用 `tabletop_place_yaw_variant_deg`：

```text
0, -45, +45, -90, +90, -135, +135, 180 deg
```

每个 yaw 选一个排序最优的 place candidate，再与原来的 grasp relation 做 hover/release IK 检查。

偏好排序：

- `gluestick`: 优先靠近 `±135 deg`
- `hongshupian`: 根据 slot 位置偏好 `0 / 45 / -90` 的不同顺序

### bi spin 扩展

`bi` 使用 `_bi_fast_insert_values` 产生插入相关的 spin/roll 候选。偏好大致为：

```text
0, -15, +15, -90, +90, -120, -135, -150 ...
```

### tennis release 扩展

`tennis` 不靠 yaw 匹配，而是扩展 release 端的 tilt/roll。这样可以在球体抓取关系不变的情况下，给 release endpoint 更多可达姿态。

## 评分和 top pair 选择

四图交集中的每个完整链路会生成一个 score。主要组成：

```text
transport_score = ||q_hover - q_grasp||
release_score   = ||q_release - q_hover||
joint7_score    = abs(delta joint7)

place_pair_score =
    transport_score
  + 0.35 * release_score
  + 0.15 * joint7_score
  + selection_penalty
  + place_pref_penalty
  + clearance_penalty
  + yaw_expansion_preference_penalty
  + ik_error_score

final_score =
    place_pair_score
  + grasp_joint_score
  + pregrasp_ik_score
  + grasp_ik_score
```

排序后保留 `--fast-chain-top-pairs` 条。默认 `fast_chain_diverse_top_grasps=True`，会优先保留不同 grasp label 的 top pair，避免 top-K 全是几乎相同的抓法。

最终输出到网页/日志的核心行类似：

```text
fast-chain gluestick；
pregrasp图 10/13: ...
grasp图 9/13: ...
hover图 1/13: ...
release图 1/13: ...
四图交集 1/13: ...
最终链路 ...
状态 已选中完整链路
```

## 进入运输规划后的执行方式

fast-chain 选中 pair 后，不再走候选阶段的大范围 grasp MotionGen。

### 抓取阶段

`_make_ik_preselected_grasp_success` 会直接使用：

- cached `q_pregrasp`
- cached `q_grasp`
- cached `pregrasp -> grasp` approach path

它只构造当前 q 到 `q_pregrasp` 的简单 joint path，并把真正的最终 approach 推迟到选中 transport pair 后执行。

### 抓后上抬

pair-first 模式会先做强制短 straight lift，然后再进入运输。这样避免物体刚抓住时直接做大范围运输导致碰撞。

### transport hover

transport 阶段优先复用 fast-chain 的 `q_hover`：

```text
plan_to_joint_state(start_q, q_hover)
```

对应参数：

```bash
--transport-use-prefilter-q-goal
--transport-prefilter-q-goal-max-trials 1
--transport-prefilter-q-goal-timeout 2.0
--transport-prefilter-q-goal-num-trajopt-seeds 1
```

这个 joint-space MotionGen 成功后，还会检查末端 TCP 是否真的落到原始 hover pose 附近。若 terminal pose 对不上，则不能接受。

如果 q-goal transport 不成功，才继续走原有 pose MotionGen / fallback 逻辑，具体是否允许取决于当前 fallback 参数。

### final contact

final contact 顺序是：

1. 用 cached `q_release` 构造 `hover -> release` 直线段。
2. 如果失败，用 release endpoint IK 再构造直线段。
3. 如果仍失败，走 constrained MotionGen / segmented fallback。

这里的 final contact 是真正执行前的最终验证；前面的 release IK 只是帮我们提前筛 endpoint。

## cuRobo CUDA Graph IK 机制

底层 `solve_batch_start_goal_ik` 的行为：

1. `use_cuda_graph_batch=True`
2. 请求数量大于 1
3. 没有 active attached object
4. CUDA graph solver 没被禁用

满足这些条件时进入 CUDA Graph batch IK。

固定 batch 行为：

- `requested <= 16`: padding 到 16，复用 `(batch=16, seeds=32)` 的 solver。
- `requested > 16`: 拆成多个 16-size chunk。
- 每个 chunk 的输出只取真实 requested 部分，padding 结果丢弃。

solver cache key：

```text
(fixed_batch_size, num_seeds)
```

世界模型变化时会尝试 update world；如果 world 更新失败，清空 solver cache 并记录 disabled reason。抓取 contact 阶段会显式 invalidate graph solver，因为 active object 从 collision world 中移除。

profile 里会记录：

- `ik_batch_call_count`
- `ik_goal_count`
- `ik_cuda_graph_solve_count`
- `ik_cuda_graph_requested_goal_count`
- `ik_cuda_graph_padded_goal_count`
- IK batch size histogram

## Prefetch 与 fast-chain 的关系

如果 next-cycle prefetch 命中，当前 cycle 可以直接拿后台预计算好的：

- `selected_joint_chain`
- `grasp_choice`
- `place_choice`

此时会跳过当前 cycle 的 winner-chain IK 和 transport MotionGen，前提是 consume 时验证当前起点和场景仍在容差内。

如果 prefetch 没命中或结构无效，则正常跑 fast-chain 四图交集。

## 成功与失败分支

### 成功条件

至少存在一个 relation 满足：

```text
pregrasp IK 成功
grasp IK 成功
pregrasp->grasp 直线 approach 成功
hover IK 成功
release IK 成功
```

并且后续 transport / final contact / release 执行阶段没有失败。

### 常见失败状态

- `NO_GRASP_IK`: pregrasp 有解但 grasp 无解，或 grasp approach 直线验证失败。
- `NO_VALID_PAIRED_RELATION`: grasp 侧有解，但 hover/release 侧没有可匹配 relation。
- `NO_PLACE_CANDIDATES`: 该 `T_tcp_obj` 生成不了合法 place candidate。
- `NO_PLACE_IK_PAIR`: 有 grasp IK，但没有 hover/release IK pair。

### legacy fallback

当前默认：

```bash
--no-fast-chain-allow-legacy-fallback
```

也就是 fast-chain 找不到完整链路就 fail-fast，不再用旧的大范围 fallback 把问题藏起来。

如果打开：

```bash
--fast-chain-allow-legacy-fallback
```

则会回到旧的 per-grasp place IK ranking / broader MotionGen 搜索。但这会变慢，也会让 fast-chain 的真实命中率不容易分析。

## 这套逻辑的优点

1. 大部分失败在 IK 层提前暴露，避免 MotionGen 长时间失败。
2. 固定 batch size 16 适合 CUDA Graph，避免动态 shape 导致 10 倍级别变慢。
3. 抓取 relation 和放置 relation 通过 key 绑定，避免“抓法和放法不一致”的错误链路。
4. `q_hover/q_release` 可以复用于 transport/final contact，减少 MotionGen 自己重新找 IK 分支。
5. 第二阶段只在特定失败模式触发，不会无脑扩大所有搜索。

## 主要限制

1. IK 交集只证明端点可达，不证明中间运输路径一定可行。
2. release IK 是 endpoint check，真正 `hover -> release` 仍要在 final contact 阶段验证。
3. 增加 yaw/spin 扩展会增加 IK chunk 数；虽然每个 chunk 固定 16 很快，但候选太多仍会堆时间。
4. 固定 batch 16 对速度友好，但如果某类物体需要更多 relation，必须通过第二阶段扩展或分 chunk 处理。
5. 碰撞世界切换会影响 CUDA graph solver cache；尤其 grasp contact 前后 active object 是否在 world 里要保持一致。

## 当前推荐调试视角

看日志时优先看：

```text
fast-chain xxx；
pregrasp图 A/N
grasp图 B/N
hover图 C/M
release图 D/M
四图交集 E/M
最终链路 ...
状态 ...
```

判断方式：

- `pregrasp图` 高、`grasp图` 低：抓取接触姿态或 approach 有问题。
- `grasp图` 高、`hover/release` 全 0：放置姿态/yaw/spin/release relation 不可达，需要看第二阶段扩展。
- `四图交集` 有值但 transport 失败：IK 端点可达，但运输路径绕不过障碍或起点碰撞。
- `final_contact` 失败：release endpoint 可达，但 hover 到 release 的直线/受限下降不可行。

这也是现在优化的主线：先让四图交集尽量覆盖真实可行关系，再只把少量 top pair 送进昂贵的路径规划。

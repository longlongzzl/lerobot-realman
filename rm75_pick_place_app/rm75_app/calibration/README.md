# RM75 Calibration Workflow

推荐顺序：

1. 固定/主相机视觉标定

   机械臂走 `base_reference_qpos_deg`，采集图像、FK link pose 和机器人 mask，用 EasyHEC/nvdiffrast 优化主相机外参 `T_R_Cg`。

   ```bash
   python -m rm75_app calib-base -- \
     --config assets/calibration/rm75_calibration_config.example.json \
     --execute-real \
     --mask-dir <mask_dir> \
     --detect-board-after
   ```

   也可以手摆机械臂采集，不让程序自动移动：

   ```bash
   python -m rm75_app calib-base -- \
     --config assets/calibration/rm75_calibration_config.example.json \
     --manual-capture \
     --manual-count 8 \
     --interactive-sam2 \
     --detect-board-after \
     --save-to-app-assets
   ```

2. 主相机真机点板精度检查

   这一步只检查全局/主相机外参链路：主相机识别 ChArUco 板，把板上固定角点变到机器人基座坐标，然后可选让机器人移动到该点上方。默认只预览和记录，不会动机械臂。

   ```bash
   python -m rm75_app calib-base-point-check
   ```

   真机只移动到安全悬停。默认是物理尖端在板面法向上方 30 cm；如果没有配置尖端偏移，程序会假设控制器 TCP 就是物理尖端，所以这一步只用于高位粗看。

   ```bash
   python -m rm75_app calib-base-point-check -- --execute-real
   ```

   如果夹爪或探针尖不等于控制器当前 TCP，必须先配置从控制器 TCP 到真实接触尖端的工具坐标偏移。比如尖端在当前 tool -Z 方向 12 cm 处，可以先用高悬停验证符号：

   ```bash
   python -m rm75_app calib-base-point-check -- \
     --execute-real \
     --tip-offset-tool-m 0 0 -0.12 \
     --hover-height-m 0.30
   ```

   确认高位投影和方向都对之后，再做低速接近。默认最终仍停在板面上方 5 cm；低于 2 cm 需要显式加 `--allow-low-final-offset`：

   ```bash
   python -m rm75_app calib-base-point-check -- \
     --execute-real \
     --touch \
     --tip-offset-tool-m 0 0 -0.12 \
     --final-offset-m 0.05
   ```

   移动标定板后反复按 Enter，程序每次都会重新识别板位姿并指向同一个板坐标角点。输出 `point_accuracy_report.json`、原图和叠图。注意：如果夹爪/探针尖没有设成 TCP 或没有通过 `--tip-offset-tool-m` 补偿，不能做低位接触误差判断。

3. 推荐：双相机移动板腕带标定

   主相机外参先固定准确；之后你可以移动机械臂，也可以在每张之间移动标定板。每次采样同时保存主相机图、腕带图和当前机械臂位姿。主相机为每一帧提供当时的 `T_R_P_i`，腕带相机用同一帧角点联合优化 `T_E_Cw`，最后直接输出交叉误差。

   ```bash
   python -m rm75_app calib-dual-board -- \
     --config assets/calibration/rm75_calibration_config.example.json \
     --base-camera-run-dir <base_run_dir> \
     --manual-count 40 \
     --preview
   ```

   输出 `ee_T_wrist_camera.npy/json`、`calibration_report.json`、`dual_board_report.json` 和双相机叠图。报告里的 `cross_camera_translation_mm_*` / `cross_camera_rotation_deg_*` 是腕带反推标定板和主相机反推标定板之间的残差。

4. 可选：固定板锚定

   主/固定相机观测桌上固定 ChArUco 板，把板固定到机器人基座坐标系，输出 `T_R_P`。

   ```bash
   python -m rm75_app calib-board-anchor -- \
     --config assets/calibration/rm75_calibration_config.example.json \
     --base-camera-run-dir <base_run_dir> \
     --manual-capture \
     --manual-count 20 \
     --preview
   ```

5. 可选：腕带锚定标定

   腕带相机从多个末端姿态观测同一块固定板，使用固定 `T_R_P` 和原始角点重投影优化 `T_E_Cw`。

   ```bash
   python -m rm75_app calib-wrist-anchor -- \
     --config assets/calibration/rm75_calibration_config.example.json \
     --board-anchor-run-dir <board_anchor_run_dir> \
     --manual-capture \
     --manual-count 40 \
     --preview
   ```

6. 可选双相机固定板联合优化

   主相机和腕带相机同时观测同一块板，联合细化/交叉检查 `T_R_P` 和 `T_E_Cw`。这一步应只接受漂移和重投影误差都合理的结果。

   ```bash
   python -m rm75_app calib-joint-board -- \
     --config assets/calibration/rm75_calibration_config.example.json \
     --global-board-run-dir <board_anchor_run_dir> \
     --wrist-board-run-dir <wrist_run_dir> \
     --base-camera-run-dir <base_run_dir>
   ```

7. 可选腕带本体视觉 sanity check

   如果腕带相机能看到机械臂本体，可以在标定板结果之后做一轮小幅视觉微调。这个流程是辅助检查，不是主标定路线；默认不会覆盖标定板外参，只有显式 accepted 的微调才应该被 runtime 自动使用。

   ```bash
   /home/zhangzhao/anaconda3/envs/foundationpose310/bin/python -m rm75_app calib-wrist-visual-refine -- \
     --wrist-camera-run-dir <wrist_run_dir>
   ```

   默认使用 `assets/calibration/rm75_calibration_config.example.json`，自动找最新 accepted 的腕带标定作为初值，并手动采集 12 张。优化器默认用 nvdiffrast 可微 silhouette；如果没有 `--mask-dir` 或 `--mask-npy`，会退化为边缘距离目标，容易受背景边缘影响。输出 `ee_T_wrist_camera_refined.npy/json`、`delta_board_to_refined.npy/json`、`visual_refine_report.json` 和轮廓叠图。叠图里黄色是真实图像边缘，红色是初始外参投影，绿色是优化后投影。

8. 最终双相机联合校验

   主相机和腕带相机同时看同一块板，用主相机外参、腕带 `ee_T_wrist_camera`、机械臂 FK 交叉验证。

   ```bash
   python -m rm75_app calib-check -- \
     --config assets/calibration/rm75_calibration_config.example.json \
     --execute-real \
     --base-camera-run-dir <base_run_dir> \
     --wrist-camera-run-dir <wrist_run_dir> \
     --preview
   ```

现有经典腕带 ChArUco 手眼标定仍可作为 fallback：

```bash
python -m rm75_app calib-wrist -- \
  --config assets/calibration/rm75_calibration_config.example.json \
  --auto-generate-samples \
  --dry-run-plan

python -m rm75_app calib-wrist -- \
  --config assets/calibration/rm75_calibration_config.example.json \
  --execute-real \
  --auto-generate-samples \
  --preview
```

手动采集版本不会自动移动机械臂。它默认打开实时预览，显示角点、坐标轴和重投影误差；你把机械臂摆到不同姿态，按回车或空格拍照，输入 `q` 或按 Esc 后结束并自动求解。没有图形界面时加 `--manual-terminal` 退回纯终端采集。

```bash
/home/zhangzhao/anaconda3/envs/foundationpose310/bin/python -m rm75_app calib-wrist -- \
  --config assets/calibration/rm75_calibration_config.example.json \
  --manual-capture \
  --manual-count 18
```

输出：

- `planned_qpos_deg.json`: 腕带采样计划。
- `observations.json`: 原始帧、qpos、FK、板检测结果。
- `calibration_report.json`: 统一 schema 报告，包含 `schema_version`、`run_kind`、`accepted`、`transforms` 和 `metrics`。
- `visual_refine_report.json`: 腕带本体视觉 sanity check 报告，同样包含 `accepted` 和拒绝原因。
- `report.html`: 离线可视化报告，包含矩阵、accepted/rejected 状态、帧数、指标、异常帧和叠图。
- `camera_extrinsic_opencv.npy/json`: 主相机视觉标定结果。
- `base_T_board.npy/json` 或 `T_R_P` 等价输出：固定板在机器人基座下的位姿。
- `ee_T_wrist_camera.npy/json`: 腕带相机相对末端的标定结果。
- `ee_T_wrist_camera_joint.npy/json`: 双相机联合优化后的腕带相机外参。
- `ee_T_wrist_camera_refined.npy/json`: accepted 的本体轮廓微调外参。

Runtime 腕带外参查找优先级：

1. 显式配置或 CLI 传入的外参路径。
2. accepted 的 `*_wrist_camera_dual_board/ee_T_wrist_camera.npy`。
3. accepted 的 `*_wrist_camera_board_anchor/ee_T_wrist_camera.npy`。
4. accepted 的 `*_two_camera_joint_board/ee_T_wrist_camera_joint.npy`。
5. accepted 且带 metadata 的 `*_wrist_camera_board/ee_T_wrist_camera_refined.npy`。
6. 非 rejected、非 identity 的普通 `*_wrist_camera_board/ee_T_wrist_camera.npy`。

建议：

- 腕带采样至少 15 帧，末端姿态要有明显旋转变化；只平移会让手眼标定退化。
- `--preview` 里绿点/坐标轴稳定贴住板时再继续，橙色表示检测到了但质量未过门槛。
- 联合校验默认 `3cm / 5deg` 以内算通过；如果夹具或板固定不牢，应收紧机械结构再调阈值。
- 插入前/入孔前的局部视觉伺服是后续独立层，不属于这套全局相机和腕带相机标定栈。

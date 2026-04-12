#!/usr/bin/env bash
set -euo pipefail

# ====== 需要打包的 commit 列表 ======
COMMITS=(
  3a46cfcaabe4309604b74390c76b9cee7c54be8a
  5c2d44babd5ddc9840b4696dffa130b175fd059e
  a61bb3ec2c04f73c682282151615fcc351f8e483
  3dc7221e4535cc4821335e01eb963755f858bf72
  48d3cdbc560af0ccbf120c688eb32b4e83a035ad
  25f5137048410b5402641cb0835afe9657b22b9d
)

OUT_ZIP="easyhec_changed_files.zip"
TMP_LIST="$(mktemp)"

echo "[INFO] Collecting changed files from commits..."

# 收集所有修改过的文件（Added / Modified / Renamed）
for c in "${COMMITS[@]}"; do
  git diff-tree --no-commit-id --name-only -r "$c" >> "$TMP_LIST"
done

# 去重 + 排序
sort -u "$TMP_LIST" > "${TMP_LIST}.uniq"

echo "[INFO] Files to be packed:"
cat "${TMP_LIST}.uniq"

echo "[INFO] Creating zip archive: ${OUT_ZIP}"

# 打包当前工作区中对应路径的文件
git archive \
  --format=zip \
  --output="$OUT_ZIP" \
  HEAD \
  $(cat "${TMP_LIST}.uniq")

rm -f "$TMP_LIST" "${TMP_LIST}.uniq"

echo "[DONE] Archive created: ${OUT_ZIP}"

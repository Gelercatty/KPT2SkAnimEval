#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量将 22 关节 SMPL 动作从世界坐标转为本地坐标（根关节固定到原点）。
输入目录下仅包含 .npy 的动作序列文件（[F,22,3] 或 [22,3]）。

必选参数：
  --globle_dir   世界坐标序列所在目录（保持拼写，与需求一致）
  --local_dir    输出目录

可选参数：
  --remove-root-rot   在去平移基础上，估计每帧根坐标系并与第0帧对齐（去根旋转）
  --overwrite         若输出已存在则覆盖
"""

import argparse
import os
import sys
import numpy as np

# 关节索引（如你的顺序不同，改这里）
PELVIS = 0
L_HIP  = 1
R_HIP  = 2
SPINE1 = 3

def safe_normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def build_root_basis(j):
    """
    用关节估计根坐标系:
      up      ~ normalize(spine1 - pelvis)
      right   ~ normalize(R_hip - L_hip)
      forward ~ normalize(cross(up, right))
    返回 3x3，行向量为 [right; up; forward]
    """
    pelvis  = j[PELVIS]
    up_vec  = j[SPINE1] - pelvis
    right_v = j[R_HIP] - j[L_HIP]

    up      = safe_normalize(up_vec)
    right   = safe_normalize(right_v)

    forward = np.cross(up, right)
    forward = safe_normalize(forward)
    right   = np.cross(forward, up)
    right   = safe_normalize(right)
    up      = safe_normalize(up)

    return np.stack([right, up, forward], axis=0)  # (3,3)

def to_local_translation_only(arr):
    """每帧减去 pelvis 平移。"""
    return arr - arr[:, [PELVIS], :]

def to_local_translation_rotation(arr):
    """
    去平移 + 去根旋转：将每帧对齐到第0帧的根坐标系。
      centered_f = X_f - pelvis_f
      B0 = 第0帧根基
      Bf = 第f帧根基
      R_f = B0 @ Bf^T
      X_local_f = centered_f @ R_f^T
    """
    F = arr.shape[0]
    centered = to_local_translation_only(arr)
    bases = np.zeros((F, 3, 3), dtype=np.float64)
    for f in range(F):
        bases[f] = build_root_basis(arr[f])
    B0 = bases[0]
    out = np.empty_like(centered)
    for f in range(F):
        Rf = B0 @ bases[f].T
        out[f] = centered[f] @ Rf.T
    return out

def convert_file(in_path, out_path, remove_root_rot=False, overwrite=False):
    if (not overwrite) and os.path.exists(out_path):
        print(f"跳过（已存在）：{out_path}")
        return

    try:
        arr = np.load(in_path)
    except Exception as e:
        print(f"读取失败 {in_path}: {e}", file=sys.stderr)
        return

    if arr.ndim == 2 and arr.shape == (22,3):
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (22,3):
        print(f"形状不符，跳过 {in_path}，got {arr.shape}", file=sys.stderr)
        return

    arr = arr.astype(np.float64)

    if remove_root_rot:
        local = to_local_translation_rotation(arr)
        suffix = "_local_TR"
    else:
        local = to_local_translation_only(arr)
        suffix = "_local"

    # 保持原文件名，添加后缀
    base = os.path.splitext(os.path.basename(in_path))[0]
    # out_name = base + suffix + ".npy"
    out_name = base + ".npy"
    out_dir  = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, out_name)
    np.save(save_path, local)

    # 简短校验输出
    pelvis = local[:, PELVIS, :]
    d0 = np.linalg.norm(pelvis, axis=1)
    step = np.linalg.norm(np.diff(pelvis, axis=0), axis=1) if len(local) > 1 else np.array([0.0])
    print(f"✅ {os.path.basename(in_path)} -> {out_name} | mean|pelvis|={d0.mean():.4f}, std={d0.std():.4f}, max_step={step.max():.4f}")

def main():
    ap = argparse.ArgumentParser(description="Batch convert SMPL(22) sequences from world to local coordinates.")
    ap.add_argument("--globle_dir", required=True, type=str, help="世界坐标序列目录（保持拼写）")
    ap.add_argument("--local_dir",  required=True, type=str, help="输出目录")
    ap.add_argument("--remove-root-rot", action="store_true", help="去平移 + 去根旋转（对齐到第0帧根坐标系）")
    ap.add_argument("--overwrite", action="store_true", help="若输出存在则覆盖")
    args = ap.parse_args()

    in_dir  = os.path.abspath(args.globle_dir)
    out_dir = os.path.abspath(args.local_dir)

    if not os.path.isdir(in_dir):
        print(f"输入目录不存在：{in_dir}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if f.lower().endswith(".npy")]
    if not files:
        print("输入目录中未找到 .npy 文件。")
        sys.exit(0)

    print(f"共 {len(files)} 个文件，开始转换（remove_root_rot={args.remove_root_rot}) …")
    for fname in files:
        in_path  = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)
        convert_file(in_path, out_path, remove_root_rot=args.remove_root_rot, overwrite=args.overwrite)

    print("全部完成。")

if __name__ == "__main__":
    main()

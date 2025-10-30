#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys

def decide_world_or_local(arr, eps_center=0.05, eps_step=0.01):
    """
    eps_center: 判断是否围绕原点的阈值（米），默认 5cm
    eps_step:   判断每步帧间漂移是否近乎 0 的阈值（米），默认 1cm
    """
    if arr.ndim == 2 and arr.shape == (22,3):
        arr = arr[None,...]
    assert arr.ndim == 3 and arr.shape[1:] == (22,3), f"shape expected [F,22,3], got {arr.shape}"
    F = arr.shape[0]
    pelvis = arr[:,0,:]  # 约定 0 为 pelvis

    # 距原点的统计
    d0 = np.linalg.norm(pelvis, axis=1)             # 每帧 pelvis 到原点的距离
    d0_mean, d0_std, d0_max = d0.mean(), d0.std(), d0.max()

    # 帧间位移（是否有全局轨迹）
    step = np.linalg.norm(np.diff(pelvis, axis=0), axis=1) if F>1 else np.array([0.0])
    step_med, step_max = np.median(step), step.max() if step.size>0 else (0.0, 0.0)

    # 去掉 pelvis 平移后的包围盒尺度（人体尺寸参考）
    centered = arr - arr[:,[0],:]                  # 每帧减去 pelvis
    bbox_min = centered.reshape(F,-1,3).min(axis=(0,1))
    bbox_max = centered.reshape(F,-1,3).max(axis=(0,1))
    bbox_size = bbox_max - bbox_min               # 典型应在 ~[0.3, 2] 米范围

    # 简单启发式判断
    local_like = (d0_mean < eps_center and d0_std < eps_center and d0_max < 3*eps_center and step_med < eps_step)
    guess = "本地坐标 (local/body)" if local_like else "世界坐标 (world/global)"

    print("=== Pelvis 原点统计（米） ===")
    print(f"mean|pelvis|: {d0_mean:.4f}, std: {d0_std:.4f}, max: {d0_max:.4f}")
    print("=== Pelvis 帧间位移（米） ===")
    print(f"median step:  {step_med:.4f}, max step: {step_max:.4f}")
    print("=== 去平移后的人体包围盒尺寸（米） ===")
    print(f"bbox size xyz: {bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f}")
    print("=== 结论（启发式） ===")
    print(f"猜测：{guess}")
    return guess

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python detect_coord_space.py your_seq.npy")
        sys.exit(1)
    arr = np.load(sys.argv[1])
    decide_world_or_local(arr)

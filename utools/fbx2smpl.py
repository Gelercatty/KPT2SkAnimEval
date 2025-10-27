#!/usr/bin/env python3
"""
fbx2smpl.py — 将单个 FBX 动画导出为 SMPL-24 关节序列 [frames, 24, 3]

依赖：Blender 3.x（可无界面运行）。脚本会清空当前场景后导入 FBX。

用法示例（Windows / PowerShell）：
  blender -b -P fbx2smpl.py -- \
    --input "C:/data/walk.fbx" \
    --out   "C:/out/walk.json" \
    --format json \
    --space root \
    --fps 0 \
    
参数说明：
  --input   输入 FBX 文件路径（单文件）。
  --out     输出文件路径（json/csv/npy）。
  --format  导出格式：json（默认）/csv/npy。
  --space   坐标系：world 或 root（以 Pelvis 为原点；默认 root）。
  --invert-z 可选，Z 取反以切换左右手系（Blender 为右手系，Unity 常用左手系）。
  --fps     采样帧率，0 表示“逐格采样”使用时间线原帧（默认 0）。>0 时将按指定 FPS 线性采样（忽略场景 FPS）。

说明：
- 该脚本使用“骨骼节点的 head 点”作为关节位置近似；SMPL 官方关节通常由网格回归得来，
  若需严格对齐请提供回归权重或在 DCC 中烘焙到匹配的骨点。
- 映射表兼容常见命名（Mixamo / HumanIK / UE），找不到的骨将输出 [0,0,0]。
- 输出关节顺序固定为 SMPL-24（见 SMPL24_NAMES）。
"""

import argparse
import os
import sys
import json

try:
    import numpy as np
except Exception:
    np = None

import bpy
from mathutils import Vector

SMPL24_NAMES = [
    'Pelvis',        # 0
    'L_Hip',         # 1
    'R_Hip',         # 2
    'Spine1',        # 3
    'L_Knee',        # 4
    'R_Knee',        # 5
    'Spine2',        # 6
    'L_Ankle',       # 7
    'R_Ankle',       # 8
    'Spine3',        # 9
    'L_Foot',        # 10 (toe)
    'R_Foot',        # 11 (toe)
    'Neck',          # 12
    'L_Collar',      # 13 (clavicle)
    'R_Collar',      # 14
    'Head',          # 15
    'L_Shoulder',    # 16 (upper arm start)
    'R_Shoulder',    # 17
    'L_Elbow',       # 18
    'R_Elbow',       # 19
    'L_Wrist',       # 20
    'R_Wrist',       # 21
    'L_Hand',        # 22 (掌心近似: 五指近端均值，回退 Wrist)
    'R_Hand',        # 23
]

# 关节候选名（常见：Mixamo, HumanIK, UE4/UE5, 通用），按 SMPL 索引给出
C = lambda *xs: list(xs)
CANDS = {
    0:  C('Hips','hip','pelvis','root','mixamorig:Hips','pelvis_M'),
    1:  C('LeftUpLeg','LeftUpperLeg','thigh_l','thigh.L','mixamorig:LeftUpLeg','pelvis.L','UE4_Mannequin:thigh_l'),
    2:  C('RightUpLeg','RightUpperLeg','thigh_r','thigh.R','mixamorig:RightUpLeg','pelvis.R','UE4_Mannequin:thigh_r'),
    3:  C('Spine','spine','spine_01','mixamorig:Spine'),
    4:  C('LeftLeg','LeftLowerLeg','calf_l','shin_l','shin.L','mixamorig:LeftLeg','UE4_Mannequin:calf_l'),
    5:  C('RightLeg','RightLowerLeg','calf_r','shin_r','shin.R','mixamorig:RightLeg','UE4_Mannequin:calf_r'),
    6:  C('Chest','Spine1','spine_02','mixamorig:Spine1'),
    7:  C('LeftFoot','foot_l','foot.L','mixamorig:LeftFoot','UE4_Mannequin:foot_l'),
    8:  C('RightFoot','foot_r','foot.R','mixamorig:RightFoot','UE4_Mannequin:foot_r'),
    9:  C('UpperChest','Spine2','spine_03','chest','mixamorig:Spine2'),
    10: C('LeftToeBase','LeftToe','ball_l','toe_l','toe.L','mixamorig:LeftToeBase'),
    11: C('RightToeBase','RightToe','ball_r','toe_r','toe.R','mixamorig:RightToeBase'),
    12: C('Neck','neck','mixamorig:Neck'),
    13: C('LeftShoulder','clavicle_l','clavicle.L','mixamorig:LeftShoulder','shoulder_l','shoulder.L'),
    14: C('RightShoulder','clavicle_r','clavicle.R','mixamorig:RightShoulder','shoulder_r','shoulder.R'),
    15: C('Head','head','mixamorig:Head','head_top'),
    16: C('LeftArm','LeftUpperArm','upperarm_l','upper_arm_l','upper_arm.L','mixamorig:LeftArm'),
    17: C('RightArm','RightUpperArm','upperarm_r','upper_arm_r','upper_arm.R','mixamorig:RightArm'),
    18: C('LeftForeArm','LeftLowerArm','lowerarm_l','forearm_l','forearm.L','mixamorig:LeftForeArm'),
    19: C('RightForeArm','RightLowerArm','lowerarm_r','forearm_r','forearm.R','mixamorig:RightForeArm'),
    20: C('LeftHand','hand_l','hand.L','mixamorig:LeftHand'),
    21: C('RightHand','hand_r','hand.R','mixamorig:RightHand'),
}

LEFT_PROX = [
    'LeftMiddle1','LeftIndex1','LeftRing1','LeftPinky1','LeftThumb1',
    'mixamorig:LeftHandMiddle1','mixamorig:LeftHandIndex1','mixamorig:LeftHandRing1','mixamorig:LeftHandPinky1','mixamorig:LeftHandThumb1',
    'middle_metacarpal.L','index_metacarpal.L','ring_metacarpal.L','pinky_metacarpal.L','thumb_metacarpal.L',
]
RIGHT_PROX = [
    'RightMiddle1','RightIndex1','RightRing1','RightPinky1','RightThumb1',
    'mixamorig:RightHandMiddle1','mixamorig:RightHandIndex1','mixamorig:RightHandRing1','mixamorig:RightHandPinky1','mixamorig:RightHandThumb1',
    'middle_metacarpal.R','index_metacarpal.R','ring_metacarpal.R','pinky_metacarpal.R','thumb_metacarpal.R',
]


def reset_scene():
    bpy.ops.wm.read_homefile(use_empty=True)


def import_fbx(path: str):
    bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=False)


def pick_armature():
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def build_maps(pose_bones):
    nm = {pb.name: pb for pb in pose_bones}
    low = {pb.name.lower(): pb for pb in pose_bones}
    return nm, low


def find_pbone(cands, name_map, lower_map):
    for n in cands:
        if n in name_map:
            return name_map[n]
        ln = n.lower()
        if ln in lower_map:
            return lower_map[ln]
    return None


def head_world(obj_eval, pbone):
    return obj_eval.matrix_world @ pbone.head


def hand_center(obj_eval, left: bool, name_map, lower_map, wrist_p):
    cands = LEFT_PROX if left else RIGHT_PROX
    pts = []
    for n in cands:
        pb = find_pbone([n], name_map, lower_map)
        if pb is not None:
            pts.append(head_world(obj_eval, pb))
    if pts:
        v = Vector((0.0,0.0,0.0))
        for p in pts:
            v += p
        return v / len(pts)
    return wrist_p


def sample_fbx_to_array(fbx_path: str, fps: float, space: str, invert_z: bool):
    reset_scene()
    import_fbx(fbx_path)
    arm = pick_armature()
    if arm is None:
        raise RuntimeError('未找到 Armature（骨架）')

    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    arm_eval = arm.evaluated_get(deps)

    name_map, lower_map = build_maps(arm_eval.pose.bones)

    # 预取 22 个直接骨
    pbones = [None]*22
    for i in range(22):
        pbones[i] = find_pbone(CANDS.get(i, []), name_map, lower_map)

    # 时间采样：逐格 or 按 FPS
    t_start = scene.frame_start
    t_end   = scene.frame_end

    frames = []
    if fps and fps > 0:
        # 以秒为单位采样，使用场景 FPS 做 frame_to_time 映射
        scene_fps = scene.render.fps / scene.render.fps_base
        dur_sec = (t_end - t_start) / scene_fps
        step = 1.0 / float(fps)
        k = 0.0
        while k <= dur_sec + 1e-8:
            f = t_start + int(round(k * scene_fps))
            f = max(t_start, min(t_end, f))
            if not frames or f != frames[-1]:
                frames.append(f)
            k += step
    else:
        frames = list(range(t_start, t_end+1))

    data = [[[0.0,0.0,0.0] for _ in range(24)] for __ in frames]

    for i, f in enumerate(frames):
        scene.frame_set(f)
        deps.update()
        arm_eval = arm.evaluated_get(deps)

        pelvis = pbones[0]
        pelvis_w = head_world(arm_eval, pelvis) if pelvis is not None else arm_eval.matrix_world.translation

        # 0..21
        for j in range(22):
            pb = pbones[j]
            if pb is None:
                p = Vector((0.0,0.0,0.0))
            else:
                p = head_world(arm_eval, pb)
            if space == 'root':
                p = p - pelvis_w
            if invert_z:
                p = Vector((p.x, p.y, -p.z))
            data[i][j] = [float(p.x), float(p.y), float(p.z)]

        # 22/23: 手掌中心
        lwrist = Vector(data[i][20])
        rwrist = Vector(data[i][21])
        lc = hand_center(arm_eval, True,  name_map, lower_map, lwrist)
        rc = hand_center(arm_eval, False, name_map, lower_map, rwrist)
        if space == 'root':
            lc -= pelvis_w
            rc -= pelvis_w
        if invert_z:
            lc = Vector((lc.x, lc.y, -lc.z))
            rc = Vector((rc.x, rc.y, -rc.z))
        data[i][22] = [float(lc.x), float(lc.y), float(lc.z)]
        data[i][23] = [float(rc.x), float(rc.y), float(rc.z)]

    return data, frames


def save_array(data, out_path, fmt):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    base, ext = os.path.splitext(out_path)
    if fmt == 'json':
        with open(out_path if ext else base+'.json', 'w', encoding='utf-8') as f:
            json.dump(data, f)
    elif fmt == 'csv':
        with open(out_path if ext else base+'.csv', 'w', encoding='utf-8') as f:
            f.write('frame,joint,x,y,z\n')
            for fi in range(len(data)):
                for ji in range(24):
                    x,y,z = data[fi][ji]
                    f.write(f"{fi},{ji},{x:.7g},{y:.7g},{z:.7g}\n")
    elif fmt == 'npy':
        if np is None:
            raise RuntimeError('numpy 不可用，无法写入 .npy')
        arr = np.asarray(data, dtype=np.float32)
        np.save(out_path if ext else base+'.npy', arr)
    else:
        raise ValueError('未知格式')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='输入 FBX 文件')
    ap.add_argument('--out',   required=True, help='输出文件路径（含扩展名或不含）')
    ap.add_argument('--format', choices=['json','csv','npy'], default='json')
    ap.add_argument('--space', choices=['world','root'], default='root')
    ap.add_argument('--invert-z', action='store_true', help='Z 取反切换左右手系')
    ap.add_argument('--fps', type=float, default=0.0, help='采样帧率；0 表示逐格采样')
    args, _ = ap.parse_known_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])

    in_path = os.path.abspath(args.input)
    out_path = os.path.abspath(args.out)

    if not os.path.isfile(in_path):
        raise FileNotFoundError(in_path)

    data, frames = sample_fbx_to_array(in_path, args.fps, args.space, args.invert_z)
    save_array(data, out_path, args.format)
    print(f"[OK] Exported {len(frames)} frames to {out_path}")

if __name__ == '__main__':
    main()

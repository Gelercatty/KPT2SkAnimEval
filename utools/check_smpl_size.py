import argparse
import json
import numpy as np
import sys

# 常见的 SMPL-22 关节顺序对应的边 (parent, child, name)
SMPL22_EDGES = [
    (0, 1,  "pelvis->L_hip"),
    (1, 4,  "L_hip->L_knee"),
    (4, 7,  "L_knee->L_ankle"),
    (7, 10, "L_ankle->L_foot"),

    (0, 2,  "pelvis->R_hip"),
    (2, 5,  "R_hip->R_knee"),
    (5, 8,  "R_knee->R_ankle"),
    (8, 11, "R_ankle->R_foot"),

    (0, 3,  "pelvis->spine1"),
    (3, 6,  "spine1->spine2"),
    (6, 9,  "spine2->spine3"),
    (9, 12, "spine3->neck"),
    (12, 15,"neck->head"),

    (12, 13,"neck->L_collar"),
    (13, 16,"L_collar->L_shoulder"),
    (16, 18,"L_shoulder->L_elbow"),
    (18, 20,"L_elbow->L_wrist"),

    (12, 14,"neck->R_collar"),
    (14, 17,"R_collar->R_shoulder"),
    (17, 19,"R_shoulder->R_elbow"),
    (19, 21,"R_elbow->R_wrist"),
]

def load_edges(path):
    if not path:
        return SMPL22_EDGES
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    edges = []
    for it in data:
        if isinstance(it, dict):
            a = it["a"]; b = it["b"]; name = it.get("name", f"{a}->{b}")
        else:
            a, b = it[0], it[1]
            name = it[2] if len(it) > 2 else f"{a}->{b}"
        edges.append((int(a), int(b), str(name)))
    return edges

def compute_lengths(arr, edges):
    """arr: [F,22,3] -> lengths [F,B], names[list]"""
    F = arr.shape[0]
    B = len(edges)
    L = np.zeros((F, B), dtype=np.float64)
    names = []
    for k, (a, b, name) in enumerate(edges):
        if max(a, b) >= 22 or min(a, b) < 0:
            raise ValueError(f"边索引越界: ({a},{b})")
        v = arr[:, b, :] - arr[:, a, :]
        L[:, k] = np.linalg.norm(v, axis=-1)
        names.append(name)
    return L, names

def main():
    ap = argparse.ArgumentParser(description="Print SMPL-22 bone lengths from [F,22,3] .npy")
    ap.add_argument("npy_path", type=str, help=".npy 文件路径（[F,22,3] 或 [22,3]）")
    ap.add_argument("-f","--frame", type=int, default=None, help="仅输出第 N 帧（0-based）")
    ap.add_argument("--summary-only", action="store_true", help="只输出统计汇总")
    ap.add_argument("--custom-edges", type=str, default=None, help="自定义边 JSON")
    args = ap.parse_args()

    # 载入
    try:
        arr = np.load(args.npy_path)
    except Exception as e:
        print(f"读取失败：{e}", file=sys.stderr); sys.exit(1)

    # 标准化形状
    if arr.ndim == 2 and arr.shape == (22, 3):
        arr = arr[None, ...]  # -> [1,22,3]
    if arr.ndim != 3 or arr.shape[1:] != (22, 3):
        print(f"数据形状不符，期望 [F,22,3]，实际 {arr.shape}", file=sys.stderr); sys.exit(1)

    edges = load_edges(args.custom_edges)

    # 计算
    try:
        lengths, names = compute_lengths(arr, edges)
    except Exception as e:
        print(f"计算失败：{e}", file=sys.stderr); sys.exit(1)

    F, B = lengths.shape

    # 打印骨骼拓扑
    print("Bone edges (parent->child):")
    for (a,b,name) in edges:
        print(f"  [{a:02d}->{b:02d}] {name}")
    print("-"*50)

    # 打印逐帧
    if not args.summary_only:
        if args.frame is not None:
            if args.frame < 0 or args.frame >= F:
                print(f"--frame 越界：0 <= N < {F}", file=sys.stderr); sys.exit(1)
            i = args.frame
            print(f"[Frame {i}] bone lengths:")
            for k in range(B):
                print(f"  {names[k]:25s} : {lengths[i, k]:.6f}")
        else:
            # 全部帧
            for i in range(F):
                print(f"[Frame {i}] bone lengths:")
                for k in range(B):
                    print(f"  {names[k]:25s} : {lengths[i, k]:.6f}")
                if i != F-1:
                    print("-"*50)

    # 打印统计汇总
    print("="*50)
    print("Summary over frames (mean / std / min / max):")
    mean = lengths.mean(axis=0)
    std  = lengths.std(axis=0, ddof=0)
    minv = lengths.min(axis=0)
    maxv = lengths.max(axis=0)
    for k in range(B):
        print(f"  {names[k]:25s} : {mean[k]:.6f}  / {std[k]:.6f}  / {minv[k]:.6f}  / {maxv[k]:.6f}")

if __name__ == "__main__":
    main()
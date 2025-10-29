from __future__ import annotations
import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any
import joblib
import numpy as np
import torch
from tqdm import tqdm

try:
    import smplx
except Exception as e:
    raise SystemExit("smplx is required. Install with `pip install smplx`. Error: %s" % e)

SMPL24_NAMES = [
"pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
"spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
"neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
"left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]

#--------------------------------------- IO Helper ---------------------------------------#
def _load_any(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    if ext == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {k:data[k] for k in data.files}
    if ext == ".npy":
        arr = np.load(path, allow_pickle= True)

        return {"poses": arr}
    if ext == ".pkl" or ext == ".pickle":
        with open(path, "rb") as f:
            obj = joblib.load(f)
        if isinstance(obj, dict):
            return obj
        raise ValueError(f"Unsupported pickle object type:{type(obj)} in {path}")
    if ext == ".json":
        with open(path, "r", encoding= "utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported file extension: {ext}")


def _as_np(a: Any) -> np.ndarray:
    if a is None:
        return None
    a = np.asarray(a)
    return a


def _first_present(d: Dict[str, Any], keys) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


# --------------------------- Param canonicalzation --------------------------------------#

def canoicalize_params(raw: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    pose72 = _as_np(_first_present(raw, ["poses","pose","full_pose","pose_aa"]) )
    body69 = _as_np(_first_present(raw, ["body_pose", "pose_body"]))
    orient3 = _as_np(_first_present(raw, ["global_orient", "root_orient", "Rh", "orient"]))
    betas10 = _as_np(_first_present(raw, ["betas", "shape", "shape_params"]))
    transl3 = _as_np(_first_present(raw, ["transl", "Th", "trans", "translation"]))

    def squeeze_maybe(x):
        if x is None: return None
        x = np.asarray(x)
        while x.ndim > 2 and x.shape[1] == 1:
            x = x[:, 0]
        return x
    
    pose72 = squeeze_maybe(pose72)
    body69 = squeeze_maybe(body69)
    orient3 = squeeze_maybe(orient3)
    betas10 = squeeze_maybe(betas10)
    transl3 = squeeze_maybe(transl3)

    lengths = []
    for arr in (pose72, body69, orient3, betas10, transl3):
        if arr is None:
            continue
        if arr.ndim == 1:
        # single vector
            lengths.append(1)
        elif arr.ndim == 2:
            lengths.append(arr.shape[0])
        else:
            # Try to flatten leading dims into T
            arr = arr.reshape(arr.shape[0], -1)
            lengths.append(arr.shape[0])
    T = max(lengths) if lengths else 1

    if pose72 is not None:
        pose72 = pose72.reshape((-1, 72)) if pose72.ndim == 1 else pose72.reshape((-1, 72))
        go = pose72[:, :3]
        bp = pose72[:, 3:]
    else:
        if orient3 is None:
            go = np.zeros((T, 3), dtype=np.float32)
        else:
            go = orient3.reshape((-1, 3)) if orient3.ndim > 1 else orient3.reshape((1, 3))
        if body69 is None:
            bp = np.zeros((T, 69), dtype=np.float32)
        else:
            bp = body69.reshape((-1, 69)) if body69.ndim > 1 else body69.reshape((1, 69))

    # Betas
# Betas
    if betas10 is None:
        betas10 = np.zeros((1, 10), dtype=np.float32)
    else:
        betas10 = betas10.reshape((-1, betas10.shape[-1])) if betas10.ndim > 1 else betas10.reshape((1, -1))
        # Clamp to first 10 if longer
        if betas10.shape[-1] > 10:
            betas10 = betas10[:, :10]
        elif betas10.shape[-1] < 10:
        # pad with zeros
            pad = np.zeros((betas10.shape[0], 10 - betas10.shape[-1]), dtype=betas10.dtype)
            betas10 = np.concatenate([betas10, pad], axis=-1)

    # Translation
    if transl3 is None:
        transl3 = np.zeros((1, 3), dtype=np.float32)
    else:
        transl3 = transl3.reshape((-1, 3)) if transl3.ndim > 1 else transl3.reshape((1, 3))

    def bcast(x, dim):
        if x.shape[0] == 1 and T > 1:
            x = np.repeat(x, T, axis=0)
        assert x.shape[0] == T, f"Mismatched time length: expected {T}, got {x.shape[0]} for dim {dim}"
        return x.astype(np.float32)
    
    go = bcast(go, 3)
    bp = bcast(bp, 69)
    betas10 = bcast(betas10, 10)
    transl3 = bcast(transl3, 3)

    return go, bp, betas10, transl3, T


def build_smpl(model_dir: str, gender: str, device: torch.device):
    if not model_dir:
        env_dir = os.environ.get("SMPL_MODEL_DIR")
        if env_dir:
            model_dir = env_dir

    if not model_dir:
        raise FileNotFoundError(
        "SMPL model directory not provided. Use --model-dir or set SMPL_MODEL_DIR env var."
        )
    
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(model_dir)
    model = smplx.create(
        model_path=model_dir,
        model_type="smpl",
        gender = gender,
        use_pca = False,
        batch_size = 1,
    )
    return model.to(device)


def to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


@torch.no_grad()
def smpl_to_joints(model, go: np.ndarray, bp: np.ndarray, betas: np.ndarray, transl: np.ndarray,
                   device: torch.device, batch_size: int = 512) -> np.ndarray:
    T = go.shape[0]
    joints_out = np.zeros((T, 24, 3), dtype= np.float32)
    for  s in range(0, T, batch_size):
        e = min(T, s+batch_size)
        go_t = to_torch(go[s:e], device)
        bp_t = to_torch(bp[s:e], device)
        betas_t = to_torch(betas[s:e], device)
        transl_t = to_torch(transl[s:e], device)
        out = model(
            betas=betas_t,
            body_pose=bp_t,
            global_orient=go_t,
            transl=transl_t,
            return_verts=False,
        )


        j = out.joints # [B, 24, 3]
        joints_out[s:e] = j.detach().cpu().numpy().astype(np.float32)
    return joints_out


def find_files(root: Path) -> list[Path]:
    allow = {".npz", ".npy", ".pkl", ".pickle", ".json"}
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in allow:
            files.append(p)
    return files


def process_file(path: Path, out_dir: Path, model, device: torch.device, batch: int) -> Tuple[Path, Path]:
    raw = _load_any(path)
    go, bp, betas, transl, T = canoicalize_params(raw)
    joints = smpl_to_joints(model, go, bp, betas, transl, device=device, batch_size=batch)


    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    npy_path = out_dir / f"{stem}_kpt3d.npy"
    meta_path = out_dir / f"{stem}_meta.json"
    np.save(npy_path, joints)
    meta = {
    "source_file": str(path),
    "num_frames": int(joints.shape[0]),
    "joints_format": "SMPL_24",
    "units": "meters",
    "joint_names": SMPL24_NAMES,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return npy_path, meta_path

def parse_args():
    ap = argparse.ArgumentParser(
    description="Convert SMPL parameter sequences to 3D keypoint sequences (SMPL-24).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dir", required=True, type=str, help="Directory with SMPL parameter files")
    ap.add_argument("--out", required=True, type=str, help="Output directory for keypoints")
    ap.add_argument("--cuda", type=int, default=-1, help="GPU index (e.g., 0). Use -1 for CPU")
    ap.add_argument("--model-dir", type=str, default=None, help="Path to SMPL model files (or set SMPL_MODEL_DIR)")
    ap.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"], help="SMPL gender")
    ap.add_argument("--batch-size", type=int, default=512, help="Batch size for forward pass")
    return ap.parse_args()


def main():
    args = parse_args()


    device = torch.device("cpu")
    if args.cuda is not None and int(args.cuda) >= 0:
        cuda_idx = int(args.cuda)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_idx}")
    else:
        print("[WARN] CUDA requested but not available. Falling back to CPU.")


    model = build_smpl(args.model_dir, args.gender, device)


    in_root = Path(args.dir)
    out_root = Path(args.out)
    files = find_files(in_root)
    if not files:
        raise SystemExit(f"No parameter files found under: {in_root}")


    print(f"Found {len(files)} files. Writing outputs to: {out_root}")
    for f in tqdm(files, desc="Converting", unit="file"):
        try:
            npy_path, meta_path = process_file(f, out_root, model, device, args.batch_size)
        except Exception as e:
            print(f"[ERROR] {f}: {e}")
        continue
    print("Done.")



if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Convert a folder of SMPL parameter sequences into 3D keypoint (joint) sequences.

Supported inputs per file (auto-detected):
  - .npz/.npy/.pkl/.json containing any of the following keys (common from SPIN, SMPLify, etc.):
      * 72D pose vector:  'poses', 'pose', 'full_pose', 'pose_aa'  (global_orient[:3] + body_pose[69])
      * 69D body pose:    'body_pose', 'pose_body'
      * 3D root orient:  'global_orient', 'root_orient', 'Rh', 'orient'
      * 3D translation:  'transl', 'Th', 'trans', 'translation'
      * 10D shape:       'betas', 'shape', 'shape_params'

Outputs per input file:
  - <stem>_kpt3d.npy        shape: (T, 24, 3)  (meters, SMPL-24 joint set)
  - <stem>_meta.json        metadata (joint names, units, etc.)

Examples
--------
python smpl_param2kpt.py \
  --dir ./params_seq \
  --out ./kpt_seq \
  --model-dir /path/to/SMPL/models \
  --gender neutral \
  --cuda 0

Requirements
------------
  pip install numpy torch smplx tqdm
  # And download SMPL model files (from smpl.is.tue.mpg.de), e.g.:
  #   SMPL_NEUTRAL.pkl, SMPL_MALE.pkl, SMPL_FEMALE.pkl
  # Put them under --model-dir or export SMPL_MODEL_DIR

Notes
-----
* This script produces **3D joints** only. If you need 2D projections, you can add your camera intrinsics/extrinsics and project downstream.
* If a parameter is missing, reasonable defaults are used: betas=0, transl=0, global_orient=0.
* Mixed per-sequence/per-frame shapes are supported; single vectors are broadcast across frames.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle, io, gzip, zlib, lzma, warnings
import joblib
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

try:
    import smplx
except Exception as e:  # pragma: no cover
    raise SystemExit("smplx is required. Install with `pip install smplx`. Error: %s" % e)

SMPL24_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]

# --------------------------- IO helpers ---------------------------

def _load_any(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    if ext == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        # wrap with a generic key
        return {"poses": arr}
    if ext in {".pkl", ".pickle", ".pt", ".pth"}:
        return _load_pickle_like(path)
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
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


# --------------------------- Param canonicalization ---------------------------

def canonicalize_params(raw: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (global_orient[T,3], body_pose[T,69], betas[T,10], transl[T,3], T) in axis-angle (rad) and meters.

    The function tolerates common naming variants and shapes (per-frame or per-sequence). Missing
    parts are filled with zeros.
    """
    # Try different common keys
    pose72 = _as_np(_first_present(raw, ["smpl_pose", "poses", "pose", "full_pose", "pose_aa"]))
    body69 = _as_np(_first_present(raw, ["body_pose", "pose_body"]))
    orient3 = _as_np(_first_present(raw, ["global_orient", "root_orient", "Rh", "orient"]))
    betas10 = _as_np(_first_present(raw, ["betas", "shape", "shape_params"]))
    transl3 = _as_np(_first_present(raw, ["smpl_trans", "transl", "Th", "trans", "translation"]))

    # Some outputs wrap arrays in dicts, lists, or have nested dim [T,1,dim]
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

    # Decide T
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

    # Extract orient/body
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

    # Broadcast to T
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


# --------------------------- SMPL forward ---------------------------

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
    model = smplx.create(
        model_path=model_dir,
        model_type="smpl",
        gender=gender,
        use_pca=False,
        batch_size=1,  # we will pass arbitrary batch sizes later
    )
    return model.to(device)


def to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


@torch.no_grad()
def smpl_to_joints(model, go: np.ndarray, bp: np.ndarray, betas: np.ndarray, transl: np.ndarray,
                   device: torch.device, batch_size: int = 512, out_mode: str = "24") -> np.ndarray:
    """Forward SMPL and return joints.

    out_mode:
      - "24": return the first 24 joints (SMPL-24 convention)
      - "all": return all joints provided by the model (e.g., 45)
    """
    T = go.shape[0]
    chunks = []
    for s in range(0, T, batch_size):
        e = min(T, s + batch_size)
        out = model(
            betas=to_torch(betas[s:e], device),
            body_pose=to_torch(bp[s:e], device),
            global_orient=to_torch(go[s:e], device),
            transl=to_torch(transl[s:e], device),
            return_verts=False,
        )
        j = out.joints  # (B, J, 3)
        if out_mode == "24":
            J = j.shape[1]
            if J >= 24:
                j = j[:, :24, :]
            else:
                pad = torch.zeros((j.shape[0], 24 - J, 3), dtype=j.dtype, device=j.device)
                j = torch.cat([j, pad], dim=1)
        chunks.append(j.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)

# --------------------------- Main pipeline ---------------------------

def _load_pickle_like(path: Path) -> Dict[str, Any]:
    """Loader for 'pickle-like' files prioritizing joblib.load as requested.
    Tries in order:
      1) joblib.load (strict requirement)
      2) torch.load
      3) pickle.load
      4) decompress [gzip/zlib/lzma] + pickle.loads
      5) np.load(npz)
      6) json
    Returns a dict; wraps non-dicts.
    """
    # 1) joblib.load (preferred)
    try:
        obj = joblib.load(str(path))
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "_asdict"):
            return dict(obj._asdict())
        if isinstance(obj, (list, tuple)):
            return {"data": obj}
        return {"data": obj}
    except Exception:
        pass

    # 2) torch.load
    try:
        import torch
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "state_dict"):
            return {"state_dict": obj.state_dict()}
        return {"data": obj}
    except Exception:
        pass

    # 3) pickle.load
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "_asdict"):
            return dict(obj._asdict())
        if isinstance(obj, (list, tuple)):
            return {"data": obj}
        return {"data": obj}
    except Exception:
        pass

    # Read raw bytes once for possible decompress
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception as e:
        raise ValueError(f"Failed to read bytes from {path}: {e}")

    # 4) decompressors + pickle
    for name, decompress in [("gzip", gzip.decompress), ("zlib", zlib.decompress), ("lzma", lzma.decompress)]:
        try:
            obj = pickle.loads(decompress(raw))
            warnings.warn(f"[loader] {path.name}: decoded via {name}+pickle")
            if isinstance(obj, dict):
                return obj
            return {"data": obj}
        except Exception:
            continue

    # 5) Try npz (zip container)
    try:
        bio = io.BytesIO(raw)
        with np.load(bio, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
    except Exception:
        pass

    # 6) Try JSON
    try:
        txt = raw.decode("utf-8", errors="ignore")
        return json.loads(txt)
    except Exception:
        pass

    head = raw[:8]
    hint = f"bytes head={list(head)} (ASCII head='{''.join(chr(b) if 32<=b<127 else '.' for b in head)}')"
    raise ValueError(f"Unrecognized pickle-like file: {path}. Consider gunzip/zlib/lzma decompress, or convert to npz/json. {hint}")


def find_files(root: Path) -> list[Path]:
    allow = {".npz", ".npy", ".pkl", ".pickle", ".pt", ".pth", ".json"}
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in allow:
            files.append(p)
    return files


def process_file(path: Path, out_dir: Path, model, device: torch.device, batch: int, apply_scaling: bool = False, joints_mode: str = "24", humanml22: bool = False) -> Tuple[Path, Path]:
    raw = _load_any(path)

    # Extract canonical params
    go, bp, betas, transl, T = canonicalize_params(raw)

    # Optional dataset-specific global scaling
    s = 1.0
    scale_raw = _first_present(raw, ["smpl_scaling", "scale", "scaling"])
    if scale_raw is not None:
        try:
            s = float(np.asarray(scale_raw).reshape(-1)[0])
        except Exception:
            s = 1.0

    joints = smpl_to_joints(model, go, bp, betas, transl, device=device, batch_size=batch, out_mode=joints_mode)

    # Optional: map to HumanML3D 22 joints (SMPL-24 minus hand end-effectors)
    if humanml22:
        if joints.shape[1] >= 24:
            joints = joints[:, :22, :]
        elif joints.shape[1] == 22:
            pass  # already 22
        else:
            raise ValueError(f"Cannot form HumanML3D-22 from {joints.shape[1]} joints; need >=24 or ==22")

    if apply_scaling and (abs(s - 1.0) > 1e-8):
        joints *= s

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    npy_path = out_dir / f"{stem}_kpt3d.npy"
    meta_path = out_dir / f"{stem}_meta.json"
    np.save(npy_path, joints)

    # Joint naming
    J = int(joints.shape[1])
    if humanml22:
        joints_format = "HUMANML3D_22"
        joint_names = SMPL24_NAMES[:22] if J == 22 else SMPL24_NAMES[:22]
    elif joints_mode == "24" and J >= 24:
        joint_names = SMPL24_NAMES
        joints_format = "SMPL_24"
    else:
        names = getattr(model, "joint_names", None)
        if isinstance(names, (list, tuple)) and len(names) == J:
            joint_names = list(map(str, names))
        else:
            joint_names = [f"J{i}" for i in range(J)]
        joints_format = f"MODEL_{J}"

    meta = {
        "source_file": str(path),
        "num_frames": int(joints.shape[0]),
        "joints_format": joints_format,
        "units": "meters",
        "joint_names": joint_names,
        "scaling_factor": float(s),
        "scaling_applied": bool(apply_scaling and (abs(s - 1.0) > 1e-8)),
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
    ap.add_argument("--ignore-scaling", action="store_true", help="Ignore `smpl_scaling` from inputs if present")
    ap.add_argument("--joints", choices=["24", "all"], default="24", help="Output joint set: '24' keeps SMPL-24 (first 24 joints), 'all' keeps all model joints (e.g., 45)")
    ap.add_argument("--humanml22", action="store_true", help="After forward, convert joints to HumanML3D 22-joint set (drops SMPL left/right hand end-effectors)")
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
            npy_path, meta_path = process_file(f, out_root, model, device, args.batch_size, apply_scaling=(not args.ignore_scaling), joints_mode=args.joints, humanml22=args.humanml22)
        except Exception as e:
            print(f"[ERROR] {f}: {e}")
            continue

    print("Done.")


if __name__ == "__main__":
    main()

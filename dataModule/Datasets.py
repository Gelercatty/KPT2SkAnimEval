import os
from glob import glob
from enum import Enum
import numpy as np
from pathlib  import Path
import torch
from torch.utils.data import Dataset
import lightning  as pl
from typing import List, Dict, Any, Literal, Optional
from tqdm import tqdm

# class KptDataset(Dataset):
#     def __init__(
#                 self, 
#                 gt_dir: str, 
#                 pred_dir: str,
#                 device: Optional[torch.device|str] = "cuda",
#                 dtype: torch.dtype = torch.float32,
#                 strict_pair: bool = True  
#         ):
#         self.gt_dir, self.pred_dir = Path(gt_dir), Path(pred_dir)
#         self.files = sorted([p.name for p in self.gt_dir.glob("*.npy") if (self.pred_dir/p.name).exists()])
#         if not self.files:
#             raise RuntimeError(f"No paired files under: gt={gt_dir}, pred={pred_dir}")
#         # check missing pair
#         if strict_pair:
#             missing = [p.name for p in self.gt_dir.glob("*.npy") if not (self.pred_dir / p.name).exists()]
#             if missing:
#                 print(f"[KptDatasetPreloaded] Warning: {len(missing)} gt files have no pred; they were skipped.")
        
#         self.device = torch.device(device) if device is not None else torch.device("cpu")
#         self.dtype  = dtype

#         gts: List[torch.Tensor] = []
#         preds: List[torch.Tensor] = []
#         print("loading dataset: ")
#         for name in tqdm(self.files):
#             gt    = torch.from_numpy(np.load(self.gt_dir   / name)).to(self.device).to(self.dtype)
#             pred  = torch.from_numpy(np.load(self.pred_dir / name)).to(self.device).to(self.dtype)
#             gts.append(gt)
#             preds.append(pred)

#         self.gts = gts
#         self.preds = preds

#     def __len__(self): return len(self.files)
#     def __getitem__(self, i):
#         return {"gt": self.gts[i],
#                 "pred": self.preds[i],
#                 "name": self.files[i]}




# class MultiMethodKptDataset(Dataset):
#     """
#     一个数据集（task）= 一份 GT + 多个方法的预测
#     - GT 一次性读入并常驻 (device/CPU)
#     - 预测支持三种策略：lazy / preload_cpu / preload_device
#     """
#     def __init__(
#         self,
#         gt_dir: str,
#         methods: Dict[str, str],                 # method -> pred_dir
#         device: Optional[torch.device | str] = "cuda",
#         dtype: torch.dtype = torch.float32,
#         pred_load: str = "lazy",                 # "lazy" | "preload_cpu" | "preload_device"
#         strict_pair: bool = True,
#         name: str = None                       # for debug
#     ):
#         self.gt_dir   = Path(gt_dir)
#         self.methods  = {m: Path(p) for m, p in methods.items()}
#         self.device   = torch.device(device) if device else torch.device("cpu")
#         self.dtype    = dtype
#         self.pred_load = pred_load

#         # 确定文件列表：以 GT 为准，要求每个方法都存在同名文件
#         gt_files = sorted([p.name for p in self.gt_dir.glob("*.npy")])
#         files = []
#         # for name in gt_files:
#         #     # if all((pred_dir / name).exists() for pred_dir in self.methods.values()):
#         #         # files.append(name)
#         #     print(name)
#         #     stem = Path(name).stem
#         #     print(stem)                      # 000000
            

#         #     # elif strict_pair:
#         #     #     # 严格模式：缺任何一个方法的就跳过该样本
#         #     #     continue
#         for name in gt_files:
#             stem = Path(name).stem  # 例如 "000000"
#             # 只要每个方法目录中都能找到“包含 stem 且 .npy 结尾”的文件，就认为这条样本可用
#             ok_all = True
#             for pred_dir in self.methods.values():
#                 print(pred_dir)
#                 # 在 pred_dir 下查是否有任意一个文件名包含 stem 且以 .npy 结尾
#                 if not any((stem in p.stem) and (p.suffix == ".npy") for p in Path(pred_dir).glob("*.npy")):
#                     ok_all = False
#                     break
#             if ok_all:
#                 files.append(name)

#         if not files:
#             raise RuntimeError("No paired samples after checking all methods.")
#         self.files = files

#         # ---- 读取 GT 到 device（一次性）----
#         self.gt_tensors: List[torch.Tensor] = []
#         if name:
#             print(f"loading {name} gt")
#         else:
#             print("loading dataset gt, not given name, not error")
#         for name in tqdm(self.files,"loading gt"):
#             gt = torch.from_numpy(np.load(self.gt_dir / name)).to(torch.float32)
#             # 需要半精度可在这里 .to(self.dtype)
#             if self.dtype != torch.float32:
#                 gt = gt.to(self.dtype)
#             self.gt_tensors.append(gt.to(self.device, non_blocking=False))

#         # ---- 预测的缓存策略 ----
#         self.pred_cache_cpu: Dict[str, List[torch.Tensor]] | None = None
#         self.pred_cache_dev: Dict[str, List[torch.Tensor]] | None = None

#         if self.pred_load == "preload_cpu":
#             self.pred_cache_cpu = {m: [] for m in self.methods}
#             print("loading preds")
#             for i, name in enumerate(self.files):
#                 for m, pred_dir in self.methods.items():
#                     arr = np.load(pred_dir / name)
#                     t = torch.from_numpy(arr).to(torch.float32)
#                     if self.dtype != torch.float32:
#                         t = t.to(self.dtype)
#                     self.pred_cache_cpu[m].append(t)  # 先留在 CPU
#         elif self.pred_load == "preload_device":
#             self.pred_cache_dev = {m: [] for m in self.methods}
#             print("loading preds")
#             for i, name in enumerate(self.files):
#                 for m, pred_dir in self.methods.items():
#                     arr = np.load(pred_dir / name)
#                     t = torch.from_numpy(arr).to(torch.float32)
#                     if self.dtype != torch.float32:
#                         t = t.to(self.dtype)
#                     self.pred_cache_dev[m].append(t.to(self.device, non_blocking=False))

#     def __len__(self): return len(self.files)

#     def __getitem__(self, idx: int):
#         name = self.files[idx]
#         gt = self.gt_tensors[idx]  # 已在 device

#         preds_by_method = {}
#         if self.pred_load == "lazy":
#             # 运行时按需读取，并直接搬到 device
#             for m, pred_dir in self.methods.items():
#                 arr = np.load(pred_dir / name)
#                 t = torch.from_numpy(arr).to(torch.float32)
#                 if self.dtype != torch.float32:
#                     t = t.to(self.dtype)
#                 preds_by_method[m] = t.to(self.device, non_blocking=False)
#         elif self.pred_load == "preload_cpu":
#             for m in self.methods:
#                 preds_by_method[m] = self.pred_cache_cpu[m][idx].to(self.device, non_blocking=False)
#         elif self.pred_load == "preload_device":
#             for m in self.methods:
#                 preds_by_method[m] = self.pred_cache_dev[m][idx]
#         else:
#             raise ValueError(f"Unknown pred_load: {self.pred_load}")

#         return {"name": name, "gt": gt, "preds": preds_by_method}



class MultiMethodKptDataset(Dataset):
    def __init__(
        self,
        gt_dir: str,
        methods: Dict[str, str],   # method -> pred_dir
        device: Optional[torch.device | str] = "cuda",
        dtype: torch.dtype = torch.float32,
        pred_load: str = "lazy",   # "lazy" | "preload_cpu" | "preload_device"
        strict_pair: bool = True,
        name: str | None = None,
    ):
        self.gt_dir   = Path(gt_dir)
        self.methods  = {m: Path(p) for m, p in methods.items()}
        self.device   = torch.device(device) if device else torch.device("cpu")
        self.dtype    = dtype
        self.pred_load = pred_load

        # 1) 先把每个方法目录建立“stem -> Path”的快速索引（方便用 GT 的 stem 去查）
        #    如果一个 stem 对应多个文件，取排序后第一个（保证确定性）
        method_index: Dict[str, Dict[str, Path]] = {}
        for m, d in self.methods.items():
            idx: Dict[str, Path] = {}
            for p in d.glob("*.npy"):
                st = p.stem
                # 为了支持包含关系（如 SMPLPose_000000_kpt3d），也记录所有“数字段”的包含
                # 但这里按“直接包含”来做：后面用 gt_stem in pred_stem 判断
                idx[st] = p
            method_index[m] = idx

        # 2) 以 GT 为准，找到每个 GT 的各方法真实预测文件 Path
        gt_files = sorted([p.name for p in self.gt_dir.glob("*.npy")])
        if not gt_files:
            raise RuntimeError(f"No GT .npy under {self.gt_dir}")

        matched_gt_names: List[str] = []
        pred_paths_by_method: Dict[str, List[Path]] = {m: [] for m in self.methods}

        for gt_name in gt_files:
            stem = Path(gt_name).stem  # 例如 "000000"

            ok = True
            chosen: Dict[str, Path] = {}
            for m, d in self.methods.items():
                # 在该方法目录下找“包含 stem 的 .npy”
                # 为避免每次 glob，可直接遍历索引表；这里还是用 glob 写法更直观
                cand = sorted([p for p in d.glob("*.npy") if stem in p.stem])
                if not cand:
                    ok = False
                    break
                chosen[m] = cand[0]   # 选第一个，或自定义规则挑选

            if ok:
                matched_gt_names.append(gt_name)
                for m in self.methods:
                    pred_paths_by_method[m].append(chosen[m])

        if not matched_gt_names:
            # 打点调试信息，看看前若干个
            print("[DEBUG] First 5 GT:", gt_files[:5])
            for m, d in self.methods.items():
                ex = [p.name for p in d.glob("*.npy")][:5]
                print(f"[DEBUG] Method {m} examples:", ex)
            raise RuntimeError("No paired samples after matching by stem.")

        # 保存对齐后的索引：index i 对应 1 个 gt + 每个方法 1 个 pred
        self.files: List[str] = matched_gt_names               # GT 文件名列表
        self.pred_paths: Dict[str, List[Path]] = pred_paths_by_method  # method -> [Path,...] 与 files 对齐

        # 3) 预加载 GT（一次性，上 device）
        self.gt_tensors: List[torch.Tensor] = []
        if name:
            print(f"loading {name} gt")
        else:
            print("loading dataset gt")
        for gt_name in tqdm(self.files, desc="loading gt"):
            t = torch.from_numpy(np.load(self.gt_dir / gt_name)).to(torch.float32)
            if self.dtype != torch.float32:
                t = t.to(self.dtype)
            self.gt_tensors.append(t.to(self.device, non_blocking=False))

        # 4) 预测的缓存（按对齐后的路径来读）
        self.pred_cache_cpu: Dict[str, List[torch.Tensor]] | None = None
        self.pred_cache_dev: Dict[str, List[torch.Tensor]] | None = None

        if self.pred_load == "preload_cpu":
            print("loading preds (CPU)")
            self.pred_cache_cpu = {m: [] for m in self.methods}
            for i in tqdm(range(len(self.files)), desc="preload cpu"):
                for m in self.methods:
                    arr = np.load(self.pred_paths[m][i])
                    t = torch.from_numpy(arr).to(torch.float32)
                    if self.dtype != torch.float32:
                        t = t.to(self.dtype)
                    self.pred_cache_cpu[m].append(t)
        elif self.pred_load == "preload_device":
            print("loading preds (device)")
            self.pred_cache_dev = {m: [] for m in self.methods}
            for i in tqdm(range(len(self.files)), desc="preload device"):
                for m in self.methods:
                    arr = np.load(self.pred_paths[m][i])
                    t = torch.from_numpy(arr).to(torch.float32)
                    if self.dtype != torch.float32:
                        t = t.to(self.dtype)
                    self.pred_cache_dev[m].append(t.to(self.device, non_blocking=False))

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx: int):
        gt = self.gt_tensors[idx]  # 已在 device
        preds: Dict[str, torch.Tensor] = {}

        if self.pred_load == "lazy":
            # 用“对齐后的真实路径”去读，**不要**再用 pred_dir / name
            for m in self.methods:
                arr = np.load(self.pred_paths[m][idx])
                t = torch.from_numpy(arr).to(torch.float32)
                if self.dtype != torch.float32:
                    t = t.to(self.dtype)
                preds[m] = t.to(self.device, non_blocking=False)
        elif self.pred_load == "preload_cpu":
            for m in self.methods:
                preds[m] = self.pred_cache_cpu[m][idx].to(self.device, non_blocking=False)
        elif self.pred_load == "preload_device":
            for m in self.methods:
                preds[m] = self.pred_cache_dev[m][idx]
        else:
            raise ValueError(f"Unknown pred_load: {self.pred_load}")

        return {"name": self.files[idx], "gt": gt, "preds": preds}
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




class MultiMethodKptDataset(Dataset):
    """
    一个数据集（task）= 一份 GT + 多个方法的预测
    - GT 一次性读入并常驻 (device/CPU)
    - 预测支持三种策略：lazy / preload_cpu / preload_device
    """
    def __init__(
        self,
        gt_dir: str,
        methods: Dict[str, str],                 # method -> pred_dir
        device: Optional[torch.device | str] = "cuda",
        dtype: torch.dtype = torch.float32,
        pred_load: str = "lazy",                 # "lazy" | "preload_cpu" | "preload_device"
        strict_pair: bool = True,
        name: str = None                       # for debug
    ):
        self.gt_dir   = Path(gt_dir)
        self.methods  = {m: Path(p) for m, p in methods.items()}
        self.device   = torch.device(device) if device else torch.device("cpu")
        self.dtype    = dtype
        self.pred_load = pred_load

        # 确定文件列表：以 GT 为准，要求每个方法都存在同名文件
        gt_files = sorted([p.name for p in self.gt_dir.glob("*.npy")])
        files = []
        for name in gt_files:
            if all((pred_dir / name).exists() for pred_dir in self.methods.values()):
                files.append(name)
            elif strict_pair:
                # 严格模式：缺任何一个方法的就跳过该样本
                continue
        if not files:
            raise RuntimeError("No paired samples after checking all methods.")
        self.files = files

        # ---- 读取 GT 到 device（一次性）----
        self.gt_tensors: List[torch.Tensor] = []
        if name:
            print(f"loading {name} gt")
        else:
            print("loading dataset gt, not given name, not error")
        for name in tqdm(self.files,"loading gt"):
            gt = torch.from_numpy(np.load(self.gt_dir / name)).to(torch.float32)
            # 需要半精度可在这里 .to(self.dtype)
            if self.dtype != torch.float32:
                gt = gt.to(self.dtype)
            self.gt_tensors.append(gt.to(self.device, non_blocking=False))

        # ---- 预测的缓存策略 ----
        self.pred_cache_cpu: Dict[str, List[torch.Tensor]] | None = None
        self.pred_cache_dev: Dict[str, List[torch.Tensor]] | None = None

        if self.pred_load == "preload_cpu":
            self.pred_cache_cpu = {m: [] for m in self.methods}
            print("loading preds")
            for i, name in enumerate(self.files):
                for m, pred_dir in self.methods.items():
                    arr = np.load(pred_dir / name)
                    t = torch.from_numpy(arr).to(torch.float32)
                    if self.dtype != torch.float32:
                        t = t.to(self.dtype)
                    self.pred_cache_cpu[m].append(t)  # 先留在 CPU
        elif self.pred_load == "preload_device":
            self.pred_cache_dev = {m: [] for m in self.methods}
            print("loading preds")
            for i, name in enumerate(self.files):
                for m, pred_dir in self.methods.items():
                    arr = np.load(pred_dir / name)
                    t = torch.from_numpy(arr).to(torch.float32)
                    if self.dtype != torch.float32:
                        t = t.to(self.dtype)
                    self.pred_cache_dev[m].append(t.to(self.device, non_blocking=False))

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        gt = self.gt_tensors[idx]  # 已在 device

        preds_by_method = {}
        if self.pred_load == "lazy":
            # 运行时按需读取，并直接搬到 device
            for m, pred_dir in self.methods.items():
                arr = np.load(pred_dir / name)
                t = torch.from_numpy(arr).to(torch.float32)
                if self.dtype != torch.float32:
                    t = t.to(self.dtype)
                preds_by_method[m] = t.to(self.device, non_blocking=False)
        elif self.pred_load == "preload_cpu":
            for m in self.methods:
                preds_by_method[m] = self.pred_cache_cpu[m][idx].to(self.device, non_blocking=False)
        elif self.pred_load == "preload_device":
            for m in self.methods:
                preds_by_method[m] = self.pred_cache_dev[m][idx]
        else:
            raise ValueError(f"Unknown pred_load: {self.pred_load}")

        return {"name": name, "gt": gt, "preds": preds_by_method}
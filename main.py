import numpy as np
from EvalBanchmark import MultiMethodEvalModule, build_metrics_from_cfg
from dataModule.Datasets import MultiMethodKptDataset
from eval.metrics import MPJEP, PAMPJPE, PCK, BLE
from omegaconf import OmegaConf
from typing import Dict, List, Tuple
import csv
import argparse
import lightning as pl
import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path
parents_22 = [
    -1,  # 0  pelvis / root
     0,  # 1
     0,  # 2
     0,  # 3
     1,  # 4
     2,  # 5
     3,  # 6
     4,  # 7
     5,  # 8
     6,  # 9
     7,  # 10
     8,  # 11
     9,  # 12
     9,  # 13
     9,  # 14
    12,  # 15
    13,  # 16
    14,  # 17
    16,  # 18
    17,  # 19
    18,  # 20
    19
]

def expand_tasks(ds_cfg: Dict) -> List[Tuple[str, str, Dict[str, str]]]:
    tasks = []
    for ds_name, block in (ds_cfg or {}).items():
        if not block or "gt" not in block or not block["gt"]:
            continue
        gt_dir = block["gt"]
        methods = {k: v for k, v in block.items() if k!="gt" and v}
        if methods:
            tasks.append((ds_name, gt_dir, methods))
    if not tasks:
        raise ValueError("No valid tasks under 'dataset'. Need gt + at least one method.")

    return tasks


def prase_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, default="configs/base.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    return cfg
def test():
    metric_PAMPJPE = PAMPJPE(unit_scale=1)
    metric_MPJPE   = MPJEP(0)
    metric_PCK     = PCK(threshold=0.25)
    metric_BLE     = BLE(parents=parents_22, mode="rel")
    data_gt = r"C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\humanML3D\000000.npy"
    data_pred = r"C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\hyberik_reKpt\SMPLPoss_000000_kpt3d.npy"

    gt = np.load(data_gt)  # [T, J, 3]
    pred = np.load(data_pred)  # [T, J, 3]

    metric_PAMPJPE.update(gt, pred)
    metric_MPJPE.update(gt, pred)
    metric_PCK.update(gt, pred)
    metric_BLE.update(gt, pred)
    print(metric_PAMPJPE.compute())
    print(metric_MPJPE.compute())
    print(metric_PCK.compute())
    print(metric_BLE.compute())

def run(cfg):
    metrics_cfg = cfg.metrics
    metric_factories = build_metrics_from_cfg(metrics_cfg)
    tasks = expand_tasks(cfg.dataset)

    dev = "cuda" if (isinstance(cfg.device, int) or str(cfg.device).lower() in {"cuda","gpu","auto"}) and torch.cuda.is_available() else "cpu"
    device_ids = [int(cfg.device)] if isinstance(cfg.device, int) else 0
    
    trainer = pl.Trainer(accelerator=("gpu" if dev=="cuda" else "cpu"), devices=device_ids, logger=False)

    out_dir = Path(cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    rows = []

    for ds_name, gt_dir, methods in tasks:
        print(f"\n=== Task: {ds_name} ===")

        ds = MultiMethodKptDataset(
            gt_dir=gt_dir,
            methods=methods,
            device=dev,
            dtype=torch.float32,
            pred_load=getattr(cfg, "pred_load", "lazy")
        )
        dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

        model = MultiMethodEvalModule(metric_factories, method_names=list(methods.keys()))

        trainer.validate(model, dataloaders=dl, verbose=False)

        result = model.get_epoch_values()

        for method, metrics in result.items():
            row = {"dataset": ds_name, "method": method, **metrics}
            rows.append(row)

    if rows:
        headers = list(rows[0].keys())
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader(); w.writerows(rows)
        print(f"\nSummary saved to: {summary_csv}\n")
        for r in rows: print(r)

def main():
    # test()
    cfg = prase_args()
    run(cfg)
if __name__ == "__main__":
    main()

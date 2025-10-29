import numpy as np
from EvalBanchmark import Kpt2SkeleEvalModel
from dataModule.DataMoudle import EvalDataModule,DatasetType
from eval.metrics import MPJEP, PAMPJPE, PCK, BLE
from omegaconf import OmegaConf
import argparse
import lightning as pl
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

    dataModule = EvalDataModule(cfg)
    model      = Kpt2SkeleEvalModel(cfg)

    trainer    = pl.Trainer(
        accelerator="auto",
        devices=cfg.devices,
        logger=True,
        enable_progress_bar=True
    )
def main():
    # test()
    cfg = prase_args()
    run(cfg)
if __name__ == "__main__":
    main()

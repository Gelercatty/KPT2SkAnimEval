from .Datasets import KptDataset, DatasetType
from torch.utils.data import DataLoader
import lightning as pl
from typing import Dict

def build_metrics_from_cfg(metrics_cfg: Dict):
    parents_22 = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]

class EvalDataMoudle(pl.LightningDataModule):
    def __init__(self, gt_dir: str, pred_dir: str, batch_size=1, num_workers=0):
        super().__init__()
        self.gt_dir, self.pred_dir = gt_dir, pred_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self._ds = None
    def setup(self, stage = None):
        self._ds = KptDataset(self.gt_dir, self.pred_dir)


# # 抽象了模型获取到的数据，在这里更改数据集
# class EvalDataModule(pl.LightningDataModule):
#     # def __init__(self, gt_type, pred_dir, batch_size=1, num_workers=0, files = None):
#     def __init__(self, metrics_cfg: Dict):
#         super().__init__() 
#         self.gt_type     = cfg.gt_type
#         self.pred_dir    = cfg.pred_dir
#         self.batch_size  = cfg.batch_size
#         self.num_workers = cfg.num_workers
#         self.files       = cfg.files


#     def setup(self, stage = None):
#         # 使用哪个数据集
#         ds = KptDataset(gt_type=self.gt_type , pred_dir=self.pred_dir, files=self.files)
        
#         self.val_ds  = ds
#         self.test_ds = ds

#     def val_dataloader(self):
#         return DataLoader(self.val_ds,batch_size=self.batch_size, num_workers=self.num_workers)
#     def test_dataloader(self):
#         return DataLoader(self.test_ds,batch_size=self.batch_size, num_workers=self.num_workers)
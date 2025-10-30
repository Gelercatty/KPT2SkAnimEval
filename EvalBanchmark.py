import lightning as pl
from typing import Dict, List
import torch
from eval.metrics import MPJEP, PAMPJPE, PCK, BLE,PAPCK,BDE
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
def build_metrics_from_cfg(metrics_cfg: Dict):
    # factory mode, return build func
    REG = {
        "mpjep":  lambda c: (lambda: MPJEP(root_index=c.get("root_index", 0), unit_scale=c.get("unit_scale", 1.1))),
        "pampjep":lambda c: (lambda: PAMPJPE(unit_scale=c.get("unit_scale", 1.1))),
        "pck":    lambda c: (lambda: PCK(threshold=c.get("threshold", 2), unit_scale=c.get("unit_scale", 1))),
        "ble":    lambda c: (lambda: BLE(parents=parents_22,
                                     mode=c.get("mode", "rel"),
                                     unit_scale=c.get("unit_scale", 1.1))),
        "papck":  lambda c: (lambda: PAPCK(threshold=c.get("threshold", 2), unit_scale=c.get("unit_scale", 1))),
        "bde":     lambda c: (lambda: BDE(parents=parents_22,
                                          strict_shape=c.get("strict_shape", True),
                                          eps=c.get("eps", 1e-8))),
    
    }
    metrics = {}
    for name, subcfg in (metrics_cfg or {}).items():
        name_l = name.lower()
        if name_l in REG:
            metrics[name] = REG[name_l](subcfg or {})
    if not metrics:
        raise ValueError("No metrics enabled in YAML (metrics: {}).".format(list(metrics_cfg.keys())))
    
    return metrics
    

# 定义了模型的行为
class Kpt2SkeleEvalModel(pl.LightningModule):
    def __init__(self, metrics_cfg):
        super.__init__()

        self.metrics = build_metrics_from_cfg(metrics_cfg)

        self._epoch_values = {}


    def validation_step(self, batch, _):
        pred   = batch["pred"]
        target = batch["gt"]

        for m in self.metrics.values():
            m.update(pred, target)

    def on_validation_epoch_end(self):
        self._epoch_values = {name: metric.compute().item() for name, metric in self.metrics.items()}
        for k, v in self._epoch_values.items():
            self.log(f"val/{k}", v, prog_bar=False, on_step=False, on_epoch=True)
        for m in self.metrics.values():
            m.reset()
    def configure_optimizers(self):
        return None




class MultiMethodEvalModule(pl.LightningModule):
    def __init__(self, metric_factories: Dict[str, callable], method_names: List[str]):
        super().__init__()
        # self.metrics = torch.nn.ModuleDict({
        #     method: torch.nn.ModuleDict({name: factory() for name, factory in metric_factories().items()})
        #     for method in method_names
        # })
        self.metrics = torch.nn.ModuleDict({
            method: torch.nn.ModuleDict({name: factory() for name, factory in metric_factories.items()})
            for method in method_names
        })
        self._epoch_values: Dict[str, Dict[str, float]] = {} # method_name -> metric_name -> value

    def validation_step(self, batch, _):
        gt = batch["gt"]
        preds_by_method = batch["preds"]

        for method, preds in preds_by_method.items():
            for name, metric in self.metrics[method].items():
                metric.update(preds, gt)

    def on_validation_epoch_end(self):

        results = {}
        for method, metric in self.metrics.items():
            one = {name: m.compute().item() for name , m in metric.items()}
            results[method] = one
            for k, v in one.items():
                self.log(f"val/{method}/{k}", v, prog_bar=False, on_step=False, on_epoch=True)
            for m in metric.values():
                m.reset()
        self._epoch_values = results

        print("\nEpoch Results:")
        for method, one in results.items():
            pretty = "  ".join([f"{k}={v:.6f}" for k, v in one.items()])
            print(f"  [{method}] {pretty}")

    def get_epoch_values(self) -> Dict[str, Dict[str, float]]:
        return self._epoch_values
    
    def configure_optimizers(self):
        return None
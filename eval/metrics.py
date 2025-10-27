import torch
from torchmetrics import Metric
from torch import Tensor

class MPJEP(Metric):
    """
    Mean Per Joint Position Error (单位由 unit_scale 决定，默认与输入一致)

    适用输入:
      preds, target: [..., J, 3]
      mask (可选):   [..., J] or 可广播到 [..., J] 的 bool 张量，True 表示有效

    特性:
      - 支持任意批/时间等前置维度，一次性累计
      - 可选根关节对齐: root_index != None 时，先对两者减去根关节坐标
      - 分布式/多设备安全: 通过 add_state(sum_error, total_count) 聚合

    参数:
      root_index: int | None        # 若给出，对两侧做 root-relative
      unit_scale: float             # 将距离乘以该系数再累计。若你的数据是米，想输出毫米，设为 1000.
      strict_shape: bool            # True 时输入形状不符直接报错；False 时给出更友好的报错信息

    返回:
      compute() -> 标量张量（float32）
    """
    full_state_update: bool = False
    higher_is_better:  bool = False
    is_differentiable: bool = False 

    def __init__(
        self,
        root_index: int | None = None,
        unit_scale: float = 1.0,
        strict_shape: bool = True,
        dist_sync_on_step: bool = False,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step = dist_sync_on_step, **kwargs)
        self.root_index = root_index
        self.unit_scale = float(unit_scale)
        self.strict_shape = strict_shape

        self.add_state("sum_error", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx = "sum")
        self.add_state("total_count", default=torch.tensor(0, dtype=torch.tensor(0, dtype=torch.long), dist_reduce_fx = "sum"))

    @staticmethod
    def _ensure_shape(preds: Tensor, target: Tensor):
        if preds.shape != target.shape:
            raise ValueError(f"preds/target 形状不一致: {preds.shape} vs {target.shape}")
        if preds.ndim < 3 or preds.size(-1) != 3:
            raise ValueError(f"期望输入为 [..., J, 3]，但得到 {preds.shape}")
        
    
    def _root_align(self, x:Tensor) -> Tensor:
        if self.root_index is None:
            return x
        if x.size(-2) <= self.root_index:
            return ValueError(f"root_index={self.root_index} 超出关节数 J={x.size(-2)}")
        root = x[..., self.root_index:self.root_index+1, :] # [..., 1, 3]
        return x - root
    

    @torch.no_grad()
    def update(self, preds: Tensor, target:Tensor, mask: Tensor | None = None) -> None:
        if self.strict_shape:
            self._ensure_shape(preds, target)

        device = self.sum_error.device
        preds  = preds.to(device)
        target = target.to(device)

        preds  = self._root_align(preds)
        target = self._root_align(target)

        err = torch.linalg.vector_norm(preds-target, dim = -1)

        if mask is None:
            valid = torch.isfinite(err)
        else:
            mask  = mask.to(device)
            try: 
                mask = mask.expand_as(err)
            except RuntimeError:
                raise ValueError(f"mask 形状 {mask.shape} 不能广播到误差形状 {err.shape}")
            valid = mask & torch.isfinite(err)
        
        if valid.any():
            sel = err[valid].to(torch.float64) * self.unit_scale
            self.sum_error += sel.sum()
            self.total_count += valid.sum().to(torch.long)

    @torch.no_grad()
    def compute(self) -> Tensor:
        if self.total_count == 0:
            return torch.tensor(float('nan'), dtype= torch.float32, device= self.sum_error.device)
        return (self.sum_error / self.total_count).to(torch.float32)
    
    @torch.no_grad()
    def reset(self) -> None:
        super().reset()
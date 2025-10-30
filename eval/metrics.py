import torch
from torchmetrics import Metric
from torch import Tensor
from typing import Optional, Literal, Tuple, Sequence
import numpy
class MPJEP(Metric):
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
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        self.root_index = root_index
        self.unit_scale = float(unit_scale)
        self.strict_shape = strict_shape

        self.add_state("sum_error",
                       default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("total_count",
                       default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")

    @staticmethod
    def _ensure_shape(preds: Tensor, target: Tensor):
        if preds.shape != target.shape:
            raise ValueError(f"preds/target 形状不一致: {preds.shape} vs {target.shape}")
        if preds.ndim < 3 or preds.shape[-1] != 3:   # ← 用 shape[-1]
            raise ValueError(f"期望输入为 [..., J, 3]，但得到 {preds.shape}")

    def _root_align(self, x: Tensor) -> Tensor:
        if self.root_index is None:
            return x
        if x.shape[-2] <= self.root_index:          # ← 用 shape[-2]
            raise ValueError(f"root_index={self.root_index} 超出关节数 J={x.shape[-2]}")
        root = x[..., self.root_index:self.root_index+1, :]  # [..., 1, 3]
        return x - root

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor, mask: Tensor | None = None) -> None:
        # 先吃成 torch.Tensor，兼容 numpy 输入
        preds  = torch.as_tensor(preds)
        target = torch.as_tensor(target)
        if mask is not None:
            mask = torch.as_tensor(mask).bool()

        if self.strict_shape:
            self._ensure_shape(preds, target)

        device = self.sum_error.device
        preds  = preds.to(device)
        target = target.to(device)

        preds  = self._root_align(preds)
        target = self._root_align(target)

        err = torch.linalg.vector_norm(preds - target, dim=-1)

        if mask is None:
            valid = torch.isfinite(err)
        else:
            try:
                mask = mask.to(device).expand_as(err)
            except RuntimeError:
                raise ValueError(f"mask 形状 {mask.shape} 不能广播到误差形状 {err.shape}")
            valid = mask & torch.isfinite(err)

        if valid.any().item():                       # ← 用 .item()
            sel = err[valid].to(torch.float64) * self.unit_scale
            self.sum_error   += sel.sum()
            self.total_count += valid.sum().to(torch.long)

    @torch.no_grad()
    def compute(self) -> Tensor:
        if self.total_count.item() == 0:
            return torch.tensor(float('nan'), dtype=torch.float32, device=self.sum_error.device)
        return (self.sum_error / self.total_count).to(torch.float32)

    @torch.no_grad()
    def reset(self) -> None:
        super().reset()
class PAMPJPE(Metric):
    """
    PA-MPJPE (Procrustes Aligned Mean Per Joint Position Error)

    功能
    ----
    - 对每个样本的预测与 GT 进行相似变换（旋转R、尺度s、平移t）的 Procrustes 对齐，
      然后计算三维关节的欧氏距离平均值。
    - 可选 mask（True 表示该关节有效）。
    - 支持任意批/时间等前置维度：输入形状 [..., J, 3]。
    - 分布式/多设备安全：内部通过 add_state(sum_error, total_count) 聚合。
    - 单位缩放 unit_scale（例如从米到毫米设为 1000）。

    重要说明
    --------
    - PA 对齐已包含平移（t），因此**不需要**再做 root 对齐；本类不提供 root_index 选项。
    - 若使用 mask，则对齐时仅基于有效关节子集估计 (R, s, t)。

    参数
    ----
    unit_scale: float = 1.0
        输出误差的单位缩放（例如从 m -> mm 设为 1000）
    strict_shape: bool = True
        若为 True，输入形状不符直接报错
    use_mask_in_align: bool = True
        若提供 mask，是否在对齐步骤中仅使用有效关节参与求解相似变换

    输入
    ----
    update(preds, target, mask=None)
      preds, target: [..., J, 3]
      mask（可选）:  [..., J] 或可广播到 [..., J] 的 bool 张量（True=有效）

    输出
    ----
    compute() -> 标量张量（float32）
    """
    full_state_update: bool = False
    higher_is_better:  bool = False
    is_differentiable: bool = False

    def __init__(
        self,
        unit_scale: float = 1.0,
        strict_shape: bool = True,
        use_mask_in_align: bool = True,
        dist_sync_on_step: bool = False,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        self.unit_scale = float(unit_scale)
        self.strict_shape = strict_shape
        self.use_mask_in_align = bool(use_mask_in_align)

        self.add_state("sum_error",
                       default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("total_count",
                       default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")

    # ---------- 工具函数 ----------

    @staticmethod
    def _ensure_shape(preds: Tensor, target: Tensor):
        if preds.shape != target.shape:
            raise ValueError(f"preds/target 形状不一致: {preds.shape} vs {target.shape}")
        if preds.ndim < 3 or preds.shape[-1] != 3:
            raise ValueError(f"期望输入为 [..., J, 3]，但得到 {preds.shape}")

    @staticmethod
    @torch.no_grad()
    def _procrustes_align_batched(X: Tensor, Y: Tensor, M: Tensor | None, eps: float = 1e-8) -> Tensor:
        """
        对输入批次做相似变换 Procrustes 对齐。
        X, Y: (B, J, 3)
        M:    (B, J) 的 bool，True 表示该关节有效；若为 None 则全有效。
        返回: X 对齐到 Y 后的坐标，形状 (B, J, 3)
        """
        B, J, C = X.shape
        dtype = torch.float64  # 对齐计算用双精度更稳
        X = X.to(dtype)
        Y = Y.to(dtype)

        if M is None:
            # 全有效
            ones = torch.ones((B, J, 1), dtype=dtype, device=X.device)
            wsum = torch.full((B, 1, 1), J, dtype=dtype, device=X.device)  # 每个样本权重和=J
            WX = X
            WY = Y
        else:
            # 仅有效点参与对齐：按 mask 作为权重
            M = M.to(X.device, dtype=dtype).unsqueeze(-1)     # (B,J,1)
            wsum = M.sum(dim=1, keepdim=True).clamp_min(1.0)  # 防止除 0
            WX = X * M
            WY = Y * M

        muX = WX.sum(dim=1, keepdim=True) / wsum        # (B,1,3)
        muY = WY.sum(dim=1, keepdim=True) / wsum

        X0 = X - muX                                     # (B,J,3)
        Y0 = Y - muY
        if M is not None:
            X0m = X0 * M
            Y0m = Y0 * M
        else:
            X0m = X0
            Y0m = Y0

        # 跨关节的加权协方差 H = X0^T Y0
        H = X0m.transpose(1, 2) @ Y0m                    # (B,3,3)

        # SVD 分解
        U, S, Vh = torch.linalg.svd(H)                   # H = U @ diag(S) @ Vh
        R = Vh.transpose(-2, -1) @ U.transpose(-2, -1)   # (B,3,3)

        # 反射修正，确保 det(R)=+1
        detR = torch.det(R)
        idx = detR < 0
        if idx.any():
            Vh[idx, -1, :] *= -1
            R = Vh.transpose(-2, -1) @ U.transpose(-2, -1)

        # 尺度：trace(R @ H) / ||X0||^2（只统计有效关节）
        varX = (X0m ** 2).sum(dim=(1, 2)).clamp_min(eps)          # (B,)
        scale = (R * H).sum(dim=(1, 2)) / varX                    # (B,)

        # 应用相似变换：s * (X0 @ R) + muY
        X_aligned = scale.view(B, 1, 1) * (X0 @ R) + muY          # (B,J,3)
        return X_aligned.to(X.dtype)

    # ---------- Metric 接口 ----------

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor, mask: Tensor | None = None) -> None:
        # 吃成 torch.Tensor（兼容 numpy），并进行基本检查
        preds  = torch.as_tensor(preds)
        target = torch.as_tensor(target)
        if self.strict_shape:
            self._ensure_shape(preds, target)

        # 展平批维到 B
        *batch_dims, J, C = preds.shape
        B = int(torch.tensor(batch_dims).prod().item()) if batch_dims else 1
        preds  = preds.reshape(B, J, C)
        target = target.reshape(B, J, C)

        # mask 处理到 (B,J)
        if mask is not None:
            mask = torch.as_tensor(mask).bool()
            try:
                mask = mask.expand(*batch_dims, J).reshape(B, J)
            except RuntimeError:
                raise ValueError(f"mask 形状 {tuple(mask.shape)} 不能广播到 {tuple((*batch_dims, J))}")

        # 搬到正确设备
        device = self.sum_error.device
        preds  = preds.to(device)
        target = target.to(device)
        if mask is not None:
            mask = mask.to(device)

        # Procrustes 对齐
        preds_aligned = self._procrustes_align_batched(preds, target, mask if self.use_mask_in_align else None)

        # 逐关节欧氏距离
        err = torch.linalg.vector_norm(preds_aligned - target, dim=-1)  # (B,J)

        # 有效性：mask（若有）与数值合法性共同决定
        valid = torch.isfinite(err) if mask is None else (mask & torch.isfinite(err))

        if valid.any().item():
            sel = err[valid].to(torch.float64) * self.unit_scale
            self.sum_error   += sel.sum()
            self.total_count += valid.sum().to(torch.long)

    @torch.no_grad()
    def compute(self) -> Tensor:
        if self.total_count.item() == 0:
            return torch.tensor(float('nan'), dtype=torch.float32, device=self.sum_error.device)
        return (self.sum_error / self.total_count).to(torch.float32)

    @torch.no_grad()
    def reset(self) -> None:
        super().reset()
        


class PCK(Metric):
    """
    PCK (Percentage of Correct Keypoints)

    判定逻辑
    --------
    - 计算每个关节的欧氏距离 d = ||pred - target||。
    - 如指定归一化尺度 n，则用 d' = d / n；否则 d' = d。
    - 若 d' <= threshold 则该关节计为命中。
    - PCK = 命中数 / 有效关节数。

    归一化选项
    ----------
    - ref_pair: Optional[Tuple[int,int]]
        若给定，将以 target 中这对关节的欧氏距离作为每样本归一化尺度。
    - norm_scale (update 的可选实参):
        外部传入每样本或每关节尺度，用于 d 的归一化。优先级高于 ref_pair。
        形状可为 [...], 或 [..., 1], 或 [..., J]；会广播到 (B, J)。

    其他
    ----
    - 支持 mask（True=有效），仅对有效关节计数/统计。
    - 输入支持任意批前维：[..., J, C]，C ∈ {2,3}。
    - unit_scale 在比较前作用于距离（例如数据是米，阈值按米定义；若想以毫米阈值比较，可 unit_scale=1000）。

    参数
    ----
    threshold: float
        阈值（绝对或归一化后判定用）
    unit_scale: float
        距离单位缩放
    strict_shape: bool
        若 True，preds/target 形状不一致直接报错
    ref_pair: Optional[(i, j)]
        参考关节对（用于归一化）
    """

    full_state_update: bool = False
    higher_is_better:  bool = True
    is_differentiable: bool = False

    def __init__(
        self,
        threshold: float = 0.05,
        unit_scale: float = 1.0,
        strict_shape: bool = True,
        ref_pair: Optional[Tuple[int, int]] = None,
        dist_sync_on_step: bool = False,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        self.threshold = float(threshold)
        self.unit_scale = float(unit_scale)
        self.strict_shape = bool(strict_shape)
        self.ref_pair = ref_pair

        self.add_state("correct_count",
                       default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")
        self.add_state("total_count",
                       default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")

    # ---------- 工具函数 ----------

    @staticmethod
    def _ensure_shape(preds: Tensor, target: Tensor):
        if preds.shape != target.shape:
            raise ValueError(f"preds/target 形状不一致: {preds.shape} vs {target.shape}")
        if preds.ndim < 3:
            raise ValueError(f"期望输入为 [..., J, C]，但得到 {preds.shape}")
        C = preds.shape[-1]
        if C not in (2, 3):
            raise ValueError(f"最后一维 C 只能为 2 或 3，但得到 C={C}")

    @torch.no_grad()
    def _compute_ref_scale(self, target: Tensor) -> Optional[Tensor]:
        """
        若设置了 ref_pair=(i,j)，返回形状 (B,1) 的每样本尺度；
        否则返回 None。
        target: (B, J, C)
        """
        if self.ref_pair is None:
            return None
        i, j = self.ref_pair
        J = target.shape[-2]
        if not (0 <= i < J and 0 <= j < J):
            raise ValueError(f"ref_pair 索引越界: (i={i}, j={j})，J={J}")
        vi = target[:, i, :]          # (B, C)
        vj = target[:, j, :]
        scale = torch.linalg.vector_norm(vi - vj, dim=-1).clamp_min(1e-12)  # (B,)
        return scale.view(-1, 1)      # (B,1)

    # ---------- Metric 接口 ----------

    @torch.no_grad()
    def update(
        self,
        preds: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        norm_scale: Optional[Tensor] = None,
    ) -> None:
        """
        参数
        ----
        preds, target: [..., J, C] (C=2或3)
        mask:         [..., J] 的 bool（True=有效），可广播
        norm_scale:   每样本或每关节归一化尺度；形状可为 [...], [...,1], 或 [...,J]，
                      将广播到 (B, J)。若给定，优先于 ref_pair。
        """
        preds  = torch.as_tensor(preds)
        target = torch.as_tensor(target)

        if self.strict_shape:
            self._ensure_shape(preds, target)

        # 展平批维
        *batch_dims, J, C = preds.shape
        B = int(torch.tensor(batch_dims).prod().item()) if batch_dims else 1
        preds  = preds.reshape(B, J, C)
        target = target.reshape(B, J, C)

        # mask -> (B, J)
        if mask is not None:
            mask = torch.as_tensor(mask).bool()
            try:
                mask = mask.expand(*batch_dims, J).reshape(B, J)
            except RuntimeError:
                raise ValueError(f"mask 形状 {tuple(mask.shape)} 不能广播到 {tuple((*batch_dims, J))}")

        device = self.correct_count.device
        preds  = preds.to(device)
        target = target.to(device)
        if mask is not None:
            mask = mask.to(device)

        # 距离（单位缩放）
        dist = torch.linalg.vector_norm(preds - target, dim=-1).to(torch.float64)  # (B, J)
        if self.unit_scale != 1.0:
            dist = dist * self.unit_scale

        # 归一化尺度
        if norm_scale is not None:
            s = torch.as_tensor(norm_scale, device=device, dtype=torch.float64)
            # 广播到 (B, J)
            try:
                s = s.expand(*batch_dims, J).reshape(B, J)
            except RuntimeError:
                # 允许只给每样本尺度（形状 (B,) 或 (*batch,1)）
                try:
                    s = s.expand(*batch_dims, 1).reshape(B, 1).expand(B, J)
                except RuntimeError:
                    raise ValueError(f"norm_scale 形状 {tuple(s.shape)} 不能广播到 {tuple((*batch_dims, J))}")
            s = s.clamp_min(1e-12)
            dist_norm = dist / s
        else:
            s_ref = self._compute_ref_scale(target)  # (B,1) or None
            if s_ref is not None:
                dist_norm = dist / s_ref.clamp_min(1e-12)
            else:
                dist_norm = dist  # 绝对阈值

        # 有效性与命中
        valid = torch.isfinite(dist_norm)
        if mask is not None:
            valid = valid & mask

        if valid.any().item():
            hit = (dist_norm <= self.threshold) & valid
            self.correct_count += hit.sum().to(torch.long)
            self.total_count   += valid.sum().to(torch.long)

    @torch.no_grad()
    def compute(self) -> Tensor:
        if self.total_count.item() == 0:
            return torch.tensor(float('nan'), dtype=torch.float32, device=self.correct_count.device)
        pck = (self.correct_count.to(torch.float64) / self.total_count).to(torch.float32)
        return pck

    @torch.no_grad()
    def reset(self) -> None:
        super().reset()




class BLE(Metric):
    """
    BLE (Bone Length Error)

    功能
    ----
    比较预测骨架与 GT 的骨长差异（绝对/相对）。
    - 输入形状: [..., J, 3]
    - 骨架由 `bones=[(i,j), ...]` 或 `parents=[-1, p1, p2, ...]` 指定（两者给其一即可）。
    - 有 mask 时，仅当骨骼两端关节均有效才统计该骨骼。
    - 'abs' 模式返回平均绝对骨长误差；'rel' 模式返回平均相对误差 |lp-lt|/max(lt, eps)。

    参数
    ----
    bones: Optional[Sequence[Tuple[int,int]]]
        明确的骨段关节对列表（i<->j）
    parents: Optional[Sequence[int]]
        每个关节的父索引，根为 -1；将自动生成 bones={(i, parents[i]) | parents[i]>=0}
    mode: Literal['abs','rel']
        绝对误差或相对误差
    unit_scale: float
        单位缩放（仅在 'abs' 模式下作用到误差；'rel' 模式下误差已无量纲）
    strict_shape: bool
        若 True，preds/target 形状不符直接报错
    eps: float
        相对误差与数值稳定性的极小值

    输出
    ----
    compute() -> 标量张量（float32）
    """

    full_state_update: bool = False
    higher_is_better:  bool = False
    is_differentiable: bool = False

    def __init__(
        self,
        bones: Optional[Sequence[Tuple[int, int]]] = None,
        parents: Optional[Sequence[int]] = None,
        mode: Literal["abs", "rel"] = "abs",
        unit_scale: float = 1.0,
        strict_shape: bool = True,
        eps: float = 1e-8,
        dist_sync_on_step: bool = False,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        if bones is None and parents is None:
            raise ValueError("必须提供 bones 或 parents 之一来定义骨架")
        if bones is not None and parents is not None:
            raise ValueError("bones 与 parents 只能提供其一")
        self.bones = None if bones is None else [(int(i), int(j)) for i, j in bones]
        self.parents = None if parents is None else [int(p) for p in parents]
        self.mode = mode
        self.unit_scale = float(unit_scale)
        self.strict_shape = bool(strict_shape)
        self.eps = float(eps)

        self.add_state("sum_error",
                       default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("total_count",
                       default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")

    # ---------- 工具函数 ----------

    @staticmethod
    def _ensure_shape(preds: Tensor, target: Tensor):
        if preds.shape != target.shape:
            raise ValueError(f"preds/target 形状不一致: {preds.shape} vs {target.shape}")
        if preds.ndim < 3 or preds.shape[-1] != 3:
            raise ValueError(f"期望输入为 [..., J, 3]，但得到 {preds.shape}")

    @staticmethod
    def _build_bones_from_parents(parents: Sequence[int]) -> Sequence[Tuple[int, int]]:
        bones = []
        for i, p in enumerate(parents):
            if p is None:
                continue
            if p >= 0:
                bones.append((i, p))
        if not bones:
            raise ValueError("由 parents 生成的骨段为空，请检查 parents 是否正确（根为 -1，其余为父索引）")
        return bones

    # ---------- Metric 接口 ----------

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> None:
        preds  = torch.as_tensor(preds)
        target = torch.as_tensor(target)

        if self.strict_shape:
            self._ensure_shape(preds, target)

        # 展平批维
        *batch_dims, J, C = preds.shape
        B = int(torch.tensor(batch_dims).prod().item()) if batch_dims else 1
        preds  = preds.reshape(B, J, C)
        target = target.reshape(B, J, C)

        # 准备骨段索引
        bones = self.bones if self.bones is not None else self._build_bones_from_parents(self.parents)
        # 验证索引合法
        for (i, j) in bones:
            if not (0 <= i < J and 0 <= j < J):
                raise ValueError(f"骨段索引越界: ({i},{j})，J={J}")

        # mask -> (B, J)，骨段有效性需要两端都有效
        if mask is not None:
            mask = torch.as_tensor(mask).bool()
            try:
                mask = mask.expand(*batch_dims, J).reshape(B, J)
            except RuntimeError:
                raise ValueError(f"mask 形状 {tuple(mask.shape)} 不能广播到 {tuple((*batch_dims, J))}")

        device = self.sum_error.device
        preds  = preds.to(device)
        target = target.to(device)
        if mask is not None:
            mask = mask.to(device)

        # 取两端点并计算骨长
        dtype = torch.float64
        idx_i = torch.tensor([i for i, _ in bones], device=device, dtype=torch.long)
        idx_j = torch.tensor([j for _, j in bones], device=device, dtype=torch.long)

        pi = preds[:, idx_i, :].to(dtype)   # (B, |E|, 3)
        pj = preds[:, idx_j, :].to(dtype)
        ti = target[:, idx_i, :].to(dtype)
        tj = target[:, idx_j, :].to(dtype)

        lp = torch.linalg.vector_norm(pi - pj, dim=-1)  # (B, |E|)
        lt = torch.linalg.vector_norm(ti - tj, dim=-1)  # (B, |E|)

        # 有效性：数值合法 +（若提供）关节 mask 两端都有效
        valid = torch.isfinite(lp) & torch.isfinite(lt)
        if mask is not None:
            mi = mask[:, idx_i]
            mj = mask[:, idx_j]
            valid = valid & mi & mj

        if valid.any().item():
            if self.mode == "abs":
                err = (lp - lt).abs()                    # 绝对长度误差
                if self.unit_scale != 1.0:
                    err = err * self.unit_scale
            elif self.mode == "rel":
                # 相对误差：对 target 骨长归一化；lt 为 0 的位置被忽略（valid 已要求 finite，但仍加 clamp）
                lt_safe = lt.clamp_min(self.eps)
                err = ((lp - lt).abs() / lt_safe)
            else:
                raise ValueError(f"不支持的 mode: {self.mode}")

            sel = err[valid].to(torch.float64)
            self.sum_error   += sel.sum()
            self.total_count += valid.sum().to(torch.long)

    @torch.no_grad()
    def compute(self) -> Tensor:
        if self.total_count.item() == 0:
            return torch.tensor(float('nan'), dtype=torch.float32, device=self.sum_error.device)
        return (self.sum_error / self.total_count).to(torch.float32)

    @torch.no_grad()
    def reset(self) -> None:
        super().reset()


class PAPCK(Metric):
    """
    PCK (Percentage of Correct Keypoints) with Procrustes Alignment (PA)

    流程
    ----
    1) 对齐：对每个样本，用(相似)Procrustes将 preds 对齐到 target（可选是否估计尺度）。
       - 对齐仅使用 mask=True 的关节（若给定），无效点不参与对齐。
       - 若有效点数少于 2，则跳过对齐（等价于恒等变换）。
    2) 计算距离：对齐后的 preds 与 target 的每关节欧氏距离 d，并按 unit_scale 缩放。
    3) 归一化：若给定 norm_scale，以其归一化；否则若给定 ref_pair，以该对关节在 target 中的距离做每样本尺度；
       否则使用绝对阈值。
    4) PCK：d' <= threshold 计为命中，仅统计 mask=True 且有限值的关节。

    参数
    ----
    threshold: float
        阈值（绝对或归一化后）
    unit_scale: float
        距离单位缩放（在比较阈值前施加）
    strict_shape: bool
        是否强检 preds/target 形状一致、且为 [..., J, C], C∈{2,3}
    ref_pair: Optional[(i, j)]
        每样本归一化参考关节对（当未给 norm_scale 时生效）
    align_with_scale: bool
        PA 对齐是否估计尺度（True=相似变换；False=刚体变换）
    dist_sync_on_step: bool
        torchmetrics 选项
    """

    full_state_update: bool = False
    higher_is_better:  bool = True
    is_differentiable: bool = False

    def __init__(
        self,
        threshold: float = 0.05,
        unit_scale: float = 1.0,
        strict_shape: bool = True,
        ref_pair: Optional[Tuple[int, int]] = None,
        align_with_scale: bool = True,
        dist_sync_on_step: bool = False,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        self.threshold = float(threshold)
        self.unit_scale = float(unit_scale)
        self.strict_shape = bool(strict_shape)
        self.ref_pair = ref_pair
        self.align_with_scale = bool(align_with_scale)

        self.add_state("correct_count",
                       default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")
        self.add_state("total_count",
                       default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")

    # ---------- 工具函数 ----------

    @staticmethod
    def _ensure_shape(preds: Tensor, target: Tensor):
        if preds.shape != target.shape:
            raise ValueError(f"preds/target 形状不一致: {preds.shape} vs {target.shape}")
        if preds.ndim < 3:
            raise ValueError(f"期望输入为 [..., J, C]，但得到 {preds.shape}")
        C = preds.shape[-1]
        if C not in (2, 3):
            raise ValueError(f"最后一维 C 只能为 2 或 3，但得到 C={C}")

    @torch.no_grad()
    def _compute_ref_scale(self, target: Tensor) -> Optional[Tensor]:
        """
        若设置了 ref_pair=(i,j)，返回形状 (B,1) 的每样本尺度；否则返回 None。
        target: (B, J, C)
        """
        if self.ref_pair is None:
            return None
        i, j = self.ref_pair
        J = target.shape[-2]
        if not (0 <= i < J and 0 <= j < J):
            raise ValueError(f"ref_pair 索引越界: (i={i}, j={j})，J={J}")
        vi = target[:, i, :]          # (B, C)
        vj = target[:, j, :]
        scale = torch.linalg.vector_norm(vi - vj, dim=-1).clamp_min(1e-12)  # (B,)
        return scale.view(-1, 1)      # (B,1)

    @torch.no_grad()
    def _pa_align_single(
        self, X: Tensor, Y: Tensor, valid: Optional[Tensor], with_scale: bool
    ) -> Tensor:
        """
        对单个样本做 Procrustes 对齐，将 X (J,C) 对齐到 Y (J,C)，返回对齐后的 X'。
        valid: (J,) 的 bool，指示用于估计对齐的关节；若 None 则全部使用。
        """
        J, C = X.shape
        if valid is None:
            valid = torch.ones(J, dtype=torch.bool, device=X.device)
        idx = valid.nonzero(as_tuple=False).flatten()
        # 有效点不足，跳过对齐
        if idx.numel() < 2:
            return X

        x = X[idx]  # (K, C)
        y = Y[idx]  # (K, C)

        # 去均值
        mu_x = x.mean(dim=0, keepdim=True)  # (1, C)
        mu_y = y.mean(dim=0, keepdim=True)
        x_c = x - mu_x
        y_c = y - mu_y

        # 协方差与 SVD
        # 注意：torch.linalg.svd 在低秩时也可用
        # M = x_c^T y_c
        M = x_c.transpose(0, 1) @ y_c  # (C, C)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)  # M ≈ U @ diag(S) @ Vh
        V = Vh.transpose(0, 1)
        R = V @ U.transpose(0, 1)  # 初始旋转

        # 处理反射（保证 det(R) > 0）
        if torch.linalg.det(R) < 0:
            V_corr = V.clone()
            V_corr[:, -1] *= -1
            R = V_corr @ U.transpose(0, 1)
            # S 的最后一项也视作取反，但尺度计算里用的是 S.sum()，下面会自然体现

        # 尺度
        if with_scale:
            x_var = (x_c**2).sum()  # Frobenius 范数平方
            # 防止除零
            if x_var.item() < 1e-12:
                s = 1.0
            else:
                s = (S.sum() / x_var).item()
        else:
            s = 1.0

        # 平移
        t = (mu_y.squeeze(0) - s * (R @ mu_x.squeeze(0)))  # (C,)

        # 应用于所有关节
        X_aligned = (s * (X @ R)) + t  # 这里使用右乘：X(J,C) @ R(C,C)

        return X_aligned

    @torch.no_grad()
    def _pa_align_batch(
        self, preds: Tensor, target: Tensor, mask: Optional[Tensor]
    ) -> Tensor:
        """
        逐样本对齐：preds/target: (B, J, C)；mask: (B, J) 或 None
        返回对齐后的 preds_aligned: (B, J, C)
        """
        B, J, C = preds.shape
        preds_aligned = torch.empty_like(preds, dtype=preds.dtype)
        for b in range(B):
            valid_b = mask[b] if mask is not None else None
            preds_aligned[b] = self._pa_align_single(
                preds[b], target[b], valid_b, self.align_with_scale
            )
        return preds_aligned

    # ---------- Metric 接口 ----------

    @torch.no_grad()
    def update(
        self,
        preds: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        norm_scale: Optional[Tensor] = None,
    ) -> None:
        """
        参数
        ----
        preds, target: [..., J, C] (C=2或3)
        mask:         [..., J] 的 bool（True=有效），可广播
        norm_scale:   每样本或每关节归一化尺度；形状可为 [...], [...,1], 或 [...,J]，
                      将广播到 (B, J)。若给定，优先于 ref_pair。
        """
        preds  = torch.as_tensor(preds)
        target = torch.as_tensor(target)

        if self.strict_shape:
            self._ensure_shape(preds, target)

        # 展平批维
        *batch_dims, J, C = preds.shape
        B = int(torch.tensor(batch_dims).prod().item()) if batch_dims else 1
        preds  = preds.reshape(B, J, C)
        target = target.reshape(B, J, C)

        # mask -> (B, J)（用于对齐与评估）
        if mask is not None:
            mask = torch.as_tensor(mask).bool()
            try:
                mask = mask.expand(*batch_dims, J).reshape(B, J)
            except RuntimeError:
                raise ValueError(f"mask 形状 {tuple(mask.shape)} 不能广播到 {tuple((*batch_dims, J))}")

        device = self.correct_count.device
        preds  = preds.to(device)
        target = target.to(device)
        if mask is not None:
            mask = mask.to(device)

        # --------- 先做 PA 对齐（逐样本，基于 mask=True 的关节） ---------
        preds_aligned = self._pa_align_batch(preds, target, mask)

        # --------- 距离（单位缩放） ---------
        dist = torch.linalg.vector_norm(preds_aligned - target, dim=-1).to(torch.float64)  # (B, J)
        if self.unit_scale != 1.0:
            dist = dist * self.unit_scale

        # --------- 归一化尺度 ---------
        if norm_scale is not None:
            s = torch.as_tensor(norm_scale, device=device, dtype=torch.float64)
            # 广播到 (B, J)
            try:
                s = s.expand(*batch_dims, J).reshape(B, J)
            except RuntimeError:
                # 允许只给每样本尺度（形状 (B,) 或 (*batch,1)）
                try:
                    s = s.expand(*batch_dims, 1).reshape(B, 1).expand(B, J)
                except RuntimeError:
                    raise ValueError(f"norm_scale 形状 {tuple(s.shape)} 不能广播到 {tuple((*batch_dims, J))}")
            s = s.clamp_min(1e-12)
            dist_norm = dist / s
        else:
            s_ref = self._compute_ref_scale(target)  # (B,1) or None
            if s_ref is not None:
                dist_norm = dist / s_ref.clamp_min(1e-12)
            else:
                dist_norm = dist  # 绝对阈值

        # --------- 有效性与命中 ---------
        valid = torch.isfinite(dist_norm)
        if mask is not None:
            valid = valid & mask

        if valid.any().item():
            hit = (dist_norm <= self.threshold) & valid
            self.correct_count += hit.sum().to(torch.long)
            self.total_count   += valid.sum().to(torch.long)

    @torch.no_grad()
    def compute(self) -> Tensor:
        if self.total_count.item() == 0:
            return torch.tensor(float('nan'), dtype=torch.float32, device=self.correct_count.device)
        pck = (self.correct_count.to(torch.float64) / self.total_count).to(torch.float32)
        return pck

    @torch.no_grad()
    def reset(self) -> None:
        super().reset()



class BDE(Metric):
    """
    Bone Direction Error (角度, 单位: 度)
    - 输入: [..., J, 3]
    - 每条骨段向量 v = p[i]-p[j]，与 GT 的夹角 acos( clamp( dot/(||v||*||vt||) ) )
    - 仅统计两端关节均有效的骨段；零长度骨段将被跳过（不计入统计）
    """
    full_state_update: bool = False
    higher_is_better:  bool = False
    is_differentiable: bool = False

    def __init__(
        self,
        bones: Optional[Sequence[Tuple[int, int]]] = None,
        parents: Optional[Sequence[int]] = None,
        strict_shape: bool = True,
        eps: float = 1e-8,
        dist_sync_on_step: bool = False,
        **kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        if bones is None and parents is None:
            raise ValueError("必须提供 bones 或 parents 之一")
        if bones is not None and parents is not None:
            raise ValueError("bones 与 parents 只能提供其一")
        self.bones = None if bones is None else [(int(i), int(j)) for i, j in bones]
        self.parents = None if parents is None else [int(p) for p in parents]
        self.strict_shape = bool(strict_shape)
        self.eps = float(eps)

        self.add_state("sum_deg",
                       default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("total_count",
                       default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")

    @staticmethod
    def _ensure_shape(preds: Tensor, target: Tensor):
        if preds.shape != target.shape:
            raise ValueError(f"preds/target 形状不一致: {preds.shape} vs {target.shape}")
        if preds.ndim < 3 or preds.shape[-1] != 3:
            raise ValueError(f"期望输入为 [..., J, 3]，但得到 {preds.shape}")

    @staticmethod
    def _build_bones_from_parents(parents: Sequence[int]) -> Sequence[Tuple[int, int]]:
        bones = []
        for i, p in enumerate(parents):
            if p is None:
                continue
            if p >= 0:
                bones.append((i, p))
        if not bones:
            raise ValueError("parents 生成的骨段为空")
        return bones

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> None:
        preds = torch.as_tensor(preds)
        target = torch.as_tensor(target)

        if self.strict_shape:
            self._ensure_shape(preds, target)

        *batch_dims, J, C = preds.shape
        B = int(torch.tensor(batch_dims).prod().item()) if batch_dims else 1
        preds  = preds.reshape(B, J, C)
        target = target.reshape(B, J, C)

        bones = self.bones if self.bones is not None else self._build_bones_from_parents(self.parents)
        for (i, j) in bones:
            if not (0 <= i < J and 0 <= j < J):
                raise ValueError(f"骨段越界: ({i},{j}), J={J}")

        if mask is not None:
            mask = torch.as_tensor(mask).bool()
            try:
                mask = mask.expand(*batch_dims, J).reshape(B, J)
            except RuntimeError:
                raise ValueError(f"mask 形状 {tuple(mask.shape)} 不能广播到 {tuple((*batch_dims, J))}")

        device = self.sum_deg.device
        preds  = preds.to(device, dtype=torch.float64)
        target = target.to(device, dtype=torch.float64)
        if mask is not None:
            mask = mask.to(device)

        idx_i = torch.tensor([i for i, _ in bones], device=device, dtype=torch.long)
        idx_j = torch.tensor([j for _, j in bones], device=device, dtype=torch.long)

        pv = preds[:, idx_i, :] - preds[:, idx_j, :]   # (B, E, 3)
        tv = target[:, idx_i, :] - target[:, idx_j, :]

        lp = torch.linalg.vector_norm(pv, dim=-1)  # (B, E)
        lt = torch.linalg.vector_norm(tv, dim=-1)

        # 有效性：长度都>eps，且（若给定）两端关节有效
        valid = (lp > self.eps) & (lt > self.eps) & torch.isfinite(lp) & torch.isfinite(lt)
        if mask is not None:
            mi = mask[:, idx_i]
            mj = mask[:, idx_j]
            valid = valid & mi & mj

        if valid.any().item():
            dot = (pv * tv).sum(-1)
            cos = (dot / (lp * lt).clamp_min(self.eps)).clamp(-1.0, 1.0)
            ang = torch.rad2deg(torch.arccos(cos))  # (B, E)
            sel = ang[valid]
            self.sum_deg    += sel.sum()
            self.total_count += valid.sum().to(torch.long)

    @torch.no_grad()
    def compute(self) -> Tensor:
        if self.total_count.item() == 0:
            return torch.tensor(float('nan'), dtype=torch.float32, device=self.sum_deg.device)
        return (self.sum_deg / self.total_count).to(torch.float32)
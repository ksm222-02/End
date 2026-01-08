from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class OnlineRegressionMetrics:
    """
    Streaming regression metrics for multi-dimensional outputs.

    Tracks per-dimension:
      - MAE, MSE, RMSE
      - max absolute error
      - R2 (computed at finalize; NaN if variance ~ 0)
    """

    num_dims: int
    eps: float = 1e-12
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        d = int(self.num_dims)
        self.count = 0
        self.sum_abs_err = torch.zeros(d, device=self.device, dtype=torch.float64)
        self.sum_sq_err = torch.zeros(d, device=self.device, dtype=torch.float64)
        self.max_abs_err = torch.zeros(d, device=self.device, dtype=torch.float64)
        self.sum_y = torch.zeros(d, device=self.device, dtype=torch.float64)
        self.sum_y_sq = torch.zeros(d, device=self.device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Args:
          predictions: (N, D) or (B, T, D)
          targets:     (N, D) or (B, T, D)
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: pred={tuple(predictions.shape)} target={tuple(targets.shape)}")

        if predictions.dim() == 3:
            predictions = predictions.reshape(-1, predictions.size(-1))
            targets = targets.reshape(-1, targets.size(-1))
        if predictions.dim() != 2:
            raise ValueError("predictions/targets must be shaped (N,D) or (B,T,D)")
        if predictions.size(-1) != self.num_dims:
            raise ValueError(f"Expected D={self.num_dims}, got {predictions.size(-1)}")

        pred = predictions.detach().to(device=self.device, dtype=torch.float64)
        y = targets.detach().to(device=self.device, dtype=torch.float64)

        err = pred - y
        abs_err = err.abs()
        sq_err = err.square()

        self.count += int(pred.size(0))
        self.sum_abs_err += abs_err.sum(dim=0)
        self.sum_sq_err += sq_err.sum(dim=0)
        self.max_abs_err = torch.maximum(self.max_abs_err, abs_err.max(dim=0).values)
        self.sum_y += y.sum(dim=0)
        self.sum_y_sq += y.square().sum(dim=0)

    def finalize(
        self,
        prefix: str = "",
        names: Optional[list[str]] = None,
        overall_weights: Optional[list[float]] = None,
    ) -> Dict[str, float]:
        if self.count <= 0:
            raise RuntimeError("No samples were added to metrics.")

        n = float(self.count)
        mse = self.sum_sq_err / n
        mae = self.sum_abs_err / n
        rmse = torch.sqrt(mse)

        mean_y = self.sum_y / n
        ss_tot = self.sum_y_sq - n * mean_y.square()
        ss_res = self.sum_sq_err
        r2 = 1.0 - (ss_res / torch.clamp(ss_tot, min=self.eps))
        r2 = torch.where(ss_tot < self.eps, torch.full_like(r2, float("nan")), r2)

        out: Dict[str, float] = {}
        if names is None:
            names = [f"dim{i}" for i in range(self.num_dims)]
        if len(names) != self.num_dims:
            raise ValueError(f"names must have length {self.num_dims}")

        for i, name in enumerate(names):
            base = f"{prefix}{name}"
            out[f"{base}_mae"] = float(mae[i].item())
            out[f"{base}_mse"] = float(mse[i].item())
            out[f"{base}_rmse"] = float(rmse[i].item())
            out[f"{base}_max_error"] = float(self.max_abs_err[i].item())
            out[f"{base}_r2"] = float(r2[i].item()) if torch.isfinite(r2[i]) else float("nan")

        out[f"{prefix}mse_mean"] = float(mse.mean().item())
        out[f"{prefix}mae_mean"] = float(mae.mean().item())

        if overall_weights is not None:
            if len(overall_weights) != self.num_dims:
                raise ValueError(f"overall_weights must have length {self.num_dims}")
            w = torch.tensor(overall_weights, device=mse.device, dtype=mse.dtype)
            out[f"{prefix}overall_weighted_mse"] = float((mse * w).sum().item())
        return out

from typing import Any, Callable, Dict, Optional

import torch
from torchmetrics import Metric

from .utils import sort_predictions


class minADE(Metric):
    """Minimum Average Displacement Error
    minADE: The average L2 distance between the best forecasted trajectory and the ground truth.
            The best here refers to the trajectory that has the minimum endpoint error.
            
            用于计算 Minimum Average Displacement Error (minADE)，通常用于轨迹预测任务中评价预测路径的准确性
    """

    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        k=6,#k: 从模型预测中选择的前 k 条轨迹，用于计算 minADE
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(minADE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.k = k
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        with torch.no_grad():
            pred, _ = sort_predictions(
                outputs["trajectory"], outputs["probability"], k=self.k
            )
            ade = torch.norm(
                pred[..., :2] - target.unsqueeze(1)[..., :2], p=2, dim=-1
            ).mean(-1)
            min_ade = ade.min(-1)[0]
            self.sum += min_ade.sum()
            self.count += pred.size(0)#size(0) 获取张量第一个维度的大小，即 批次大小（batch_size）

    def compute(self) -> torch.Tensor:
        return self.sum / self.count

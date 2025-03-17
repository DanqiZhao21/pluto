import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection
from src.metrics import MR, minADE, minFDE
from src.metrics.prediction_avg_ade import PredAvgADE
from src.metrics.prediction_avg_fde import PredAvgFDE
from src.optim.warmup_cos_lr import WarmupCosLR

from .loss.esdf_collision_loss import ESDFCollisionLoss
from pytorch_lightning.callbacks import TensorBoardLogger
logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        use_collision_loss=True,
        use_contrast_loss=False,
        regulate_yaw=False,
        objective_aggregate_mode: str = "mean",
    ) -> None:
        
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.objective_aggregate_mode = objective_aggregate_mode
        self.history_steps = model.history_steps
        self.use_collision_loss = use_collision_loss
        self.use_contrast_loss = use_contrast_loss
        self.regulate_yaw = regulate_yaw

        self.radius = model.radius #120
        self.num_modes = model.num_modes #12 #6每一种参考线下面会有6条轨迹
        self.mode_interval = self.radius / self.#模型横向间隔为 10/20   120/12=10
        
        # 初始化 TensorBoardLogger
        self.tb_logger = TensorBoardLogger('logs', name='training_logs')

        if use_collision_loss:
            self.collision_loss = ESDFCollisionLoss()

    def on_fit_start(self) -> None:
        metrics_collection = MetricCollection(
            [
                minADE().to(self.device),
                minFDE().to(self.device),
                MR(miss_threshold=2).to(self.device),
                PredAvgADE().to(self.device),
                PredAvgFDE().to(self.device),
            ]
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }
        # 字典通过键快速访问对应的值，例如 self.metrics["train"]。,动态储存数据
        #参数 prefix="train/" 为这个集合中所有指标加上前缀，例如 train/minADE。

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        
        
        batch：一个包含输入特征、目标值、场景列表的元组。
        类型：Tuple[FeaturesType, TargetsType, ScenarioListType]，用来严格指定元组中每个元素的类型。
        实际数据：features 是输入数据，targets 是监督学习目标，scenarios 表示场景（具体内容依赖于上下文）。
        prefix：字符串，用于在记录日志时为名称添加前缀（如 "train/" 或 "val/" 等）。

        """
        import ipdb; ipdb.set_trace()
        print(f" batch in _step() is { batch}")
        features, targets, scenarios = batch
        print(f" scenarios in _step() is { scenarios}")
        print(f" type of scenarios  is { type(scenarios)}")
        
        res = self.forward(features["feature"].data)

        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"] if self.training else 0.0

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        #res: 包含模型的预测输出，可能包括轨迹预测、概率预测
        #data: 包含输入数据和目标标签的信息，比如目标位置、速度、掩码等。
        
        bs, _, T, _ = res["prediction"].shape

        if self.use_contrast_loss:
            train_num = (bs // 3) * 2 if self.training else bs
        else:
            train_num = bs
            
#从res 中提取了不同的模型输出
# train_num就是bs
# 表示从数据的第 0 个索引开始（默认起始索引为 0），一直到（但不包括）索引 train_num 的部分
# batch_size：样本数量。
# R：参考线数量（Reference Lines）。
# M：每条参考线的候选轨迹数量。
# T：时间步数（未来预测的时间步）。
# 4：轨迹信息，例如 [x, y, vx, vy] 或 [x, y, dx/dt, dy/dt]

        trajectory, probability, prediction = (
            res["trajectory"][:train_num],  #(train_num, R, M, T, 6)
            res["probability"][:train_num],#(train_num, R, M)
            res["prediction"][:train_num], #(train_num, A-1, T, 2)
        )
        ref_free_trajectory = res.get("ref_free_trajectory", None)
        
        
#从data中提取目标数据
        targets_pos = data["agent"]["target"][:train_num] #(train_num, num_agents, 3)最后一维可能是3个维度（x,y,theta)
        valid_mask = data["agent"]["valid_mask"][:train_num, :, -T:] #(train_num, num_agents, T)
        targets_vel = data["agent"]["velocity"][:train_num, :, -T:] #(train_num, num_agents, T)
        
#通常在多目标预测任务中，targets_pos 是一个三维张量 [batch_size, time_steps, features]，features 的最后一维定义了每个时间步的特征。
    #torch.cat 用于将多个张量按指定维度 (dim) 拼接。
    #参数 dim=-1 表示沿着最后一个维度进行拼接（例如，如果张量是三维的 [B, T, F]，则在特征维度 F 上拼接）。
    #torch.stack 用于创建一个新的维度，将输入张量列表沿指定维度堆叠成一个新张量。
    #:2 是切片操作，选择最后一维的索引范围 [0, 1]
    #2 是索引，表示提取最后一个维度（通常是表示 x, y, θ 的维度）中的第 3 个元素
    
# torch.stack 是 PyTorch 中的一个操作，用于将一组张量沿新的维度进行拼接。它与 
# torch.cat 的主要区别在于，cat 是沿现有的维度进行拼接，
# 而 stack 会在指定的新的维度上创建一个额外的维度。

        target = torch.cat(
            [
                targets_pos[..., :2],#x,y
                torch.stack(
                    [targets_pos[..., 2].cos(), targets_pos[..., 2].sin()], dim=-1
                ),
                targets_vel,
            ],
            dim=-1,
        )
        #(train_num, num_agents, 6)  最后一个维度6： target [x,y,cos(theta),sin(theta),velocity_X,V_Y]
                
        #ego_reg_loss: 回归损失，用于评估预测轨迹与目标轨迹之间的位置偏差。
        # : 分类损失，用于评估轨迹概率的准确性。
        # collision_loss: 碰撞损失，用于评估预测轨迹是否与障碍物发生碰撞。

        #切片与索引：
        # 切片和索引的语法
        # tensor[:, 0]：表示在所有批次（第一维）中，选择目标（【第二维】）索引为 0 的目标，
            # 保留所有时间步（第三维），其结果的形状是 (N, T, ...)，
            # 这里的 N 是批次大小，T 是时间步数，... 是其他维度的内容。
            # tensor 是 (4, 3, 5, 2)，那么 tensor[:, 0] 将返回 (4, 5, 2)
        # tensor[..., 0]：表示在所有维度中，选择最后一维（即最后一列）的第 0 项。
            # ... 表示“保留”其他所有维度的内容。这通常用于提取某个特征，比如获取最后一维的某个特定元素。
            #tensor 是 (4, 3, 5, 2)-->所有时间步的第一个特征（例如，位置坐标）。这会返回形状为 (4, 3, 5) 
    
        # planning loss
        ego_reg_loss, ego_cls_loss, collision_loss = self.get_planning_loss(
            data, trajectory, probability, valid_mask[:, 0], target[:, 0], train_num
        )
        # valid_mask[:, 0](第二维只取自车)  (bs, T)
        # trajectory:(bs, R, M, T, 4)
        # target[:, 0]  (第二维只取自车)
        # best_trajectory[Bs T fearture_dim] 
        if ref_free_trajectory is not None:
            ego_ref_free_reg_loss = F.smooth_l1_loss(
                ref_free_trajectory[:train_num],
                target[:, 0, :, : ref_free_trajectory.shape[-1]],
                reduction="none",
            ).sum(-1)
            ego_ref_free_reg_loss = (
                ego_ref_free_reg_loss * valid_mask[:, 0]
            ).sum() / valid_mask[:, 0].sum()
        else:
            ego_ref_free_reg_loss = ego_reg_loss.new_zeros(1)
        #    ego_cls_loss


        # prediction loss
        prediction_loss = self.get_prediction_loss(
            data, prediction, valid_mask[:, 1:], target[:, 1:]
        )
        #valid_mask[:, 1:] 除开自车的掩码
        #prediction ：res["prediction"][:train_num], #(train_num, A-1, T, 2) 周车的预测信息
        if self.training and self.use_contrast_loss:
            contrastive_loss = self._compute_contrastive_loss(
                res["hidden"], data["data_n_valid_mask"]
            )
        else:
            contrastive_loss = prediction_loss.new_zeros(1)

        loss = (
            ego_reg_loss
            + ego_cls_loss
            + prediction_loss
            + contrastive_loss
            + collision_loss
            + ego_ref_free_reg_loss
        )

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.item(),
            "cls_loss": ego_cls_loss.item(),
            "ref_free_reg_loss": ego_ref_free_reg_loss.item(),
            "collision_loss": collision_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
        }

    def get_prediction_loss(self, data, prediction, valid_mask, target):
        """
        prediction: (bs, A-1, T, 6)
        valid_mask: (bs, A-1, T)
        target: (bs, A-1, 6)
        """
        
        # bs 是批次大小。
        # A-1 是预测的目标数量（通常是除自车以外的其他代理对象）。
        # T 是时间步数。
        # 6 是预测的特征维度（如位置、速度等）。

        # 掩码是一种布尔矩阵（或者 0/1 的数值矩阵），其维度与需要过滤的数据一致。
        # 1（True）：表示有效位置，这些位置的值会被保留，用于后续计算。
        # 0（False）：表示无效位置，这些位置的值会被忽略。

        prediction_loss = F.smooth_l1_loss(
            prediction[valid_mask], target[valid_mask], reduction="none"
        ).sum(-1)
        prediction_loss = prediction_loss.sum() / valid_mask.sum()

        return prediction_loss

    def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs):
        
        """
        trajectory: (bs, R, M, T, 4)
        valid_mask: (bs, T)
        """
        # trajectory：模型生成的轨迹预测，形状为 (bs, R, M, T, 4)。
            # bs 是批次大小。
            # R 是参考线数量（每条轨迹关联一条参考线）。
            # M 是模式数量（轨迹的模式，如左转、直行等）。
            # T 是时间步数。
            # 4 是轨迹特征（如位置和速度等）。
        
        # probability： (bs, R, M)。
        # valid_mask： (bs, T)
        # target： (bs, T, 6)

        #valid_mask = torch.tensor([
        #     [1, 1, 0, 0],  # 第 1 个样本的有效时间步
        #     [1, 1, 1, 1],  # 第 2 个样本的有效时间步
        #     [0, 0, 0, 0],  # 第 3 个样本没有有效时间步
        # ]) 
        #num_valid_points = tensor([2, 4, 0])  [.sum是对最后一个维度求和，两维矩阵的话就是按行求和]
                #有效时间步长计算
      

        num_valid_points = valid_mask.sum(-1) #(batch_size, num_agents) 计算每个目标在所有时间步中有效的点数
        endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # max 8s #yTorch 的 .long() 方法相当于向下取整（floor 操作）。
        #(80/10=8)
        #参考线处理
            #【1】 data["reference_line"]["valid_mask"] 的形状是 (4, 10, 5)，
            # 表示批次大小为 4，每个批次有 10 条参考线，每条参考线有 5 个时间步的数据。 
            #【2】any(-1) 是 PyTorch 中的一个操作，用于沿着最后一个维度（即时间步维度 num_time_steps）
            # 检查每个参考线在该时间步是否有效。返回的是一个布尔张量，
            #如果 valid_mask 的形状是 (3, 10, 5)，则 valid_mask.any(-1) 
            # 的结果是一个形状为 (3, 10) 的布尔张量
            # (bs, R)#标记无效的参考线
            # 【3】data["reference_line"]["future_projection"][:bs] 
            # 返回一个形状为 (bs, num_reference_line（R), num_time_steps(T), feature_dim) 的张量。
            # 【B R T F】-->[B R F]  （endpoint_index使得T变为1，让第三个维度消失了）
            # 使用 torch.arange(bs) 和 endpoint_index 进行索引：
            # 从每个批次中选择所有参考线 (R)，
            # 从每个批次中选择一个 endpoint_index 对应的时间步（T），
          
        r_padding_mask = ~data["reference_line"]["valid_mask"][:bs].any(-1)  
        future_projection = data["reference_line"]["future_projection"][:bs][
            torch.arange(bs), :, endpoint_index
        ]
        
        # data["reference_line"]["future_projection"] 的形状是一个 4 维张量，'
        # 形状为 (batch_size, num_reference_lines, num_future_points, 2)，其中
        
        # future_projection 的形状是 (bs,R, feature_dim(F))
        # r_padding_mask 的维度是 (bs, R)，表示每个批次中参考线的有效性。
        
        target_r_index = torch.argmin(
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
        )
        # future_projection[..., 1] 的结果--> (bs, R) 
        # 选择最后一个维度（即F)上的第二个特征作为最小值
        
        target_m_index = (
            future_projection[torch.arange(bs), target_r_index, 0] / self.mode_interval
        ).long()#未来投影点
        # 【B R F】-->[B 1 1]-->[B]
        # 对应的是target_refline的最后一个维度的第一个特征值
        # 通过除以 self.mode_interval 来将该值映射为离散的模式索引，最后 .long() 将其转换为整数索引。
        # target_m_index 是一个索引值的序列，而不仅仅是单个值
        
        target_m_index.clamp_(min=0, max=self.num_modes - 1)

        target_label = torch.zeros_like(probability)#【Bs R M]
        target_label[torch.arange(bs), target_r_index, target_m_index] = 1

        best_trajectory = trajectory[torch.arange(bs), target_r_index, target_m_index]
        #选择最佳参考线最佳模式下的那一种轨迹，包含剩下的两个维度：时间+特征
        # trajectory [Bs R M T feature_dim=4]
        # best_trajectory[Bs T fearture_dim=4]

#计算碰撞损失
# data["cost_maps"] 返回一个张量。

        if self.use_collision_loss:
            collision_loss = self.collision_loss(
                best_trajectory, data["cost_maps"][:bs, :, :, 0].float()
            )
        else:
            collision_loss = trajectory.new_zeros(1)
#回归损失
        reg_loss = F.smooth_l1_loss(best_trajectory, target, reduction="none").sum(-1)
        reg_loss = (reg_loss * valid_mask).sum() / valid_mask.sum()
        
        # target-->(bs, A-1, 6)
        # 此处 target=target[:, 0]-->[Bs  6](只有自车了)
        # best_trajectory[Bs T fearture_dim]
        # reduction="none" 表示返回每个元素的损失，而不是对所有元素的损失进行求和或平均
        

        
        
#分类损失——cls_loss：轨迹的分类损失（选择最佳参考线和模式的误差）？

        speed_obj=[]#keep accelerate decelerate
        path_obj=[]#straight right_change left_change right_turn left_turn
        Bs, R, M = probability.shape
        # teacher_label = torch.zeros_like(probability)#【Bs R M]
        lane_cur=0
        teacher_label=create_teacher_label(Bs, R, M, speed_obj, path_obj,lane_cur)
        
        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)
        
        #软损失+硬损失
        cls_loss_hard = F.cross_entropy(
            probability.reshape(bs, -1), target_label.reshape(bs, -1).detach()
        )
        
        cls_loss_soft = F.cross_entropy(
            probability.reshape(bs, -1), teacher_label.reshape(bs, -1).detach()
        )
        cls_loss=cls_loss_hard+cls_loss_soft

        # probability:[Bs R M]
        # target_label"【Bs R M]  独热向量，只有最佳参考线最佳轨迹下为1 ，其余都为0
        # probability.reshape(bs, -1) 变成 [Bs, R * M]，相当于把每个样本的 R 条参考线和 M 个轨迹合并，变成 一个 R*M 维度的分类任务。
        
#航偏角正则化损失
        if self.regulate_yaw:
            heading_vec_norm = torch.norm(best_trajectory[..., 2:4], dim=-1)
            yaw_regularization_loss = F.l1_loss(
                heading_vec_norm, heading_vec_norm.new_ones(heading_vec_norm.shape)
            )
            reg_loss += yaw_regularization_loss

        return reg_loss, cls_loss, collision_loss

    def _compute_contrastive_loss(
        self, hidden, valid_mask, normalize=True, tempreture=0.1
    ):
        """
        Compute triplet loss

        Args:
            hidden: (3*bs, D)
        """
        if normalize:
            hidden = F.normalize(hidden, dim=1, p=2)

        if not valid_mask.any():
            return hidden.new_zeros(1)

        x_a, x_p, x_n = hidden.chunk(3, dim=0)

        x_a = x_a[valid_mask]
        x_p = x_p[valid_mask]
        x_n = x_n[valid_mask]

        logits_ap = (x_a * x_p).sum(dim=1) / tempreture
        logits_an = (x_a * x_n).sum(dim=1) / tempreture
        labels = x_a.new_zeros(x_a.size(0)).long()

        triplet_contrastive_loss = F.cross_entropy(
            torch.stack([logits_ap, logits_an], dim=1), labels
        )
        return triplet_contrastive_loss

#下面是指标计算
    def _compute_metrics(self, res, data, prefix) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
# res：包含模型的预测结果，通常包括轨迹预测和概率分布（如 trajectory 和 probability）。
# data：真实的输入数据和标注（ground truth），用于计算与预测结果的差异。
# prefix：一个字符串，用于标识评价指标所属的任务或阶段，可能用于区分训练和验证过程的指标。

        # get top 6 modes选择最好的6条 K=6
        trajectory, probability = res["trajectory"], res["probability"]
        r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        bs, R, M, T, _ = trajectory.shape
        trajectory = trajectory.reshape(bs, R * M, T, -1)
        probability = probability.reshape(bs, R * M)
        top_k_prob, top_k_index = probability.topk(6, dim=-1)
        top_k_traj = trajectory[torch.arange(bs)[:, None], top_k_index]

        outputs = {
            "trajectory": top_k_traj[..., :2],
            "probability": top_k_prob,
            "prediction": res["prediction"][..., :2],
            "prediction_target": data["agent"]["target"][:, 1:],
            "valid_mask": data["agent"]["valid_mask"][:, 1:, self.history_steps :],
        }
        target = data["agent"]["target"][:, 0]

        metrics = self.metrics[prefix](outputs, target)
        return metrics

    def _log_step(
        self,
        loss,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True if prefix == "train" else False,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]

    # def on_before_optimizer_step(self, optimizer) -> None:
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print("unused param", name)



#生成teacher矩阵信息
def create_teacher_label(Bs, R, M, speed_obj, path_obj,lane_cur):
    """
    生成 teacher_label 矩阵
    
    :param bs: batch size
    :param R: 左右车道数R (R)
    :param M: 模式数
    :param speed_obj: 纵向动作 ["keep", "accelerate", "decelerate"]
    :param path_obj: 横向动作 ["straight", "right_change", "left_change", "right_turn", "left_turn"]
    :param target_r_index: 目标时间索引 (bs,)
    :param target_m_index: 目标车道索引 (bs,)
    
    :return: teacher_label (bs, num_timesteps, num_lanes)
    """
    # 初始化 teacher_label
    teacher_label = torch.ones(Bs, R, M) * 10  # 默认填充为 10

    # 处理纵向行为 (speed_obj)
    for i in range(Bs):
        if speed_obj[i] == "keep":
            continue
        elif speed_obj[i] == "accelerate":
            teacher_label[i, :, :M//3] = 30       # 赋值给 0 ~ M/3 这一段
            teacher_label[i, :, M//3:2*M//3] = 20 # 赋值给 M/3 ~ 2M/3 这一段
            teacher_label[i, :, 2*M//3:] = 10     # 赋值给 2M/3 ~ M 这一段
        elif speed_obj[i] == "decelerate":
            teacher_label[i, :, :M//3] =10       # 赋值给 0 ~ M/3 这一段
            teacher_label[i, :, M//3:2*M//3] = 20 # 赋值给 M/3 ~ 2M/3 这一段
            teacher_label[i, :, 2*M//3:] = 30     # 赋值给 2M/3 ~ M 这一段

    #先不处理横向行为，横向行为已经很准确了
    # # 处理横向行为 (path_obj)
    # for i in range(Bs):
    #     lane = target_m_index[i]  # 当前位置车道索引

    #     if path_obj[i] == "straight":
    #         teacher_label[i, :, lane] = 1  # 当前位置设为 1
    #     elif path_obj[i] == "right_change" and lane < num_lanes - 1:
    #         teacher_label[i, :, lane] = 5
    #         teacher_label[i, :, lane + 1] = 1
    #     elif path_obj[i] == "left_change" and lane > 0:
    #         teacher_label[i, :, lane] = 5
    #         teacher_label[i, :, lane - 1] = 1
    #     elif path_obj[i] == "right_turn" and lane < num_lanes - 1:
    #         teacher_label[i, :, lane + 1] = 1
    #     elif path_obj[i] == "left_turn" and lane > 0:
    #         teacher_label[i, :, lane - 1] = 1
    return teacher_label

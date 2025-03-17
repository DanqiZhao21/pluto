import logging
import json
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

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario, CameraChannel, LidarChannel
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
import os
import numpy as np

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection
from src.metrics import MR, minADE, minFDE
from src.metrics.prediction_avg_ade import PredAvgADE
from src.metrics.prediction_avg_fde import PredAvgFDE
from src.optim.warmup_cos_lr import WarmupCosLR

from .loss.esdf_collision_loss import ESDFCollisionLoss
import time

logger = logging.getLogger(__name__)

MM = 0  # 全局变量
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
        self.mode_interval = self.radius / self.num_modes#模型横向间隔为 10/20   120/12=10

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
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str) -> torch.Tensor:
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

        log_db=[]
        token_db=[]
        
        features, targets, scenarios = batch
        
        
        for scenario in scenarios:  
            log_db.append(scenario.log_db)  
            token_db.append(scenario.token)
        
        
        
        
        
        
        
        
        skip_batch = True  # 标记是否要跳过整个 batch

        res={}
        for i in range(7):  # 0 表示 cur, 1-6 表示 cur-1, cur-2, ..., cur-6
            # print(f"cur -{i}")
            cur_key = "cur" if i == 0 else f"cur-{i}"  # 处理 cur 和 cur-1 ~ cur-6
            res[cur_key] = self.forward(features["feature"].data[cur_key])#轨迹预测的结果预测的，是只有80步长的呢！
            # print(f"res[cur_key] is {res[cur_key]}")
            if res[cur_key]["trajectory"] is None:
                # print(f"some of res[cur_key] is None")
                # print(f"res[cur_key] is {res[cur_key]}")
                skip_batch = True 
                print(f"skip_batch is {skip_batch}")
                break  # 只要有一个有效数据，就不跳过 batch
            else:
                skip_batch =False
        
        
        if skip_batch:
            print("Skipping entire batch because some forward() results are None")
            return None
                    
    
        losses = self._compute_objectives(res, features["feature"].data)#预测结果和真值之间计算Loss  这个feature里面应该也只有每一帧的信息，不然应该会多一个维度150
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)
        return losses["loss"] if self.training else 0.0
       
    

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        '''
        32个scenario 8个reference 12个mode 8s刚好为8/0.1=80步
        trajectory: ([32, 7, 12, 80, 6])  
        probability: ([32, 7, 12])
        prediction: ([32, 48, 80, 6])
        
        res.keys()ipdb> p res.keys()
        dict_keys(['trajectory', 'probability', 'prediction', 'output_prediction', 'output_trajectory', 'candidate_trajectories'])
        
        features["feature"].data.kets()yys()ipdb> p features["feature"].data.keys()即data.keys()
        dict_keys(['agent', 'map', 'reference_line', 'static_objects', 'current_state', 'origin', 'angle', 'cost_maps'])

        <class 'src.features.pluto_feature.PlutoFeature'>   #type(features["feature"])
        
         dir(features["feature"])
         '__abstractmethods__', '__annotations__', '__class__', '__dataclass_fields__', '__dataclass_params__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', 
        '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', 
        '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl', 'collate', 'data', 'data_n', 'data_n_info', 'data_p', 'deserialize', 'is_valid', 'normalize', 'serialize', 
        'to_device', 'to_feature_tensor', 'to_numpy', 'unpack']
        '''
        #res: 包含模型的预测输出，可能包括轨迹预测、概率预测
        #data: 包含输入数据和目标标签的信息，比如目标位置、速度、掩码等。
        
        
        
        bs, _, T, _ = res["cur"]["prediction"].shape  #torch.Size([32, 48, 80, 6])

        if self.use_contrast_loss:
            train_num = (bs // 3) * 2 if self.training else bs
        else:
            train_num = bs

        trajectory={}
        probability={}
        prediction={}
        ref_free_trajectory={}
        targets_pos={}
        valid_mask={}
        targets_vel={}
        
        for i in range(7):  # 0 表示 cur, 1-6 表示 cur-1, cur-2, ..., cur-6
            cur_key = "cur" if i == 0 else f"cur-{i}"  # 处理 cur 和 cur-1 ~ cur-6
            
            trajectory[cur_key], probability[cur_key], prediction[cur_key] = (
                res[cur_key]["trajectory"][:train_num],  #(train_num, R, M, T, 6)  ([32, 7, 12, 80, 6]) 32个scenario 8个reference 12个mode 8s刚好为8/0.1=80步 
                res[cur_key]["probability"][:train_num],#(train_num, R, M) ([32, 7, 12])
                res[cur_key]["prediction"][:train_num], #(train_num, A-1, T, 6) ([32, 48, 80, 6])
            )
            
            # print(f" probility shape of {cur_key} is {probability[cur_key].shape}")
                #从data中提取目标数据
            targets_pos[cur_key] = data[cur_key]["agent"]["target"][:train_num] #(train_num, num_agents, T, 3)  Size([32, 49, 80, 3]) 最后一维度（x,y,theta)#总共有49个agent其中第0个为ego-car
            valid_mask[cur_key] = data[cur_key]["agent"]["valid_mask"][:train_num, :, -T:] #(train_num, num_agents, T,3) Size([32, 49, 80, 3])
            targets_vel[cur_key] = data[cur_key]["agent"]["velocity"][:train_num, :, -T:] #(train_num, num_agents, T,3) Size([32, 49, 80, 3])

        ref_free_trajectory["cur"] = res["cur"].get("ref_free_trajectory", None)

###############################################################################
        #为了减轻计算量，仅仅需要当前时cur时刻的taget算cls_matrix
        target = torch.cat(
            [
                targets_pos["cur"][..., :2],#x,y
                torch.stack(
                    [targets_pos["cur"][..., 2].cos(), targets_pos["cur"][..., 2].sin()], dim=-1
                ),
                targets_vel["cur"],
            ],
            dim=-1,
        )

#ego_reg_loss, ego_cls_loss, collision_loss, cls_loss_inter

        #data trajectory probability valid_mask均为字典， target[:, 0]仅仅包含当前时刻信息
        valid_mask_sliced = {key: value[:, 0] for key, value in valid_mask.items()}
        ego_reg_loss, ego_cls_loss, collision_loss, cls_loss_inter = self.get_planning_loss(
            data, trajectory, probability, valid_mask_sliced, target[:, 0], train_num
        )

# ego_ref_free_reg_loss
        if ref_free_trajectory["cur"] is not None:
            ego_ref_free_reg_loss = F.smooth_l1_loss(
                ref_free_trajectory["cur"][:train_num],
                target[:, 0, :, : ref_free_trajectory["cur"].shape[-1]],
                reduction="none",
            ).sum(-1)
            ego_ref_free_reg_loss = (
                ego_ref_free_reg_loss * valid_mask_sliced["cur"]
            ).sum() / valid_mask_sliced["cur"].sum()
        else:
            ego_ref_free_reg_loss = ego_reg_loss.new_zeros(1)



# prediction_loss
        prediction_loss = self.get_prediction_loss(
            data["cur"], prediction["cur"], valid_mask["cur"][:, 1:], target[:, 1:]
        )
        #valid_mask[:, 1:] 除开自车的掩码
        #prediction ：res["prediction"][:train_num], #(train_num, A-1, T, 2) 周车的预测信息
        
# contrastive_loss
        if self.training and self.use_contrast_loss:
            contrastive_loss = self._compute_contrastive_loss(
                res["cur"]["hidden"], data["cur"]["data_n_valid_mask"]
            )
        else:
            contrastive_loss = prediction_loss.new_zeros(1)
            
            
        
            
#####################################full loss############################################

        # loss = (
        #     ego_reg_loss
        #     + ego_cls_loss
        #     + prediction_loss
        #     + contrastive_loss
        #     + collision_loss
        #     + cls_loss_inter
        #     + ego_ref_free_reg_loss
        # )
        
#############################################################################################3
        
###############################3precise loss ###################################################

        loss = (
            ego_reg_loss*0.3
            + ego_cls_loss
            + prediction_loss* 0.01
            + contrastive_loss* 0.01
            + collision_loss* 0.01
            + cls_loss_inter
            + ego_ref_free_reg_loss* 0.01
        )

################################################################################
        
        
        # print(f"ego_reg_loss:{ego_reg_loss}")
        # print(f"ego_cls_loss:{ego_cls_loss}")
        # print(f"prediction_loss:{prediction_loss}")
        # print(f"contrastive_loss:{contrastive_loss}")
        # print(f"collision_loss:{collision_loss}")
        # print(f"cls_loss_inter:{cls_loss_inter}")
        # print(f"ego_ref_free_reg_loss:{ego_ref_free_reg_loss}")
        # print(f"loss:{loss}")

        # return {
        #     "loss": loss,
        #     "reg_loss": ego_reg_loss.item(),
        #     "cls_loss": ego_cls_loss.item(),
        #     "cls_loss_inter": cls_loss_inter.item(),
        #     "ref_free_reg_loss": ego_ref_free_reg_loss.item(),
        #     "collision_loss": collision_loss.item(),
        #     "prediction_loss": prediction_loss.item(),
        #     "contrastive_loss": contrastive_loss.item(),
        # }
        
        
        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.item()* 0.3,
            "cls_loss": ego_cls_loss.item(),
            "cls_loss_inter": cls_loss_inter.item(),
            "ref_free_reg_loss": ego_ref_free_reg_loss.item()* 0.01,
            "collision_loss": collision_loss.item()* 0.01,
            "prediction_loss": prediction_loss.item()* 0.01,
            "contrastive_loss": contrastive_loss.item()* 0.01,
        }

    def get_prediction_loss(self, data, prediction, valid_mask, target):
        """
        prediction: (bs, A-1, T, 6)
        valid_mask: (bs, A-1, T)
        target: (bs, A-1, 6)
        """

        prediction_loss = F.smooth_l1_loss(
            prediction[valid_mask], target[valid_mask], reduction="none"
        ).sum(-1)
        prediction_loss = prediction_loss.sum() / valid_mask.sum()

        return prediction_loss

    def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs):
        
        """
        
        trajectory: (bs, R, M, T, 4)R不一定为7
        valid_mask: (bs, T)
        
        
        #scenario_info是一个关于场景的字典
        # trajectory:(bs, R, M, Timestep, 6) : [32, 7, 12, 80, 6]
        # probablity: (train_num, R, M) : ([32, 7, 12])
        # valid_mask[:, 0](第二维只取自车) :(bs, T) Size([32, 80])
        # target[:, 0]  (第二维只取自车) : ([32, 80, 6])
        # data["reference_line"]["future_projection"]shape: Size([32, 7, 8, 2]) [bs R M feature_dim]

        """
        num_valid_points_cur = valid_mask["cur"].sum(-1) #(batch_size, Timestep) => (1, bachsize) 计算每个目标在所有时间步中有效的点数  tensor(1× bs)  [[1 1 0], [0 0 0], [1 0 0]]=>[2,0,1]
        endpoint_index_cur = (num_valid_points_cur / 10).long().clamp_(min=0, max=7)  # shape: ( bachsize，1)  max 8s #pyTorch 的 .long() 方法相当于向下取整（floor 操作）。tensor([2, 40, 80])=> tensor([0, 4, 7])

        r_padding_mask={}
        # VALID_MASK--》[bs R 120]
        #R_PADDING_MASK--> [bs R ]
        # Probability --> [bs R M]
        for i in range(7):  # 0 表示 cur, 1-6 表示 cur-1, cur-2, ..., cur-6
            cur_key = "cur" if i == 0 else f"cur-{i}"  # 处理 cur 和 cur-1 ~ cur-6
            r_padding_mask[cur_key] = ~data[cur_key]["reference_line"]["valid_mask"][:bs].any(-1)#对张量进行索引操作
        future_projection_cur = data["cur"]["reference_line"]["future_projection"][:bs][torch.arange(bs), :, endpoint_index_cur]
        
        target_r_index = torch.argmin(
            future_projection_cur[..., 1] + 1e6 * r_padding_mask["cur"], dim=-1
        ) #Size([32])
        
        target_m_index = (
            future_projection_cur[torch.arange(bs), target_r_index, 0] / self.mode_interval
        ).long()
        target_m_index.clamp_(min=0, max=self.num_modes - 1)

        target_label = torch.zeros_like(probability["cur"])   #[Bs R M]  (train_num, R, M) : ([32, 7, 12])
        target_label[torch.arange(bs), target_r_index, target_m_index] = 1

        best_trajectory = trajectory["cur"][torch.arange(bs), target_r_index, target_m_index] #torch.Size([32, 80, 6])

#计算碰撞损失
# data["cost_maps"] 返回一个张量。

        if self.use_collision_loss:
            collision_loss = self.collision_loss(
                best_trajectory, data["cur"]["cost_maps"][:bs, :, :, 0].float()
            )
        else:
            collision_loss = trajectory["cur"].new_zeros(1)
#回归损失
        reg_loss = F.smooth_l1_loss(best_trajectory, target, reduction="none").sum(-1)
        reg_loss = (reg_loss * valid_mask["cur"]).sum() / valid_mask["cur"].sum()

        
        
        
#分类损失？——cls_loss：轨迹的分类损失（选择最佳参考线和模式的误差）？原地操作
        probability["cur"].masked_fill_(r_padding_mask["cur"].unsqueeze(-1), -1e6)
        # VALID_MASK--》[bs R 120]
        #R_PADDING_MASK--> [bs R ] --》【bs R 1]
        # Probability --> [bs R M]
        #将probability["cur"]变成了--》[bs R M]
        #[f f f t t t t]
        # [f f t t t t t]
        # [f f f f f t t](bs=3 R=7)
        # # --》 【【num num num ....12个...num num】
        #        【 num num num ....12个...num num】
        #        【 num num num ....12个...num num】
        #        【 -1e6 -1e6 -1e6  ....12个...-1e6】
                    #    .....省略7个】】
                    # 。。。省略32个
                #    【 】
        
        # print(f"probability['cur'].shape is {probability['cur'].reshape(bs, -1).shape}")
        # print(f"target_label.reshape(bs, -1).detach().shape is {target_label.reshape(bs, -1).detach().shape}")
        

        cls_loss_orig = F.cross_entropy(probability["cur"].reshape(bs, -1), target_label.reshape(bs, -1).detach())
        probability_reshaped_cur=probability["cur"].reshape(bs, -1)
        cls_losses_inter = {}
        alpha = 0.8  # 设置递减速率
        for i in range(1, 7):
            cur_key = f"cur-{i}"
            # 使用 masked_fill 处理每个概率
            probability[cur_key].masked_fill_(r_padding_mask[cur_key].unsqueeze(-1), -1e6)
            
            # 重新调整概率和标签的形状
            probability_reshaped = probability[cur_key].reshape(bs, -1)
            # target_label_reshaped = target_label.reshape(bs, -1).detach()
            
            # 检查形状是否兼容
            if probability_reshaped.shape != probability_reshaped_cur.shape:
                cls_losses_inter[cur_key] = 2 * cls_loss_orig  # 使用默认损失
                print(f"Shapes are not compatible for cross_entropy in {cur_key}, using default cls_loss_orig loss")
            else:
                # 计算交叉熵损失
                cls_losses_inter[cur_key] = F.cross_entropy(probability_reshaped, probability_reshaped_cur)
                # print(f"Calculated cross_entropy loss for {cur_key}: {cls_losses_inter[cur_key]}")
                if abs(cls_losses_inter[cur_key]) > 32:
                    # print(f"异常值cls_losses_inter[{cur_key}] is {cls_losses_inter[cur_key]}")
                    # print(f"{cur_key}: probability shape {probability_reshaped.shape}, target shape {probability_reshaped_cur.shape}")
                    # print(f"{cur_key}: probability min {probability_reshaped.min()}, max {probability_reshaped.max()}")
                    # print(f"{cur_key}: target min {probability_reshaped_cur.min()}, max {probability_reshaped_cur.max()}")
                    cls_losses_inter[cur_key]=cls_loss_orig*1.5
                
                
            weight = 0.5 * (alpha ** (i - 1))  # 计算每个损失的权重
            cls_losses_inter[cur_key] *= weight  # 用加权损失替代原来的损失
        scaled_inter_weight=0.42
        cls_loss_inter = sum(cls_losses_inter.values())*scaled_inter_weight
        
#航偏角正则化损失
        if self.regulate_yaw:
            heading_vec_norm = torch.norm(best_trajectory[..., 2:4], dim=-1)
            yaw_regularization_loss = F.l1_loss(
                heading_vec_norm, heading_vec_norm.new_ones(heading_vec_norm.shape)
            )
            reg_loss += yaw_regularization_loss

        return reg_loss, cls_loss_orig, collision_loss,cls_loss_inter

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

        # get top 6 modes选择最好的6条 K=6
        trajectory, probability = res["cur"]["trajectory"], res["cur"]["probability"]
        r_padding_mask = ~data["cur"]["reference_line"]["valid_mask"].any(-1)
        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        bs, R, M, T, _ = trajectory.shape
        trajectory = trajectory.reshape(bs, R * M, T, -1)
        probability = probability.reshape(bs, R * M)
        top_k_prob, top_k_index = probability.topk(6, dim=-1)
        top_k_traj = trajectory[torch.arange(bs)[:, None], top_k_index]

        outputs = {
            "trajectory": top_k_traj[..., :2],
            "probability": top_k_prob,
            "prediction": res["cur"]["prediction"][..., :2],
            "prediction_target": data["cur"]["agent"]["target"][:, 1:],
            "valid_mask": data["cur"]["agent"]["valid_mask"][:, 1:, self.history_steps :],
        }
        target = data["cur"]["agent"]["target"][:, 0]

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
        speed_longi_plans=[]
        path_lateral_plans=[]
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
        speed_longi_plans=[]
        path_lateral_plans=[]
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


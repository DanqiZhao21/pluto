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
        # start_time = time.time()  # 记录开始时间
        # print(f"start_time is {start_time}")/
        features, targets, scenarios = batch
        skip_batch = True  # 标记是否要跳过整个 batch
        # import ipdb; ipdb.set_trace()  # 进入调试模式#TODO
        # #####################################################################
        # print(f"features type: {type(features)}")
        
        
        #新改过后的ipdb
        # ipdb> p features["feature"].keys()ipdb> p features["feature"].keys()
        # # *** AttributeError: 'PlutoFeature' object has no attribute 'keys'
        # ipdb> p type(features["feature"].data)ipdb> p type(features["feature"].data)
        # <class 'dict'>
        # ipdb> p features["feature"].data.keys()
        # dict_keys(['cur', 'cur-1', 'cur-2', 'cur-3', 'cur-4', 'cur-5', 'cur-6'])
        
        #修改之后
        # qipdb> p features["feature"].data['cur'].keys()ipdb> p features["feature"].data['cur'].keys()
        # dict_keys(['agent', 'map', 'data_n_valid_mask', 'data_n_type', 'current_state', 'origin', 'angle'])
        
        # p features["feature"].data["cur-5"].keys()ipdb> p features["feature"].data["cur-5"].keys()
        # dict_keys(['agent', 'map', 'data_n_valid_mask', 'data_n_type', 'current_state', 'origin', 'angle'])
        
        #为什么村的数据和最开始的data不太一样呢
        # features["feature"].data.kets()yys()ipdb> p features["feature"].data.keys()即data.keys()
        # dict_keys(['agent', 'map', 'reference_line', 'static_objects', 'current_state', 'origin', 'angle', 'cost_maps'])
        
        
        # ipdb> p features["feature"].data['cur-1'].keys()
        #（agent下面的内容也为字典）
        # p features["feature"].data['cur-3']['agent'].keys()ipdb> p features["feature"].data['cur-3']['agent'].keys()
        # dict_keys(['position', 'heading', 'velocity', 'shape', 'category', 'valid_mask', 'target'])
        
        
        # <class 'tuple'> batch
        # <class 'dict'> features
        #feature["feature"].data.keys()  (['agent', 'map', 'reference_line', 'static_objects', 'current_state', 'origin', 'angle', 'cost_maps'])

#******************************************************************************
        # current_times = [scenario.iteration.time_us for scenario in scenarios]
        # print(f"当前 batch 处理的时间步是: {current_times}")
        # with torch.autograd.detect_anomaly():
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
            
            
            
            
        #TODO
        #与res相关的全都化成了{}字典额形式，除了在外部有一个单独的Prediction
        #与data相关
        ref_free_trajectory["cur"] = res["cur"].get("ref_free_trajectory", None)
        
        
#从data中提取目标数据
        # targets_pos = data["agent"]["target"][:train_num] #(train_num, num_agents, T, 3)  Size([32, 49, 80, 3]) 最后一维度（x,y,theta)#总共有49个agent其中第0个为ego-car
        # valid_mask = data["agent"]["valid_mask"][:train_num, :, -T:] #(train_num, num_agents, T,3) Size([32, 49, 80, 3])
        # targets_vel = data["agent"]["velocity"][:train_num, :, -T:] #(train_num, num_agents, T,3) Size([32, 49, 80, 3])

        ###############################################################
        # import ipdb; ipdb.set_trace()  # 进入调试模式
        # print("Batch size (bs):", data["agent"]["target"][:train_num].shape)  # 打印第一个维度，即 batch size   torch.Size([32, 49, 80, 3])
        # print("Batch size (bs):", data["agent"]["valid_mask"][:train_num, :, -T:].shape)#torch.Size([32, 49, 101])->  ([32, 49, 80])
        # print("Batch size (bs):", data["agent"]["velocity"][:train_num, :, -T:].shape) #torch.Size([32, 49, 101, 2])没有-T:的时候->有T ([32, 49, 80, 2])
        ###########################################################################
        
        #语法解释：
        #通常在多目标预测任务中，targets_pos 是一个三维张量 [batch_size, time_steps, features]，features 的最后一维定义了每个时间步的特征。
            #torch.cat 用于将多个张量按指定维度 (dim) 拼接。
            #参数 dim=-1 表示沿着最后一个维度进行拼接（例如，如果张量是三维的 [B, T, F]，则在特征维度 F 上拼接）。
            #torch.stack 用于创建一个新的维度，将输入张量列表沿指定维度堆叠成一个新张量。
            #:2 是切片操作，选择最后一维的索引范围 [0, 1]
            #2 是索引，表示提取最后一个维度（通常是表示 x, y, θ 的维度）中的第 3 个元素
            
        # torch.stack 是 PyTorch 中的一个操作，用于将一组张量沿新的维度进行拼接。它与 
        # torch.cat 的主要区别在于，cat 是沿现有的维度进行拼接，
        # 而 stack 会在指定的新的维度上创建一个额外的维度。
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
    ##################################################################################
        # import ipdb; ipdb.set_trace()  # 进入调试模式
        # print("Batch size (bs):", valid_mask[:, 0].shape)  # torch.Size([32, 80])
        # print("Batch size (bs):", target[:, 0].shape)    # torch.Size([32, 80, 6])
    ######################################################################################################
        # valid_mask[:, 0].shape : torch.Size([32, 80])
        # target[:, 0].shape : torch.Size([32, 80, 6])
        #(train_num, num_agents, 6)  最后一个维度6： target [x,y,cos(theta),sin(theta),velocity_X,V_Y]
    #################################################################################################################       
        #ego_reg_loss: 回归损失，用于评估预测轨迹与目标轨迹之间的位置偏差。
        # : 分类损失，用于评估轨迹概率的准确性。
        # collision_loss: 碰撞损失，用于评估预测轨迹是否与障碍物发生碰撞。

        #切片与索引：
        # 切片和索引的语法
        # tensor[:, 0]：表示在所有批次（第一维）中，选择目标（【第二维】）索引为 0 的目标，
            # 保留所有时间步（第三维），其结果的形状是 (N, T, ...)，
            # 这里的 N 是批次大小，T 是时间步数，... 是其他维度的内容。
            # tensor 是 (4, 3, 5, 2)，那么 tensor[:, 0] 将返回 (4, 5, 2)
            #即：自车的信息呢！
        # tensor[..., 0]：表示在所有维度中，选择最后一维（即最后一列）的第 0 项。
            # ... 表示“保留”其他所有维度的内容。这通常用于提取某个特征，比如获取最后一维的某个特定元素。
            #tensor 是 (4, 3, 5, 2)-->所有时间步的第一个特征（例如，位置坐标）。这会返回形状为 (4, 3, 5) 
    ####################################################################################################
# planning loss【ego_reg_loss+ego_cls_loss+collision_loss+ cls_loss_inter】
        
        #data trajectory probability valid_mask均为字典， target[:, 0]仅仅包含当前时刻信息
        valid_mask_sliced = {key: value[:, 0] for key, value in valid_mask.items()}
        ego_reg_loss, ego_cls_loss, collision_loss, cls_loss_inter = self.get_planning_loss(
            data, trajectory, probability, valid_mask_sliced, target[:, 0], train_num
        )

        
        
        #scenario_info是一个关于场景的字典
        # trajectory:(bs, R, M, Timestep, 6) ： [32, 7, 12, 80, 6]
        # probablity: (train_num, R, M) :  ([32, 7, 12])
        # valid_mask[:, 0](第二维只取自车) ：(bs, T) Size([32, 80])
        # target[:, 0]  (第二维只取自车) ： ([32, 80, 6])
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
            
            
        
            

        loss = (
            ego_reg_loss
            + ego_cls_loss
            + prediction_loss
            + contrastive_loss
            + collision_loss
            + cls_loss_inter
            + ego_ref_free_reg_loss
        )
        
        
        
        print(f"ego_reg_loss:{ego_reg_loss}")
        print(f"ego_cls_loss:{ego_cls_loss}")
        print(f"prediction_loss:{prediction_loss}")
        print(f"contrastive_loss:{contrastive_loss}")
        print(f"collision_loss:{collision_loss}")
        print(f"cls_loss_inter:{cls_loss_inter}")
        print(f"ego_ref_free_reg_loss:{ego_ref_free_reg_loss}")
        print(f"loss:{loss}")

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.item(),
            "cls_loss": ego_cls_loss.item(),
            "cls_loss_inter": cls_loss_inter.item(),
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
        endpoint_index_cur = (num_valid_points_cur / 10).long().clamp_(min=0, max=7)  # shape: (1, bachsize)  max 8s #pyTorch 的 .long() 方法相当于向下取整（floor 操作）。tensor([2, 40, 80])=> tensor([0, 4, 7])
        #(80/10=8)
        #8s÷（0.1s)=80 time_step
        #这个valid应该有一定的时间因果性，0之后再无 1
        ####################################################################################################################
        #参考线处理
            #【1】 data["reference_line"]["valid_mask"] 的形状是 (bs ,R ,timestep)，
            #      valid_mask.any(-1) valid_any=(bs ,R )
            #【2】any(-1) 在最后一个维度（时间步 T）上检查是否 存在至少一个有效时间步，返回的是一个布尔张量，
            #     ~：逻辑取反
            #    [[[1, 0, 0], [0, 0, 0]],  # 样本 1：第一条参考线有效，第二条无效
            #    [[1, 1, 1], [0, 1, 0]]]  # 样本 2：两条参考线都有有效时间步
            #    则：
            #     [[ True, False],  # 样本 1：第一条参考线有效，第二条无效
            #     [ True,  True]]   # 样本 2：两条参考线均有效
            # 【3】data["reference_line"]["future_projection"][:bs]   :  (bs, R, T, F)
            # 返回一个形状为 (bs, num_reference_line（R), num_time_steps(T), feature_dim) 的张量。
            # 【B R T F】-->[B R F]  （endpoint_index使得T变为1，让第三个维度消失了）
            # 使用 torch.arange(bs) 和 endpoint_index 进行索引：
            # 从每个批次中选择所有参考线 (R)，
            # 从每个批次中选择一个 endpoint_index 对应的时间步（T）
        ################################################################################################################################
        r_padding_mask={}
        for i in range(7):  # 0 表示 cur, 1-6 表示 cur-1, cur-2, ..., cur-6
            cur_key = "cur" if i == 0 else f"cur-{i}"  # 处理 cur 和 cur-1 ~ cur-6
            r_padding_mask[cur_key] = ~data[cur_key]["reference_line"]["valid_mask"][:bs].any(-1)
            
            
        # r_padding_mask["cur"] = ~data["cur"]["reference_line"]["valid_mask"][:bs].any(-1)  #
        future_projection_cur = data["cur"]["reference_line"]["future_projection"][:bs][torch.arange(bs), :, endpoint_index_cur]
        # data["reference_line"]["valid_mask"]shape : torch.Size([32, 7, 120])   [bs R timestep]  所有timestep总结成一个false或者true
        # r_padding_mask shape : Size([32, 7])
        # data["reference_line"]["future_projection"]shape: Size([32, 7, 8, 2]) [bs R M feature_dim]
        # future_projection : Size([32, 7, 2])  [bs R feature_dim]
        #################################################################################################################################
        # import ipdb; ipdb.set_trace()  # 进入调试模式

        # print("Batch size (bs):",  data["reference_line"]["future_projection"].shape)  # Size([32, 7, 8, 2])
        # print("Batch size (bs):", future_projection.shape)    # ([32, 7, 2])
        ##############################################################################
        target_r_index = torch.argmin(
            future_projection_cur[..., 1] + 1e6 * r_padding_mask["cur"], dim=-1
        ) #Size([32])
        #地面真实轨迹的终点τ^gt相对于参考线进行投影，选择在横向距离上最接近的参考线作为目标参考线
        #######################################################################
        # future_projection[..., 1] 的结果--> (bs, R) 
        # 选择最后一个维度（即F)上的第二个特征作为最小值
        #######################################################################
        
        target_m_index = (
            future_projection_cur[torch.arange(bs), target_r_index, 0] / self.mode_interval
        ).long()
        #Size([32])
        # 该目标参考线随后被按距离分割成N_L-1等分，每个段落对应一个纵向查询所管理的区域，最后一个查询对应超出目标参考线的区域。
        # 包含投影终点的查询被指定为目标查询。通过结合目标参考线和目标纵向查询，我们得到了目标监督轨迹τ ̂。
        #######################################################################
        #未来投影点
        # 【B R F】-->[B 1 1]-->[B]
        # 对应的是target_refline的最后一个维度的第一个特征值
        # 通过除以 self.mode_interval 来将该值映射为离散的模式索引，最后 .long() 将其转换为整数索引。
        # target_m_index 是一个索引值的序列，而不仅仅是单个值
        ############################################################################
        target_m_index.clamp_(min=0, max=self.num_modes - 1)

        target_label = torch.zeros_like(probability["cur"])   #[Bs R M]  (train_num, R, M) : ([32, 7, 12])
        target_label[torch.arange(bs), target_r_index, target_m_index] = 1

        best_trajectory = trajectory["cur"][torch.arange(bs), target_r_index, target_m_index] #torch.Size([32, 80, 6])
        
        # trajectory: (bs, R, M, T, 6)
        # valid_mask: (bs, T)
        # best_trajectory：(bs, T, 6)
        
        #选择最佳参考线最佳模式下的那一种轨迹，包含剩下的两个维度：时间+特征
        # trajectory [Bs R M T feature_dim=4]
        # best_trajectory[Bs T fearture_dim=4]

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
        # target-->(bs, A-1, 6)
        # 此处 target=target[:, 0]-->[Bs  6](只有自车了)
        # best_trajectory[Bs T fearture_dim]
        # reduction="none" 表示返回每个元素的损失，而不是对所有元素的损失进行求和或平均
        
        
        
#分类损失？——cls_loss：轨迹的分类损失（选择最佳参考线和模式的误差）？
        #TODO
        #尝试调用api
        # 获取当前场景的图片和prompt（需从输入数据中提取）
        
        
        # scenario_image = data["scene_image"]      # 假设场景图片存储在 data 中
        # prompt = "Generate trajectory probability based on traffic rules and obstacles."
        
        # # 调用大模型接口生成 teacher_label
        # teacher_label = large_model.generate_teacher_matrix(
        #     image=scenario_image, 
        #     prompt=prompt, 
        #     bs=bs, R=R, M=M  # 传入维度参数
        # )
    
        
        
        
        
        #此处计算cls损失
        # cur时刻的probability与cur-1.....cur-6时刻的做损失分别获得cls_inter_loss_1-6
        #然后让大模型也给一个probability 分别于cur，cur-1,cur-2....cur-6时刻进行损失获得cls_teacher_loss0-6
        #############################################################
        probability["cur"].masked_fill_(r_padding_mask["cur"].unsqueeze(-1), -1e6)
        
        # print(f"probability['cur'].shape is {probability['cur'].reshape(bs, -1).shape}")
        # print(f"target_label.reshape(bs, -1).detach().shape is {target_label.reshape(bs, -1).detach().shape}")
        
        cls_loss_orig = F.cross_entropy(probability["cur"].reshape(bs, -1), target_label.reshape(bs, -1).detach())
        cls_losses_inter = {}
        alpha = 0.8  # 设置递减速率
        for i in range(1, 7):
            cur_key = f"cur-{i}"
            # 使用 masked_fill 处理每个概率
            probability[cur_key].masked_fill_(r_padding_mask[cur_key].unsqueeze(-1), -1e6)
            
            # 重新调整概率和标签的形状
            probability_reshaped = probability[cur_key].reshape(bs, -1)
            target_label_reshaped = target_label.reshape(bs, -1).detach()
            
            # 检查形状是否兼容
            if probability_reshaped.shape != target_label_reshaped.shape:
                cls_losses_inter[cur_key] = 2 * cls_loss_orig  # 使用默认损失
                print(f"Shapes are not compatible for cross_entropy in {cur_key}, using default cls_loss_orig loss")
            else:
                # 计算交叉熵损失
                cls_losses_inter[cur_key] = F.cross_entropy(probability_reshaped, target_label_reshaped)
                # print(f"Calculated cross_entropy loss for {cur_key}: {cls_losses_inter[cur_key]}")
                if cls_losses_inter[cur_key] > 100:
                    print(f"异常值cls_losses_inter[{cur_key}] is {cls_losses_inter[cur_key]}")
                    print(f"{cur_key}: probability shape {probability_reshaped.shape}, target shape {target_label_reshaped.shape}")
                    print(f"{cur_key}: probability min {probability_reshaped.min()}, max {probability_reshaped.max()}")
                    print(f"{cur_key}: target min {target_label_reshaped.min()}, max {target_label_reshaped.max()}")

                
                
            weight = 0.5 * (alpha ** (i - 1))  # 计算每个损失的权重
            cls_losses_inter[cur_key] *= weight  # 用加权损失替代原来的损失
    
        cls_loss_inter = sum(cls_losses_inter.values())

        # if cls_loss_inter > 100:
        #     for i in range(1,7):
        #         cur_key = f"cur-{i}"
        #         print(f"cls_losses_inter[{cur_key}] is {cls_losses_inter[cur_key]}")
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # probability["cur-1"].masked_fill_(r_padding_mask["cur-1"].unsqueeze(-1), -1e6)
        # # print(f"probability['cur-1'].shape is {probability['cur-1'].reshape(bs, -1).shape}")
        # # print(f"target_label.reshape(bs, -1).detach().shape is {target_label.reshape(bs, -1).detach().shape}")
        
        # probability_reshaped = probability["cur-1"].reshape(bs, -1)
        # target_label_reshaped = target_label.reshape(bs, -1).detach()
        # if probability_reshaped.shape != target_label_reshaped.shape:
        #     cls_inter_loss1 = 2 * cls_loss_orig  # 使用默认损失
        #     print(f"Shapes are not compatible for cross_entropy, using default loss: {cls_inter_loss1}")
        # else:
        #     cls_inter_loss1 = F.cross_entropy(probability_reshaped, target_label_reshaped)
        #     # print(f"Calculated cross_entropy loss: {cls_inter_loss1}"
        # # cls_inter_loss1=F.cross_entropy(probability["cur-1"].reshape(bs, -1), target_label.reshape(bs, -1).detach())
        
        # probability["cur-2"].masked_fill_(r_padding_mask["cur-2"].unsqueeze(-1), -1e6)
        # print(f"probability['cur-2'].shape is {probability['cur-2'].reshape(bs, -1).shape}")
        # print(f"target_label.reshape(bs, -1).detach().shape is {target_label.reshape(bs, -1).detach().shape}")
        # cls_inter_loss2=F.cross_entropy(probability["cur-2"].reshape(bs, -1), target_label.reshape(bs, -1).detach())
        
        # probability["cur-3"].masked_fill_(r_padding_mask["cur-3"].unsqueeze(-1), -1e6)
        # print(f"probability['cur-3'].shape is {probability['cur-3'].reshape(bs, -1).shape}")
        # print(f"target_label.reshape(bs, -1).detach().shape is {target_label.reshape(bs, -1).detach().shape}")
        # cls_inter_loss3=F.cross_entropy(probability["cur-3"].reshape(bs, -1), target_label.reshape(bs, -1).detach())
        
        # probability["cur-4"].masked_fill_(r_padding_mask["cur-4"].unsqueeze(-1), -1e6)
        # print(f"probability['cur-4'].shape is {probability['cur-4'].reshape(bs, -1).shape}")
        # print(f"target_label.reshape(bs, -1).detach().shape is {target_label.reshape(bs, -1).detach().shape}")
        # cls_inter_loss4=F.cross_entropy(probability["cur-4"].reshape(bs, -1), target_label.reshape(bs, -1).detach())
        
        # probability["cur-5"].masked_fill_(r_padding_mask["cur-5"].unsqueeze(-1), -1e6)
        # print(f"probability['cur-5'].shape is {probability['cur-5'].reshape(bs, -1).shape}")
        # print(f"target_label.reshape(bs, -1).detach().shape is {target_label.reshape(bs, -1).detach().shape}")
        # cls_inter_loss5=F.cross_entropy(probability["cur-5"].reshape(bs, -1), target_label.reshape(bs, -1).detach())
        
        # probability["cur-6"].masked_fill_(r_padding_mask["cur-6"].unsqueeze(-1), -1e6)
        # print(f"probability['cur-6'].shape is {probability['cur-6'].reshape(bs, -1).shape}")
        # print(f"target_label.reshape(bs, -1).detach().shape is {target_label.reshape(bs, -1).detach().shape}")
        # cls_inter_loss6=F.cross_entropy(probability["cur-6"].reshape(bs, -1), target_label.reshape(bs, -1).detach())
        
        # cls_loss_inter=cls_inter_loss1+cls_inter_loss2+cls_inter_loss3+cls_inter_loss4+cls_inter_loss5+cls_inter_loss6
        
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
# res：包含模型的预测结果，通常包括轨迹预测和概率分布（如 trajectory 和 probability）。
# data：真实的输入数据和标注（ground truth），用于计算与预测结果的差异。
# prefix：一个字符串，用于标识评价指标所属的任务或阶段，可能用于区分训练和验证过程的指标。

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
        features, targets, scenarios = batch
        len_scenario=len(scenarios)
        # print(f"此次训练总共使用了 {len_scenario} 个scenario")
        
        # len_scenario=len(scenarios)
        scenarios_info = []
        nuplan_scenarios = []
        speed_longi_plans = []
        path_lateral_plans=[]

        # #TODO
        # #*************************************************************************
        # # 存储所有 scenario 信息的列表
        
        # # 遍历 scenarios 列表，提取每个 scenario 的信息
        
        # NUPLAN_DATA_ROOT11 = "/mnt/data/nuplan/dataset"
        # NUPLAN_MAP_VERSION11 = "nuplan-maps-v1.0"
        # NUPLAN_MAPS_ROOT11 = "/mnt/data/nuplan/dataset/maps"
        # NUPLAN_SENSOR_ROOT11 = f"{NUPLAN_DATA_ROOT11}/nuplan-v1.1/sensor_blobs"
        # TEST_DB_FILE = f"{NUPLAN_DATA_ROOT11}/nuplan-v1.1/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"
        # MAP_NAME = "us-nv-las-vegas"
        # # TEST_INITIAL_LIDAR_PC = "58ccd3df9eab54a3"
        # # TEST_INITIAL_TIMESTAMP = 1620858198150622
        # os.environ["NUPLAN_DATA_STORE"] = "None"  # 禁用远程存储
        
        # data_root = "/mnt/data/nuplan/dataset"  # 这里使用你的数据根路径
        # map_root = "/mnt/data/nuplan/dataset/maps"   # 地图文件路径
        # sensor_root = "/mnt/data/nuplan/dataset/nuplan-v1.1/splits/mini"  # 传感器数据路径(sensor_blob和mini里面的一样)
        
        # for scenario in scenarios:
        #     scenario_data = {
        #         "initial_LidarPc_token": scenario.token, # 获取 该scenario 起始的lidarPc的信息  "58ccd3df9eab54a3"
        #         "log_name": scenario.log_name,          # 获取 log 文件名称2021.05.12.22.00.38_veh-35_01008_01518
        #         "scenario_type": scenario.scenario_type # 获取 scenario 类型
        #     }
            
        #     db_files=f"/mnt/data/nuplan/dataset/nuplan-v1.1/splits/mini/{scenario_data['log_name']}.db"
        #     nuplan_db = NuPlanDB(
        #         data_root=data_root,
        #         maps_db=map_root,
        #         load_path=db_files,
        #         verbose=True
        #         )
            
        #     lidar_pc_map = {record.token: record for record in nuplan_db.lidar_pc}#由lidar_pc_token获得Lidar_pc对象
        #     ego_pose_map = {record.token: record for record in nuplan_db.ego_pose}#由ego_pose_token获得ego_pose对象
        #     lidar_pc_obj=lidar_pc_map.get(scenario_data["initial_LidarPc_token"])
        #     Initial_lidar_timestamp=lidar_pc_obj.timestamp
            
        #     scenarios_info.append(scenario_data)  # 存入列表
        # #***************************************************************************
        #     # 创建 NuPlanScenario 实例
        #     nuplan_scenario = NuPlanScenario(
        #         data_root=f"{NUPLAN_DATA_ROOT11}/nuplan-v1.1/splits/mini",  # 数据根目录
        #         log_file_load_path=  db_files,  # 测试数据库文件路径
        #         initial_lidar_token=scenario_data["initial_LidarPc_token"],  # 获取起始 Lidar PC token
        #         initial_lidar_timestamp=Initial_lidar_timestamp,  # 起始时间戳
        #         scenario_type=scenario_data["scenario_type"],  # scenario 类型
        #         map_root=NUPLAN_MAPS_ROOT11,  # 地图数据根目录
        #         map_version=NUPLAN_MAP_VERSION11,  # 地图版本
        #         map_name=MAP_NAME,  # 地图名称
        #         scenario_extraction_info=ScenarioExtractionInfo(
        #             scenario_name="scenario_name", scenario_duration=15, extraction_offset=1, subsample_ratio=0.5  # 采样信息
        #         ),
        #         ego_vehicle_parameters=get_pacifica_parameters(),  # 自车的车辆参数
        #         sensor_root=f"{NUPLAN_DATA_ROOT11}/nuplan-v1.1/sensor_blobs",  # 传感器数据路径
        #     )
        #     # 将构建的 NuPlanScenario 实例添加到列表中
        #     nuplan_scenarios.append(nuplan_scenario)
        #     iteration=20
        #     lidar_pc_obj = nuplan_scenario.get_sensors_at_iteration_edit(iteration, channels=None)  # 获取传感器数据
        #     egoPose_token=lidar_pc_obj.ego_pose_token
            
            
        #     speed_longi_plans=[]
        #     path_lateral_plans=[]

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
        return self._step(batch, "val", speed_longi_plans,path_lateral_plans)

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
        return self._step(batch, "test", speed_longi_plans,path_lateral_plans)

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
    :param speed_obj: 纵向动作 ["keep", "accelerate", "decelerate","STOP"咋办？]
    :param path_obj: 横向动作 ["straight", "right_change", "left_change", "right_turn", "left_turn"]
    :param target_r_index: 目标时间索引 (bs,)
    :param target_m_index: 目标车道索引 (bs,)
    
    :return: teacher_label (bs, num_timesteps, num_lanes)
    """
    # 初始化 teacher_label
    teacher_label = torch.ones(Bs, R, M) * 10  # 默认填充为 10

    # 处理纵向行为 (speed_obj)
    for i in range(Bs):
        if speed_obj[i] == "KEEP":
            continue
        elif speed_obj[i] == "ACCELERATE":
            teacher_label[i, :, :M//3] = 30       # 赋值给 0 ~ M/3 这一段
            teacher_label[i, :, M//3:2*M//3] = 20 # 赋值给 M/3 ~ 2M/3 这一段
            teacher_label[i, :, 2*M//3:] = 10     # 赋值给 2M/3 ~ M 这一段
        elif speed_obj[i] == "DECELERATE":
            teacher_label[i, :, :M//3] =10       # 赋值给 0 ~ M/3 这一段
            teacher_label[i, :, M//3:2*M//3] = 20 # 赋值给 M/3 ~ 2M/3 这一段
            teacher_label[i, :, 2*M//3:] = 30     # 赋值给 2M/3 ~ M 这一段

    #先不处理横向行为，横向行为已经很准确了
    # # 处理横向行为 (path_obj)
    # for i in range(Bs):
    #     lane = target_m_index[i]  # 当前位置车道索引

    #     if path_obj[i] == "STRAIGHT":
    #         teacher_label[i, :, lane] = 1  # 当前位置设为 1
    #     elif path_obj[i] == "right_change" and lane < num_lanes - 1:
    #         teacher_label[i, :, lane] = 5
    #         teacher_label[i, :, lane + 1] = 1
    #     elif path_obj[i] == "left_change" and lane > 0:
    #         teacher_label[i, :, lane] = 5
    #         teacher_label[i, :, lane - 1] = 1
    #     elif path_obj[i] == "RIGHT_TURN" and lane < num_lanes - 1:
    #         teacher_label[i, :, lane + 1] = 1
    #     elif path_obj[i] == "left_turn" and lane > 0:
    #         teacher_label[i, :, lane - 1] = 1
    return teacher_label
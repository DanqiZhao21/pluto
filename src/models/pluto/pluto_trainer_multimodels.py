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
        #TODO
        model: TorchModuleWrapper,
        model_class,  # 传入一个模型类，而不是单个模型实例
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        use_collision_loss=True,
        use_contrast_loss=False,
        regulate_yaw=False,
        objective_aggregate_mode: str = "mean",
        num_models=6,  # 6 个模型 6帧的信息
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

        # self.model = model
        #TODO
        self.num_models = num_models 
        self.models = nn.ModuleList([model_class() for _ in range(num_models)])
        
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
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str,speed_longi_plan,path_lateral_plan) -> torch.Tensor:
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
        
        # import ipdb; ipdb.set_trace()
        # # print(f" batch in _step() is { batch}")
        
        # # print(f" scenarios in _step() is { scenarios}")
        # # print(f" type of scenarios  is { type(scenarios)}")
        # # print(f" batch in _step() is { batch}")\
        
        start_time = time.time()  # 记录开始时间
        print(f"start_time is {start_time}")
        
        
        
        global MM  # 声明使用全局变量 MM
        MM+=1
        features, targets, scenarios = batch
        
        scenario_data=[]
        for scenario in scenarios:
            scenario_data = {
                "initial_LidarPc_token": scenario.token, # 获取 该scenario 起始的lidarPc的信息  "58ccd3df9eab54a3"
                "log_name": scenario.log_name,          # 获取 log 文件名称2021.05.12.22.00.38_veh-35_01008_01518
                "scenario_type": scenario.scenario_type # 获取 scenario 类型
            }
            print(f"scenario_data 起始的lidarPc is {scenario_data['initial_LidarPc_token']}")
            print(f"scenario_data 起始的log_name is {scenario_data['log_name']}")   
        
        print(f"MM = {MM}")
        # # len_scenario=len(scenarios)
        scenarios_info = []
        nuplan_scenarios = []
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
        
        
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算时间差
        print(f"创建 NuPlanScenario 实例运行时间: {elapsed_time:.4f} 秒")
        
        # # import ipdb ; ipdb.set_trace()
        # iterations_per_scenario = [scenario.get_number_of_iterations() for scenario in nuplan_scenarios]
        #         # 打印结果
        # iteration=0
        # # for idx, iterationsPerScenario in enumerate(iterations_per_scenario):
        #     # print(f"Scenario {idx + 1} has {iterationsPerScenario} iterations.")
        #     # Scenario 1 has 150 iterations.
        #     # Scenario 2 has 151 iterations.
        #     # Scenario 19 has 150 iterations.
        #     # ......
        # for Scenario in nuplan_scenarios:
        #     sensors = Scenario.get_sensors_at_iteration_edit(iteration, channels=None)  # 获取传感器数据
        #     lidar_pc_obj = sensors
        #     egoPose_token=lidar_pc_obj.ego_pose_token
        #     logi_meta=[]#根据egoPose_token获取logi_meta
        #     lateral_meta=[]#应该是1×32的tensor
        
    
        
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算时间差
        print(f"获得lidar_pc_token信息 运行时间: {elapsed_time:.4f} 秒")

#******************************************************************************
        # current_times = [scenario.iteration.time_us for scenario in scenarios]
        # print(f"当前 batch 处理的时间步是: {current_times}")

        
        res = self.forward(features["feature"].data)#轨迹预测的结果预测的，是只有80步长的呢！
        # print("Shape of res:", res.shape)  # 打印整个形状AttributeError: 'dict' object has no attribute 'shape'
        
        # import ipdb; ipdb.set_trace()
        # # print(f"Shape of features[feature].data: is {features["feature"].data}")
        # print(f"Shape of features[feature].data: is {features['feature'].data}")
        
        # p res.keys()ipdb> p res.keys()
        #     dict_keys(['trajectory', 'probability', 'prediction', 'output_prediction', 'output_trajectory', 'candidate_trajectories'])
    #   p features["feature"].data.kets()yys()ipdb> p features["feature"].data.keys()
    #     dict_keys(['agent', 'map', 'reference_line', 'static_objects', 'current_state', 'origin', 'angle', 'cost_maps'])
        
        
        losses = self._compute_objectives(res, features["feature"].data,speed_longi_plan,path_lateral_plan)#预测结果和真值之间计算Loss  这个feature里面应该也只有每一帧的信息，不然应该会多一个维度150
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)
        
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算时间差
        print(f"总step 运行时间: {elapsed_time:.4f} 秒")
        
        
        
        return losses["loss"] if self.training else 0.0
    
    
    
            
        
        
        #dict dict list
        # features keys: dict_keys(['feature'])
        # targets keys: dict_keys(['trajectory'])
        #scenario list
        #len(scenario 32)
        # import ipdb; ipdb.set_trace()  # 进入调试模式#TODO
        # #####################################################################
        # print(f"features type: {type(features)}")
        # print(f"targets type: {type(targets)}")
        # print(f"scenarios type: {type(scenarios)}")
        # print(f"scenarios length: {len(scenarios)}")
        # if len(scenarios) > 0:
        #     print(f"First scenario: {scenarios[0]}")
        #     # First scenario: <nuplan.planning.scenario_builder.cache.cached_scenario.CachedScenario object at 0x7b5a913ebcd0>
        #     print(f"First scenario type: {type(scenarios[0])}")
        #     # First scenario type: <class 'nuplan.planning.scenario_builder.cache.cached_scenario.CachedScenario'>n
        #     print(f"First scenario (other type): {scenarios[0]}")
        #     #First scenario (other type): <nuplan.planning.scenario_builder.cache.cached_scenario.CachedScenario object at 0x750e37c79040>
        
        
        
        #     print(scenarios[0].token)         # 获取 token:eb9eab4d9a8a537c
        #     print(scenarios[0].log_name)      # 获取 log_name:2021.06.08.16.31.33_veh-38_01589_02072
        #     print(scenarios[0].scenario_type) # 获取 scenario_type:stationary

        # #################################################################################
        # # 如果是字典，可以打印 keys
        # if isinstance(features, dict):
        #     print(f"features keys: {features.keys()}")
        #     #features keys: dict_keys(['feature'])
        #     print(f"Type of features['feature']: {type(features['feature'])}")
        #     # Type of features['feature']: <class 'src.features.pluto_feature.PlutoFeature'>
        #     print(dir(features['feature'])) 
        #     #['__abstractmethods__', '__annotations__', '__class__', '__dataclass_fields__', '__dataclass_params__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', 
        #     # '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl', 'collate', 'data', 'data_n', 
        #     # 'data_n_info', 'data_p', 'deserialize', 'is_valid', 'normalize', 'serialize', 'to_device', 'to_feature_tensor', 'to_numpy', 'unpack']
        #     print(features['feature']) 
        #     #'future_projection'
        #     # 'static_objects': 
        #     #  'heading': tensor
        #     # 'shape': tensor
        #     # 'category': tensor
        #     # 'valid_mask': tensor
        #     # 'current_state':
        #     #'origin': tensor
        #     # 'angle': tensor(
        #     # 'cost_maps': tensor
        #     #错误的
        #     # print(f"Shape of features['feature']: {features['feature'].shape}")
        #     #AttributeError: 'PlutoFeature' object has no attribute 'shape'
        #     ## print(f"First few elements of features['feature']: {features['feature'][:5]}")
        #     # TypeError: 'PlutoFeature' object is not subscriptable
            
        # if isinstance(targets, dict):
        #     print(f"targets keys: {targets.keys()}")
        #     #targets keys: dict_keys(['trajectory'])
        #     print(f"Type of targets['trajectory']: {type(targets['trajectory'])}")
        #     #Type of targets['trajectory']: <class 'nuplan.planning.training.preprocessing.features.trajectory.Trajectory'>
        #     print(targets['trajectory']) 
        #     #Trajectory(data=tensor([[[
        # if isinstance(scenarios, dict):
        #     print(f"scenarios keys: {scenarios.keys()}")
######################################################################################################################
    
    

    def _compute_objectives(self, res, data,logi_meta,lateral_meta) -> Dict[str, torch.Tensor]:
        '''
        32个scenario 8个reference 12个mode 8s刚好为8/0.1=80步
        trajectory: ([32, 7, 12, 80, 6])  
        probability: ([32, 7, 12])
        prediction: ([32, 48, 80, 6])
        
        res.keys()ipdb> p res.keys()
        dict_keys(['trajectory', 'probability', 'prediction', 'output_prediction', 'output_trajectory', 'candidate_trajectories'])
        
        features["feature"].data.kets()yys()ipdb> p features["feature"].data.keys()即data.keys()
        dict_keys(['agent', 'map', 'reference_line', 'static_objects', 'current_state', 'origin', 'angle', 'cost_maps'])

        '''
        #res: 包含模型的预测输出，可能包括轨迹预测、概率预测
        #data: 包含输入数据和目标标签的信息，比如目标位置、速度、掩码等。

        
        
        bs, _, T, _ = res["prediction"].shape  #torch.Size([32, 48, 80, 6])

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
        # 6：轨迹信息
        ########################################################################
        # import ipdb; ipdb.set_trace()  # 进入调试模式
        # print("Batch size (bs):", res["trajectory"][:train_num].shape)  # 打印第一个维度，即 batch size
        # print("Batch size (bs):", res["probability"][:train_num].shape)
        # print("Batch size (bs):", res["prediction"][:train_num].shape)
        ###########################################################################

        trajectory, probability, prediction = (
            res["trajectory"][:train_num],  #(train_num, R, M, T, 6)  ([32, 7, 12, 80, 6]) 32个scenario 8个reference 12个mode 8s刚好为8/0.1=80步 
            res["probability"][:train_num],#(train_num, R, M) ([32, 7, 12])
            res["prediction"][:train_num], #(train_num, A-1, T, 6) ([32, 48, 80, 6])
        )

        ref_free_trajectory = res.get("ref_free_trajectory", None)
        
        
#从data中提取目标数据
        targets_pos = data["agent"]["target"][:train_num] #(train_num, num_agents, T, 3)  Size([32, 49, 80, 3]) 最后一维度（x,y,theta)
        valid_mask = data["agent"]["valid_mask"][:train_num, :, -T:] #(train_num, num_agents, T,3) Size([32, 49, 80, 3])
        targets_vel = data["agent"]["velocity"][:train_num, :, -T:] #(train_num, num_agents, T,3) Size([32, 49, 80, 3])

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
        # planning loss
        ego_reg_loss, ego_cls_loss, collision_loss = self.get_planning_loss(
            data, trajectory, probability, valid_mask[:, 0], target[:, 0], train_num, logi_meta,lateral_meta
        )
        #scenario_info是一个关于场景的字典
        # trajectory:(bs, R, M, Timestep, 6) ： [32, 7, 12, 80, 6]
        # probablity: (train_num, R, M) :  ([32, 7, 12])
        # valid_mask[:, 0](第二维只取自车) ：(bs, T) Size([32, 80])
        # target[:, 0]  (第二维只取自车) ： ([32, 80, 6])
        
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

    def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs,logi_meta,lateral_meta):
        
        """
        #scenario_info是一个关于场景的字典
        # trajectory:(bs, R, M, Timestep, 6) : [32, 7, 12, 80, 6]
        # probablity: (train_num, R, M) : ([32, 7, 12])
        # valid_mask[:, 0](第二维只取自车) :(bs, T) Size([32, 80])
        # target[:, 0]  (第二维只取自车) : ([32, 80, 6])
        # data["reference_line"]["future_projection"]shape: Size([32, 7, 8, 2]) [bs R M feature_dim]

        """
        num_valid_points = valid_mask.sum(-1) #(batch_size, Timestep) => (1, bachsize) 计算每个目标在所有时间步中有效的点数  tensor(1× bs)  [[1 1 0], [0 0 0], [1 0 0]]=>[2,0,1]
        endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # shape: (1, bachsize)  max 8s #pyTorch 的 .long() 方法相当于向下取整（floor 操作）。tensor([2, 40, 80])=> tensor([0, 4, 7])
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
        r_padding_mask = ~data["reference_line"]["valid_mask"][:bs].any(-1)  #
        future_projection = data["reference_line"]["future_projection"][:bs][
            torch.arange(bs), :, endpoint_index
        ]
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
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
        ) #Size([32])
        #地面真实轨迹的终点τ^gt相对于参考线进行投影，选择在横向距离上最接近的参考线作为目标参考线
        #######################################################################
        # future_projection[..., 1] 的结果--> (bs, R) 
        # 选择最后一个维度（即F)上的第二个特征作为最小值
        #######################################################################
        
        target_m_index = (
            future_projection[torch.arange(bs), target_r_index, 0] / self.mode_interval
        ).long()#Size([32])
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

        target_label = torch.zeros_like(probability)   #[Bs R M]  (train_num, R, M) : ([32, 7, 12])
        target_label[torch.arange(bs), target_r_index, target_m_index] = 1

        best_trajectory = trajectory[torch.arange(bs), target_r_index, target_m_index] #torch.Size([32, 80, 6])
        
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
        
        
        
#分类损失？——cls_loss：轨迹的分类损失（选择最佳参考线和模式的误差）？
        #TODO
        #**************************************************************************
        # speed_obj=[]#keep accelerate decelerate
        # path_obj=[]#straight right_change left_change right_turn left_turn
        # Bs, R, M = probability.shape
        # # teacher_label = torch.zeros_like(probability)#【Bs R M]
        # lane_cur=0
        # teacher_label=create_teacher_label(Bs, R, M, speed_obj, path_obj,lane_cur)
        # #应该结合一下真值信息，在真值预测信息？的基础上再加入元动作及逆行全局矩阵的编写
        #**************************************************************************

        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)
        cls_loss = F.cross_entropy(
            probability.reshape(bs, -1), target_label.reshape(bs, -1).detach()
        )
        
        # probability:[Bs R M]
        # target_label"[Bs R M]  独热向量，只有最佳参考线最佳轨迹下为1 ，其余都为0
        #unsqueeze(-1)在dim=-1即最后一个维度增加一个新的维度
        #################################################################################
        # probability = torch.tensor([
        #     [[2.1, 1.3, -0.5, 0.8],  # 样本 1, 参考线 1
        #     [1.2, 3.5, 0.6, -1.0],  # 样本 1, 参考线 2
        #     [0.5, -0.8, 1.4, 2.1]], # 样本 1, 参考线 3
        
        #     [[0.2, 1.8, -0.1, 2.5],  # 样本 2, 参考线 1
        #     [-0.3, 2.1, 1.7, 0.4],  # 样本 2, 参考线 2
        #     [1.2, 0.3, 1.5, -0.6]]  # 样本 2, 参考线 3
        # ])

        # # r_padding_mask: True 表示该参考线无效，需要屏蔽
        # r_padding_mask = torch.tensor([
        #     [False, True, False],  # 样本 1: 第 2 条参考线无效
        #     [True, False, False]   # 样本 2: 第 1 条参考线无效
        # ])
        
        # tensor
        # ([[[ 2.1000e+00,  1.3000e+00, -5.0000e-01,  8.0000e-01],  # 样本 1, 参考线 1（保留）
        #  [-1.0000e+06, -1.0000e+06, -1.0000e+06, -1.0000e+06],  # 样本 1, 参考线 2（屏蔽）
        #  [ 5.0000e-01, -8.0000e-01,  1.4000e+00,  2.1000e+00]], # 样本 1, 参考线 3（保留）

        # [[-1.0000e+06, -1.0000e+06, -1.0000e+06, -1.0000e+06],  # 样本 2, 参考线 1（屏蔽）
        #  [-3.0000e-01,  2.1000e+00,  1.7000e+00,  4.0000e-01],  # 样本 2, 参考线 2（保留）
        #  [ 1.2000e+00,  3.0000e-01,  1.5000e+00, -6.0000e-01]]]) # 样本 2, 参考线 3（保留）

        ################################################################################
        
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
        features, targets, scenarios = batch
        len_scenario=len(scenarios)
        print(f"此次训练总共使用了 {len_scenario} 个scenario")
        # len_scenario=len(scenarios)
        scenarios_info = []
        nuplan_scenarios = []
        speed_longi_plans = []
        path_lateral_plans=[]

        #TODO
        #*************************************************************************
        # 存储所有 scenario 信息的列表
        
        # 遍历 scenarios 列表，提取每个 scenario 的信息
        
        NUPLAN_DATA_ROOT11 = "/mnt/data/nuplan/dataset"
        NUPLAN_MAP_VERSION11 = "nuplan-maps-v1.0"
        NUPLAN_MAPS_ROOT11 = "/mnt/data/nuplan/dataset/maps"
        NUPLAN_SENSOR_ROOT11 = f"{NUPLAN_DATA_ROOT11}/nuplan-v1.1/sensor_blobs"
        TEST_DB_FILE = f"{NUPLAN_DATA_ROOT11}/nuplan-v1.1/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"
        MAP_NAME = "us-nv-las-vegas"
        # TEST_INITIAL_LIDAR_PC = "58ccd3df9eab54a3"
        # TEST_INITIAL_TIMESTAMP = 1620858198150622
        os.environ["NUPLAN_DATA_STORE"] = "None"  # 禁用远程存储
        
        data_root = "/mnt/data/nuplan/dataset"  # 这里使用你的数据根路径
        map_root = "/mnt/data/nuplan/dataset/maps"   # 地图文件路径
        sensor_root = "/mnt/data/nuplan/dataset/nuplan-v1.1/splits/mini"  # 传感器数据路径(sensor_blob和mini里面的一样)
        
        for scenario in scenarios:
            scenario_data = {
                "initial_LidarPc_token": scenario.token, # 获取 该scenario 起始的lidarPc的信息  "58ccd3df9eab54a3"
                "log_name": scenario.log_name,          # 获取 log 文件名称2021.05.12.22.00.38_veh-35_01008_01518
                "scenario_type": scenario.scenario_type # 获取 scenario 类型
            }
            
            db_files=f"/mnt/data/nuplan/dataset/nuplan-v1.1/splits/mini/{scenario_data['log_name']}.db"
            nuplan_db = NuPlanDB(
                data_root=data_root,
                maps_db=map_root,
                load_path=db_files,
                verbose=True
                )
            
            lidar_pc_map = {record.token: record for record in nuplan_db.lidar_pc}#由lidar_pc_token获得Lidar_pc对象
            ego_pose_map = {record.token: record for record in nuplan_db.ego_pose}#由ego_pose_token获得ego_pose对象
            lidar_pc_obj=lidar_pc_map.get(scenario_data["initial_LidarPc_token"])
            Initial_lidar_timestamp=lidar_pc_obj.timestamp
            
            scenarios_info.append(scenario_data)  # 存入列表
        #***************************************************************************
            # 创建 NuPlanScenario 实例
            nuplan_scenario = NuPlanScenario(
                data_root=f"{NUPLAN_DATA_ROOT11}/nuplan-v1.1/splits/mini",  # 数据根目录
                log_file_load_path=  db_files,  # 测试数据库文件路径
                initial_lidar_token=scenario_data["initial_LidarPc_token"],  # 获取起始 Lidar PC token
                initial_lidar_timestamp=Initial_lidar_timestamp,  # 起始时间戳
                scenario_type=scenario_data["scenario_type"],  # scenario 类型
                map_root=NUPLAN_MAPS_ROOT11,  # 地图数据根目录
                map_version=NUPLAN_MAP_VERSION11,  # 地图版本
                map_name=MAP_NAME,  # 地图名称
                scenario_extraction_info=ScenarioExtractionInfo(
                    scenario_name="scenario_name", scenario_duration=15, extraction_offset=1, subsample_ratio=0.5  # 采样信息
                ),
                ego_vehicle_parameters=get_pacifica_parameters(),  # 自车的车辆参数
                sensor_root=f"{NUPLAN_DATA_ROOT11}/nuplan-v1.1/sensor_blobs",  # 传感器数据路径
            )
            # 将构建的 NuPlanScenario 实例添加到列表中
            nuplan_scenarios.append(nuplan_scenario)
            iteration=20
            lidar_pc_obj = nuplan_scenario.get_sensors_at_iteration_edit(iteration, channels=None)  # 获取传感器数据
            egoPose_token=lidar_pc_obj.ego_pose_token
            
            
            # /home/zhaodanqi/clone/pluto/jsonfiles/2021.05.12.22.00.38_veh-35_01008_01518.json
            eval_data_path='/home/zhaodanqi/clone/pluto/jsonfiles'
            log_name = scenario_data['log_name']#2021.05.12.22.00.38_veh-35_01008_01518
            file_extension = ".json"
            new_eval_data_path = os.path.join(eval_data_path, log_name + file_extension)
            # new_eval_data_path = os.path.splitext(eval_data_path)[0] + f'_{log_name}' + file_extension
            # with open(new_eval_data_path, 'r') as file:
            #     eval_data = json.load(file)
            # 遍历查找对应的 token
            speed_plan=None
            path_plan=None
            gt_answer = None
            
            #TODO
            # ？接API进行训练
            #获得20/21/22/23/24/25/26的图片信息，然后让api输出一个元动作。
            # 能获得信息，但是无法进行连续训练？
            #同时运行多个scenario/其实也可以同时获得多个轨迹输出，那么我就让这个几个轨迹输出享查不要太大就行；也就是帧与帧之间的轨迹计算Loss；
            # 
            # tong'shi
            # 检查文件是否存在
            if os.path.exists(new_eval_data_path):
                with open(new_eval_data_path, 'r') as file:
                    eval_data = json.load(file)
                for sample in eval_data:
                    if sample.get("token") == egoPose_token:
                        speed_plan= sample["speed_plan"]
                        path_plan=sample["path_plan"]
                        print(f"{egoPose_token} Speed Plan:", speed_plan)
                        print(f"{egoPose_token} Path Plan:", path_plan)
                        break  # 找到后立即跳出循环
                # 如果没有找到对应的 token，输出未找到的提示信息
                if speed_plan is None and path_plan is None:
                    print(f"Ego pose token {egoPose_token} not found in the JSON file.")
            else:
                print("待补充此Log的信息")  # 只打印提示信息，不影响后续代码执行
            
                        # 如果没有找到对应的 token，输出未找到的提示信息
            if speed_plan is None and path_plan is None:
                print(f"Ego pose token {egoPose_token} not found in the JSON file.")
            
            speed_longi_plans.append(speed_plan)
            path_lateral_plans.append(path_plan)
            
            
            
            
            
            
            
            
            
            
            
            
            # for sample in eval_data:
            #     if sample.get("token") == egoPose_token:
            #         speed_plan= sample["speed_plan"]
            #         path_plan=sample["path_plan"]
            #         print(f"{egoPose_token} Speed Plan:", speed_plan)
            #         print(f"{egoPose_token} Path Plan:", path_plan)
            #         break  # 找到后立即跳出循环
            # # 如果没有找到对应的 token，输出未找到的提示信息
            # if speed_plan is None and path_plan is None:
            #     print(f"Ego pose token {egoPose_token} not found in the JSON file.")
            
            # speed_longi_plans.append(speed_plan)
            # path_lateral_plans.append(path_plan)

        return self._step(batch, "train",speed_longi_plans,path_lateral_plans)

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
        # print("Input shape:", features.shape)
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
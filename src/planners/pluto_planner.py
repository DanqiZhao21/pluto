import os
import time
from pathlib import Path
from typing import List, Optional, Type

import numpy as np
import numpy.typing as npt
import shapely
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType
from scipy.special import softmax

from src.feature_builders.nuplan_scenario_render import NuplanScenarioRender

from ..post_processing.emergency_brake import EmergencyBrake
from ..post_processing.trajectory_evaluator import TrajectoryEvaluator
from ..scenario_manager.scenario_manager import ScenarioManager
from .ml_planner_utils import global_trajectory_to_states, load_checkpoint
#..表示相对导入

'''
利用深度学习模型生成最优驾驶轨迹，同时评估候选轨迹的安全性和规则合规性。
'''


class PlutoPlanner(AbstractPlanner):
    requires_scenario: bool = True

    def __init__(
        self,
        planner: TorchModuleWrapper,
        scenario: AbstractScenario = None,
        planner_ckpt: str = None,
        render: bool = False,
        use_gpu=True,
        save_dir=None,
        candidate_subsample_ratio: int = 0.5,
        candidate_min_num: int = 1,#候选轨迹限制
        candidate_max_num: int = 20,
        eval_dt: float = 0.1,
        eval_num_frames: int = 80,
        learning_based_score_weight: float = 0.25,
        use_prediction: bool = True,
    ) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        self._render = render
        self._imgs = []
        self._scenario = scenario
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self._use_prediction = use_prediction

        self._planner = planner
        self._planner_feature_builder = planner.get_list_of_required_feature()[0]
        self._planner_ckpt = planner_ckpt

        self._initialization: Optional[PlannerInitialization] = None
        self._scenario_manager: Optional[ScenarioManager] = None

        self._future_horizon = 8.0
        self._step_interval = 0.1
        self._eval_dt = eval_dt
        self._eval_num_frames = eval_num_frames
        self._candidate_subsample_ratio = candidate_subsample_ratio
        self._candidate_min_num = candidate_min_num
        self._topk = candidate_max_num

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        self._scenario_type = scenario.scenario_type

        # post-processing
        self._trajectory_evaluator = TrajectoryEvaluator(eval_dt, eval_num_frames)
        self._emergency_brake = EmergencyBrake()
        self._learning_based_score_weight = learning_based_score_weight

        if render:
            self._scene_render = NuplanScenarioRender()
            if save_dir is not None:
                self.video_dir = Path(save_dir)
            else:
                self.video_dir = Path(os.getcwd())
            self.video_dir.mkdir(exist_ok=True, parents=True)

    @torch.no_grad()
    
    def _infer_model(self, features: FeaturesType) -> npt.NDArray[np.float32]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        输入：利用深度学习模型对输入特征进行推理，
        输出：返回预测的驾驶轨迹
        """
        # Propagate model
        action = self._model_loader.infer(features)
        action = action[0].cpu().numpy()[0]

        return action.astype(np.float64)

    def initialize(self, initialization: PlannerInitialization) -> None:
        
        """Inherited, see superclass.
        初始化规划器，包括加载模型、设置场景管理器和规划器特性构建器。"""
        
        torch.set_grad_enabled(False)
        #预测输出：利用模型参数对输入数据进行前向传播  ，生成预测结果。
        # 效率优先：推理时并不需要优化模型参数，因此可以跳过梯度计算 。

        if self._planner_ckpt is not None:
            self._planner.load_state_dict(load_checkpoint(self._planner_ckpt))
            
        #设置模型为评估模式（禁用 dropout、batch normalization 的训练特性）
        # 并将其加载到指定的计算设备（CPU/GPU）。
        self._planner.eval()
        self._planner = self._planner.to(self.device)

        self._initialization = initialization

        self._scenario_manager = ScenarioManager(
            map_api=initialization.map_api,
            ego_state=None,
            route_roadblocks_ids=initialization.route_roadblock_ids,
            radius=self._eval_dt * self._eval_num_frames * 60 / 4.0,
        )
        self._planner_feature_builder.scenario_manager = self._scenario_manager

        if self._render:
            self._scene_render.scenario_manager = self._scenario_manager

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        start_time = time.perf_counter()
        self._feature_building_runtimes.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()

        #更新车辆状态与可行域
        ego_state = current_input.history.ego_states[-1]
        self._scenario_manager.update_ego_state(ego_state)
        self._scenario_manager.update_drivable_area_map()
        
        #轨迹计算的核心逻辑
        planning_trajectory = self._run_planning_once(current_input)

        self._inference_runtimes.append(time.perf_counter() - start_time)

        return planning_trajectory

    def _run_planning_once(self, current_input: PlannerInput):
        
        
        #接收一个 PlannerInput 类型的参数 current_input，
        # PlannerInput：
        # iteration: SimulationIteration  # Iteration and time in a simulation progress
        # history: SimulationHistoryBuffer  # Rolling buffer containing past observations and states.
        # traffic_light_data: Optional[List[TrafficLightStatusData]] = None  # The traffic light status data
        
        ego_state = current_input.history.ego_states[-1]
        #提取自车状态
        planner_feature = self._planner_feature_builder.get_features_from_simulation(
            current_input, self._initialization
        )
        #提取所需规划特征
        
        planner_feature_torch = planner_feature.collate_for_simulation(
            [planner_feature.to_feature_tensor()]
        ).to_device(self.device)
        #转化为pytorch张量
        
        out = self._planner.forward(planner_feature_torch.data)#前向推理
        
        candidate_trajectories = (
            out["candidate_trajectories"][0].cpu().numpy().astype(np.float64)
        )
        probability = out["probability"][0].cpu().numpy()
        
        #是否需要预测
        if self._use_prediction:
            predictions = out["output_prediction"][0].cpu().numpy()
        else:
            predictions = None
            
        #是否包含无关参考轨迹
        ref_free_trajectory = (
            (out["output_ref_free_trajectory"][0].cpu().numpy().astype(np.float64))
            if "output_ref_free_trajectory" in out
            else None
        )
        
        # 对候选轨迹进行裁剪，去除不合适的轨迹，
        # 通常是根据概率、参考无关轨迹等进行筛选。
        # 返回裁剪后的轨迹和学习评分。
        candidate_trajectories, learning_based_score = self._trim_candidates(
            candidate_trajectories,
            probability,
            current_input.history.ego_states[-1],
            ref_free_trajectory,
        )
        print(f"learning-score{learning_based_score}")
        
        #计算基于规则的评分。评估指标包括交通信号、检测到的障碍物、车道信息等。
        rule_based_scores = self._trajectory_evaluator.evaluate(
            candidate_trajectories=candidate_trajectories,
            init_ego_state=current_input.history.ego_states[-1],
            detections=current_input.history.observations[-1],
            traffic_light_data=current_input.traffic_light_data,
            agents_info=self._get_agent_info(
                planner_feature.data, predictions, ego_state
            ),
            route_lane_dict=self._scenario_manager.get_route_lane_dicts(),
            drivable_area_map=self._scenario_manager.drivable_area_map,
            baseline_path=self._get_ego_baseline_path(
                self._scenario_manager.get_cached_reference_lines(), ego_state
            ),
        )
        print(f"rule-score{rule_based_scores}")
        
#######################################Fomat1 rule+ learning############################################

        # final_scores = (
        #     rule_based_scores + self._learning_based_score_weight * learning_based_score
        # )#确定最好的路径评分  #  learning_based_score_weight: 0.3
        # print(f"【 rule+learning 】finel-score : {rule_based_scores} + {self._learning_based_score_weight} * {learning_based_score} = {final_scores}")
        # best_candidate_idx = final_scores.argmax()
        
###########################################################################################################

########################################Format2 only learning############################################

        final_scores = learning_based_score
        print(f"【only learning 】 finel-score : { final_scores}")
        best_candidate_idx = final_scores.argmax()
        
###########################################################################################################

        
        #检查是否会发生碰撞，否则刹车
        trajectory = self._emergency_brake.brake_if_emergency(
            ego_state,
            self._trajectory_evaluator.time_to_at_fault_collision(best_candidate_idx),
            candidate_trajectories[best_candidate_idx],
        )

        # no emergency无碰撞   则进行产值并输出平滑轨迹
        if trajectory is None:
            # candidate_trajectories: (n_ref, n_mode, 80, 3)
            # probability: (n_ref, n_mode)
            trajectory = candidate_trajectories[best_candidate_idx, 1:]
            trajectory = InterpolatedTrajectory(
                global_trajectory_to_states(
                    global_trajectory=trajectory,
                    ego_history=current_input.history.ego_states,
                    future_horizon=len(trajectory) * self._step_interval,
                    step_interval=self._step_interval,
                    include_ego_state=False,
                )
            )

        if self._render:
            self._imgs.append(
                self._scene_render.render_from_simulation(
                    current_input=current_input,
                    initialization=self._initialization,
                    route_roadblock_ids=self._scenario_manager.get_route_roadblock_ids(),
                    scenario=self._scenario,
                    iteration=current_input.iteration.index,
                    planning_trajectory=self._global_to_local(trajectory, ego_state),
                    candidate_trajectories=self._global_to_local(
                        candidate_trajectories[rule_based_scores > 0], ego_state
                    ),
                    candidate_index=best_candidate_idx,
                    predictions=predictions,
                    return_img=True,
                )
            )

        return trajectory

    #self._topk]选择前k条轨迹
    def _trim_candidates(
        self,
        candidate_trajectories: np.ndarray,
        probability: np.ndarray,
        ego_state: EgoState,
        ref_free_trajectory: np.ndarray = None,
    ) -> npt.NDArray[np.float32]:
        """
        candidate_trajectories: (n_ref, n_mode, 80, 3)
        probability: (n_ref, n_mode)
        _topk=20
        """
        # import ipdb; ipdb.set_trace() 
        if len(candidate_trajectories.shape) == 4:
            n_ref, n_mode, T, C = candidate_trajectories.shape
            candidate_trajectories = candidate_trajectories.reshape(-1, T, C) # (n_ref*n_mode, 80, 3)
            probability = probability.reshape(-1) # # (1,n_ref*n_mode)
            #reshape(-1) 默认按照 行优先（C 语言风格，row-major order） 展开，
            # 即 先按行 (row) 展开，再按列 (column) 展开，再按更高维展开。
        sorted_idx = np.argsort(-probability)
        sorted_candidate_trajectories = candidate_trajectories[sorted_idx][: self._topk]#选择前top_k条
        sorted_probability = probability[sorted_idx][: self._topk]
        sorted_probability = softmax(sorted_probability)

        if ref_free_trajectory is not None:
            sorted_candidate_trajectories = np.concatenate(
                [sorted_candidate_trajectories, ref_free_trajectory[None, ...]],
                axis=0,
            )
            sorted_probability = np.concatenate([sorted_probability, [0.25]], axis=0)

        # to global
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        sorted_candidate_trajectories[..., :2] = (
            np.matmul(sorted_candidate_trajectories[..., :2], rot_mat) + origin
        )
        sorted_candidate_trajectories[..., 2] += angle

        sorted_candidate_trajectories = np.concatenate(
            [sorted_candidate_trajectories[..., 0:1, :], sorted_candidate_trajectories],
            axis=-2,
        )

        return sorted_candidate_trajectories, sorted_probability

    def _get_agent_info(self, data, predictions, ego_state: EgoState):
        """
        predictions: (n_agent, 80, 2 or 3)
        """
        current_velocity = np.linalg.norm(data["agent"]["velocity"][1:, -1], axis=-1)
        current_state = np.concatenate(
            [data["agent"]["position"][1:, -1], data["agent"]["heading"][1:, -1, None]],
            axis=-1,
        )
        velocity = None

        if predictions is None:  # constant velocity
            timesteps = np.linspace(0.1, 8, 80).reshape(1, 80, 1)
            displacement = data["agent"]["velocity"][1:, None, -1] * timesteps
            positions = current_state[:, None, :2] + displacement
            angles = current_state[:, None, 2:3].repeat(80, axis=1)
            predictions = np.concatenate([positions, angles], axis=-1)
            predictions = np.concatenate([current_state[:, None], predictions], axis=1)
            velocity = current_velocity[:, None].repeat(81, axis=1)
        elif predictions.shape[-1] == 2:
            predictions = np.concatenate(
                [current_state[:, None, :2], predictions], axis=1
            )
            diff = predictions[:, 1:] - predictions[:, :-1]
            start_end_dist = np.linalg.norm(
                predictions[:, -1, :2] - predictions[:, 0, :2], axis=-1
            )
            near_stop_mask = start_end_dist < 1.0
            angle = np.arctan2(diff[..., 1], diff[..., 0])
            angle = np.concatenate([current_state[:, None, -1], angle], axis=1)
            angle = np.where(
                near_stop_mask[:, None], current_state[:, 2:3].repeat(81, axis=1), angle
            )
            predictions = np.concatenate(
                [predictions[..., :2], angle[..., None]], axis=-1
            )
        elif predictions.shape[-1] == 3:
            predictions = np.concatenate([current_state[:, None], predictions], axis=1)
        elif predictions.shape[-1] == 5:
            velocity = np.linalg.norm(predictions[..., 3:5], axis=-1)
            predictions = np.concatenate(
                [current_state[:, None], predictions[..., :3]], axis=1
            )
            velocity = np.concatenate([current_velocity[:, None], velocity], axis=-1)
        else:
            raise ValueError("Invalid prediction shape")

        # to global
        predictions_global = self._local_to_global(predictions, ego_state)

        if velocity is None:
            velocity = (
                np.linalg.norm(np.diff(predictions_global[..., :2], axis=-2), axis=-1)
                / 0.1
            )
            velocity = np.concatenate([current_velocity[..., None], velocity], axis=-1)

        return {
            "tokens": data["agent_tokens"][1:],
            "shape": data["agent"]["shape"][1:, -1],
            "category": data["agent"]["category"][1:],
            "velocity": velocity,
            "predictions": predictions_global,
        }

    def _get_ego_baseline_path(self, reference_lines, ego_state: EgoState):
        '''
        根据给定的参考线数据和车辆当前的状态（ego_state），
        确定车辆当前位置最接近的参考线，并从该参考线生成一条基准路径
        '''
        init_ref_points = np.array([r[0] for r in reference_lines], dtype=np.float64)

        init_distance = np.linalg.norm(
            init_ref_points[:, :2] - ego_state.rear_axle.array, axis=-1
        )
        nearest_idx = np.argmin(init_distance)
        reference_line = reference_lines[nearest_idx]
        baseline_path = shapely.LineString(reference_line[:, :2])

        return baseline_path

    def _local_to_global(self, local_trajectory: np.ndarray, ego_state: EgoState):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(local_trajectory[..., :2], rot_mat) + origin
        heading = local_trajectory[..., 2] + angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    def _global_to_local(self, global_trajectory: np.ndarray, ego_state: EgoState):
        if isinstance(global_trajectory, InterpolatedTrajectory):
            states: List[EgoState] = global_trajectory.get_sampled_trajectory()
            global_trajectory = np.stack(
                [
                    np.array(
                        [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
                    )
                    for state in states
                ],
                axis=0,
            )

        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(global_trajectory[..., :2] - origin, rot_mat)
        heading = global_trajectory[..., 2] - angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        if self._render:
            import imageio

            imageio.mimsave(
                self.video_dir
                / f"{self._scenario.log_name}_{self._scenario.token}.mp4",
                self._imgs,
                fps=10,
            )
            print("\n video saved to ", self.video_dir / "video.mp4\n")

        return report

from copy import deepcopy
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import (
    AbstractAugmentor,
)
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

from src.features.pluto_feature import PlutoFeature
from src.utils.collision_checker import CollisionChecker
from src.utils.utils import crop_img_from_center
from src.utils.utils import shift_and_rotate_img

MAP_CONTRAST_TYPE = 0
AGENT_CONTRAST_TYPE = 1


class ContrastiveScenarioGenerator(AbstractAugmentor):
    def __init__(
        self,
        history_steps=21,
        max_interaction_horizon=40,
        low: List[float] = [0.0, -1.5, -0.35, -1, -0.5, -0.2, -0.2],
        high: List[float] = [2.0, 1.5, 0.35, 1, 0.5, 0.2, 0.2],
        use_negative_sample: bool = True,
    ) -> None:
        """
        Initialize the augmentor,
        state: [x, y, yaw, vel, acc, steer, steer_rate, angular_vel, angular_acc],
        :param dt: Time interval between trajectory points.
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        self.history_steps = history_steps
        self.max_interaction_horizon = max_interaction_horizon
        self._random_offset_generator = UniformNoise(low, high)
        self._collision_checker = CollisionChecker()
        self._rear_to_cog = get_pacifica_parameters().rear_axle_to_center
        self.use_negative_sample = use_negative_sample

    def augment(
        self,
        features: FeaturesType,
        targets: TargetsType = None,
        scenario: Optional[AbstractScenario] = None,
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        print

        feature: PlutoFeature = features["feature"]

        feature.data_p = self.generate_positive_sample(feature.data)
        # feature.data的类型为<class 'dict'> dict_keys(['cur', 'cur-1', 'cur-2', 'cur-3', 'cur-4', 'cur-5', 'cur-6'])
        if "cost_maps" in feature.data:
            for i in range(7):  # 0 表示 cur, 1-6 表示 cur-1, cur-2, ..., cur-6
                cur_key = "cur" if i == 0 else f"cur-{i}"  # 处理 cur 和 cur-1 ~ cur-6
                feature.data[cur_key]["cost_maps"] = crop_img_from_center(
                    feature.data[cur_key]["cost_maps"], (500, 500)
                )
        if self.use_negative_sample:
            feature.data_n, feature.data_n_info = self.generate_negative_sample(
                feature.data
            )

        features["feature"] = feature

        return features, targets

    def generate_positive_sample(self, data):
        new_data = deepcopy(data)
        # new_data=data
        # print(f"new_data的类型和结构是 {type(new_data)} {new_data.keys()}")
        # new_data的类型和结构是 <class 'dict'> dict_keys(['cur', 'cur-1', 'cur-2', 'cur-3', 'cur-4', 'cur-5', 'cur-6'])

        #cur的时刻的所有情况
        
        
        for i in range(7):  # 0 表示 cur, 1-6 表示 cur-1, cur-2, ..., cur-6
            cur_key = "cur" if i == 0 else f"cur-{i}"  # 处理 cur 和 cur-1 ~ cur-6
            
            
            current_state = data[cur_key]["current_state"]
            noise = self._random_offset_generator.sample()

            num_tries, scale = 0, 1.0
            agents_position = data[cur_key]["agent"]["position"][1:11, self.history_steps - 1]
            agents_shape = data[cur_key]["agent"]["shape"][1:11, self.history_steps - 1]
            agents_heading = data[cur_key]["agent"]["heading"][1:11, self.history_steps - 1]
            agents_shape = data[cur_key]["agent"]["shape"][1:11, self.history_steps - 1]

            while num_tries < 5:
                new_noise = noise * scale
                new_state = current_state + new_noise
                new_state[3] = max(0.0, new_state[3])

                if self.safety_check(
                    ego_position=new_state[:2],
                    ego_heading=new_state[2],
                    agents_position=agents_position,
                    agents_heading=agents_heading,
                    agents_shape=agents_shape,
                ):
                    break
                
                num_tries += 1
                scale *= 0.5
                
            if "cost_maps" in data:
                new_data[cur_key]["cost_maps"] = crop_img_from_center(
                    shift_and_rotate_img(
                        img=new_data[cur_key]["cost_maps"].astype(np.float32),
                        shift=np.array([new_noise[1], -new_noise[0], 0]),
                        angle=-new_noise[2],
                        resolution=0.2,
                        cval=-200,
                    ),
                    (500, 500),
                )
            
            new_data[cur_key]["current_state"] = new_state
            new_data[cur_key]["agent"]["position"][0, self.history_steps - 1] = new_state[:2]
            new_data[cur_key]["agent"]["heading"][0, self.history_steps - 1] = new_state[2]


            non_interacting_agent_mask = data[cur_key]["causal"]["interaction_label"] <= 0
            if non_interacting_agent_mask.sum() > 1 and np.random.uniform(0, 1) < 0.5:
                non_interacting_agent_mask[0] = False  # exclude ego
                non_interacting_agent_mask[data[cur_key]["causal"]["leading_agent_mask"]] = False
                drop_portion = np.random.uniform(low=0.1, high=1.0)
                noise = np.random.uniform(0, 1, len(non_interacting_agent_mask))
                noise[~non_interacting_agent_mask] = 2
                drop_mask = noise <= drop_portion

                for k, v in new_data[cur_key]["agent"].items():
                    new_data[cur_key]["agent"][k] = v[~drop_mask]

        new_data = PlutoFeature.normalize(new_data).data
        
    
        return new_data

    def generate_negative_sample(self, data):
        available_generators = []
        data_n={}
        data_info={}
        for i in range(7):  # 0 表示 cur, 1-6 表示 cur-1, cur-2, ..., cur-6
            cur_key = "cur" if i == 0 else f"cur-{i}"  # 处理 cur 和 cur-1 ~ cur-6
        # 移除交互中的其他车辆（neg_interacting_agent_dropout）
        # 改变交通信号灯状态（neg_traffic_light_inversion）
        # 在自车前方插入一辆车（neg_leading_agent_insertion）
            # print(f"NOW IS {cur_key}")
            interacting_agent_mask = (data[cur_key]["causal"]["interaction_label"] > 0) & (
                data[cur_key]["causal"]["interaction_label"] < self.max_interaction_horizon
            )

            if not data[cur_key]["causal"]["is_waiting_for_red_light_without_lead"]:
                if (
                    data[cur_key]["causal"]["leading_agent_mask"].any()
                    or interacting_agent_mask.any()
                ):
                    data[cur_key]["causal"]["interacting_agent_mask"] = interacting_agent_mask
                    available_generators.append(self.neg_interacting_agent_dropout)
            else:
                available_generators.append(self.neg_traffic_light_inversion)

            if (
                len(data[cur_key]["causal"]["free_path_points"]) > 0
                and data[cur_key]["agent"]["position"].shape[0] > 1
            ):
                available_generators.append(self.neg_leading_agent_insertion)

            #因为 self.neg_leading_agent_insertion 被作为函数引用添加到列表，而不是立即调用函数。
            
            if len(available_generators) > 0:
                generator = np.random.choice(available_generators)
                data_n[cur_key], contrast_type = generator(data[cur_key])
                data_n_valid_mask = True
            else:
                data_n[cur_key] = data[cur_key]
                contrast_type = 0
                data_n_valid_mask = False
            data_info[cur_key] = {"valid_mask": data_n_valid_mask,"type": contrast_type}
        # return data_n, {"valid_mask": data_n_valid_mask, "type": contrast_type}
        return data_n, data_info

    def safety_check(
        self,
        ego_position: npt.NDArray[np.float32],
        ego_heading: npt.NDArray[np.float32],
        agents_position: npt.NDArray[np.float32],
        agents_heading: npt.NDArray[np.float32],
        agents_shape: npt.NDArray[np.float32],
    ) -> bool:
        if len(agents_position) == 0:
            return True

        ego_center = (
            ego_position
            + np.stack([np.cos(ego_heading), np.sin(ego_heading)], axis=-1)
            * self._rear_to_cog
        )
        ego_state = torch.from_numpy(
            np.concatenate([ego_center, [ego_heading]], axis=-1)
        ).unsqueeze(0)
        objects_state = torch.from_numpy(
            np.concatenate([agents_position, agents_heading[..., None]], axis=-1)
        ).unsqueeze(0)

        collisions = self._collision_checker.collision_check(
            ego_state=ego_state,
            objects=objects_state,
            objects_width=torch.from_numpy(agents_shape[:, 0]).unsqueeze(0),
            objects_length=torch.from_numpy(agents_shape[:, 1]).unsqueeze(0),
        )

        return not collisions.any()

    def neg_traffic_light_inversion(self, data):
        new_data = deepcopy(data)
        # new_data=data
        ego_care_red_light_mask = data["causal"]["ego_care_red_light_mask"]
        choices = [TrafficLightStatusType.GREEN, TrafficLightStatusType.UNKNOWN]
        new_status = np.random.choice(choices, size=ego_care_red_light_mask.sum())
        new_data["map"]["polygon_tl_status"][ego_care_red_light_mask] = new_status
        return new_data, MAP_CONTRAST_TYPE

    def neg_interacting_agent_dropout(self, data):
        new_data = deepcopy(data)
        # new_data=data
        # print(f" neg_里面 data(即data[])的键为{data.keys()}")
        # print(f" neg_里面 data['causal'](即data[])的键为{data['causal'].keys()}") 
#           data(即data[cur-x])的键为dict_keys(['agent', 'current_state', 'static_objects', 'map', 'cost_maps', 'causal', 'reference_line', 'origin', 'angle'])
#           data['causal'](即data[cur-x]['causal'])的键为dict_keys(['is_waiting_for_red_light_without_lead', 'leading_agent_mask', 'leading_distance', 
#           'ego_care_red_light_mask', 'fixed_ego_future_valid_mask', 'free_path_points', 'interaction_label', 'interacting_agent_mask'])

        if "interacting_agent_mask" not in data["causal"]:
            print(" warning : interacting_agent_mask does nont exist")
            # data["causal"]["interacting_agent_mask"]
            dropout_mask = (
            data["causal"]["leading_agent_mask"])
        else:
            dropout_mask = (
                data["causal"]["leading_agent_mask"]
                | data["causal"]["interacting_agent_mask"]
            )
        for k, v in new_data["agent"].items():
            new_data["agent"][k] = v[~dropout_mask]
        return new_data, AGENT_CONTRAST_TYPE

    def neg_leading_agent_insertion(self, data):
        new_data = deepcopy(data)
        # new_data=data
        
        if len(data["causal"]["free_path_points"]) == 0:
    # 如果没有路径点，使用一个默认值
            path_point = np.array([1, 1,1])  # 使用一个默认值或其他合理的路径点
            print("Warning: path_point is (one,one,one).") 
        
        else:
            path_point = data["causal"]["free_path_points"][np.random.choice(len(data["causal"]["free_path_points"]))]
        
        
        
        
        # path_point = data["causal"]["free_path_points"][
        #     np.random.choice(len(data["causal"]["free_path_points"]))
        # ]

        agents_velocity = np.linalg.norm(
            data["agent"]["velocity"][:, self.history_steps - 1], axis=-1
        )
        agents_velocity_diff = np.abs(agents_velocity[1:] - agents_velocity[0])
        
        if len(agents_velocity_diff) > 0:
            similar_agent_idx = np.argmin(agents_velocity_diff)
        else:
            # 处理 agents_velocity_diff 为空的情况
            similar_agent_idx = -1  # 或者其他适合的默认值
            # print("Warning: agents_velocity_diff is empty.") 
        
        similar_agent_idx = np.argmin(agents_velocity_diff)
        if agents_velocity_diff[similar_agent_idx] < 2:
            copy_agent_idx = similar_agent_idx + 1
        else:
            copy_agent_idx = 0

        if agents_velocity[copy_agent_idx] < 0.1:
            scale_coeff = 1.0
        else:
            scale_coeff = agents_velocity[0] / agents_velocity[copy_agent_idx]

        generated_agent = self._generate_agent_from_idx(
            data["agent"], copy_agent_idx, scale_coeff, path_point
        )

        for k, v in new_data["agent"].items():
            new_data["agent"][k] = np.concatenate(
                [v, generated_agent[k][None, ...]], axis=0
            )
        return new_data, AGENT_CONTRAST_TYPE

    def _generate_agent_from_idx(
        self, agent, idx, scale_coeff, path_point, shape_scale=[0.9, 1.1]
    ):
        scale_coeff *= np.random.uniform(low=0.0, high=0.8)
        current_position = agent["position"][idx][self.history_steps - 1]
        hist_position = agent["position"][idx][: self.history_steps]
        fut_position = agent["position"][idx][self.history_steps - 1 :]
        hist_diff = np.concatenate(
            [scale_coeff * np.diff(hist_position, axis=0), np.zeros((1, 2))], axis=0
        )
        fut_diff = scale_coeff * np.diff(fut_position, axis=0)
        scaled_position = np.concatenate(
            [
                -np.cumsum(hist_diff[::-1], axis=0)[::-1] + current_position,
                np.cumsum(fut_diff, axis=0) + current_position,
            ],
            axis=0,
        )
        heading = agent["heading"][idx]
        velocity = scale_coeff * agent["velocity"][idx]
        shape = agent["shape"][idx]

        delta_angle = heading[self.history_steps - 1] - path_point[2]
        cos, sin = np.cos(delta_angle), np.sin(delta_angle)
        rot_mat = np.array([[cos, -sin], [sin, cos]])

        new_position = (
            np.matmul(scaled_position - current_position[None, :2], rot_mat)
            + path_point[None, :2]
        )
        new_heading = heading - heading[self.history_steps - 1] + path_point[2]
        new_velocity = np.matmul(velocity, rot_mat)
        new_shape = shape * np.random.uniform(*shape_scale, size=shape.shape)
        new_category = agent["category"][idx] if idx != 0 else np.array(1)
        new_target = np.concatenate(
            (
                new_position[self.history_steps :]
                - new_position[self.history_steps - 1],
                (
                    new_heading[self.history_steps :]
                    - new_heading[self.history_steps - 1]
                )[:, None],
            ),
            axis=-1,
        )

        return {
            "position": new_position,
            "heading": new_heading,
            "velocity": new_velocity,
            "shape": new_shape,
            "category": new_category,
            "valid_mask": agent["valid_mask"][idx],
            "target": new_target,
        }

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(
            param=self._augment_prob,
            param_name=f"{self._augment_prob=}".partition("=")[0].split(".")[1],
            scaling_direction=ScalingDirection.MAX,
        )

    @property
    def get_schedulable_attributes(self) -> List[ParameterToScale]:
        """Inherited, see superclass."""
        return cast(
            List[ParameterToScale],
            self._random_offset_generator.get_schedulable_attributes(),
        )

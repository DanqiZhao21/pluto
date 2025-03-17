from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from torch.nn.utils.rnn import pad_sequence

from src.utils.utils import to_device, to_numpy, to_tensor

'''
自动驾驶或轨迹预测任务中的场景数据预处理和操作
'''


@dataclass
class PlutoFeature(AbstractModelFeature):
    data: Dict[str, Any]  # anchor sample
    data_p: Dict[str, Any] = None  # positive sample
    data_n: Dict[str, Any] = None  # negative sample
    data_n_info: Dict[str, Any] = None  # negative sample info


        # p features["feature"].data["cur-5"].keys()ipdb> p features["feature"].data["cur-5"].keys()
        # dict_keys(['agent', 'map', 'data_n_valid_mask', 'data_n_type', 'current_state', 'origin', 'angle'])
        
        #为什么村的数据和最开始的data不太一样呢
        # features["feature"].data.kets()yys()ipdb> p features["feature"].data.keys()即data.keys()
        # dict_keys(['agent', 'map', 'reference_line', 'static_objects', 'current_state', 'origin', 'angle', 'cost_maps'])



    @classmethod
    def collate(cls, feature_list: List[PlutoFeature]) -> PlutoFeature:
        batch_data = {}

        pad_keys = ["agent", "map"]
        stack_keys = ["current_state", "origin", "angle"]

        if "reference_line" in feature_list[0].data["cur"]:
            pad_keys.append("reference_line")
        if "static_objects" in feature_list[0].data["cur"]:
            pad_keys.append("static_objects")
        if "cost_maps" in feature_list[0].data["cur"]:
            stack_keys.append("cost_maps")
            
        for i in range(7):  # 0 表示 cur, 1-6 表示 cur-1, cur-2, ..., cur-6
            cur_key = "cur" if i == 0 else f"cur-{i}"  # 处理 cur 和 cur-1 ~ cur-6
            batch_data[cur_key] = {}
            if feature_list[0].data_n is not None:
                for key in pad_keys:
                    batch_data[cur_key][key] = {
                        k: pad_sequence(
                            [f.data[cur_key][key][k] for f in feature_list]
                            + [f.data_p[cur_key][key][k] for f in feature_list]
                            + [f.data_n[cur_key][key][k] for f in feature_list],
                            batch_first=True,
                        )
                        for k in feature_list[0].data[cur_key][key].keys()
                    }

                batch_data[cur_key]["data_n_valid_mask"] = torch.Tensor(
                    [f.data_n_info[cur_key]["valid_mask"] for f in feature_list]
                ).bool()
                batch_data[cur_key]["data_n_type"] = torch.Tensor(
                    [f.data_n_info[cur_key]["type"] for f in feature_list]
                ).long()

                for key in stack_keys:
                    batch_data[cur_key][key] = torch.stack(
                        [f.data[cur_key][key] for f in feature_list]
                        + [f.data_p[cur_key][key] for f in feature_list]
                        + [f.data_n[cur_key][key] for f in feature_list],
                        dim=0,
                    )
            elif feature_list[0].data_p is not None:
                for key in pad_keys:
                    batch_data[cur_key][key] = {
                        k: pad_sequence(
                            [f.data[cur_key][key][k] for f in feature_list]
                            + [f.data_p[cur_key][key][k] for f in feature_list],
                            batch_first=True,
                        )
                        for k in feature_list[0].data[cur_key][key].keys()
                    }

                for key in stack_keys:
                    batch_data[cur_key][key] = torch.stack(
                        [f.data[cur_key][key] for f in feature_list]
                        + [f.data_p[cur_key][key] for f in feature_list],
                        dim=0,
                    )
            else:
                for key in pad_keys:
                    batch_data[cur_key][key] = {
                        k: pad_sequence(
                            [f.data[cur_key][key][k] for f in feature_list], batch_first=True
                        )
                        for k in feature_list[0].data[cur_key][key].keys()
                    }

                for key in stack_keys:
                    batch_data[cur_key][key] = torch.stack(
                        [f.data[cur_key][key] for f in feature_list], dim=0
                    )

        return PlutoFeature(data=batch_data)
    
    #############################################Collect for simulation#############################################
    def collate_for_simulation(cls, feature_list: List[PlutoFeature]) -> PlutoFeature:
        batch_data = {}

        pad_keys = ["agent", "map"]
        stack_keys = ["current_state", "origin", "angle"]
        
        # for f in feature_list:
        print("注意一下")
        print(feature_list[0].data.keys())

        if "reference_line" in feature_list[0].data:
            pad_keys.append("reference_line")
        if "static_objects" in feature_list[0].data:
            pad_keys.append("static_objects")
        if "cost_maps" in feature_list[0].data:
            stack_keys.append("cost_maps")

        if feature_list[0].data_n is not None:
            for key in pad_keys:
                batch_data[key] = {
                    k: pad_sequence(
                        [f.data[key][k] for f in feature_list]
                        + [f.data_p[key][k] for f in feature_list]
                        + [f.data_n[key][k] for f in feature_list],
                        batch_first=True,
                    )
                    for k in feature_list[0].data[key].keys()
                }

            batch_data["data_n_valid_mask"] = torch.Tensor(
                [f.data_n_info["valid_mask"] for f in feature_list]
            ).bool()
            batch_data["data_n_type"] = torch.Tensor(
                [f.data_n_info["type"] for f in feature_list]
            ).long()

            for key in stack_keys:
                batch_data[key] = torch.stack(
                    [f.data[key] for f in feature_list]
                    + [f.data_p[key] for f in feature_list]
                    + [f.data_n[key] for f in feature_list],
                    dim=0,
                )
        elif feature_list[0].data_p is not None:
            for key in pad_keys:
                batch_data[key] = {
                    k: pad_sequence(
                        [f.data[key][k] for f in feature_list]
                        + [f.data_p[key][k] for f in feature_list],
                        batch_first=True,
                    )
                    for k in feature_list[0].data[key].keys()
                }

            for key in stack_keys:
                batch_data[key] = torch.stack(
                    [f.data[key] for f in feature_list]
                    + [f.data_p[key] for f in feature_list],
                    dim=0,
                )
        else:
            for key in pad_keys:
                batch_data[key] = {
                    k: pad_sequence(
                        [f.data[key][k] for f in feature_list], batch_first=True
                    )
                    for k in feature_list[0].data[key].keys()
                }

            for key in stack_keys:
                batch_data[key] = torch.stack(
                    [f.data[key] for f in feature_list], dim=0
                )

        return PlutoFeature(data=batch_data)
    
    #############################################Collect for simulation#############################################

    def to_feature_tensor(self) -> PlutoFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_tensor(v)

        if self.data_p is not None:
            new_data_p = {}
            for k, v in self.data_p.items():
                new_data_p[k] = to_tensor(v)
        else:
            new_data_p = None

        if self.data_n is not None:
            new_data_n = {}
            new_data_n_info = {}
            for k, v in self.data_n.items():
                new_data_n[k] = to_tensor(v)
            for k, v in self.data_n_info.items():
                new_data_n_info[k] = to_tensor(v)
        else:
            new_data_n = None
            new_data_n_info = None

        return PlutoFeature(
            data=new_data,
            data_p=new_data_p,
            data_n=new_data_n,
            data_n_info=new_data_n_info,
        )

    def to_numpy(self) -> PlutoFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_numpy(v)
        if self.data_p is not None:
            new_data_p = {}
            for k, v in self.data_p.items():
                new_data_p[k] = to_numpy(v)
        else:
            new_data_p = None
        if self.data_n is not None:
            new_data_n = {}
            for k, v in self.data_n.items():
                new_data_n[k] = to_numpy(v)
        else:
            new_data_n = None
        return PlutoFeature(data=new_data, data_p=new_data_p, data_n=new_data_n)

    def to_device(self, device: torch.device) -> PlutoFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_device(v, device)
        return PlutoFeature(data=new_data)

    def serialize(self) -> Dict[str, Any]:
        return {"data": self.data}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PlutoFeature:
        return PlutoFeature(data=data["data"])

    def unpack(self) -> List[AbstractModelFeature]:
        raise NotImplementedError

    @property
    def is_valid(self) -> bool:
        if "reference_line" in self.data:
            return self.data["cur"]["reference_line"]["valid_mask"].any()
        else:
            return self.data["cur"]["map"]["point_position"].shape[0] > 0

    @classmethod
    #对数据进行归一化处理，使得所有坐标和方向信息相对于当前帧自车的位置 (center_xy) 和航向角 center_angle 进行平移和旋转对齐
    def normalize(
        self, data, first_time=False, radius=None, hist_steps=21
    ) -> PlutoFeature:      
        #直全部变成循环
        for i in range(7):  # 0 表示 cur, 1-6 表示 cur-1, cur-2, ..., cur-6
            cur_key = "cur" if i == 0 else f"cur-{i}"  # 处理 cur 和 cur-1 ~ cur-6

            if cur_key not in data:
                continue  # 如果某一帧不存在，则跳过

            # print(f"In normalize :  cur -{i}")
            cur_state = data[cur_key]["current_state"]
            center_xy, center_angle = cur_state[:2].copy(), cur_state[2].copy()

            rotate_mat = np.array(
                [
                    [np.cos(center_angle), -np.sin(center_angle)],
                    [np.sin(center_angle), np.cos(center_angle)],
                ],
                dtype=np.float64,
            )

            data[cur_key]["current_state"][:3] = 0
            data[cur_key]["agent"]["position"] = np.matmul(
                data[cur_key]["agent"]["position"] - center_xy, rotate_mat
            )
            data[cur_key]["agent"]["velocity"] = np.matmul(data[cur_key]["agent"]["velocity"], rotate_mat)
            data[cur_key]["agent"]["heading"] -= center_angle

            data[cur_key]["map"]["point_position"] = np.matmul(
                data[cur_key]["map"]["point_position"] - center_xy, rotate_mat
            )
            data[cur_key]["map"]["point_vector"] = np.matmul(data[cur_key]["map"]["point_vector"], rotate_mat)
            data[cur_key]["map"]["point_orientation"] -= center_angle

            data[cur_key]["map"]["polygon_center"][..., :2] = np.matmul(
                data[cur_key]["map"]["polygon_center"][..., :2] - center_xy, rotate_mat
            )
            data[cur_key]["map"]["polygon_center"][..., 2] -= center_angle
            data[cur_key]["map"]["polygon_position"] = np.matmul(
                data[cur_key]["map"]["polygon_position"] - center_xy, rotate_mat
            )
            data[cur_key]["map"]["polygon_orientation"] -= center_angle

            if "causal" in data[cur_key]:
                if len(data[cur_key]["causal"]["free_path_points"]) > 0:
                    data[cur_key]["causal"]["free_path_points"][..., :2] = np.matmul(
                        data[cur_key]["causal"]["free_path_points"][..., :2] - center_xy, rotate_mat
                    )
                    data[cur_key]["causal"]["free_path_points"][..., 2] -= center_angle
            if "static_objects" in data[cur_key]:
                data[cur_key]["static_objects"]["position"] = np.matmul(
                    data[cur_key]["static_objects"]["position"] - center_xy, rotate_mat
                )
                data[cur_key]["static_objects"]["heading"] -= center_angle
            if "route" in data[cur_key]:
                data[cur_key]["route"]["position"] = np.matmul(
                    data[cur_key]["route"]["position"] - center_xy, rotate_mat
                )
            if "reference_line" in data[cur_key]:
                data[cur_key]["reference_line"]["position"] = np.matmul(
                    data[cur_key]["reference_line"]["position"] - center_xy, rotate_mat
                )
                data[cur_key]["reference_line"]["vector"] = np.matmul(
                    data[cur_key]["reference_line"]["vector"], rotate_mat
                )
                data[cur_key]["reference_line"]["orientation"] -= center_angle

            target_position = (
                data[cur_key]["agent"]["position"][:, hist_steps:]
                - data[cur_key]["agent"]["position"][:, hist_steps - 1][:, None]
            )
            target_heading = (
                data[cur_key]["agent"]["heading"][:, hist_steps:]
                - data[cur_key]["agent"]["heading"][:, hist_steps - 1][:, None]
            )
            target = np.concatenate([target_position, target_heading[..., None]], -1)
            target[~data[cur_key]["agent"]["valid_mask"][:, hist_steps:]] = 0
            data[cur_key]["agent"]["target"] = target

            if first_time:
                point_position = data[cur_key]["map"]["point_position"]
                x_max, x_min = radius, -radius
                y_max, y_min = radius, -radius
                valid_mask = (
                    (point_position[:, 0, :, 0] < x_max)
                    & (point_position[:, 0, :, 0] > x_min)
                    & (point_position[:, 0, :, 1] < y_max)
                    & (point_position[:, 0, :, 1] > y_min)
                )
                valid_polygon = valid_mask.any(-1)
                data[cur_key]["map"]["valid_mask"] = valid_mask

                for k, v in data[cur_key]["map"].items():
                    data[cur_key]["map"][k] = v[valid_polygon]

                if "causal" in data[cur_key]:
                    data[cur_key]["causal"]["ego_care_red_light_mask"] = data[cur_key]["causal"][
                        "ego_care_red_light_mask"
                    ][valid_polygon]

                data[cur_key]["origin"] = center_xy
                data[cur_key]["angle"] = center_angle


        return PlutoFeature(data=data)

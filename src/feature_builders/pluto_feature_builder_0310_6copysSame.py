import warnings
from typing import List, Type

import numpy as np
import shapely
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox, in_collision
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap, PolygonMapObject
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from shapely import LineString, Point

from src.features.pluto_feature import PlutoFeature
from src.scenario_manager.cost_map_manager import CostMapManager
from src.scenario_manager.scenario_manager import OccupancyType, ScenarioManager
from . import common


class PlutoFeatureBuilder(AbstractFeatureBuilder):
    def __init__(
        self,
        radius: float = 100,
        history_horizon: float = 2,
        future_horizon: float = 8,
        sample_interval: float = 0.1,
        max_agents: int = 64,
        max_static_obstacles: int = 10,
        build_reference_line: bool = False,
        disable_agent: bool = False,
    ) -> None:
        super().__init__()

        self.radius = radius
        self.history_horizon = history_horizon#2时间
        self.future_horizon = future_horizon#8
        self.history_samples = int(self.history_horizon / sample_interval)#2/0.1=20
        self.future_samples = int(self.future_horizon / sample_interval)#8/0.1=80
        self.sample_interval = sample_interval
        self.ego_params = get_pacifica_parameters()
        self.length = self.ego_params.length
        self.width = self.ego_params.width
        self.max_agents = max_agents
        self.max_static_obstacles = max_static_obstacles
        
        # self.scenario_manager = None
        self.scenario_manager = {"cur": None, **{f"cur-{i}": None for i in range(1, 7)}}#创建一个字典，包含当前场景和前6个场景

        self.build_reference_line = build_reference_line
        self.disable_agent = disable_agent
        self.inference = None
        self.simulation = False

        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        self.static_objects_types = [
            TrackedObjectType.CZONE_SIGN,
            TrackedObjectType.BARRIER,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.GENERIC_OBJECT,
        ]
        self.polygon_types = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.CROSSWALK,
        ]

    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return PlutoFeature  # type: ignore

    def get_class(self) -> Type[AbstractFeatureBuilder]:
        return PlutoFeatureBuilder

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "feature"

    def get_features_from_scenario(
        self,
        scenario: AbstractScenario,
        iteration=0,
    ) -> AbstractModelFeature:
        
        # p iterations_per_scenario
        # (wrapped_fn pid=505885) 1
        # (wrapped_fn pid=505885) ipdb> 
        # p iteration
        # (wrapped_fn pid=505917) 0
    
        # import ipdb; ipdb.set_trace()
        # print(f"持续时间为{scenario.duration_s}")
        #持续时间为TimeDuration(0.0s)
        #  p scenario.map_api
        # (wrapped_fn pid=2078763) <nuplan.common.maps.nuplan_map.nuplan_map.NuPlanMap object at 0x74867350b730>
        # (wrapped_fn pid=2078763) ipdb> 
        # p scenario.log_name
        # (wrapped_fn pid=2078744) '2021.06.09.12.39.51_veh-26_05620_06003'
        # (wrapped_fn pid=2078744) ipdb> 
        # p scenario.token
        # (wrapped_fn pid=2078752) '33813f3dcfc65fef'
        # (wrapped_fn pid=2078752) ipdb> 
        # p scenario.scenario_name
        # (wrapped_fn pid=2078760) '2d0b0d2913e65fdf'
        
        # ego_cur_state = scenario.get_ego_state_at_iteration(iteration)
        ego_cur_state = scenario.initial_ego_state ###似乎有问题，不应该是第0帧，而是第iterration帧？
        all_past_ego_trajectory = scenario.get_ego_past_trajectory(
            iteration=iteration,
            time_horizon=self.history_horizon,
            num_samples=self.history_samples, 
        )
        all_future_ego_trajectory = scenario.get_ego_future_trajectory(
            iteration=iteration,
            time_horizon=self.future_horizon,
            num_samples=self.future_samples,
        )
        all_ego_state_list = (
            list(all_past_ego_trajectory) + [ego_cur_state] + list(all_future_ego_trajectory)
        )
        
        if any(state is None for state in all_ego_state_list):
            print("Warning: Some ego states are None!")

        
        
        # import ipdb; ipdb.set_trace()
        
        # p len(list(all_past_ego_trajectory))
        # (wrapped_fn pid=2078756) 30
        
        
        
        
        # 将all_ego_state_list中的元素按照时间顺序排列
        ego_state_list = all_ego_state_list[-101:]
        ego_state_list_29 = ego_state_list
        ego_state_list_28 = ego_state_list
        ego_state_list_27 =ego_state_list
        ego_state_list_26 = ego_state_list
        ego_state_list_25 = ego_state_list
        ego_state_list_24 = ego_state_list
        
        
       ###########################################################################################################
        
        # all_present_tracked_objects=scenario.get_tracked_objects_at_iteration(
        #         iteration=iteration
        #     )#  -> DetectionsTracks
        all_present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        all_past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=iteration,
                time_horizon=self.history_horizon+1,
                num_samples=self.history_samples+10,
            )  #-> Generator[DetectionsTracks, None, None]:
        ]
        all_future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=iteration,
                time_horizon=self.future_horizon,
                num_samples=self.future_samples,
            )
        ]
        
        all_tracked_objects_list = (
            all_past_tracked_objects + [all_present_tracked_objects] + all_future_tracked_objects
        )
        
        # 将all_tracked_objects_list中的元素按照时间顺序排列
        tracked_objects_list = all_tracked_objects_list[-101:]
        tracked_objects_list_29 =  tracked_objects_list
        tracked_objects_list_28 =  tracked_objects_list
        tracked_objects_list_27 = tracked_objects_list
        tracked_objects_list_26 = tracked_objects_list
        tracked_objects_list_25 = tracked_objects_list
        tracked_objects_list_24 =  tracked_objects_list
        
        ##########################################################################################
        #历史的交通信息     
        traffic_light_status=scenario.get_traffic_light_status_at_iteration(iteration)

        traffic_light_status=list(traffic_light_status)
        all_past_traffic_light_status = list(scenario.get_past_traffic_light_status_history(iteration,1, 10))#准确来说只需要过去6帧（不到一秒）此除保险器件算了过去10帧（1s）

        all_traffic_lights_status_data=[]
        for status in all_past_traffic_light_status:

            all_traffic_lights_status_data.append(status.traffic_lights)

        traffic_light_status_29= traffic_light_status
        traffic_light_status_28= traffic_light_status
        traffic_light_status_27= traffic_light_status
        traffic_light_status_26= traffic_light_status
        traffic_light_status_25= traffic_light_status
        traffic_light_status_24= traffic_light_status
        
        
        
        # # traffic_light_status_29= all_past_traffic_light_status[-1]
        # print(f"typr of traffic_light_status_29: {type(traffic_light_status_29)}")
        # # typr of traffic_light_status_29: <class 'nuplan.common.maps.maps_datatypes.TrafficLightStatuses'>
        # print(dir(traffic_light_status_29))
        #         typr of traffic_light_status_29: <class 'nuplan.common.maps.maps_datatypes.TrafficLightStatuses'>
        # (wrapped_fn pid=757038) ['__annotations__', '__class__', '__dataclass_fields__', '__dataclass_params__', 
        #                          '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
        #                          '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', 
        #                          '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', 
        #                          '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'traffic_lights']
                
        
        # 获取 route_roadblock_ids
        route_roadblocks_ids_value = scenario.get_route_roadblock_ids()

        # 创建字典，所有键对应的值都为 route_roadblocks_ids_value
        route_roadblocks_ids = {f"cur-{i}": route_roadblocks_ids_value for i in range(6 + 1)}

        # 额外添加 cur 键
        route_roadblocks_ids["cur"] = route_roadblocks_ids_value

        data = self._build_feature(
            # present_idx=self.history_samples, #20==》30
            present_idx=self.history_samples, #20
          #过去20+现在+未来 （我直接将这里的增加以下，改一下iteration即可） 将改成curent=30;然后往后倒6帧；24，25，26，27，28，29
            ego_state_list=ego_state_list,  #过去20+现在+未来 （我直接将这里的增加以下，改一下iteration即可） 将改成curent=30;然后往后倒6帧；24，25，26，27，28，29
            ego_state_list_29=ego_state_list_29,
            ego_state_list_28=ego_state_list_28,
            ego_state_list_27=ego_state_list_27,
            ego_state_list_26=ego_state_list_26,
            ego_state_list_25=ego_state_list_25,
            ego_state_list_24=ego_state_list_24,

            tracked_objects_list=tracked_objects_list, ##过去20+现在+未来的跟踪信息
            tracked_objects_list_29=tracked_objects_list_29,
            tracked_objects_list_28=tracked_objects_list_28,
            tracked_objects_list_27=tracked_objects_list_27,
            tracked_objects_list_26=tracked_objects_list_26,
            tracked_objects_list_25=tracked_objects_list_25,
            tracked_objects_list_24=tracked_objects_list_24,
            
            # route_roadblocks_ids=scenario.get_route_roadblock_ids(), #获得整段场景下的障碍物信息
            route_roadblocks_ids=route_roadblocks_ids,  # 传递这个字典
            
            map_api=scenario.map_api,
            mission_goal=scenario.get_mission_goal(),#没啥用
            
            traffic_light_status=traffic_light_status,
            traffic_light_status_29=traffic_light_status_29,
            traffic_light_status_28=traffic_light_status_28,
            traffic_light_status_27=traffic_light_status_27,
            traffic_light_status_26=traffic_light_status_26,
            traffic_light_status_25=traffic_light_status_25,
            traffic_light_status_24=traffic_light_status_24,
            
            inference=False,
        )
        
        return data

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> AbstractModelFeature:
        self.simulation = True

        history = current_input.history
        tracked_objects_list = [
            observation.tracked_objects for observation in history.observations
        ]

        horizon = self.history_samples + 1
        return self._build_feature(
            present_idx=-1,
            ego_state_list=history.ego_states[-horizon:],
            tracked_objects_list=tracked_objects_list[-horizon:],
            route_roadblocks_ids=initialization.route_roadblock_ids,
            map_api=initialization.map_api,
            mission_goal=initialization.mission_goal,
            traffic_light_status=current_input.traffic_light_data,
            inference=True,
        )






    def _build_feature(
        self,
        present_idx: int,
        ego_state_list: List[EgoState],
        ego_state_list_29: List[EgoState],
        ego_state_list_28: List[EgoState],
        ego_state_list_27: List[EgoState],
        ego_state_list_26: List[EgoState],
        ego_state_list_25: List[EgoState],
        ego_state_list_24: List[EgoState],
        
        tracked_objects_list: List[TrackedObjects],
        tracked_objects_list_29: List[TrackedObjects],
        tracked_objects_list_28: List[TrackedObjects],
        tracked_objects_list_27: List[TrackedObjects],
        tracked_objects_list_26: List[TrackedObjects],
        tracked_objects_list_25: List[TrackedObjects],
        tracked_objects_list_24: List[TrackedObjects],
        
        # route_roadblocks_ids: list[int],
        route_roadblocks_ids: dict,  # 这里定义为字典类型
  
        map_api: AbstractMap,
        mission_goal: StateSE2,
        
        traffic_light_status: List[TrafficLightStatusData] = None,
        traffic_light_status_29: List[TrafficLightStatusData] = None,
        traffic_light_status_28: List[TrafficLightStatusData] = None,
        traffic_light_status_27: List[TrafficLightStatusData] = None,
        traffic_light_status_26: List[TrafficLightStatusData] = None,
        traffic_light_status_25: List[TrafficLightStatusData] = None,
        traffic_light_status_24: List[TrafficLightStatusData] = None,
        
        inference: bool = False,
    ):
        
        
        
        ego_state_lists = {
            29: ego_state_list_29,
            28: ego_state_list_28,
            27: ego_state_list_27,
            26: ego_state_list_26,
            25: ego_state_list_25,
            24: ego_state_list_24,
        }
        
        
        tracked_objects_lists={
            29: tracked_objects_list_29,
            28: tracked_objects_list_28,
            27: tracked_objects_list_27,
            26: tracked_objects_list_26,
            25: tracked_objects_list_25,
            24: tracked_objects_list_24,
        }
        # print(f"这个track——29的类型是 {type(tracked_objects_list_29)}")#已经是list了
        
        traffic_light_status_lists = {
            29: traffic_light_status_29,
            28: traffic_light_status_28,
            27: traffic_light_status_27,
            26: traffic_light_status_26,
            25: traffic_light_status_25,
            24: traffic_light_status_24,
        }
        
        if present_idx < 0:
            present_idx = len(ego_state_list) + present_idx

        present_ego_state = ego_state_list[present_idx]#20帧
        query_xy = present_ego_state.center
        traffic_light_status = list(traffic_light_status)  # note: tl is a iterator
        scenario_manager={}
        # route_roadblock_ids={}本身初始化就为字典了

        route_reference_path={}
        if self.scenario_manager["cur"] is None:
            scenario_manager["cur"] = ScenarioManager(
                map_api,
                present_ego_state,
                route_roadblocks_ids["cur"],
                radius=50,
            )
            scenario_manager["cur"].update_ego_state(present_ego_state)
            scenario_manager["cur"].update_drivable_area_map()
        else:
            scenario_manager["cur"] = self.scenario_manager["cur"]
            
        route_roadblocks_ids["cur"] = scenario_manager["cur"].get_route_roadblock_ids()
        route_reference_path["cur"] = scenario_manager["cur"].update_ego_path()
        
        # print("注意啦注意啦注意啦！！！cur信息")
        # print(type(tracked_objects_list[present_idx]))

        # (wrapped_fn pid=542627) 注意啦注意啦注意啦！！！ [repeated 3x across cluster]
        # (wrapped_fn pid=542627) <class 'nuplan.common.actor_state.tracked_objects.TrackedObjects'> [repeated 3x across cluster]
        
        scenario_manager["cur"].update_obstacle_map(
            tracked_objects_list[present_idx], traffic_light_status
        )
        
    
        for i in range(1, 7): 
            ego_state_listtt = ego_state_lists[30 - i]
            tracked_objects_listtt = tracked_objects_lists[30 - i]
            traffic_light_statusss= traffic_light_status_lists[30 - i]#已经是List了
            # print(f"注意注意注意type of loights is {type(traffic_light_status_lists[30 - i])}")
            # traffic_light_statusss=list(traffic_light_statusss)
            present_ego_statee= ego_state_listtt[present_idx]
            if self.scenario_manager[f"cur-{i}"] is None:
                scenario_manager[f"cur-{i}"] = ScenarioManager(
                    map_api,
                    present_ego_statee,
                    route_roadblocks_ids[f"cur-{i}"],
                    radius=50,
                )
                scenario_manager[f"cur-{i}"].update_ego_state(present_ego_statee)
                scenario_manager[f"cur-{i}"].update_drivable_area_map()
            else:
                scenario_manager[f"cur-{i}"] = self.scenario_manager[f"cur-{i}"]
                
            route_roadblocks_ids[f"cur-{i}"] = scenario_manager[f"cur-{i}"].get_route_roadblock_ids()
            route_reference_path[f"cur-{i}"] = scenario_manager[f"cur-{i}"].update_ego_path()
            scenario_manager[f"cur-{i}"].update_obstacle_map(
                tracked_objects_listtt[present_idx], traffic_light_statusss
            )# 从29帧到24帧  30其实是31帧； 29其实是30帧
        
        #TODO

        ego_features = {}
        agent_features={}
        agent_tokens={}
        agents_polygon={}
        map_polygon_tokens={}
#自车状态  
        #还是在最开始增加一个维度
        data = {}
        data["cur"]={}
        data["cur-1"]={}
        data["cur-2"]={}
        data["cur-3"]={}
        data["cur-4"]={}
        data["cur-5"]={}
        data["cur-6"]={}
        data["cur"]["agent"] = {}
        data["cur-1"]["agent"]={}
        data["cur-2"]["agent"]={}
        data["cur-3"]["agent"]={}
        data["cur-4"]["agent"]={}
        data["cur-5"]["agent"]={}
        data["cur-6"]["agent"]={}
        
        data["cur"]["current_state"] = self._get_ego_current_state(
            ego_state_list[present_idx], ego_state_list[present_idx - 1]
        )
        
        # present_idx=30
        ego_features["cur"]=self._get_ego_features(ego_states=ego_state_list)
        for i in range(1, 7):  # 从29帧到24帧  30其实是31帧； 29其实是30帧
            
            # print(f"注意啦注意啦注意啦！！！ cur - {i}  信息")
            
            tracked_objects_listtt=tracked_objects_lists[30 - i]
            ego_state_listtt = ego_state_lists[30 - i]  
            present_ego_statee= ego_state_listtt[present_idx]
            traffic_light_statusss= traffic_light_status_lists[30 - i]
            traffic_light_statusss=list(traffic_light_statusss)
            data[f"cur-{i}"]["current_state"] = self._get_ego_current_state(ego_state_listtt[present_idx ], ego_state_listtt[present_idx  - 1])
            ego_features[f"cur-{i}"] = self._get_ego_features(ego_states=ego_state_listtt)
            agent_features[f"cur-{i}"], agent_tokens[f"cur-{i}"], agents_polygon[f"cur-{i}"] = self._get_agent_features(
            query_xy=ego_state_listtt[present_idx].center,
            present_idx=present_idx,
            tracked_objects_list=tracked_objects_listtt,
            ) 
            data[f"cur-{i}"]["static_objects"] = self._get_static_objects_features(present_ego_statee, scenario_manager[f"cur-{i}"], tracked_objects_listtt[present_idx])

            for k in agent_features[f"cur-{i}"].keys():
                data[f"cur-{i}"]["agent"][k]  = np.concatenate(
                    [ego_features[f"cur-{i}"][k] [None, ...], agent_features[f"cur-{i}"][k]], axis=0
                )
            agent_tokens[f"cur-{i}"] = ["ego"] + agent_tokens[f"cur-{i}"]
            
            
            data[f"cur-{i}"]["map"], map_polygon_tokens[f"cur-{i}"] = self._get_map_features(
                map_api=map_api,
                query_xy=present_ego_statee.center,
                route_roadblock_ids=route_roadblocks_ids[f"cur-{i}"],
                traffic_light_status=traffic_light_statusss,
                radius=self.radius,
            )
            
            
        agent_features["cur"], agent_tokens["cur"], agents_polygon["cur"] = self._get_agent_features(
            query_xy=query_xy,
            present_idx=present_idx,
            tracked_objects_list=tracked_objects_list,
        )

        
        data["cur"]["agent"] = {}
        for k in agent_features["cur"].keys():
            data["cur"]["agent"][k] = np.concatenate(
                [ego_features["cur"][k][None, ...], agent_features["cur"][k]], axis=0
            )
        agent_tokens["cur"] = ["ego"] + agent_tokens["cur"]
        

        #是否推理模式：
        if inference:
            data["cur"]["agent_tokens"] = agent_tokens["cur"]

        #上面循环获得
        
        data["cur"]["static_objects"] = self._get_static_objects_features(
            present_ego_state, scenario_manager["cur"], tracked_objects_list[present_idx]
        )
    
        
        data["cur"]["map"], map_polygon_tokens["cur"] = self._get_map_features(
            map_api=map_api,
            query_xy=query_xy,
            route_roadblock_ids=route_roadblocks_ids["cur"],
            traffic_light_status=traffic_light_status,
            radius=self.radius,
        )


        #非推理  即 训练模式
        if not inference:
            data["cur"]["causal"] = self.scenario_casual_reasoning_preprocess(
                present_ego_state,
                scenario_manager["cur"],
                agent_tokens["cur"],
                map_polygon_tokens["cur"],
                ego_state_list[self.history_samples + 1 :],
            )
            data["cur"]["causal"]["interaction_label"] = self._get_interaction_label(
                ego_features["cur"], agent_features["cur"]
            )
            data["cur"]["agent"]["valid_mask"][0, self.history_samples + 1 :] = data["cur"]["causal"][ "fixed_ego_future_valid_mask"]
            
            
            cost_map_manager= CostMapManager(
                origin=present_ego_state.rear_axle.array,
                angle=present_ego_state.rear_axle.heading,
                height=600,
                width=600,
                resolution=0.2,
                map_api=map_api,
            )
            cost_maps = cost_map_manager.build_cost_maps(
                static_objects=tracked_objects_list[present_idx].get_static_objects(),
                agents=agent_features["cur"],
                agents_polygon=agents_polygon["cur"],
                route_roadblock_ids=set(route_roadblocks_ids["cur"]),
            )
            data["cur"]["cost_maps"] = cost_maps["cost_maps"]
            

            for i in range(1, 7):     
            # for i in range(1, 7):  # 从29帧到24帧  30其实是31帧； 29其实是30帧  
                ego_state_listtt = ego_state_lists[30 - i]
                present_ego_statee=ego_state_listtt[present_idx]
                tracked_objects_listtt=tracked_objects_lists[30 - i]
                
                data[f"cur-{i}"]["causal"] = self.scenario_casual_reasoning_preprocess(
                present_ego_statee,
                scenario_manager[f"cur-{i}"],
                agent_tokens[f"cur-{i}"],
                map_polygon_tokens[f"cur-{i}"],
                ego_state_listtt[self.history_samples + 1 :],
                )
                
                
                data[f"cur-{i}"]["causal"]["interaction_label"] = self._get_interaction_label(
                ego_features[f"cur-{i}"], agent_features[f"cur-{i}"]
                )
                data[f"cur-{i}"]["agent"]["valid_mask"][0, self.history_samples + 1 :] = data[f"cur-{i}"]["causal"][ "fixed_ego_future_valid_mask"]
                
                cost_map_manager = CostMapManager(
                    origin=present_ego_statee.rear_axle.array,
                    angle=present_ego_statee.rear_axle.heading,
                    height=600,
                    width=600,
                    resolution=0.2,
                    map_api=map_api,
                )
                cost_maps = cost_map_manager.build_cost_maps(
                    static_objects=tracked_objects_listtt[present_idx].get_static_objects(),
                    agents=agent_features[f"cur-{i}"],
                    agents_polygon=agents_polygon[f"cur-{i}"],
                    route_roadblock_ids=set(route_roadblocks_ids[f"cur-{i}"]),
                )
                data[f"cur-{i}"]["cost_maps"] = cost_maps["cost_maps"]

        if self.build_reference_line:
            data["cur"]["reference_line"] = self._get_reference_line_feature(
                scenario_manager["cur"], ego_features["cur"]
            )

            for i in range(1, 7): 
                data[f"cur-{i}"] ["reference_line"]= self._get_reference_line_feature(
                    scenario_manager[f"cur-{i}"], ego_features[f"cur-{i}"]
                    
                )


        return PlutoFeature.normalize(data, first_time=True, radius=self.radius)

    def scenario_casual_reasoning_preprocess(
        self,
        ego_state: EgoState,
        scenario_manager: ScenarioManager,
        agents_tokens: List[str],
        map_polygon_tokens: List[int],
        ego_future_trajectory: List[EgoState] = None,
    ):
        is_waiting_for_red_light_without_lead = False
        leading_agent_mask = np.zeros(len(agents_tokens), dtype=bool)
        leading_distance = np.zeros(len(agents_tokens), dtype=np.float64)
        ego_care_red_light_mask = np.zeros(len(map_polygon_tokens), dtype=bool)
        fixed_ego_future_valid_mask = np.ones(len(ego_future_trajectory), dtype=bool)
        free_path_points = np.array([], dtype=np.float64)

        leading_objects = scenario_manager.get_leading_objects()
        nearest_leading_agent_idx = None
        nearest_leading_red_light = None
        nearest_leading_red_light_distance = None

        if (
            len(leading_objects) > 0
            and leading_objects[0][1] == OccupancyType.RED_LIGHT
        ):
            is_waiting_for_red_light_without_lead = True

        for leading_object in leading_objects:
            token, occupancy_type, distance = leading_object
            if occupancy_type == OccupancyType.DYNAMIC:
                try:
                    idx = agents_tokens.index(token)
                except ValueError:
                    continue
                if nearest_leading_agent_idx is None:
                    nearest_leading_agent_idx = idx
                leading_agent_mask[idx] = True
                leading_distance[idx] = distance
            if occupancy_type == OccupancyType.RED_LIGHT:
                idx = map_polygon_tokens.index(token)
                ego_care_red_light_mask[idx] = True
                if nearest_leading_red_light is None:
                    nearest_leading_red_light = scenario_manager.get_occupancy_object(
                        token
                    )
                    nearest_leading_red_light_distance = distance

        if nearest_leading_red_light is not None:
            for i, state in enumerate(ego_future_trajectory):
                if nearest_leading_red_light.contains(Point(*state.center.array)):
                    fixed_ego_future_valid_mask[i:] = False
                    break

        ego_velocity = ego_state.dynamic_car_state.speed
        free_path_start = ego_velocity**2 / (2 * 5) + self.ego_params.length / 2
        free_path_end = max(7, ego_velocity**2 / (2 * 1.5))
        if nearest_leading_agent_idx is not None:
            free_path_end = leading_distance[nearest_leading_agent_idx]
        if nearest_leading_red_light_distance is not None:
            free_path_end = min(free_path_end, nearest_leading_red_light_distance)
        free_path_points = scenario_manager.get_ego_path_points(
            free_path_start + 3, free_path_end - 3
        )

        return {
            "is_waiting_for_red_light_without_lead": is_waiting_for_red_light_without_lead,
            "leading_agent_mask": leading_agent_mask,
            "leading_distance": leading_distance,
            "ego_care_red_light_mask": ego_care_red_light_mask,
            "fixed_ego_future_valid_mask": fixed_ego_future_valid_mask,
            "free_path_points": free_path_points,
        }

    def _get_ego_current_state(self, ego_state: EgoState, prev_state: EgoState):
        state = np.zeros(7, dtype=np.float64)
        state[0:2] = ego_state.rear_axle.array
        state[2] = ego_state.rear_axle.heading
        state[3] = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        state[4] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x

        if self.simulation:
            steering_angle, yaw_rate = (
                ego_state.tire_steering_angle,
                ego_state.dynamic_car_state.angular_velocity,
            )
        else:
            steering_angle, yaw_rate = self.calculate_additional_ego_states(
                ego_state, prev_state
            )

        state[5] = steering_angle
        state[6] = yaw_rate

        return state

    def _get_ego_features(self, ego_states: List[EgoState]):
        """note that rear axle velocity and acceleration are in ego local frame,
        and need to be transformed to the global frame.
        """
        T = len(ego_states)

        position = np.zeros((T, 2), dtype=np.float64)
        heading = np.zeros((T), dtype=np.float64)
        velocity = np.zeros((T, 2), dtype=np.float64)
        acceleration = np.zeros((T, 2), dtype=np.float64)
        shape = np.zeros((T, 2), dtype=np.float64)
        valid_mask = np.ones(T, dtype=np.bool_)

        for t, state in enumerate(ego_states):
            position[t] = state.rear_axle.array
            heading[t] = state.rear_axle.heading
            velocity[t] = common.rotate_round_z_axis(
                state.dynamic_car_state.rear_axle_velocity_2d.array,
                -state.rear_axle.heading,
            )
            acceleration[t] = common.rotate_round_z_axis(
                state.dynamic_car_state.rear_axle_acceleration_2d.array,
                -state.rear_axle.heading,
            )
            shape[t] = np.array([self.width, self.length])

        category = np.array(
            self.interested_objects_types.index(TrackedObjectType.EGO), dtype=np.int8
        )

        return {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "acceleration": acceleration,
            "shape": shape,
            "category": category,
            "valid_mask": valid_mask,
        }

    def _get_agent_features(
        self,
        query_xy: Point2D,
        present_idx: int,
        tracked_objects_list: List[TrackedObjects],
    ):
        present_tracked_objects = tracked_objects_list[present_idx]
        present_agents = present_tracked_objects.get_tracked_objects_of_types(
            self.interested_objects_types
        )
        N, T = min(len(present_agents), self.max_agents), len(tracked_objects_list)

        position = np.zeros((N, T, 2), dtype=np.float64)
        heading = np.zeros((N, T), dtype=np.float64)
        velocity = np.zeros((N, T, 2), dtype=np.float64)
        shape = np.zeros((N, T, 2), dtype=np.float64)
        category = np.zeros((N,), dtype=np.int8)
        valid_mask = np.zeros((N, T), dtype=np.bool_)
        polygon = [None] * N

        if N == 0 or self.disable_agent:
            return (
                {
                    "position": position,
                    "heading": heading,
                    "velocity": velocity,
                    "shape": shape,
                    "category": category,
                    "valid_mask": valid_mask,
                },
                [],
                [],
            )

        agent_ids = np.array([agent.track_token for agent in present_agents])
        agent_cur_pos = np.array([agent.center.array for agent in present_agents])
        distance = np.linalg.norm(agent_cur_pos - query_xy.array[None, :], axis=1)
        agent_ids_sorted = agent_ids[np.argsort(distance)[: self.max_agents]]
        agent_ids_dict = {agent_id: i for i, agent_id in enumerate(agent_ids_sorted)}

        for t, tracked_objects in enumerate(tracked_objects_list):
            for agent in tracked_objects.get_tracked_objects_of_types(
                self.interested_objects_types
            ):
                if agent.track_token not in agent_ids_dict:
                    continue

                idx = agent_ids_dict[agent.track_token]
                position[idx, t] = agent.center.array
                heading[idx, t] = agent.center.heading
                velocity[idx, t] = agent.velocity.array
                shape[idx, t] = np.array([agent.box.width, agent.box.length])
                valid_mask[idx, t] = True

                if t == present_idx:
                    category[idx] = self.interested_objects_types.index(
                        agent.tracked_object_type
                    )
                    polygon[idx] = agent.box.geometry

        agent_features = {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "shape": shape,
            "category": category,
            "valid_mask": valid_mask,
        }

        return agent_features, list(agent_ids_sorted), polygon

    def _get_static_objects_features(
        self,
        ego_state: EgoState,
        scenario_manager: ScenarioManager,
        tracked_objects_list: TrackedObjects,
    ):
        static_objects = []

        # only cares objects that are in drivable area
        for obj in tracked_objects_list.get_static_objects():
            if np.linalg.norm(ego_state.center.array - obj.center.array) > self.radius:
                continue
            if not scenario_manager.object_in_drivable_area(obj.box.geometry):
                continue
            static_objects.append(
                np.concatenate(
                    [
                        obj.center.array,
                        [obj.center.heading],
                        [obj.box.width, obj.box.length],
                        [self.static_objects_types.index(obj.tracked_object_type)],
                    ],
                    axis=-1,
                    dtype=np.float64,
                )
            )

        if len(static_objects) > 0:
            static_objects = np.stack(static_objects, axis=0)
            valid_mask = np.ones(len(static_objects), dtype=np.bool_)
        else:
            static_objects = np.zeros((0, 6), dtype=np.float64)
            valid_mask = np.zeros(0, dtype=np.bool_)

        return {
            "position": static_objects[:, :2],
            "heading": static_objects[:, 2],
            "shape": static_objects[:, 3:5],
            "category": static_objects[:, -1],
            "valid_mask": valid_mask,
        }

    def _get_map_features(
        self,
        map_api: AbstractMap,
        query_xy: Point2D,
        route_roadblock_ids: List[str],
        traffic_light_status: List[TrafficLightStatusData],
        radius: float,
        sample_points: int = 20,
    ):
        route_ids = set(int(route_id) for route_id in route_roadblock_ids)
        tls = {tl.lane_connector_id: tl.status for tl in traffic_light_status}

        map_objects = map_api.get_proximal_map_objects(
            query_xy,
            radius,
            [
                SemanticMapLayer.LANE,
                SemanticMapLayer.LANE_CONNECTOR,
                SemanticMapLayer.CROSSWALK,
            ],
        )
        lane_objects = (
            map_objects[SemanticMapLayer.LANE]
            + map_objects[SemanticMapLayer.LANE_CONNECTOR]
        )
        crosswalk_objects = map_objects[SemanticMapLayer.CROSSWALK]

        object_ids = [int(obj.id) for obj in lane_objects + crosswalk_objects]
        object_types = (
            [SemanticMapLayer.LANE] * len(map_objects[SemanticMapLayer.LANE])
            + [SemanticMapLayer.LANE_CONNECTOR]
            * len(map_objects[SemanticMapLayer.LANE_CONNECTOR])
            + [SemanticMapLayer.CROSSWALK]
            * len(map_objects[SemanticMapLayer.CROSSWALK])
        )

        M, P = len(lane_objects) + len(crosswalk_objects), sample_points
        point_position = np.zeros((M, 3, P, 2), dtype=np.float64)
        point_vector = np.zeros((M, 3, P, 2), dtype=np.float64)
        point_side = np.zeros((M, 3), dtype=np.int8)
        point_orientation = np.zeros((M, 3, P), dtype=np.float64)
        polygon_center = np.zeros((M, 3), dtype=np.float64)
        polygon_position = np.zeros((M, 2), dtype=np.float64)
        polygon_orientation = np.zeros(M, dtype=np.float64)
        polygon_type = np.zeros(M, dtype=np.int8)
        polygon_on_route = np.zeros(M, dtype=np.bool_)
        polygon_tl_status = np.zeros(M, dtype=np.int8)
        polygon_speed_limit = np.zeros(M, dtype=np.float64)
        polygon_has_speed_limit = np.zeros(M, dtype=np.bool_)
        polygon_road_block_id = np.zeros(M, dtype=np.int32)

        for lane in lane_objects:
            object_id = int(lane.id)
            idx = object_ids.index(object_id)
            speed_limit = lane.speed_limit_mps

            centerline = self._sample_discrete_path(
                lane.baseline_path.discrete_path, sample_points + 1
            )
            left_bound = self._sample_discrete_path(
                lane.left_boundary.discrete_path, sample_points + 1
            )
            right_bound = self._sample_discrete_path(
                lane.right_boundary.discrete_path, sample_points + 1
            )
            edges = np.stack([centerline, left_bound, right_bound], axis=0)

            point_vector[idx] = edges[:, 1:] - edges[:, :-1]
            point_position[idx] = edges[:, :-1]
            point_orientation[idx] = np.arctan2(
                point_vector[idx, :, :, 1], point_vector[idx, :, :, 0]
            )
            point_side[idx] = np.arange(3)

            polygon_center[idx] = np.concatenate(
                [
                    centerline[int(sample_points / 2)],
                    [point_orientation[idx, 0, int(sample_points / 2)]],
                ],
                axis=-1,
            )
            polygon_position[idx] = centerline[0]
            polygon_orientation[idx] = point_orientation[idx, 0, 0]
            polygon_type[idx] = self.polygon_types.index(object_types[idx])
            polygon_on_route[idx] = int(lane.get_roadblock_id()) in route_ids
            polygon_tl_status[idx] = (
                tls[object_id] if object_id in tls else TrafficLightStatusType.UNKNOWN
            )
            polygon_has_speed_limit[idx] = speed_limit is not None
            polygon_speed_limit[idx] = (
                lane.speed_limit_mps if lane.speed_limit_mps else 0
            )
            polygon_road_block_id[idx] = int(lane.get_roadblock_id())

        for crosswalk in crosswalk_objects:
            idx = object_ids.index(int(crosswalk.id))
            edges = self._get_crosswalk_edges(crosswalk)
            point_vector[idx] = edges[:, 1:] - edges[:, :-1]
            point_position[idx] = edges[:, :-1]
            point_orientation[idx] = np.arctan2(
                point_vector[idx, :, :, 1], point_vector[idx, :, :, 0]
            )
            point_side[idx] = np.arange(3)
            polygon_center[idx] = np.concatenate(
                [
                    edges[0, int(sample_points / 2)],
                    [point_orientation[idx, 0, int(sample_points / 2)]],
                ],
                axis=-1,
            )
            polygon_position[idx] = edges[0, 0]
            polygon_orientation[idx] = point_orientation[idx, 0, 0]
            polygon_type[idx] = self.polygon_types.index(object_types[idx])
            polygon_on_route[idx] = False
            polygon_tl_status[idx] = TrafficLightStatusType.UNKNOWN
            polygon_has_speed_limit[idx] = False

        map_features = {
            "point_position": point_position,
            "point_vector": point_vector,
            "point_orientation": point_orientation,
            "point_side": point_side,
            "polygon_center": polygon_center,
            "polygon_position": polygon_position,
            "polygon_orientation": polygon_orientation,
            "polygon_type": polygon_type,
            "polygon_on_route": polygon_on_route,
            "polygon_tl_status": polygon_tl_status,
            "polygon_has_speed_limit": polygon_has_speed_limit,
            "polygon_speed_limit": polygon_speed_limit,
            "polygon_road_block_id": polygon_road_block_id,
        }

        return map_features, object_ids

    def _get_reference_line_feature(
        self, scenario_manager: ScenarioManager, ego_features
    ):
        reference_lines = scenario_manager.get_reference_lines(length=self.radius)
        
        if reference_lines is None or len(reference_lines) == 0:
            print("注意注意注意注意注意注意   Warning: reference_lines is empty!")

        
        

        n_points = int(self.radius / 1.0)  #所以有120个点
        
        position = np.zeros((len(reference_lines), n_points, 2), dtype=np.float64)
        vector = np.zeros((len(reference_lines), n_points, 2), dtype=np.float64)
        orientation = np.zeros((len(reference_lines), n_points), dtype=np.float64)
        valid_mask = np.zeros((len(reference_lines), n_points), dtype=np.bool_)
        future_projection = np.zeros((len(reference_lines), 8, 2), dtype=np.float64)

        ego_future = ego_features["position"][self.history_samples + 1 :]
        if len(ego_future) > 0:
            linestring = [
                LineString(reference_lines[i]) for i in range(len(reference_lines))
            ]
            future_samples = ego_future[9::10]  # every 1s
            future_samples = [Point(xy) for xy in future_samples]

        
        
        for i, line in enumerate(reference_lines):
            subsample = line[::4][: n_points + 1]  #参考线点进行下采样（每 4 个点取 1 个），限制 最多 n_points + 1 个点。
            #就是这个n_valid不能等于0，否则全部为false,取反过后就是全部为true
            
            
            # subsample = line[:: max(1, len(line) // (n_points + 1))][: n_points + 1]
            
            n_valid = len(subsample)
            position[i, : n_valid - 1] = subsample[:-1, :2]
            vector[i, : n_valid - 1] = np.diff(subsample[:, :2], axis=0)
            orientation[i, : n_valid - 1] = subsample[:-1, 2]
            valid_mask[i, : n_valid - 1] = True

            if len(ego_future) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for j, future_sample in enumerate(future_samples):
                        future_projection[i, j, 0] = linestring[i].project(
                            future_sample
                        )
                        future_projection[i, j, 1] = linestring[i].distance(
                            future_sample
                        )

        return {
            "position": position,
            "vector": vector,
            "orientation": orientation,
            "valid_mask": valid_mask,
            "future_projection": future_projection,
        }


    def _sample_discrete_path(self, discrete_path: List[StateSE2], num_points: int):
        path = np.stack([point.array for point in discrete_path], axis=0)
        return common.interpolate_polyline(path, num_points)

    def _get_crosswalk_edges(
        self, crosswalk: PolygonMapObject, sample_points: int = 21
    ):
        bbox = shapely.minimum_rotated_rectangle(crosswalk.polygon)
        coords = np.stack(bbox.exterior.coords.xy, axis=-1)
        edge1 = coords[[3, 0]]  # right boundary
        edge2 = coords[[2, 1]]  # left boundary

        edges = np.stack([(edge1 + edge2) * 0.5, edge2, edge1], axis=0)  # [3, 2, 2]
        vector = edges[:, 1] - edges[:, 0]  # [3, 2]
        steps = np.linspace(0, 1, sample_points, endpoint=True)[None, :]
        points = edges[:, 0][:, None, :] + vector[:, None, :] * steps[:, :, None]

        return points

    def _get_interaction_label(self, ego, agents):
        ego_heading = ego["heading"][self.history_samples + 1 :]
        ego_position = ego["position"][self.history_samples + 1 :]
        agents_shape = agents["shape"][:, self.history_samples + 1 :]
        agents_heading = agents["heading"][:, self.history_samples + 1 :]
        agents_position = agents["position"][:, self.history_samples + 1 :]

        if agents_position.shape[0] == 0 or agents_position.shape[1] == 0:
            return np.zeros(1)

        N, T = agents_position.shape[:2]
        agents_invalid_mask = ~torch.from_numpy(
            agents["valid_mask"][:, self.history_samples + 1 :]
        )
        agents_invalid_mask = (
            agents_invalid_mask.unsqueeze(-1).repeat(1, 1, T).reshape(N, -1)
        )

        cdist = torch.cdist(
            torch.from_numpy(agents_position).reshape(-1, 2),
            torch.from_numpy(ego_position).reshape(-1, 2),
        ).reshape(N, -1)

        cdist[agents_invalid_mask] = 1e6
        min_dist, index = cdist.min(dim=-1)
        interact_flag = min_dist < 4  # coarse judgement

        for i in torch.arange(N)[interact_flag]:
            agent_t, ego_t = index[i].item() // T, index[i] % T
            agent_shape = agents_shape[i, agent_t]
            agent_box = OrientedBox(
                center=StateSE2(
                    agents_position[i, agent_t, 0],
                    agents_position[i, agent_t, 1],
                    agents_heading[i, agent_t],
                ),
                width=agent_shape[0],
                length=agent_shape[1],
                height=0.0,
            )
            ego_box = self._build_ego_bbox(ego_position[ego_t], ego_heading[ego_t])

            if not in_collision(agent_box, ego_box):
                interact_flag[i] = False

        interact_label = index.apply_(self._get_interact_type)
        interact_label[~interact_flag] = 0
        interact_label = np.concatenate([np.zeros(1), interact_label])

        return interact_label

    @staticmethod
    def _get_interact_type(index, T=80):
        row, col = index // T, index % T
        if row == col:
            return 0  # collision or self
        return col - row

    def _build_ego_bbox(self, xy, angle):
        center = xy + 1.67 * np.array([np.cos(angle), np.sin(angle)])
        return OrientedBox(
            center=StateSE2(center[0], center[1], angle),
            width=self.width,
            length=self.length,
            height=0.0,
        )

    def _get_ego_head_position(self, xy, angle):
        return xy + self.length * np.array([np.cos(angle), np.sin(angle)]) / 2

    def calculate_additional_ego_states(
        self, current_state: EgoState, prev_state: EgoState, dt=0.1
    ):
        cur_velocity = current_state.dynamic_car_state.rear_axle_velocity_2d.x
        angle_diff = current_state.rear_axle.heading - prev_state.rear_axle.heading
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        yaw_rate = angle_diff / dt

        if abs(cur_velocity) < 0.2:
            return 0.0, 0.0  # if the car is almost stopped, the yaw rate is unreliable
        else:
            steering_angle = np.arctan(
                yaw_rate * self.ego_params.wheel_base / abs(cur_velocity)
            )
            steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
            yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

            return steering_angle, yaw_rate
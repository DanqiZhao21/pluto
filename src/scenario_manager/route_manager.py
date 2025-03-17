import warnings
from abc import ABC
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.lane import Lane
from nuplan.common.maps.nuplan_map.lane_connector import LaneConnector
from nuplan.planning.simulation.observation.idm.utils import (
    create_path_from_se2,
    path_to_linestring,
)
from nuplan.planning.simulation.path.utils import trim_path
from shapely.geometry import Point

from .occupancy_map import OccupancyMap
from .utils.dijkstra import Dijkstra
from .utils.route_utils import normalize_angle, route_roadblock_correction


class RouteManager:
    def __init__(
        self,
        map_api: AbstractMap,
        route_roadblock_ids: List[str],
        map_radius=50,
    ) -> None:
        self._route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject] = None
        self._route_lane_dict: Dict[str, LaneGraphEdgeMapObject] = None
        self._map_api = map_api
        self._map_radius = map_radius
        self._drivable_area_map: OccupancyMap = None

        self._origin_route_roadblock_ids = route_roadblock_ids
        self.reference_lines = None
        self.route_roadblock_ids = None

        self.initialized = False

    @property
    def route_lane_ids(self) -> set[str]:
        return self._route_lane_dict.keys()

    def load_route(self, ego_state: EgoState, process=True) -> None:
        """
        Loads route dictionaries from map-api.
        :param route_roadblock_ids: ID's of on-route roadblocks
        """
        if process:
            updated_route_roadblock_ids = route_roadblock_correction(
                ego_state, self._map_api, self._origin_route_roadblock_ids
            )
        else:
            updated_route_roadblock_ids = self._origin_route_roadblock_ids

        # remove repeated ids while remaining order in list
        updated_route_roadblock_ids = list(dict.fromkeys(updated_route_roadblock_ids))

        self._route_roadblock_dict = {}
        self._route_lane_dict = {}

        for id_ in updated_route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )

            self._route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                self._route_lane_dict[lane.id] = lane

        self.route_roadblock_ids = updated_route_roadblock_ids
        self.initialized = True

    def update_drivable_area_map(self, da: OccupancyMap):
        self._drivable_area_map = da

    def get_ego_path(self, ego_state: EgoState, search_depth=15):
        current_lane = self._get_starting_lane(ego_state)
        roadblocks = list(self._route_roadblock_dict.values())
        roadblock_ids = list(self._route_roadblock_dict.keys())

        # find current roadblock index
        start_idx = np.argmax(
            np.array(roadblock_ids) == current_lane.get_roadblock_id()
        )
        roadblock_window = roadblocks[start_idx : start_idx + search_depth]

        graph_search = Dijkstra(current_lane, list(self._route_lane_dict.keys()))
        route_plan, path_found = graph_search.search(roadblock_window[-1])

        centerline_discrete_path: List[StateSE2] = []
        for lane in route_plan:
            centerline_discrete_path.extend(lane.baseline_path.discrete_path)

        return centerline_discrete_path

    def get_reference_lines(self, ego_state: EgoState, interval=1.0, length=100):
        discrete_paths = []
        #已经排除，不可能为空；
        # if ego_state is None:
        #     print("🚨 Error: ego_state is None! ")

        
        #检查candidate_lines###############################################################
        candidate_lanes = self._get_candidate_starting_lane(ego_state)
        # print(f"线线线线线线线Candidate starting lanes: {candidate_lanes}")
        # if not candidate_lanes:
            # print("#########################没有candidateline#####################")
            # assert candidate_lanes, "No candidate lanes found!"
        ##################################################################################
        

        # for lane in self._get_candidate_starting_lane(ego_state):
        #     discrete_paths.extend(
        #         self.find_all_candidate_routes(ego_state, lane, maximum_length=length)
        #     )

        ##########################################检查discrete_paths
        for lane in candidate_lanes:
            paths = self.find_all_candidate_routes(ego_state, lane, maximum_length=length)
            # print(f"线线线线线线线Found {len(paths)} paths for lane {lane}")
            discrete_paths.extend(paths)

        # if not discrete_paths:
            # print(" 没有 discrete_paths.")

        ##################################################################################
        
        
        trimmed_paths, trimmed_path_length = [], []
        for discrete_path in discrete_paths:
            path, path_len = self._trim_discrete_path(ego_state, discrete_path, length)
            trimmed_paths.append(path)
            trimmed_path_length.append(path_len)

        # print(f"线线线线线线线Trimmed paths before filtering: {len(trimmed_paths)}")
        
        length_mask = np.array(trimmed_path_length) > 0.8 * length
        # print(f"线线线线线线线Length mask: {length_mask}")

        if length_mask.any() and not length_mask.all():
            trimmed_paths = [trimmed_paths[i] for i in np.where(length_mask)[0]]
            # print(f"线线线线线线线Trimmed paths after length filtering: {len(trimmed_paths)}")
        # if not trimmed_paths:
            # print("线线线线线线线All paths were filtered out due to insufficient length.")

                
        
        remove_index = set()
        for i in range(len(trimmed_paths)):
            for j in range(i + 1, len(trimmed_paths)):
                if j in remove_index:
                    continue
                min_len = min(len(trimmed_paths[i]), len(trimmed_paths[j]))
                diff = np.abs(
                    trimmed_paths[i][:min_len, :2] - trimmed_paths[j][:min_len, :2]
                ).sum(-1)
                if np.max(diff) < 0.5:
                    remove_index.add(j)

        merged_paths = [
            trimmed_paths[i] for i in range(len(trimmed_paths)) if i not in remove_index
        ]
        self.reference_lines = merged_paths

        return merged_paths

    def find_all_candidate_routes(
        self,
        ego_state: EgoState,
        lane: LaneGraphEdgeMapObject,
        maximum_length=100,
        search_depth=15,
    ):
        candidate_route = []

        def dfs_search(cur_lane: LaneGraphEdgeMapObject, visited: List, length):
            visited.append(cur_lane)
            new_length = length + cur_lane.baseline_path.length

            in_route_next_lane = [
                lane
                for lane in cur_lane.outgoing_edges
                if lane.get_roadblock_id() in self.route_roadblock_ids
            ]

            if (
                len(in_route_next_lane) == 0
                or len(visited) == search_depth
                or new_length > maximum_length
            ):
                candidate_route.append(visited)
                return

            for next_lane in in_route_next_lane:
                dfs_search(next_lane, visited.copy(), new_length)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start_progress = lane.baseline_path.linestring.project(
                Point(*ego_state.rear_axle.point.array)
            )
        init_offset = -start_progress

        dfs_search(lane, [], init_offset)

        candidate_discrete_path = []
        for lane_list in candidate_route:
            discrete_path = []
            length = init_offset
            for lane in lane_list:
                discrete_path.extend(lane.baseline_path.discrete_path)
                length += lane.baseline_path.length
            candidate_discrete_path.append(discrete_path)

        return candidate_discrete_path

    def _route_graph_search(self, lane: LaneGraphEdgeMapObject, search_depth=15):
        if lane is None:
            return None

        roadblocks = list(self._route_roadblock_dict.values())
        roadblock_ids = list(self._route_roadblock_dict.keys())

        # find current roadblock index
        start_idx = np.argmax(np.array(roadblock_ids) == lane.get_roadblock_id())
        roadblock_window = roadblocks[start_idx : start_idx + search_depth]

        graph_search = Dijkstra(lane, list(self._route_lane_dict.keys()))
        route_plan, path_found = graph_search.search(roadblock_window[-1])

        centerline_discrete_path: List[StateSE2] = []
        for lane in route_plan:
            centerline_discrete_path.extend(lane.baseline_path.discrete_path)

        return centerline_discrete_path

    def _trim_discrete_path(
        self, ego_state: EgoState, discrete_path: List[StateSE2], length=100
    ):
        if discrete_path is None:
            return None

        path = create_path_from_se2(discrete_path)
        linestring = path_to_linestring(discrete_path)
        start_progress = path.get_start_progress()
        end_progress = path.get_end_progress()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cur_progress = linestring.project(Point(*ego_state.rear_axle.point.array))

        cut_start = max(start_progress, min(cur_progress, end_progress))
        cur_end = min(cur_progress + length, end_progress)

        trimmed_path = trim_path(path, cut_start, cur_end)
        path_length = cur_end - cut_start

        np_trimmed_path = np.array([[p.x, p.y, p.heading] for p in trimmed_path])
        return np_trimmed_path, path_length

    def _get_starting_lane(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Returns the most suitable starting lane, in ego's vicinity.
        :param ego_state: state of ego-vehicle
        :return: lane object (on-route)
        """
        starting_lane: LaneGraphEdgeMapObject = None
        on_route_lanes, heading_error = self._get_intersecting_lanes(ego_state)

        if on_route_lanes:
            # 1. Option: find lanes from lane occupancy-map
            # select lane with lowest heading error
            starting_lane = on_route_lanes[np.argmin(np.abs(heading_error))]
            return starting_lane

        else:
            # 2. Option: find any intersecting or close lane on-route
            closest_distance = np.inf
            for edge in self._route_lane_dict.values():
                if edge.contains_point(ego_state.center):
                    starting_lane = edge
                    break

                distance = edge.polygon.distance(ego_state.car_footprint.geometry)
                if distance < closest_distance:
                    starting_lane = edge
                    closest_distance = distance

        return starting_lane

    def _get_intersecting_lanes(
        self, ego_state: EgoState
    ) -> Tuple[List[LaneGraphEdgeMapObject], List[float]]:
        """
        Returns on-route lanes and heading errors where ego-vehicle intersects.
        :param ego_state: state of ego-vehicle
        :return: tuple of lists with lane objects and heading errors [rad].
        """

        ego_position_array: npt.NDArray[np.float64] = ego_state.rear_axle.array
        ego_rear_axle_point: Point = Point(*ego_position_array)
        ego_heading: float = ego_state.rear_axle.heading

        intersecting_lanes = self._drivable_area_map.intersects(ego_rear_axle_point)

        on_route_lanes, on_route_heading_errors = [], []
        for lane_id in intersecting_lanes:
            if lane_id in self._route_lane_dict.keys():
                # collect baseline path as array
                lane_object = self._route_lane_dict[lane_id]
                lane_discrete_path: List[StateSE2] = (
                    lane_object.baseline_path.discrete_path
                )
                lane_state_se2_array = np.array(
                    [state.array for state in lane_discrete_path], dtype=np.float64
                )
                # calculate nearest state on baseline
                lane_distances = (
                    ego_position_array[None, ...] - lane_state_se2_array
                ) ** 2
                lane_distances = lane_distances.sum(axis=-1) ** 0.5

                # calculate heading error
                heading_error = (
                    lane_discrete_path[np.argmin(lane_distances)].heading - ego_heading
                )
                heading_error = np.abs(normalize_angle(heading_error))

                # add lane to candidates
                on_route_lanes.append(lane_object)
                on_route_heading_errors.append(heading_error)

        return on_route_lanes, on_route_heading_errors

    def _get_candidate_starting_lane(self, ego_state: EgoState):
        lanes = self._map_api.get_proximal_map_objects(
            ego_state.center.point,
            6,#原先为3m
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        )
        
        # print(f"type of lanes: {type(lanes)}")
        # print(f"获取到的车道数量first: {len(lanes)}")#len() 函数对于字典返回的是字典中 键的数量
        # print("获取到的车道数据字典:")
        # print(lanes)
        
        
        # 查看返回的车道数据
        # print(f"LANE: {lanes.get(SemanticMapLayer.LANE)}")
        # print(f"LANE_CONNECTOR: {lanes.get(SemanticMapLayer.LANE_CONNECTOR)}")
            
        # lanes = lanes[SemanticMapLayer.LANE_CONNECTOR] + lanes[SemanticMapLayer.LANE]
        
        lanes = lanes.get(SemanticMapLayer.LANE_CONNECTOR, []) + lanes.get(SemanticMapLayer.LANE, [])
        # print(f"获取到的车道数量second: {len(lanes)}")
        #逐步放低筛选条件
        # 定义多个筛选条件，逐步放低要求
        filter_conditions = [
            {"roadblock": True, "length": 2, "angle": np.pi / 2},  # 最严格的条件
            {"roadblock": False, "length": 1, "angle": np.pi / 1.5},  # 放宽条件
            {"roadblock": False, "length": 0, "angle": (np.pi)*2},  # 最宽松的条件
        ]
        
        filtered_lanes = []
        
        for condition in filter_conditions:
            # print(f"使用筛选条件: {condition}")
            
            # 筛选符合条件的车道
            lanes = [
                lane
                for lane in lanes
                if (not condition["roadblock"] or lane.get_roadblock_id() in self._route_roadblock_dict)
                and lane.baseline_path.length > condition["length"]
                and self._get_lane_angle_error(lane, ego_state) < condition["angle"]
            ]
            
            
            # 如果有车道，返回筛选结果
            if lanes:
                filtered_lanes = lanes
                break
            
        # print(f"经过筛选后的车道数量: {len(lanes)}")

        keep_ids = [lane.id for lane in lanes]

        for lane in lanes:
            for next_lane in lane.outgoing_edges:
                if next_lane.id in keep_ids:
                    keep_ids.remove(next_lane.id)
        # print(f"经过筛选后的车道id数组的长度: {len(keep_ids)}")
        merged_lanes = [self._route_lane_dict[id] for id in keep_ids]
        # print(f"最终车道数: {len(merged_lanes)}")
        return merged_lanes

    def _get_lane_angle_error(self, lane: LaneGraphEdgeMapObject, ego_state: EgoState):
        np_discrete_path = np.array(
            [[p.x, p.y, p.heading] for p in lane.baseline_path.discrete_path]
        )[::4]
        distance = np.linalg.norm(
            np_discrete_path[:, :2] - ego_state.rear_axle.array, axis=-1
        )
        closest_point = np_discrete_path[np.argmin(distance)]
        angle_error = np.abs(
            normalize_angle(closest_point[2] - ego_state.rear_axle.heading)
        )
        return angle_error

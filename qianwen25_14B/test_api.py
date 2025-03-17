import os
from openai import OpenAI

from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
import numpy as np
import json

# 配置 NuPlan 数据路径
data_root = "/mnt/data/nuplan/dataset"  
map_root = "/mnt/data/nuplan/dataset/maps"
sensor_root = "/mnt/data/nuplan/dataset/nuplan-v1.1/splits/mini"
db_files = "/mnt/data/nuplan/dataset/nuplan-v1.1/splits/mini/2021.06.09.12.39.51_veh-26_05620_06003.db" 
log_name = os.path.splitext(os.path.basename(db_files))[0]
# 加载 NuPlan 数据库
nuplan_db = NuPlanDB(
    data_root=data_root,
    maps_db=map_root,
    load_path=db_files,
    verbose=True
)


lidar_pc_map = {record.token: record for record in nuplan_db.lidar_pc}#由lidar_pc_token获得Lidar_pc对象
ego_pose_map = {record.token: record for record in nuplan_db.ego_pose}#由ego_pose_tokien获得ego_pose对象

# # 获取所有的 LiDAR 关键帧对象
lidar_pc_objects = nuplan_db.lidar_pc  

# 提取所有 `lidar_pc_token` 对应的 `ego_state`
# ego_states = []
ego_state_info={}
all_predictions = []
for lidar_pc in lidar_pc_objects:
    # 获取当前帧ego_state
    current_ego_state = ego_pose_map.get(lidar_pc.ego_pose_token)
    # ego_state=current_ego_state
    ego_state_info["current"] = {
            "timestamp": current_ego_state.timestamp,
            "x": current_ego_state.x,
            "y": current_ego_state.y,
            "z": current_ego_state.z,
            "vx": current_ego_state.vx,
            "vy":current_ego_state.vy,
            "ax": current_ego_state.acceleration_x,
            "ay":current_ego_state.acceleration_y,
            "angular_rate_x": current_ego_state.angular_rate_x,       
            "angular_rate_y":current_ego_state.angular_rate_y,   
            "angular_rate_z": current_ego_state.angular_rate_z,     
    }

    # 用于遍历过去的帧：每一步需要两次prev_token跳转
    temp_lidar = lidar_pc  # 从当前帧开始
    for i in range(1, 7):  # 生成 current-1 到 current-6
        # 第一次跳转：获取前一帧的token
        prev_token = temp_lidar.prev_token
        if not prev_token:
            break  # 如果没有前一帧，退出循环
        prev_object = lidar_pc_map.get(prev_token)
        if not prev_object:
            break

        # 第二次跳转：获取前一帧的前一帧的token
        prevprev_token = prev_object.prev_token
        if not prevprev_token:
            break
        temp_lidar = lidar_pc_map.get(prevprev_token)
        if not temp_lidar:
            break

        # 获取当前遍历帧对应的ego_pose对象
        ego_state = ego_pose_map.get(temp_lidar.ego_pose_token)
        if not ego_state:
            break

        # 保存当前历史帧的信息
        ego_state_info[f"cur-{i}"] = {
            "timestamp": ego_state.timestamp,
            "x": ego_state.x,
            "y": ego_state.y,
            "z": ego_state.z,
            "vx": ego_state.vx,
            "vy": ego_state.vy,
            "ax": ego_state.acceleration_x,
            "ay": ego_state.acceleration_y,
            "angular_rate_x": ego_state.angular_rate_x,       
            "angular_rate_y": ego_state.angular_rate_y,   
            "angular_rate_z": ego_state.angular_rate_z,               
        }
    prompt = f"""
    Below is the self-vehicle state information (ego_state) collected from an autonomous vehicle over a period of time:current,cur-1,cur-2...cur-6:
    {json.dumps(ego_state_info, indent=2)}
    Each frame is spaced 0.1 seconds apart, and the speed is in meters per second (m/s). The acceleration is in meters per second squared (m/s^2). The angular rate is in radians per second (rad/s).   
    You are now a highly skilled driver with deep understanding of vehicle dynamics and driving behavior. 

    ### Task 1: Predict the future high-level driving intent:
    - Based on the provided data, predict the vehicle's future driving actions in both the longitudinal (y-direction) and lateral (x-direction) aspects:
        - SPEED_PLAN (longitudinal direction): Options include KEEP, ACCELERATE, DECELERATE.
        - PATH_PLAN (lateral direction): Options include STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN.

    ### Task 2: Provide a concise explanation of the high-level intent:
    - Based on the given vehicle state information, explain why you think the vehicle will take the predicted actions. Your explanation should be logical and based on the vehicle's current state, speed, acceleration, and historical driving behavior.

    ### Task 3: Predict the future trajectory of the vehicle in the y-direction for the next 8 seconds:
    - For the next 8 seconds, predict the vehicle's trajectory in the y-direction, considering a maximum distance of 120 meters, with 1 interval for every 10 meters. The output should be a vector of length 10, representing the probabilities for each interval (0m-10m, 10m-20m, ..., 110m-120m).

    Please return your predictions in the following format:
    {{     
    
        "log_name": "{log_name}",
        "predictions": [
                {{
                    "lidar_pc_token": "{lidar_pc.token}",
                    "HighlevelAction": {{
                        "SPEED_PLAN": "ACCELERATE",
                        "PATH_PLAN": "STRAIGHT"
                    }},
                    "ConciseExplanation": "Given the vehicle's current velocity and acceleration, I predict that the vehicle will continue accelerating in a straight line, as it is currently and historically moving at a relatively high speed with no significant need to change direction.",
                    "M_interval": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                }},
                ...
            ]
    }}
    """

    # 配置 OpenAI API 客户端（调用 Qwen2.5）
    client = OpenAI(
        api_key="sk-a0d255cecd2546f1b423929976ac750b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 调用大模型
    completion = client.chat.completions.create(
        model="qwen2.5-32b-instruct",
        messages=[
            {'role': 'system', 'content': 'You are an expert in autonomous driving prediction.'},
            {'role': 'user', 'content': prompt}
        ]
    )
    # 获取并存储模型预测结果
    result = completion.model_dump_json()
    all_predictions.append(result)
    

# # 打印结果
# print(completion.model_dump_json())

# 将预测结果存储到 JSON 文件
output_dir = "/home/zhaodanqi/clone/pluto/qianwen25_14B/"
# 确保目标文件夹存在
os.makedirs(output_dir, exist_ok=True)

output_file_path = os.path.join(output_dir, "predictMmode.json")
with open(output_file_path, "w") as output_file:
    json.dump(all_predictions, output_file, indent=2)

print(f"All predictions have been saved to {output_file_path}")


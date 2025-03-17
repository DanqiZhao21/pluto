# import json

# # 读取文件路径
# file_path = '/home/zhaodanqi/clone/pluto/metafile/senna_conversion_train0224.json'
# fixed_json_path = '/home/zhaodanqi/clone/pluto/metafile/fixed_json.json'
# output_data_path = '/home/zhaodanqi/clone/pluto/metafile/short_json.json'
# final_json_path = '/home/zhaodanqi/clone/pluto/metafile/final_json.json'

# # 1. 读取原始 JSON 文件
# with open(file_path, 'r', encoding='utf-8') as file:
#     content = file.read()

# # 2. 尝试解析 JSON
# try:
#     # 尝试将数据解析成合法的 JSON 格式
#     parsed_data = json.loads(content)
#     print("JSON 格式有效")
# except json.JSONDecodeError as e:
#     print(f"JSON 解析错误：{e}")
#     # 尝试修复常见的错误（如缺少逗号）
#     content = content.replace("}]}][{", "}]}, {")
    
#     try:
#         parsed_data = json.loads(content)
#         print("格式已修复")
#     except json.JSONDecodeError as e:
#         print(f"修复失败，仍然无法解析：{e}")

# # 3. 如果解析成功，保存修复后的 JSON 文件
# if 'parsed_data' in locals():
#     with open(fixed_json_path, 'w', encoding='utf-8') as outfile:
#         json.dump(parsed_data, outfile, indent=4, ensure_ascii=False)
#     print(f"修复后的数据已保存到 {fixed_json_path}")

# # 4. 读取修复后的数据并简化
# with open(fixed_json_path, 'r', encoding='utf-8') as file:
#     eval_data = json.load(file)

# # 存储简化后的数据
# simplified_data = []

# # 遍历原始数据并提取需要的部分
# for sample in eval_data:
#     token = sample.get("token")
#     conversations = sample.get("conversations")
    
#     # 确保 conversations 中有有效的 gpt 回复
#     if conversations and len(conversations) > 1:
#         gt_answer = conversations[1].get("value")
        
#         # 确保 gt_answer 存在且格式正确
#         if gt_answer:
#             # 拆分 speed_plan 和 path_plan
#             speed_plan, path_plan = gt_answer.split(", ")
            
#             # 将结果保存到简化后的数据中
#             simplified_data.append({
#                 "token": token,
#                 "speed_plan": speed_plan,
#                 "path_plan": path_plan
#             })

# # 5. 保存简化后的数据到文件
# with open(output_data_path, 'w', encoding='utf-8') as output_file:
#     json.dump(simplified_data, output_file, indent=4, ensure_ascii=False)

# print(f"简化后的数据已保存到 {output_data_path}")

# # 6. 进一步处理路径计划中的换行符，并保存最终的 JSON
# with open(output_data_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # 修改每个对象的 path_plan 字段，去掉结尾的 '\n'
# for item in data:
#     if 'path_plan' in item:
#         item['path_plan'] = item['path_plan'].rstrip('\n')

# # 将修改后的数据保存回文件
# with open(final_json_path, 'w', encoding='utf-8') as file:
#     json.dump(data, file, indent=4, ensure_ascii=False)

# print("路径计划中的换行符已被移除并保存。最终数据已保存到", final_json_path)


import json
import os


file_path = '/home/zhaodanqi/clone/pluto/METAfiles_wl/senna_conversion_train_2021.05.12.23.36.44_veh-35_00152_00504.json'
final_json_path = '/home/zhaodanqi/clone/pluto/jsonfiles/2021.05.12.23.36.44_veh-35_00152_00504.json'

# 1. 读取原始 JSON 文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 2. 尝试解析 JSON
try:
    parsed_data = json.loads(content)
    print("JSON 格式有效")
except json.JSONDecodeError as e:
    print(f"JSON 解析错误：{e}")
    content = content.replace("}]}][{", "}]}, {")  # 可能的修复尝试

    try:
        parsed_data = json.loads(content)
        print("格式已修复")
    except json.JSONDecodeError as e:
        print(f"修复失败，仍然无法解析：{e}")
        exit(1)  # 解析失败则退出

# 3. 处理数据并提取所需字段
processed_data = []

for sample in parsed_data:
    token = sample.get("token")
    conversations = sample.get("conversations")

    # 确保 conversations 存在且至少有 2 条
    if conversations and len(conversations) > 1:
        gt_answer = conversations[1].get("value")

        if gt_answer:
            speed_plan, path_plan = gt_answer.split(", ")

            # 去除 path_plan 结尾的 '\n'
            processed_data.append({
                "token": token,
                "speed_plan": speed_plan,
                "path_plan": path_plan.rstrip('\n')
            })

# 4. 仅保存最终的 JSON 文件
with open(final_json_path, 'w', encoding='utf-8') as file:
    json.dump(processed_data, file, indent=4, ensure_ascii=False)

print(f"最终数据已保存到 {final_json_path}")

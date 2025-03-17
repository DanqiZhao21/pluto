 
#   sh ./script/run_pluto_planner.sh pluto_planner nuplan_mini mini_demo_scenario pluto_1M_aux_cil.ckpt "$VIDEO_SAVE_DIR"

cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER=$1
BUILDER=$2
FILTER=$3
CKPT=$4
VIDEO_SAVE_DIR=$5


CHALLENGE="closed_loop_nonreactive_agents" #几种仿真模式
# CHALLENGE="closed_loop_reactive_agents"
# CHALLENGE="open_loop_boxes"

# 自动生成保存视频的目录名称（带时间戳）
# /home/zhaodanqi/clone/pluto/SAVE_SIMU
# VIDEO_SAVE_DIR="/home/zhaodanqi/clone/pluto/SAVE_SIMU_no_post/SAVE_SIMU_$(date +%Y%m%d%H%M%S)-$BUILDER-$FILTER"

VIDEO_SAVE_DIR="/home/zhaodanqi/clone/pluto/wandbModels/models_202412/simulations/$(date +%Y%m%d%H%M%S)-$BUILDER-$FILTER"

# 创建目录
mkdir -p "$VIDEO_SAVE_DIR"


python run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    scenario_builder=$BUILDER \
    scenario_filter=$FILTER \
    worker=sequential \
    verbose=true \
    experiment_uid="pluto_planner/$FILTER" \
    planner.pluto_planner.render=true \
    planner.pluto_planner.planner_ckpt="$CKPT_ROOT/$CKPT" \
    +planner.pluto_planner.save_dir=$VIDEO_SAVE_DIR

# PLANNER=$1
# BUILDER=$2
# FILTER=$3
# CKPT=$4
# VIDEO_SAVE_DIR=$5

# bash run.sh planner1 builder1 filter1 checkpoint1 /path/to/video_save

# 那么在脚本 run.sh 里：
# $1 = planner1 → PLANNER=planner1
# $2 = builder1 → BUILDER=builder1
# $3 = filter1 → FILTER=filter1
# $4 = checkpoint1 → CKPT=checkpoint1
# $5 = /path/to/video_save → VIDEO_SAVE_DIR=/path/to/video_save

# 然后，它们会被传递到 python run_simulation.py 命令中，等效于：
# python run_simulation.py \
#     +simulation=closed_loop_nonreactive_agents \
#     planner=planner1 \
#     scenario_builder=builder1 \
#     scenario_filter=filter1 \
#     worker=sequential \
#     verbose=true \
#     experiment_uid="pluto_planner/filter1" \
#     planner.pluto_planner.render=true \
#     planner.pluto_planner.planner_ckpt="$cwd/checkpoints/checkpoint1" \
#     +planner.pluto_planner.save_dir=/path/to/video_save

#sh ./script/run_pluto_planner.sh pluto_planner[planner] nuplan_mini[builder] mini_demo_scenario[Filter] pluto_1M_aux_cil.ckpt[CKPT] "$VIDEO_SAVE_DIR"


#能行的：

#sh ./script/run_pluto_planner.sh pluto_planner nuplan_mini mini_demo_scenario pluto_1M_aux_cil.ckpt "$VIDEO_SAVE_DIR"
#sh ./script/run_pluto_planner.sh pluto_planner nuplan mini_demo_scenario pluto_1M_aux_cil.ckpt "$VIDEO_SAVE_DIR"#不能运行成功
#sh ./script/run_pluto_planner.sh pluto_planner nuplan random14_benchmark pluto_1M_aux_cil.ckpt "$VIDEO_SAVE_DIR"
#sh ./script/run_pluto_planner.sh pluto_planner nuplan val14_benchmark pluto_1M_aux_cil.ckpt "$VIDEO_SAVE_DIR"


#不能成功的：

#   sh ./script/run_pluto_planner.sh pluto_planner nuplan val_demo_scenario pluto_1M_aux_cil.ckpt "$VIDEO_SAVE_DIR"


#PLANNER:  planner: /home/angiezhao/clone/pluto/config/planner/pluto_planner.yaml
#BUILDER:  scenario_builder=$BUILDER \???
#FILTER:  scenario_filter=/home/angiezhao/clone/pluto/config/scenario_filter
    #mini_demo_scenario
    #random14_benchmark
    #training_scenarios_1M
    #training_scenario_tiny
    #val_demo_scenario
    #val14_benchmark


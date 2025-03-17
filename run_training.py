import logging
from typing import Optional
from torchinfo import summary 
from pytorch_lightning import Trainer 
'''
NuPlan中的缓存机制是为了减少重复计算、加速数据访问以及提高整体仿真效率。
'''
'''
使用的环境是pluto :conda activate pluto?? PLUTO?
使用的python解释器是: PLUTO
'''

############################################################################################
#缓存的
'''
新服务器

export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan  \
    cache.cache_path=/mnt/data/nuplan/exp/cache_pluto_1M\
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40

#离谱pluto在新的服务器上面跑不通(能跑通了）
#下面是debug版本

export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan  \
    cache.cache_path=/mnt/data/nuplan/exp/cache_pluto_0312_6_10m\
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40









旧服务器

export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan \
    cache.cache_path=/home/angiezhao/nuplan-devkit/nuplan/exp/cache_pluto_1M \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40
'''
#>?/？？之前一直都错了我去！
###########################################################################################
# cache.cache_path=/home/angiezhao/nuplan-devkit/nuplan/exp/cache_pluto_1M \
#Training with out CIL
# 训练的\上面是老的下面是新的

# CUDA_VISIBLE_DEVICES=0 python run_training.py \
#   py_func=train +training=train_pluto \
#   worker=single_machine_thread_pool worker.max_workers=32 \
#   scenario_builder=nuplan cache.cache_path=/home/angiezhao/nuplan-devkit/nuplan/exp/cache_pluto_1M  cache.use_cache_without_dataset=true \
#   data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
#   lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
#   wandb.mode=online wandb.project=nuplan wandb.name=pluto




'''
CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/mnt/data/nuplan/exp/cache_pluto_1M  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto
  

CUDA_VISIBLE_DEVICES=6 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/mnt/data/nuplan/exp/cache_pluto_0312_6  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto_031
  
  CUDA_VISIBLE_DEVICES=1,2,3,4 python run_training.py +trainer.devices=4 +trainer.strategy=deepspeed\
  
CUDA_VISIBLE_DEVICES=6 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/mnt/data/nuplan/exp/cache_pluto_1M cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto_03155_onlyclsLoss
  
  
  
  CUDA_VISIBLE_DEVICES=1,2,3,4 python run_training.py \
  +trainer.devices=4 \
  +trainer.strategy="/home/zhaodanqi/clone/pluto/config/deepspeed_config.json" \
  py_func=train \
  +training=train_pluto \
  worker=single_machine_thread_pool \
  worker.max_workers=32 \
  scenario_builder=nuplan \
  cache.cache_path=/mnt/data/nuplan/exp/cache_pluto_0312_6 \
  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 \
  data_loader.params.num_workers=16 \
  lr=1e-3 \
  epochs=25 \
  warmup_epochs=3 \
  weight_decay=0.0001 \
  wandb.mode=disable

  
  

'''
#   trainer.gpus=4 trainer.strategy=ddp \

# Run Pluto-planner simulation
'''
 sh ./script/run_pluto_planner.sh pluto_planner nuplan_mini mini_demo_scenario pluto_1M_aux_cil.ckpt /home/angiezhao/clone/pluto/SAVE_SIMU
 sh ./script/run_pluto_planner.sh pluto_planner nuplan_mini mini_demo_scenario pluto_1M_aux_cil.ckpt "$VIDEO_SAVE_DIR"

'''



import hydra
import numpy
import pytorch_lightning as pl
from nuplan.planning.script.builders.folder_builder import (
    build_training_experiment_folder,
)
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from omegaconf import DictConfig

from src.custom_training.custom_training_builder import (
    TrainingEngine,
    build_training_engine,
    build_training_engine,
    update_config_for_training,
)


# import os
# import tensorflow as tf


logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()


# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = "./config"
CONFIG_NAME = "default_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    
    print(f"场景过滤器 is {cfg.scenario_filter}")

    
    
    # print(f"Using cache path: {cfg.cache.cache_path}")

    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)
    
    

    if cfg.py_func == "train":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)
            # engine.trainer = Trainer(accelerator="gpu", devices=1, precision=16)
        # Run training
        logger.info("Starting training...")
        # print(f"训练的模型详细信息为： {engine.model}")
        # print(f"训练的模型详细信息为: {type(engine.model)}")
        print("模型的总的层数为 Total layers:", len(list(engine.model.modules())))
        # if isinstance(engine.model, LightningModule):
        #     summary(engine.model, input_size=(1, 3, 64, 64))  # 修改 input_size 以匹配你的数据
        for name, module in engine.model.named_modules():
            print(name, ":", module.__class__.__name__)


        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            engine.trainer.fit(
                model=engine.model,
                datamodule=engine.datamodule,
                ckpt_path=cfg.checkpoint,
            )
            
        return engine
    
    if cfg.py_func == "validate":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "validate"):
            engine.trainer.validate(
                model=engine.model,
                datamodule=engine.datamodule,
                ckpt_path=cfg.checkpoint,
            )
        return engine
    elif cfg.py_func == "test":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Test model
        logger.info("Starting testing...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "testing"):
            engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    
    
    
    elif cfg.py_func == "cache":
        # Precompute and cache all features
        logger.info("Starting caching...")
        # import ipdb; ipdb.set_trace()
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_data(cfg=cfg, worker=worker)
        return None
    else:
        raise NameError(f"Function {cfg.py_func} does not exist")


if __name__ == "__main__":
    import torch
    
    torch.set_float32_matmul_precision('high')  # 或者 'medium'
    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda:6") 
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    # print(len(tf.config.experimental.list_physical_devices('GPU')))  # 这应该输出 8
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 或者你实际存在的设备编号
    main()

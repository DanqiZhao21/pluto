import wandb

run = wandb.init()
artifact = run.use_artifact('2564380679-/nuplan/model-original-little1M:v2', type='model')

# 自定义下载路径
custom_path = "/home/zhaodanqi/clone/pluto/wandb_models"
artifact_dir = artifact.download(root=custom_path)

print("Model downloaded to:", artifact_dir)

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.utils.data
from omegaconf import DictConfig
from torch.utils.data.sampler import WeightedRandomSampler

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import (
    AbstractAugmentor,
)
from nuplan.planning.training.data_loader.distributed_sampler_wrapper import (
    DistributedSamplerWrapper,
)
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    move_features_type_to_device,
)
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.feature_preprocessor import (
    FeaturePreprocessor,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

logger = logging.getLogger(__name__)

DataModuleNotSetupError = RuntimeError('Data module has not been setup, call "setup()"')

'''
这段代码是一个数据管理模块，专门用于深度学习模型训练中的数据预处理、划分、加载等任务，
基于 PyTorch Lightning 框架实现。它包括数据集的创建、划分（训练、验证、测试集）、数据增强、加权采样以及数据加载器（DataLoader）的初始化。
'''
'''
1. 主类 CustomDataModule
作用: 包装了数据准备的完整流程，兼容 PyTorch Lightning 的数据模块架构。
关键功能:
数据划分为训练、验证和测试集。
数据集的加载和数据增强处理。
数据加载器的配置和返回。
适配分布式训练和加权采样等高级需求。

2. 关键方法和模块
(a) create_dataset
功能: 创建一个自定义的数据集（ScenarioDataset）。
细节:
从样本中随机采样指定比例的子集。
应用特征预处理器和可选的数据增强器。
返回符合 torch.utils.data.Dataset 接口的对象。
(b) distributed_weighted_sampler_init
功能: 为每种场景类型指定权重，生成加权采样器（WeightedRandomSampler）。
用途:
通过加权采样控制某些场景类型的训练样本比例。
可用于处理数据不平衡问题。
(c) setup
功能: 按照训练阶段（fit, test 等）初始化不同的数据集。
细节:
通过 splitter 将完整数据集拆分为训练、验证和测试集。
按照指定比例随机选取样本。
在不同阶段加载对应的数据集。
(d) train_dataloader, val_dataloader, test_dataloader
功能: 创建并返回 PyTorch 的 DataLoader 对象。
细节:
在训练阶段，支持按场景类型的权重分布加权采样。
数据批次的整理通过自定义的 FeatureCollate 函数。
(e) transfer_batch_to_device
功能: 将数据从 CPU 转移到 GPU 或指定设备。
细节:
适配 PyTorch Lightning 的设备转移流程。
递归将特征（FeaturesType）和目标转移到目标设备上。
在分布式训练中能很好地处理数据在不同设备间的移动。

3. 应用场景
分布式训练: 适配 PyTorch Lightning 的多 GPU 或 TPU 环境。
数据增强: 提供多种数据增强器（augmentors），对样本进行动态增强。
不平衡数据: 通过 distributed_weighted_sampler_init 对不同类别（场景类型）进行加权采样，缓解数据分布不均的问题。
高效训练: 使用多进程（通过 worker 参数）提高数据加载和处理速度。

4. 改进点和新功能
transfer_batch_to_device 方法: 新增或重写的功能，适配 PyTorch Lightning 的最新版本，以确保在设备转换时特征和目标保持正确的结构和内容。
'''
def create_dataset(
    samples: List[AbstractScenario],
    feature_preprocessor: FeaturePreprocessor,
    dataset_fraction: float,
    dataset_name: str,
    augmentors: Optional[List[AbstractAugmentor]] = None,
) -> torch.utils.data.Dataset:
    """
    Create a dataset from a list of samples.
    :param samples: List of dataset candidate samples.
    :param feature_preprocessor: Feature preprocessor object.
    :param dataset_fraction: Fraction of the dataset to load.
    :param dataset_name: Set name (train/val/test).
    :param scenario_type_loss_weights: Dictionary of scenario type loss weights.
    :param augmentors: List of augmentor objects for providing data augmentation to data samples.
    :return: The instantiated torch dataset.
    """
    # Sample the desired fraction from the total samples
    num_keep = int(len(samples) * dataset_fraction)
    selected_scenarios = random.sample(samples, num_keep)

    logger.info(f"Number of samples in {dataset_name} set: {len(selected_scenarios)}")
    return ScenarioDataset(
        scenarios=selected_scenarios,
        feature_preprocessor=feature_preprocessor,
        augmentors=augmentors,
    )


def distributed_weighted_sampler_init(
    scenario_dataset: ScenarioDataset,
    scenario_sampling_weights: Dict[str, float],
    replacement: bool = True,
) -> WeightedRandomSampler:
    """
    Initiliazes WeightedSampler object with sampling weights for each scenario_type and returns it.
    :param scenario_dataset: ScenarioDataset object
    :param replacement: Samples with replacement if True. By default set to True.
    return: Initialized Weighted sampler
    """
    scenarios = scenario_dataset._scenarios
    if (
        not replacement
    ):  # If we don't sample with replacement, then all sample weights must be nonzero
        assert all(
            w > 0 for w in scenario_sampling_weights.values()
        ), "All scenario sampling weights must be positive when sampling without replacement."

    default_scenario_sampling_weight = 1.0

    scenario_sampling_weights_per_idx = [
        scenario_sampling_weights[scenario.scenario_type]
        if scenario.scenario_type in scenario_sampling_weights
        else default_scenario_sampling_weight
        for scenario in scenarios
    ]

    # Create weighted sampler
    weighted_sampler = WeightedRandomSampler(
        weights=scenario_sampling_weights_per_idx,
        num_samples=len(scenarios),
        replacement=replacement,
    )

    distributed_weighted_sampler = DistributedSamplerWrapper(weighted_sampler)
    return distributed_weighted_sampler


class CustomDataModule(pl.LightningDataModule):
    """
    Datamodule wrapping all preparation and dataset creation functionality.
    """

    def __init__(
        self,
        feature_preprocessor: FeaturePreprocessor,
        splitter: AbstractSplitter,
        all_scenarios: List[AbstractScenario],
        train_fraction: float,
        val_fraction: float,
        test_fraction: float,
        dataloader_params: Dict[str, Any],
        scenario_type_sampling_weights: DictConfig,
        worker: WorkerPool,
        augmentors: Optional[List[AbstractAugmentor]] = None,
    ) -> None:
        """
        Initialize the class.
        :param feature_preprocessor: Feature preprocessor object.
        :param splitter: Splitter object used to retrieve lists of samples to construct train/val/test sets.
        :param train_fraction: Fraction of training examples to load.
        :param val_fraction: Fraction of validation examples to load.
        :param test_fraction: Fraction of test examples to load.
        :param dataloader_params: Parameter dictionary passed to the dataloaders.
        :param augmentors: Augmentor object for providing data augmentation to data samples.
        """
        super().__init__()

        assert train_fraction > 0.0, "Train fraction has to be larger than 0!"
        assert val_fraction > 0.0, "Validation fraction has to be larger than 0!"
        assert test_fraction >= 0.0, "Test fraction has to be larger/equal than 0!"

        # Datasets
        self._train_set: Optional[torch.utils.data.Dataset] = None
        self._val_set: Optional[torch.utils.data.Dataset] = None
        self._test_set: Optional[torch.utils.data.Dataset] = None

        # Feature computation
        self._feature_preprocessor = feature_preprocessor

        # Data splitter train/test/val
        self._splitter = splitter

        # Fractions
        self._train_fraction = train_fraction
        self._val_fraction = val_fraction
        self._test_fraction = test_fraction

        # Data loader for train/val/test
        self._dataloader_params = dataloader_params

        # Extract all samples
        self._all_samples = all_scenarios
        assert len(self._all_samples) > 0, "No samples were passed to the datamodule"

        # Scenario sampling weights
        self._scenario_type_sampling_weights = scenario_type_sampling_weights

        # Augmentation setup
        self._augmentors = augmentors

        # Worker for multiprocessing to speed up initialization of datasets
        self._worker = worker

    @property
    def feature_and_targets_builder(self) -> FeaturePreprocessor:
        """Get feature and target builders."""
        return self._feature_preprocessor

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the dataset for each target set depending on the training stage.
        This is called by every process in distributed training.
        :param stage: Stage of training, can be "fit" or "test".
        """
        if stage is None:
            return

        if stage == "fit":
            # Training Dataset
            train_samples = self._splitter.get_train_samples(
                self._all_samples, self._worker
            )
            assert len(train_samples) > 0, "Splitter returned no training samples"

            self._train_set = create_dataset(
                train_samples,
                self._feature_preprocessor,
                self._train_fraction,
                "train",
                self._augmentors,
            )

            # Validation Dataset
            val_samples = self._splitter.get_val_samples(
                self._all_samples, self._worker
            )
            assert len(val_samples) > 0, "Splitter returned no validation samples"

            self._val_set = create_dataset(
                val_samples,
                self._feature_preprocessor,
                self._val_fraction,
                "validation",
            )
        elif stage == "validate":
            # Validation Dataset
            val_samples = self._splitter.get_val_samples(
                self._all_samples, self._worker
            )
            assert len(val_samples) > 0, "Splitter returned no validation samples"

            self._val_set = create_dataset(
                val_samples,
                self._feature_preprocessor,
                self._val_fraction,
                "validation",
            )
        elif stage == "test":
            # Testing Dataset
            test_samples = self._splitter.get_test_samples(
                self._all_samples, self._worker
            )
            assert len(test_samples) > 0, "Splitter returned no test samples"

            self._test_set = create_dataset(
                test_samples, self._feature_preprocessor, self._test_fraction, "test"
            )
        else:
            raise ValueError(f'Stage must be one of ["fit", "test"], got ${stage}.')

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Clean up after a training stage.
        This is called by every process in distributed training.
        :param stage: Stage of training, can be "fit" or "test".
        """
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the training dataloader.
        :raises RuntimeError: If this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._train_set is None:
            raise DataModuleNotSetupError

        # Initialize weighted sampler
        if self._scenario_type_sampling_weights.enable:
            weighted_sampler = distributed_weighted_sampler_init(
                scenario_dataset=self._train_set,
                scenario_sampling_weights=self._scenario_type_sampling_weights.scenario_type_weights,
            )
        else:
            weighted_sampler = None

        return torch.utils.data.DataLoader(
            dataset=self._train_set,
            shuffle=weighted_sampler is None,
            collate_fn=FeatureCollate(),
            sampler=weighted_sampler,
            **self._dataloader_params,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the validation dataloader.
        :raises RuntimeError: if this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._val_set is None:
            raise DataModuleNotSetupError

        return torch.utils.data.DataLoader(
            dataset=self._val_set,
            **self._dataloader_params,
            collate_fn=FeatureCollate(),
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the test dataloader.
        :raises RuntimeError: if this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._test_set is None:
            raise DataModuleNotSetupError

        return torch.utils.data.DataLoader(
            dataset=self._test_set,
            **self._dataloader_params,
            collate_fn=FeatureCollate(),
        )

    # ! Modified to adapt to newer version of pytorch-lightning
    def transfer_batch_to_device(
        self, batch: Tuple[FeaturesType, ...], device: torch.device, dataloader_idx: int
    ) -> Tuple[FeaturesType, ...]:
        """
        Transfer a batch to device.
        :param batch: Batch on origin device.
        :param device: Desired device.
        :return: Batch in new device.
        """
        return tuple(
            (
                move_features_type_to_device(batch[0], device),
                move_features_type_to_device(batch[1], device),
                batch[2],
            )
        )

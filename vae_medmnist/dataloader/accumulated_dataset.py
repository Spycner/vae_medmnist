"""
This module defines a PyTorch Lightning data module that accumulates multiple subsets of the MedMNIST dataset into a single dataset.
This is particularly useful for scenarios where training on a combination of different MedMNIST datasets is required,
such as multi-task learning or domain adaptation.

Classes:
    AccumulatedMedMNIST: A class that inherits from LightningDataModule and handles the accumulation and management of multiple MedMNIST datasets.
"""

import logging
from argparse import ArgumentParser
from typing import List

import medmnist
from medmnist import INFO
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from vae_medmnist.dataloader.medmnist_datamodule import MedMNISTDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccumulatedMedMNIST(LightningDataModule):
    """
    A data module for accumulating multiple MedMNIST datasets into a unified dataset with combined training, validation, and test splits.

    Attributes:
        datasets (List[str]): List of dataset names to be accumulated.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of subprocesses to use for data loading.
        size (int): The size of the images (assumed square).
        args (tuple): Additional positional arguments.
        kwargs (dict): Additional keyword arguments.
    """

    def __init__(
        self, datasets: List[str] = None, batch_size: int = 32, num_workers: int = 8, size: int = 28, *args, **kwargs
    ):
        """
        Initializes the AccumulatedMedMNIST data module.

        Args:
            datasets (List[str], optional): Specific subsets of MedMNIST to be used. Defaults to ['tissuemnist', 'octmnist', 'chestmnist'].
            batch_size (int, optional): Number of samples in each batch of data. Defaults to 32.
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 8.
            size (int, optional): The height and width of the images after resizing. Defaults to 28.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.datasets = datasets if datasets is not None else ['tissuemnist', 'octmnist', 'organamnist']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.args = args
        self.kwargs = kwargs

        self.infos = [INFO[dataset] for dataset in self.datasets]

        self.in_channels = set(info['n_channels'] for info in self.infos)
        assert len(self.in_channels) == 1, 'All datasets must have the same number of channels'

        self.num_classes = sum(len(info['label']) for info in self.infos)
        self.n_samples = {
            'train': sum(info['n_samples']['train'] for info in self.infos),
            'val': sum(info['n_samples']['val'] for info in self.infos),
            'test': sum(info['n_samples']['test'] for info in self.infos),
        }
        self.labels = {}

    def setup(self, stage: str = None):
        """
        Prepares the datasets for use by the data module by setting up training, validation, and test datasets.

        Args:
            stage (str, optional): Stage for which setup is being called ('fit', 'validate', 'test', or 'predict'). Defaults to None.
        """
        logger.info(f'Setting up {self.__class__.__name__} with {self.datasets} datasets')
        combined_datasets = []
        label_offset = 0

        for dataset in self.datasets:
            info = INFO[dataset]
            for _, label_name in info['label'].items():
                self.labels[len(self.labels)] = label_name

            data_module = MedMNISTDataModule(
                dataset_class=getattr(medmnist, info['python_class']),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                size=self.size,
                *self.args,  # noqa: B026
                **self.kwargs,
            )
            data_module.setup(stage)

            local_datasets = []
            if stage == 'fit' or stage is None:
                train_dataset = data_module.mnist_train
                val_dataset = data_module.mnist_val
                local_datasets.extend([train_dataset, val_dataset])

            if stage == 'test' or stage is None:
                test_dataset = data_module.mnist_test
                local_datasets.append(test_dataset)

            # Adjust labels to ensure they are unique across different datasets
            for dataset in local_datasets:
                if hasattr(dataset, 'labels'):
                    dataset.labels = dataset.labels + label_offset
                else:
                    logger.error(f"{type(dataset).__name__} does not have a 'labels' attribute")
                    raise AttributeError(f"{type(dataset).__name__} does not have a 'labels' attribute")

            combined_datasets.extend(local_datasets)
            label_offset += len(info['label'])

        if stage == 'fit' or stage is None:
            self.combined_train = ConcatDataset([ds for ds in combined_datasets if ds.split == 'train'])
            self.combined_val = ConcatDataset([ds for ds in combined_datasets if ds.split == 'val'])
        if stage == 'test' or stage is None:
            self.combined_test = ConcatDataset([ds for ds in combined_datasets if ds.split == 'test'])

        if hasattr(self, 'combined_train'):
            logger.info(f'Combined train dataset: {self.combined_train}')
        if hasattr(self, 'combined_val'):
            logger.info(f'Combined validation dataset: {self.combined_val}')
        if hasattr(self, 'combined_test'):
            logger.info(f'Combined test dataset: {self.combined_test}')
        logger.info(f'Number of channels: {self.in_channels}')
        logger.info(f'Number of classes: {self.num_classes}')
        logger.info(f'Number of samples: {self.n_samples}')

    def train_dataloader(self):
        """
        Creates a DataLoader for training data.

        Returns:
            DataLoader: The DataLoader for the training data.
        """
        return DataLoader(
            self.combined_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Creates a DataLoader for validation data.

        Returns:
            DataLoader: The DataLoader for the validation data.
        """
        return DataLoader(
            self.combined_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """
        Creates a DataLoader for test data.

        Returns:
            DataLoader: The DataLoader for the test data.
        """
        return DataLoader(
            self.combined_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Adds dataset-specific arguments to the argument parser.

        Args:
            parent_parser (ArgumentParser): The parent argument parser.

        Returns:
            ArgumentParser: The modified argument parser with added dataset-specific arguments.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=32)

        return parser

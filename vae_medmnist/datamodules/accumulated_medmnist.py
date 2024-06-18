"""
This module defines a PyTorch Lightning data module that accumulates multiple subsets of the MedMNIST dataset into a single dataset.
This is particularly useful for scenarios where training on a combination of different MedMNIST datasets is required,
such as multi-task learning or domain adaptation.

Classes:
    AccumulatedMedMNIST: A class that inherits from LightningDataModule and handles the accumulation and management of multiple MedMNIST datasets.
"""

from argparse import ArgumentParser
from typing import List

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from vae_medmnist.datamodules.medmnist_datamodule import MedMNISTDataModule


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
        super().__init__(*args, **kwargs)
        self.datasets = datasets if datasets is not None else ['tissuemnist', 'octmnist', 'chestmnist']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.args = args
        self.kwargs = kwargs

    def setup(self, stage: str = None):
        """
        Prepares the datasets for use by the data module by setting up training, validation, and test datasets.

        Args:
            stage (str, optional): Stage for which setup is being called ('fit', 'validate', 'test', or 'predict'). Defaults to None.
        """
        combined_datasets = []
        label_offset = 0

        for dataset in self.datasets:
            data_module = MedMNISTDataModule(
                dataset_class=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                size=self.size,
                *self.args,  # noqa: B026
                **self.kwargs,
            )
            data_module.setup(stage)
            train_dataset = data_module.mnist_train
            val_dataset = data_module.mnist_val
            test_dataset = data_module.mnist_test

            # Adjust labels to ensure they are unique across different datasets
            for dataset in [train_dataset, val_dataset, test_dataset]:
                dataset.targets = dataset.targets + label_offset

            combined_datasets.extend([train_dataset, val_dataset, test_dataset])
            label_offset += len(data_module.classes)

        self.combined_train = ConcatDataset([ds for ds in combined_datasets if ds.split == 'train'])
        self.combined_val = ConcatDataset([ds for ds in combined_datasets if ds.split == 'val'])
        self.combined_test = ConcatDataset([ds for ds in combined_datasets if ds.split == 'test'])

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

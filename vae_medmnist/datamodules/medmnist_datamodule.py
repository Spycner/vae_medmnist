from argparse import ArgumentParser
from typing import Callable

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms


class MedMNISTDataModule(LightningDataModule):
    def __init__(self, dataset_class, batch_size=32, num_workers=4, size=28, *args, **kwargs):
        super().__init__()
        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.args = args
        self.kwargs = kwargs

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.mnist_train = self.dataset_class(split='train', transform=self.default_transform(), size=self.size)  # noqa: B026
            self.mnist_val = self.dataset_class(split='val', transform=self.default_transform(), size=self.size)  # noqa: B026

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = self.dataset_class(split='test', transform=self.default_transform(), size=self.size)  # noqa: B026

    def default_transform(self) -> Callable:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--data_dir', type=str, default='.')
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=32)

        return parser

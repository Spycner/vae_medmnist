import medmnist
from medmnist import INFO
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader


class MedMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_flag: str = "tissuemnist", batch_size: int = 64):
        super().__init__()
        self.data_flag = data_flag
        self.batch_size = batch_size

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

    def setup(self, stage=None):
        info = INFO[self.data_flag]
        DataClass = getattr(medmnist, info["python_class"])

        self.train_dataset = DataClass(
            split="train", transform=self.transform, download=True
        )
        self.val_dataset = DataClass(
            split="val", transform=self.transform, download=True
        )
        self.test_dataset = DataClass(
            split="test", transform=self.transform, download=True
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

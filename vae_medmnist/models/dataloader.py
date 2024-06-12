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
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.Lambda(lambda x: x * 0.5 + 0.5),  # from [-1, 1] to [0, 1]
            ]
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
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=7, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=7)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=7)


class FcMedMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_flag: str = "tissuemnist", batch_size: int = 64):
        super().__init__()
        self.data_flag = data_flag
        self.batch_size = batch_size

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.Lambda(lambda x: x * 0.5 + 0.5),
                transforms.Lambda(lambda x: x.view(-1)),  # Flatten the image
            ]
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
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=7, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=7)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=7)


# # Debug step to check data range
# def check_data_range(dataloader):
#     for batch in dataloader:
#         images, _ = batch
#         print(f"Min: {images.min()}, Max: {images.max()}")
#         break


# data_module = MedMNISTDataModule()
# data_module.setup()
# check_data_range(data_module.train_dataloader())

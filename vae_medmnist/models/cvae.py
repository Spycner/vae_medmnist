import logging
from argparse import ArgumentParser
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CVAE(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = None,
        lr: float = 1e-3,
        kl_coeff: float = 0.1,
        latent_dim: int = 128,
        hidden_channels: List[int] = None,
        img_size: int = 28,
        *args,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.img_size = img_size

        assert num_classes is not None, 'Number of classes must be provided'

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256, 512]

        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(input_channels, input_channels, kernel_size=1)

        self.output_channels = self.input_channels
        self.input_channels += 1

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, hidden_channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_channels[i + 1]),
                    nn.ReLU(),
                )
                for i in range(len(hidden_channels) - 1)
            ],
        )

        self.fc_mu = nn.Linear(hidden_channels[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels[-1], latent_dim)

        self.hidden_channels = hidden_channels[::-1]
        self.decoder_input = nn.Linear(self.latent_dim + self.num_classes, self.hidden_channels[0] * 4)

        self.decoder = nn.Sequential(
            *[
                nn.Sequential(  # 3, 5, 9, 17
                    nn.ConvTranspose2d(
                        self.hidden_channels[i],
                        self.hidden_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(self.hidden_channels[i + 1]),
                    nn.ReLU(),
                )
                for i in range(len(self.hidden_channels) - 1)
            ],
            nn.ConvTranspose2d(self.hidden_channels[-1], self.hidden_channels[-1], kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(self.hidden_channels[-1]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels[-1], self.output_channels, kernel_size=3, stride=2),
            nn.Tanh(),
        )

    def forward(self, x, y):
        y_flat = y.squeeze().to(torch.int64)
        y_onehot = one_hot(y_flat, self.num_classes).to(torch.float32)
        embedded_class = self.embed_class(y_onehot)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_data = self.embed_data(x)

        encoder_input = torch.cat([embedded_data, embedded_class], dim=1)
        res = self.encoder(encoder_input)
        res = torch.flatten(res, start_dim=1)

        mu = self.fc_mu(res)
        logvar = self.fc_logvar(res)

        z = self.reparameterize(mu, logvar)

        return (self.decode(z, y_onehot), mu, logvar)

    def decode(self, z, y_onehot):
        concat = torch.cat([z, y_onehot], dim=1)
        decoder_input = self.decoder_input(concat)
        decoder_input = decoder_input.view(-1, self.hidden_channels[0], 2, 2)

        return self.decoder(decoder_input)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def step(self, batch, batch_idx):  # noqa: ARG002
        x, y = batch
        x_hat, mu, logvar = self(x, y)

        recon_loss = F.mse_loss(x_hat, x)

        kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1), dim=0)
        kl *= self.kl_coeff

        loss = recon_loss + kl

        logs = {'loss': loss, 'recon_loss': recon_loss, 'kl': kl}
        return loss, logs

    def training_step(self, batch, batch_idx):
        """Perform a single training step."""
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f'train_{k}': v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step."""
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f'val_{k}': v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        """Configure the optimizers."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def sample(self, num_samples: int, class_idx: int, device=None):
        device = device or self.device
        y_flat = torch.tensor([class_idx], dtype=torch.int64)
        y_onehot = one_hot(y_flat, self.num_classes)
        z = torch.randn(num_samples, self.latent_dim, device=device)

        return self.decode(z, y_onehot)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model-specific arguments to the parser."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--lr', type=float, default=1e-4)

        parser.add_argument('--kl_coeff', type=float, default=0.1)
        parser.add_argument('--latent_dim', type=int, default=256)

        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--num_workers', type=int, default=8)

        return parser


def cli_main(args=None):
    import yaml
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    from vae_medmnist.dataloader.accumulated_dataset import AccumulatedMedMNIST

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the configuration file')

    args, unknown = parser.parse_known_args(args)

    # Load configuration from a YAML file
    with open(args.config) as file:
        config = yaml.safe_load(file)
        logger.debug(f'Loaded configuration: {config}')

    # Update parser with options from the config file
    parser = CVAE.add_model_specific_args(parser)
    parser.set_defaults(**config)
    args = parser.parse_args(unknown)

    if isinstance(args.datasets, list):
        datamodule = AccumulatedMedMNIST(**args.__dict__)
        args.input_channels = next(iter(datamodule.in_channels))
        args.num_classes = datamodule.num_classes
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    model = CVAE(**args.__dict__)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='cvae-best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    csv_logger = CSVLogger(args.log_dir, name='training_logs')

    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[checkpoint_callback], logger=csv_logger)
    trainer.fit(model, datamodule=datamodule)
    return datamodule, model, trainer


if __name__ == '__main__':
    datamodule, model, trainer = cli_main()

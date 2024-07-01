import logging
from argparse import ArgumentParser
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_medmnist.dataloader.medmnist_datamodule import MedMNISTDataModule

logging.basicConfig(level=logging.INFO, format='%(asctime)s')
logger = logging.getLogger(__name__)


class BetaVAE(pl.LightningModule):
    """BetaVAE model class."""

    def __init__(
        self,
        input_channels: int = 1,
        lr: float = 1e-4,
        kl_coeff: float = 0.1,
        hidden_channels: List = None,
        latent_dim: int = 256,
        beta: int = 4,
        **kwargs,  # noqa: ARG002
    ):
        """Initialize the BetaVAE model."""
        super().__init__()
        self.save_hyperparameters()
        self.input_channels = input_channels
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.beta = beta 

        if self.hidden_channels is None:
            self.hidden_channels = [32, 64, 128, 256, 512]

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels[0]),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Conv2d(self.hidden_channels[i], self.hidden_channels[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(self.hidden_channels[i + 1]),
                    nn.ReLU(),
                )
                for i in range(len(self.hidden_channels) - 1)
            ],
        )

        self.fc_mu = nn.Linear(self.hidden_channels[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_channels[-1], self.latent_dim)

        self.hidden_channels = self.hidden_channels[::-1]

        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_channels[0] * 4)

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
            nn.Conv2d(self.hidden_channels[-1], self.input_channels, kernel_size=3, stride=2),
            nn.Tanh(),
        )

    def forward(self, x):
        """Forward pass through the model."""
        x = self.encoder(x)
        res = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(res)
        logvar = self.fc_logvar(res)

        z = self.reparameterize(mu, logvar)

        return (
            self.decode(z),
            mu,
            logvar,
        )

    def decode(self, x):
        """Decode the latent variables."""
        decoder_input = self.decoder_input(x)
        decoder_input = decoder_input.view(-1, self.hidden_channels[0], 2, 2)

        return self.decoder(decoder_input)

    def reparameterize(self, mu, logvar):
        """Reparameterize the latent variables."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z

    def step(self, batch, batch_idx):  # noqa: ARG002
        """Perform a single optimization step."""
        x, y = batch
        x_hat, mu, logvar = self(x)

        recon_loss = F.mse_loss(x_hat, x)
        
        kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1), dim=0)
        kl *= self.kl_coeff * self.beta
        loss = recon_loss + kl

        logs = {
            'recon_loss': recon_loss,
            'kl': kl,
            'loss': loss,
        }

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

    def sample(self, num_samples=1, device=None):
        """Sample new pictures from the VAE."""
        device = device or self.device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model-specific arguments to the parser."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--lr', type=float, default=1e-4)

        parser.add_argument('--kl_coeff', type=float, default=0.1)
        parser.add_argument('--latent_dim', type=int, default=256)

        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--num_workers', type=int, default=4)

        return parser


def cli_main(args=None):
    """Main function for command-line interface."""
    import yaml
    from medmnist import INFO
    from medmnist.dataset import TissueMNIST, ChestMNIST, OCTMNIST
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the configuration file')

    args, unknown = parser.parse_known_args(args)

    # Load configuration from a YAML file
    with open(args.config) as file:
        config = yaml.safe_load(file)
        logger.debug(f'Loaded configuration: {config}')

    # Update parser with options from the config file
    parser = BetaVAE.add_model_specific_args(parser)
    parser.set_defaults(**config)
    args = parser.parse_args(unknown)

    if args.dataset == 'tissuemnist':
        dataset_class = TissueMNIST
    elif args.dataset == 'chestmnist':
        dataset_class = ChestMNIST
    elif args.dataset == 'octmnist':
        dataset_class = OCTMNIST
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    datamodule = MedMNISTDataModule(dataset_class, **args.__dict__)
    args.input_channels = INFO[args.dataset]['n_channels']

    model = BetaVAE(**args.__dict__)

    # Setup model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='betavae-best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    # Setup CSV logging
    csv_logger = CSVLogger(save_dir=args.log_dir, name='training_logs')

    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[checkpoint_callback], logger=csv_logger)
    trainer.fit(model, datamodule=datamodule)
    return datamodule, model, trainer


if __name__ == '__main__':
    datamodule, model, trainer = cli_main()

import logging
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_medmnist.datamodules.medmnist_datamodule import MedMNISTDataModule
from vae_medmnist.models.components import resnet18_decoder, resnet18_encoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class ResNetVAE(pl.LightningModule):
    def __init__(
        self,
        input_height: int,
        lr: float = 1e-4,
        kl_coeff: float = 0.1,
        enc_out_dim: int = 512,
        latent_dim: int = 256,
        first_conv: bool = False,
        maxpool1: bool = False,
        **kwargs,  # noqa: ARG002
    ):
        super().__init__()

        self.save_hyperparameters()
        self.input_height = input_height
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim

        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, self.latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        p, q, z = self.reparameterize(mu, logvar)

        return z, self.decoder(z), p, q

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)

        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):  # noqa: ARG002
        x, y = batch
        z, x_hat, p, q = self(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        kl = torch.distributions.kl_divergence(q, p).mean()
        kl *= self.kl_coeff

        loss = recon_loss + kl

        logs = {
            'recon_loss': recon_loss,
            'kl': kl,
            'loss': loss,
        }

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f'train_{k}': v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f'val_{k}': v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--first_conv', action='store_true')
        parser.add_argument('--maxpool1', action='store_true')
        parser.add_argument('--lr', type=float, default=1e-4)

        parser.add_argument(
            '--enc_out_dim',
            type=int,
            default=512,
            help='512 for resnet18',
        )
        parser.add_argument('--kl_coeff', type=float, default=0.1)
        parser.add_argument('--latent_dim', type=int, default=256)

        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--data_dir', type=str, default='.')

        return parser


def cli_main(args=None):
    import yaml
    from medmnist.dataset import TissueMNIST
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
    parser = ResNetVAE.add_model_specific_args(parser)
    parser.set_defaults(**config)
    args = parser.parse_args(unknown)

    if args.dataset == 'tissuemnist':
        dataset_class = TissueMNIST
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    datamodule = MedMNISTDataModule(dataset_class, **args.__dict__)
    args.input_height = datamodule.size

    model = ResNetVAE(**args.__dict__)

    # Setup model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='best-checkpoint',
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

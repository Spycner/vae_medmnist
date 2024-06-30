import logging
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_medmnist.dataloader.medmnist_datamodule import MedMNISTDataModule
from vae_medmnist.models.components import betavae_encoder, betavae_decoder  # Import the new functions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class BetaVAE(pl.LightningModule):
    """BetaVAE model class."""

    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        in_channels: int = 3,
        input_height: int = 32,
        lr: float = 1e-4,
        kl_coeff: float = 1.0,  # Typically beta is represented as kl_coeff
        enc_out_dim: int = 512,
        latent_dim: int = 256,
        hidden_dims: list = None,
        beta: int = 4,
        gamma: float = 1000.,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = 'B',
        first_conv: bool = False,
        maxpool1: bool = False,
        **kwargs,
    ):
        """Initialize the BetaVAE model."""
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        self.encoder = betavae_encoder(first_conv, maxpool1)
        self.decoder = betavae_decoder(self.latent_dim, self.hparams.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    def encode(self, input: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder(z)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recons, input, mu, log_var = self(x)
        loss_dict = self.loss_function(recons, input, mu, log_var, M_N=1.0)  # Set M_N to 1.0 for now
        self.log('train_loss', loss_dict['loss'])
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recons, input, mu, log_var = self(x)
        loss_dict = self.loss_function(recons, input, mu, log_var, M_N=1.0)  # Set M_N to 1.0 for now
        self.log('val_loss', loss_dict['loss'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BetaVAE")
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--input_height', type=int, default=32)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--kl_coeff', type=float, default=1.0)
        parser.add_argument('--enc_out_dim', type=int, default=512)
        parser.add_argument('--latent_dim', type=int, default=256)
        parser.add_argument('--hidden_dims', type=list, default=[32, 64, 128, 256, 512])
        parser.add_argument('--beta', type=int, default=4)
        parser.add_argument('--gamma', type=float, default=1000.0)
        parser.add_argument('--max_capacity', type=int, default=25)
        parser.add_argument('--Capacity_max_iter', type=int, default=1e5)
        parser.add_argument('--loss_type', type=str, default='B')
        parser.add_argument('--first_conv', action='store_true')
        parser.add_argument('--maxpool1', action='store_true')
        return parent_parser


def cli_main(args=None):
    import yaml
    from medmnist.dataset import TissueMNIST
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the configuration file')

    args, unknown = parser.parse_known_args(args)

    with open(args.config) as file:
        config = yaml.safe_load(file)
        logger.debug(f'Loaded configuration: {config}')

    parser = BetaVAE.add_model_specific_args(parser)
    parser.set_defaults(**config)
    args = parser.parse_args(unknown)

    if args.dataset == 'tissuemnist':
        dataset_class = TissueMNIST
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    datamodule = MedMNISTDataModule(dataset_class, **args.__dict__)
    args.input_height = datamodule.size

    model = BetaVAE(**args.__dict__)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    csv_logger = CSVLogger(save_dir=args.log_dir, name='training_logs')

    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[checkpoint_callback], logger=csv_logger)
    trainer.fit(model, datamodule=datamodule)
    return datamodule, model, trainer


if __name__ == '__main__':
    datamodule, model, trainer = cli_main()

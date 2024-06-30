"""model.py"""
import logging
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from vae_medmnist.dataloader.medmnist_datamodule import MedMNISTDataModule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H(pl.LightningModule):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.save_hyperparameters()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    
          
    def add_model_specific_args(parent_parser):
        """Add model-specific arguments to the parser."""
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

    
  

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

def cli_main(args=None):
    """Main function for command-line interface."""
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
    parser = BetaVAE_H.add_model_specific_args(parser)
    parser.set_defaults(**config)
    args = parser.parse_args(unknown)

    if args.dataset == 'tissuemnist':
        dataset_class = TissueMNIST
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    datamodule = MedMNISTDataModule(dataset_class, **args.__dict__)
    args.input_height = datamodule.size

    model = BetaVAE_H(**args.__dict__)

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

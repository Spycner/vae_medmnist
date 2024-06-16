from typing import List

import torch
from torch import device, nn, tensor as Tensor
import torch.nn.functional as F
import pytorch_lightning as pl


class FcVAE(pl.LightningModule):
    """
    A simple Variational Autoencoder (VAE) implemented using PyTorch Lightning.

    Attributes:
        encoder (nn.Sequential): The encoder part of the VAE, which encodes the input into a latent space.
        fc_mu (nn.Linear): Fully connected layer to compute the mean of the latent space.
        fc_var (nn.Linear): Fully connected layer to compute the variance of the latent space.
        decoder (nn.Sequential): The decoder part of the VAE, which decodes the latent space back to the input space.

    Methods:
        encode(x): Encodes the input tensor into the latent space.
        reparameterize(mu, logvar): Reparameterizes the latent space using the mean and variance.
        decode(z): Decodes the latent space back to the input space.
        forward(x): Forward pass through the VAE.
        loss_function(recon_x, x, mu, logvar): Computes the VAE loss, which is a combination of reconstruction loss and KL divergence.
        training_step(batch, batch_idx): Training step for the VAE.
        validation_step(batch, batch_idx): Validation step for the VAE.
        configure_optimizers(): Configures the optimizer for training.
    """

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dims: List = [400],
        latent_dim: int = 128,
        lr: float = 1e-3,
    ):
        """
        Initializes the SimpleVAE model.

        Args:
            input_dim (int): The dimension of the input data.
            hidden_dims (List[int]): A list of integers specifying the dimensions of the hidden layers.
            latent_dim (int): The dimension of the latent space.
        """
        super(FcVAE, self).__init__()
        self.lr = lr

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU())
                for i in range(len(hidden_dims) - 1)
            ],
        )

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        hidden_dims = hidden_dims[::-1]

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU())
                for i in range(len(hidden_dims) - 1)
            ],
            nn.Linear(hidden_dims[-1], input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """
        Encodes the input tensor into the latent space.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor, torch.Tensor): The mean and variance of the latent space.
        """
        print(f"Training step: x - {x.min().item()} to {x.max().item()}")
        encoded = self.encoder(x)
        print(
            f"after encoder: encoded - {encoded.min().item()} to {encoded.max().item()}"
        )
        return self.fc_mu(encoded), self.fc_var(encoded)

    def reparameterize(self, mu, logvar):
        """
        Reparameterizes the latent space using the mean and variance.

        Args:
            mu (torch.Tensor): The mean of the latent space.
            logvar (torch.Tensor): The log variance of the latent space.

        Returns:
            torch.Tensor: The reparameterized latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decodes the latent space back to the input space.

        Args:
            z (torch.Tensor): The latent space tensor.

        Returns:
            torch.Tensor: The decoded tensor.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): The reconstructed tensor, mean, and variance.
        """
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)

        if torch.isnan(mu).any() or torch.isnan(logvar).any():
            print("NaNs detected in mu or logvar")

        if torch.isnan(z).any():
            print("NaNs detected in z")

        decoded = self.decode(z)

        if torch.isnan(decoded).any():
            print("NaNs detected in decoded output")

        return decoded, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Computes the VAE loss, which is a combination of reconstruction loss and KL divergence.

        Args:
            recon_x (torch.Tensor): The reconstructed tensor.
            x (torch.Tensor): The original input tensor.
            mu (torch.Tensor): The mean of the latent space.
            logvar (torch.Tensor): The log variance of the latent space.

        Returns:
            torch.Tensor: The computed loss.
        """
        recon_loss = F.binary_cross_entropy(
            recon_x, x.view(-1, 28 * 28), reduction="sum"
        )
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss

        return {
            "Total_Loss": loss,
            "Reconstruction_Loss": recon_loss.detach(),
            "KLD_Loss": kld_loss.detach(),
        }

    def training_step(self, batch, batch_idx):
        """
        Training step for the VAE.

        Args:
            batch (tuple): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        x, _ = batch
        x_hat, mu, log_var = self(x)

        print(f"Input range: {x.min().item()} to {x.max().item()}")
        print(f"Output range: {x_hat.min().item()} to {x_hat.max().item()}")
        print(f"mu range: {mu.min().item()} to {mu.max().item()}")
        print(f"log_var range: {log_var.min().item()} to {log_var.max().item()}")

        loss_dict = self.loss_function(x, x_hat, mu, log_var)
        self.log(
            "train_loss",
            loss_dict["Total_Loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_recon_loss",
            loss_dict["Reconstruction_Loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_kld_loss",
            loss_dict["KLD_Loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss_dict["Total_Loss"]

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the VAE.

        Args:
            batch (tuple): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        x, _ = batch
        x_hat, mu, log_var = self(x)

        print(f"Input range: {x.min().item()} to {x.max().item()}")
        print(f"Output range: {x_hat.min().item()} to {x_hat.max().item()}")
        print(f"mu range: {mu.min().item()} to {mu.max().item()}")
        print(f"log_var range: {log_var.min().item()} to {log_var.max().item()}")

        loss_dict = self.loss_function(x, x_hat, mu, log_var)
        self.log(
            "val_loss",
            loss_dict["Total_Loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_recon_loss",
            loss_dict["Reconstruction_Loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_kld_loss",
            loss_dict["KLD_Loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss_dict["Total_Loss"]

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def sample(self, num_samples: int, device: device) -> Tensor:
        """
        Samples from the latent space and decodes to generate new samples.

        Parameters:
        -----------
        num_samples : int
            Number of samples to generate.
        device : torch.device
            Device to perform sampling on (CPU or GPU).

        Returns:
        --------
        Tensor
            Generated samples.
        """

        x = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decoder(x)

    def generate(self, x: Tensor) -> Tensor:
        """
        Generates output by passing input through the VAE.

        Parameters:
        -----------
        x : Tensor
            Input tensor.

        Returns:
        --------
        Tensor
            Generated output tensor.
        """

        return self.forward(x)[0]

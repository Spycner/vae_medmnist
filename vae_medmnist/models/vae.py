from typing import List, Tuple

import pytorch_lightning as pl
import torch
from torch import device, nn, optim, tensor as Tensor


class Encoder(nn.Module):
    """
    Encoder module for a Variational Autoencoder (VAE).

    This class defines the architecture of the encoder, which takes an input image
    and maps it to the parameters of a latent space distribution (mean and log-variance).
    The encoder consists of a series of convolutional layers followed by fully connected layers
    to produce the latent space parameters.

    Attributes:
    -----------
    input_channels : int
        Number of input channels (e.g., 1 for grayscale, 3 for RGB).
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dims : List[int]
        List of integers where each integer represents the number of output channels
        for a convolutional layer in the encoder. If not provided, a default list of
        [32, 64, 128, 256, 512] will be used.

    Methods:
    --------
    forward(input: Tensor) -> List[Tensor]:
        Forward pass of the encoder. Takes an input tensor and returns the mean and
        log-variance of the latent space distribution.

    Parameters:
    -----------
    input_channels : int
        Number of input channels (e.g., 1 for grayscale, 3 for RGB).
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dims : List[int], optional
        List of integers where each integer represents the number of output channels
        for a convolutional layer in the encoder. If None, a default list of
        [32, 64, 128, 256, 512] will be used.

    Example:
    --------
    >>> encoder = Encoder(input_channels=1, latent_dim=20, hidden_dims=[32, 64, 128])
    >>> input_tensor = torch.randn(16, 1, 28, 28)  # batch of 16 grayscale images of size 28x28
    >>> mu, log_var = encoder(input_tensor)
    >>> print(mu.shape, log_var.shape)
    torch.Size([16, 20]) torch.Size([16, 20])
    """

    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        hidden_dims: List[int] = [32, 64, 128, 256, 512],
    ):
        super(Encoder, self).__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        self.encoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        input_channels,
                        out_channels=hidden_dims[0],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[0]),
                    nn.LeakyReLU(),
                ),
                *[
                    nn.Sequential(
                        nn.Conv2d(
                            hidden_dims[i],
                            out_channels=hidden_dims[i + 1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU(),
                    )
                    for i in range(len(hidden_dims) - 1)
                ],
            ]
        )

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, input: Tensor) -> List[Tensor]:
        """
        Forward pass of the encoder.

        Parameters:
        -----------
        input : Tensor
            Input tensor of shape (batch_size, input_dim, height, width).

        Returns:
        --------
        List[Tensor]
            A list containing two tensors:
            - mu: Mean of the latent space distribution.
            - log_var: Log-variance of the latent space distribution.
        """

        x = self.encoder(input)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var


class Decoder(nn.Module):
    """
    Decoder module for a Variational Autoencoder (VAE).

    This class defines the architecture of the decoder, which takes a latent space
    representation and maps it back to the original input space.

    Attributes:
    -----------
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dims : List[int]
        List of integers where each integer represents the number of output channels
        for a transposed convolutional layer in the decoder.

    Methods:
    --------
    forward(input: Tensor) -> Tensor:
        Forward pass of the decoder. Takes a latent space tensor and returns the
        reconstructed input tensor.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int] = [32, 64, 128, 256, 512],
        output_channels: int = 1,
    ):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_channels = output_channels

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4 * 4)
        hidden_dims = hidden_dims[::-1]

        self.decoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
                for i in range(len(hidden_dims) - 1)
            ]
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(
                hidden_dims[-1],
                hidden_dims[-1] // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1] // 2),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims[-1] // 2,
                out_channels=output_channels // 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1] // 4),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims[-1] // 4,
                out_channels=output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of the decoder.

        Parameters:
        -----------
        input : Tensor
            Input tensor from the latent space of shape (batch_size, latent_dim).

        Returns:
        --------
        Tensor
            Reconstructed input tensor of shape (batch_size, output_channels, height, width).
        """

        print(f"Input shape: {input.shape}")
        x = self.decoder_input(input)
        print(f"Shape after decoder_input: {x.shape}")
        x = x.view(-1, self.hidden_dims[-1], 4, 4)
        print(f"Shape after view: {x.shape}")
        x = self.decoder(x)
        print(f"Shape after decoder: {x.shape}")
        x = self.final_layer(x)
        print(f"Shape after final_layer: {x.shape}")
        return x


class VAE(pl.LightningModule):
    """
    Variational Autoencoder (VAE) model.

    This class defines the architecture and functions for the VAE, including the encoder,
    decoder, reparameterization trick, forward pass, loss calculation, sampling, and generation.

    Attributes:
    -----------
    input_channels : int
        Number of input channels (e.g., 1 for grayscale, 3 for RGB).
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dims : List[int]
        List of integers where each integer represents the number of output channels
        for a convolutional layer in the encoder.
    output_channels : int
        Number of output channels (e.g., 1 for grayscale, 3 for RGB).

    Methods:
    --------
    reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        Reparameterization trick to sample from a Gaussian distribution.

    forward(input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        Forward pass of the VAE. Encodes input, samples from latent space, and decodes.

    loss_function(input: Tensor, result: Tensor, mu: Tensor, log_var: Tensor, **kwargs) -> dict:
        Calculates the VAE loss including reconstruction loss and KL divergence.

    sample(num_samples: int, device: torch.device) -> Tensor:
        Samples from the latent space and decodes to generate new samples.

    generate(x: Tensor) -> Tensor:
        Generates output by passing input through the VAE.
    """

    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        hidden_dims: List[int] = [32, 64, 128, 256, 512],
    ):
        super(VAE, self).__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_channels = input_channels

        self.encoder = Encoder(self.input_channels, self.latent_dim, self.hidden_dims)
        self.decoder = Decoder(self.latent_dim, self.hidden_dims, self.output_channels)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from a Gaussian distribution.

        Parameters:
        -----------
        mu : Tensor
            Mean of the latent space distribution.
        log_var : Tensor
            Log-variance of the latent space distribution.

        Returns:
        --------
        Tensor
            Sampled latent space tensor.
        """

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, log_var = self.encoder(input)
        intermediate = self.reparameterize(mu, log_var)
        return self.decoder(intermediate), mu, log_var

    def loss_function(
        self, input: Tensor, result: Tensor, mu: Tensor, log_var: Tensor, **kwargs
    ) -> dict:
        """
        Calculates the VAE loss including reconstruction loss and KL divergence.

        Parameters:
        -----------
        input : Tensor
            Original input tensor.
        result : Tensor
            Reconstructed tensor from the decoder.
        mu : Tensor
            Mean of the latent space distribution.
        log_var : Tensor
            Log-variance of the latent space distribution.
        kld_weight : float
            Weight of the KL divergence term in the total loss.

        Returns:
        --------
        dict
            Dictionary containing total loss, reconstruction loss, and KL divergence.
        """

        recon_loss = nn.functional.mse_loss(result, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0
        )
        kld_weight = kwargs.get("kld_weight", 1.0)

        loss = recon_loss + kld_weight * kld_loss

        return {
            "Total_Loss": loss,
            "Reconstruction_Loss": recon_loss,
            "KLD_Loss": kld_loss,
        }

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self(x)
        loss = self.loss_function(x, x_hat, mu, log_var)
        self.log("train_loss", loss["Total_Loss"])
        return loss["Total_Loss"]

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self(x)
        loss = self.loss_function(x, x_hat, mu, log_var)
        self.log("val_loss", loss["Total_Loss"])
        return loss["Total_Loss"]

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

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

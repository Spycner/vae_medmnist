import argparse
import os
import yaml

from medmnist import INFO
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from vae_medmnist.models.vae import VAE
from vae_medmnist.models.dataloader import MedMNISTDataModule


def find_model(checkpoint_path):
    """
    Searches for a model checkpoint file in the specified directory.

    Args:
        checkpoint_path (str): Path to the directory containing the model checkpoint.

    Returns:
        str: Full path to the model checkpoint file if found, else None.
    """
    for file in os.listdir(checkpoint_path):
        if file.endswith(".ckpt"):
            return os.path.join(checkpoint_path, file)
    return None


def load_model(checkpoint_path, input_channels, latent_dim):
    """
    Loads a VAE model from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the directory containing the model checkpoint.
        input_channels (int): Number of input channels for the VAE model.
        latent_dim (int): Dimensionality of the latent space for the VAE model.

    Returns:
        VAE: The loaded VAE model.

    Raises:
        ValueError: If no model checkpoint is found in the specified directory.
    """
    checkpoint_path = find_model(checkpoint_path)
    if checkpoint_path is None:
        raise ValueError(f"No model found in {checkpoint_path}")

    model = VAE(input_channels=input_channels, latent_dim=latent_dim)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()
    return model


def plot_images(images, title, n_rows, n_cols):
    """
    Plots a grid of images.

    Args:
        images (torch.Tensor): Tensor containing the images to be plotted.
        title (str): Title of the plot.
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i].reshape(28, 28), cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()

    return fig


def calculate_metrics(original, reconstruction):
    """
    Calculates SSIM and PSNR metrics between original and reconstructed images.

    Args:
        original (torch.Tensor): Original image tensor.
        reconstruction (torch.Tensor): Reconstructed image tensor.

    Returns:
        tuple: SSIM and PSNR scores.
    """
    original = original.squeeze().cpu().numpy()
    reconstruction = reconstruction.squeeze().cpu().numpy()
    ssim_score = ssim(
        original, reconstruction, data_range=original.max() - original.min()
    )
    psnr_score = psnr(
        original, reconstruction, data_range=original.max() - original.min()
    )
    return ssim_score, psnr_score


def main():
    """
    Main function to load a VAE model, reconstruct images, and evaluate the reconstruction quality.

    This function performs the following steps:
    1. Parses command-line arguments to get the path to the configuration file.
    2. Loads the configuration file.
    3. Loads the VAE model from the specified checkpoint directory.
    4. Loads the test dataset using MedMNISTDataModule.
    5. Reconstructs images using the loaded VAE model.
    6. Plots and saves the original and reconstructed images.
    7. Calculates and saves SSIM and PSNR metrics for the reconstructed images.
    """
    parser = argparse.ArgumentParser(description="VAE MedMNIST Training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the experiments config file"
    )
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    model = load_model(
        config["checkpoint_dir"],
        INFO[config["data_flag"]]["n_channels"],
        config["latent_dim"],
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    data_module = MedMNISTDataModule(
        data_flag=config["data_flag"], batch_size=config["batch_size"]
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()

    for batch in test_loader:
        original_images, _ = batch
        original_images = original_images.to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        with torch.no_grad():
            reconstructed_images, _, _ = model(original_images)
        break

    # Plot original and reconstructed images
    n_images = 10
    original_images = original_images[:n_images].cpu()
    reconstructed_images = reconstructed_images[:n_images].cpu()

    fig_original = plot_images(
        original_images, title="Original Images", n_rows=2, n_cols=5
    )

    fig_reconstructed = plot_images(
        reconstructed_images, title="Reconstructed Images", n_rows=2, n_cols=5
    )

    # Calculate and print SSIM and PSNR for the first n_images
    ssim_scores = []
    psnr_scores = []
    for i in range(n_images):
        ssim_score, psnr_score = calculate_metrics(
            original_images[i], reconstructed_images[i]
        )
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)

    os.makedirs(config["eval_dir"], exist_ok=True)

    with open(f"{config['eval_dir']}/image_metrics.txt", "w") as file:
        file.write(f"Average SSIM: {sum(ssim_scores) / n_images}\n")
        file.write(f"Average PSNR: {sum(psnr_scores) / n_images} dB\n")

    # Optionally save the figures
    fig_original.savefig(f"{config['eval_dir']}/original_images.png")
    fig_reconstructed.savefig(f"{config['eval_dir']}/reconstructed_images.png")


if __name__ == "__main__":
    main()

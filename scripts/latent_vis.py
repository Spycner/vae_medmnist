import argparse
import os
import yaml

import matplotlib.pyplot as plt
from medmnist import INFO
from sklearn.manifold import TSNE
import seaborn as sns
import torch

from vae_medmnist.models.vae import VAE
from vae_medmnist.models.dataloader import MedMNISTDataModule


def find_model(checkpoint_path):
    """
    Searches for a model checkpoint file in the given directory.

    Args:
        checkpoint_path (str): Path to the directory containing the checkpoint files.

    Returns:
        str: Full path to the checkpoint file if found, otherwise None.
    """
    for file in os.listdir(checkpoint_path):
        if file.endswith(".ckpt"):
            return os.path.join(checkpoint_path, file)
    return None


def load_model(checkpoint_path, input_channels, latent_dim):
    """
    Loads a VAE model from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the directory containing the checkpoint files.
        input_channels (int): Number of input channels for the VAE model.
        latent_dim (int): Dimensionality of the latent space.

    Returns:
        VAE: The loaded VAE model.

    Raises:
        ValueError: If no checkpoint file is found in the given directory.
    """
    checkpoint_path = find_model(checkpoint_path)
    if checkpoint_path is None:
        raise ValueError(f"No model found in {checkpoint_path}")

    model = VAE(input_channels=input_channels, latent_dim=latent_dim)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()
    return model


def visualize_latent_space(model, data_loader, n_samples, save_path):
    """
    Visualizes the latent space of the VAE model using t-SNE.

    Args:
        model (VAE): The VAE model.
        data_loader (DataLoader): DataLoader for the dataset.
        n_samples (int): Number of samples to visualize.
        save_path (str): Directory to save the visualization plot.
    """
    model.eval()
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            images, targets = batch
            _, mu, _ = model(images)
            latent_vectors.append(mu.cpu())
            labels.append(targets.cpu())
            if len(latent_vectors) * len(images) >= n_samples:
                break
    latent_vectors = torch.cat(latent_vectors)[:n_samples]
    labels = torch.cat(labels)[:n_samples].flatten()

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="viridis"
    )
    plt.title("Latent Space Visualization")
    plt.savefig(os.path.join(save_path, "latent_space.png"))


def main():
    """
    Main function to run the script. Parses the configuration file, loads the model,
    sets up the data module, and visualizes the latent space.
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
    model.to("cpu")

    data_module = MedMNISTDataModule(
        data_flag=config["data_flag"], batch_size=config["batch_size"]
    )
    data_module.setup()
    visualize_latent_space(model, data_module.val_dataloader(), 500, config["eval_dir"])


if __name__ == "__main__":
    main()

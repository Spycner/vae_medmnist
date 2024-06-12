import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main(version: str):
    # Load the metrics file
    file_path = f"results/logs/vae_training/version_{version}/metrics.csv"
    metrics_df = pd.read_csv(file_path)
    # Plot the training KLD loss and reconstruction loss
    plt.figure(figsize=(14, 6))

    # Plot KLD loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics_df["step"], metrics_df["train_kld_loss_step"], label="KLD Loss")
    plt.xlabel("Training Step")
    plt.ylabel("KLD Loss")
    plt.title("Training KLD Loss")
    plt.legend()

    # Plot Reconstruction loss (MSE)
    plt.subplot(1, 2, 2)
    plt.plot(
        metrics_df["step"],
        metrics_df["train_recon_loss_step"],
        label="Reconstruction Loss (MSE)",
        color="orange",
    )
    plt.xlabel("Training Step")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("Training Reconstruction Loss (MSE)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"results/logs/vae_training/version_{version}/training_plot.png")
    print(
        f"Plot saved to results/logs/vae_training/version_{version}/training_plot.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize different versions of metrics"
    )
    parser.add_argument(
        "-version", type=str, help="The version of the metrics to visualize"
    )
    args = parser.parse_args()
    main(args.version)

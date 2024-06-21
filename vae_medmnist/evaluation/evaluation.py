import matplotlib.pyplot as plt
import torch


def save_metrics_plot(metrics, save_path):
    """Save plots of training and validation metrics to the specified path."""
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("The 'metrics' input must be a pandas DataFrame.")
    # Plotting training losses
    plt.figure()
    if 'train_loss' in metrics.columns:
        plt.plot(metrics['train_loss'].dropna(), label='Train Loss')
    if 'train_kl' in metrics.columns:
        plt.plot(metrics['train_kl'].dropna(), label='Train KL')
    if 'train_recon_loss' in metrics.columns:
        plt.plot(metrics['train_recon_loss'].dropna(), label='Train Recon Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig(f'{save_path}/train_losses.png')
    plt.close()

    # Plotting validation losses
    plt.figure()
    if 'val_loss' in metrics.columns:
        plt.plot(metrics['val_loss'].dropna(), label='Validation Loss')
    if 'val_kl' in metrics.columns:
        plt.plot(metrics['val_kl'].dropna(), label='Validation KL')
    if 'val_recon_loss' in metrics.columns:
        plt.plot(metrics['val_recon_loss'].dropna(), label='Validation Recon Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Losses')
    plt.legend()
    plt.savefig(f'{save_path}/val_losses.png')
    plt.close()

    # Plotting comparison of overall train and validation losses
    plt.figure()
    if 'train_loss' in metrics.columns:
        plt.plot(metrics['train_loss'].dropna(), label='Train Loss')
    if 'val_loss' in metrics.columns:
        plt.plot(metrics['val_loss'].dropna(), label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Comparison of Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{save_path}/loss_comparison.png')
    plt.close()


def save_generated_images(model, dataloader, num_samples, device, save_path):
    """Save generated images from the model to the specified path."""
    model.eval()
    samples = []
    labels = []

    with torch.no_grad():
        if isinstance(model, CVAE):
            for class_idx in range(model.num_classes):
                num_samples_per_class = max(1, num_samples // model.num_classes)
                class_samples = model.sample(num_samples_per_class, class_idx, device)
                samples.append(class_samples)
                labels.extend([class_idx] * num_samples_per_class)
            samples = torch.cat(samples, dim=0)
        else:
            samples = model.sample(num_samples, device)

    # Calculate the number of rows and columns for the grid
    num_cols = min(5, len(samples))  # Maximum 5 columns
    num_rows = (len(samples) - 1) // num_cols + 1

    # Create a larger figure with more appropriate dimensions
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    axs = axs.flatten()

    for _, (ax, img, label) in enumerate(zip(axs, samples, labels)):
        ax.imshow(img.cpu().permute(1, 2, 0), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Class {label}\n{dataloader.labels[label]}', fontsize=8)

    # Remove any unused subplots
    for i in range(len(samples), len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.suptitle('Generated Images', fontsize=16, y=1.02)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def save_reconstructions(model, dataloader, device, save_path, descriptions):
    """Save reconstructions of images from each class in the dataloader to the specified path."""
    model.eval()
    class_samples = {}

    # Collect one sample from each class
    for inputs, labels in dataloader:
        for input, label in zip(inputs, labels):
            label = label.item()
            if label not in class_samples:
                class_samples[label] = input
            if len(class_samples) == model.num_classes:
                break
        if len(class_samples) == model.num_classes:
            break

    inputs = torch.stack(list(class_samples.values())).to(device)
    labels = torch.tensor(list(class_samples.keys())).to(device)

    with torch.no_grad():
        if isinstance(model, CVAE):
            reconstructions, _, _ = model(inputs, labels)
        elif isinstance(model, ResNetVAE):
            _, reconstructions, _, _ = model(inputs)
        elif isinstance(model, VAE):
            reconstructions, _, _ = model(inputs)
        else:
            raise NotImplementedError('Model type not supported for reconstructions.')

    # Calculate the number of rows and columns for the grid
    num_rows = len(class_samples)

    fig, axs = plt.subplots(num_rows, 2, figsize=(6, 3 * num_rows))

    for i, (input, reconstruction) in enumerate(zip(inputs, reconstructions)):
        # Original image
        axs[i, 0].imshow(input.permute(1, 2, 0).cpu(), cmap='gray')
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Input\nClass {labels[i].item()}\n{descriptions[labels[i].item()]}', fontsize=8)

        # Reconstructed image
        axs[i, 1].imshow(reconstruction.permute(1, 2, 0).cpu(), cmap='gray')
        axs[i, 1].axis('off')
        axs[i, 1].set_title(f'Reconstruction\nClass {labels[i].item()}\n{descriptions[labels[i].item()]}', fontsize=8)

    plt.tight_layout()
    plt.suptitle('Original and Reconstructed Images', fontsize=16, y=1.02)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':
    import argparse
    import os

    import pandas as pd
    import yaml
    from medmnist.dataset import TissueMNIST

    from vae_medmnist.dataloader.accumulated_dataset import AccumulatedMedMNIST
    from vae_medmnist.dataloader.medmnist_datamodule import MedMNISTDataModule
    from vae_medmnist.models.cvae import CVAE
    from vae_medmnist.models.resnet_vae import ResNetVAE
    from vae_medmnist.models.vae import VAE

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, help='Path to the training logs')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for computation')
    args = parser.parse_args()

    hparams_path = os.path.join(args.log_path, 'hparams.yaml')
    with open(hparams_path) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    metrics_path = os.path.join(args.log_path, 'metrics.csv')
    metrics_df = pd.read_csv(metrics_path)

    if hparams['model'] == 'resnet_vae':
        model = ResNetVAE.load_from_checkpoint(f"{hparams['checkpoint_dir']}/best-checkpoint.ckpt")
    elif hparams['model'] == 'cvae':
        model = CVAE.load_from_checkpoint(f"{hparams['checkpoint_dir']}/cvae-best-checkpoint.ckpt")
    else:
        model = VAE.load_from_checkpoint(f"{hparams['checkpoint_dir']}/best-checkpoint.ckpt")
    model.to(args.device)

    if hparams['datasets'] == 'tissuemnist':
        datasetclass = TissueMNIST
        datamodule = MedMNISTDataModule(datasetclass, batch_size=10)
    elif isinstance(hparams['datasets'], list):
        datamodule = AccumulatedMedMNIST(hparams['datasets'], batch_size=10)
    else:
        raise ValueError(f"Unknown dataset: {hparams['datasets']}")

    datamodule.setup()

    save_metrics_plot(metrics_df, args.log_path)
    save_generated_images(
        model,
        datamodule,
        args.num_samples,
        args.device,
        os.path.join(args.log_path, 'generated_images.png'),
    )
    save_reconstructions(
        model,
        datamodule.test_dataloader(),
        args.device,
        os.path.join(args.log_path, 'reconstructions.png'),
        datamodule.labels,
    )

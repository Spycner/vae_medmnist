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


def save_generated_images(model, num_samples, device, save_path):
    """Save generated images from the model to the specified path."""
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, device)

    if num_samples % 2 == 0:
        fig, axs = plt.subplots(2, num_samples // 2, figsize=(15, 3))
    else:
        fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))

    axs = axs.flatten()

    for ax, img in zip(axs, samples):
        ax.imshow(img.cpu().permute(1, 2, 0), cmap='gray')
        ax.axis('off')

    plt.suptitle('Generated Images')
    plt.savefig(save_path)
    plt.close()


def save_reconstructions(model, dataloader, device, save_path):
    """Save reconstructions of the first batch of images from the dataloader to the specified path."""
    model.eval()
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(device)
    with torch.no_grad():
        if isinstance(model, ResNetVAE):
            _, reconstructions, _, _ = model(inputs)
        elif isinstance(model, VAE):
            reconstructions, _, _ = model(inputs)
        else:
            raise NotImplementedError('Model type not supported for reconstructions.')

    fig, axs = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axs[0, i].imshow(inputs[i].permute(1, 2, 0).cpu(), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title('Input', fontsize=8, loc='center')
        axs[1, i].imshow(reconstructions[i].permute(1, 2, 0).cpu(), cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title('Reconstruction', fontsize=8, loc='center')
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    import argparse
    import os

    import pandas as pd
    import yaml
    from medmnist.dataset import TissueMNIST

    from vae_medmnist.datamodules.medmnist_datamodule import MedMNISTDataModule
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
    else:
        model = VAE.load_from_checkpoint(f"{hparams['checkpoint_dir']}/best-checkpoint.ckpt")
    model.to(args.device)

    if hparams['dataset'] == 'tissuemnist':
        datasetclass = TissueMNIST
    else:
        raise ValueError(f"Unknown dataset: {hparams['dataset']}")

    datamodule = MedMNISTDataModule(datasetclass, batch_size=10)
    datamodule.setup()

    save_metrics_plot(metrics_df, args.log_path)
    save_generated_images(model, args.num_samples, args.device, os.path.join(args.log_path, 'generated_images.png'))
    save_reconstructions(
        model, datamodule.test_dataloader(), args.device, os.path.join(args.log_path, 'reconstructions.png')
    )

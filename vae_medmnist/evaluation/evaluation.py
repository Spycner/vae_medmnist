import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import linalg
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from skimage.metrics import normalized_mutual_information as nmi
from tqdm import tqdm


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
    plt.xlabel('Steps')
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
    plt.xlabel('Steps')
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
    plt.xlabel('Steps')
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

    # Calculate the number of rows for the grid
    num_classes = len(class_samples)
    num_cols = 4  # Two pairs of original and reconstruction
    num_rows = (num_classes + 1) // 2  # Divide classes into two rows, rounding up

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))

    for i, (input, reconstruction) in enumerate(zip(inputs, reconstructions)):
        row = i // 2
        col = (i % 2) * 2

        # Original image
        axs[row, col].imshow(input.permute(1, 2, 0).cpu(), cmap='gray')
        axs[row, col].axis('off')
        axs[row, col].set_title(f'Input\nClass {labels[i].item()}\n{descriptions[labels[i].item()]}', fontsize=8)

        # Reconstructed image
        axs[row, col + 1].imshow(reconstruction.permute(1, 2, 0).cpu(), cmap='gray')
        axs[row, col + 1].axis('off')
        axs[row, col + 1].set_title(
            f'Reconstruction\nClass {labels[i].item()}\n{descriptions[labels[i].item()]}', fontsize=8
        )

    # Remove any unused subplots
    for i in range(num_classes, num_rows * 2):
        row = i // 2
        col = (i % 2) * 2
        axs[row, col].axis('off')
        axs[row, col + 1].axis('off')

    plt.tight_layout()
    plt.suptitle('Original and Reconstructed Images', fontsize=16, y=1.02)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def save_model_comparison_reconstructions(model1, model2, dataloader, device, save_path, descriptions):
    """Save reconstructions of images from each class for two models side by side."""
    model1.eval()
    model2.eval()
    class_samples = {}

    # Collect one sample from each class
    for inputs, labels in dataloader:
        for input, label in zip(inputs, labels):
            label = label.item()
            if label not in class_samples:
                class_samples[label] = input
            if len(class_samples) == model1.num_classes:
                break
        if len(class_samples) == model1.num_classes:
            break

    inputs = torch.stack(list(class_samples.values())).to(device)
    labels = torch.tensor(list(class_samples.keys())).to(device)

    with torch.no_grad():
        # Generate reconstructions for model1
        if isinstance(model1, CVAE):
            reconstructions1, _, _ = model1(inputs, labels)
        elif isinstance(model1, ResNetVAE):
            _, reconstructions1, _, _ = model1(inputs)
        elif isinstance(model1, VAE):
            reconstructions1, _, _ = model1(inputs)
        else:
            raise NotImplementedError('Model1 type not supported for reconstructions.')

        # Generate reconstructions for model2
        if isinstance(model2, CVAE):
            reconstructions2, _, _ = model2(inputs, labels)
        elif isinstance(model2, ResNetVAE):
            _, reconstructions2, _, _ = model2(inputs)
        elif isinstance(model2, VAE):
            reconstructions2, _, _ = model2(inputs)
        else:
            raise NotImplementedError('Model2 type not supported for reconstructions.')

    # Calculate the number of rows for the grid
    num_rows = len(class_samples)

    fig, axs = plt.subplots(num_rows, 3, figsize=(9, 3 * num_rows))

    for i, (input, recon1, recon2) in enumerate(zip(inputs, reconstructions1, reconstructions2)):
        # Original image
        axs[i, 0].imshow(input.permute(1, 2, 0).cpu(), cmap='gray')
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Input\nClass {labels[i].item()}\n{descriptions[labels[i].item()]}', fontsize=8)

        # Model1 reconstructed image
        axs[i, 1].imshow(recon1.permute(1, 2, 0).cpu(), cmap='gray')
        axs[i, 1].axis('off')
        axs[i, 1].set_title(f'Model1 Reconstruction\nClass {labels[i].item()}', fontsize=8)

        # Model2 reconstructed image
        axs[i, 2].imshow(recon2.permute(1, 2, 0).cpu(), cmap='gray')
        axs[i, 2].axis('off')
        axs[i, 2].set_title(f'Model2 Reconstruction\nClass {labels[i].item()}', fontsize=8)

    plt.tight_layout()
    plt.suptitle('Original and Reconstructed Images Comparison', fontsize=16, y=1.02)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def calculate_fid(real_images: np.ndarray, generated_images: np.ndarray) -> float:
    """Calculate the Fréchet Inception Distance between two sets of images."""
    # Flatten the images to 2D arrays
    real_images_flat = real_images.reshape(real_images.shape[0], -1)
    generated_images_flat = generated_images.reshape(generated_images.shape[0], -1)

    mu1, sigma1 = np.mean(real_images_flat, axis=0), np.cov(real_images_flat, rowvar=False)
    mu2, sigma2 = np.mean(generated_images_flat, axis=0), np.cov(generated_images_flat, rowvar=False)

    try:
        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = f'fid calculation produces singular product; adding {1e-6} to diagonal of cov estimates'
            print(msg)
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        return fid
    except ValueError as e:
        print(f'Error in FID calculation: {e}')
        return float('nan')


def calculate_image_metrics(real_images: np.ndarray, generated_images: np.ndarray) -> dict:
    """Calculate MSE and PSNR between two sets of images."""
    mse_scores = []
    psnr_scores = []

    for real_image, generated_image in zip(real_images, generated_images):
        # Ensure images are 2D (grayscale)
        if real_image.ndim == 4:
            real_image = real_image.squeeze()
        if generated_image.ndim == 4:
            generated_image = generated_image.squeeze()

        if real_image.ndim == 3:
            real_image = real_image[0]  # Take the first channel if it's a 3D image
        if generated_image.ndim == 3:
            generated_image = generated_image[0]  # Take the first channel if it's a 3D image

        try:
            mse = mean_squared_error(real_image, generated_image)
            psnr = peak_signal_noise_ratio(real_image, generated_image, data_range=real_image.max() - real_image.min())

            mse_scores.append(mse)
            psnr_scores.append(psnr)
        except Exception as e:
            print(f'Error calculating metrics for an image pair: {e}')
            print(f'Image shapes: real {real_image.shape}, generated {generated_image.shape}')
            continue

    if not mse_scores:
        print('Warning: Could not calculate metrics for any image pair.')
        return {'MSE': float('nan'), 'PSNR': float('nan')}

    return {'MSE': np.mean(mse_scores), 'PSNR': np.mean(psnr_scores)}


def mutual_information(real_images: np.ndarray, generated_images: np.ndarray) -> float:
    """Calculate the Mutual Information between two sets of images."""
    nmi_scores = []
    for real_image, generated_image in zip(real_images, generated_images):
        nmi_score = nmi(real_image.flatten(), generated_image.flatten())
        nmi_scores.append(nmi_score)
    return np.mean(nmi_scores)


def sample_diversity(generated_images: np.ndarray) -> float:
    """Calculate the Sample Diversity between two sets of images."""
    return np.mean(np.std(generated_images, axis=0))


def run_metrics(model, dataloader, device):
    """Calculate and return various metrics for the VAE model."""
    model.eval()
    real_images = []
    generated_images = []
    reconstructed_images = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing batches'):
            inputs, labels = batch
            inputs = inputs.to(device)

            # Generate reconstructions
            if isinstance(model, CVAE):
                reconstructions, _, _ = model(inputs, labels)
            elif isinstance(model, ResNetVAE):
                _, reconstructions, _, _ = model(inputs)
            elif isinstance(model, VAE):
                reconstructions, _, _ = model(inputs)
            else:
                raise NotImplementedError('Model type not supported for reconstructions.')

            real_images.append(inputs.cpu().numpy())
            reconstructed_images.append(reconstructions.cpu().numpy())

            # Generate new samples
            if isinstance(model, CVAE):
                samples = model.sample(inputs.size(0), labels, device)
            else:
                samples = model.sample(inputs.size(0), device)

            generated_images.append(samples.cpu().numpy())

    real_images = np.concatenate(real_images)
    generated_images = np.concatenate(generated_images)
    reconstructed_images = np.concatenate(reconstructed_images)

    # Calculate metrics
    fid_score = calculate_fid(real_images, generated_images)
    image_metrics = calculate_image_metrics(real_images, reconstructed_images)
    mi_score = mutual_information(real_images, reconstructed_images)
    diversity_score = sample_diversity(generated_images)

    metrics = {
        'FID': fid_score,
        'MSE': image_metrics['MSE'],
        'PSNR': image_metrics['PSNR'],
        'Mutual Information': mi_score,
        'Sample Diversity': diversity_score,
    }

    return metrics


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
        model = VAE.load_from_checkpoint(f"{hparams['checkpoint_dir']}/vae-best-checkpoint.ckpt")
    model.to(args.device)

    # Load second model if compare_to is set
    if 'compare_to' in hparams and hparams['compare_to']:
        if hparams['compare_to'] == 'resnet_vae':
            model2 = ResNetVAE.load_from_checkpoint(f"{hparams['compare_dir']}/best-checkpoint.ckpt")
        elif hparams['compare_to'] == 'cvae':
            model2 = CVAE.load_from_checkpoint(f"{hparams['compare_dir']}/cvae-best-checkpoint.ckpt")
        else:
            model2 = VAE.load_from_checkpoint(f"{hparams['compare_dir']}/vae-best-checkpoint.ckpt")
        model2.to(args.device)
    else:
        model2 = None

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
    if model2 is not None:
        save_model_comparison_reconstructions(
            model,
            model2,
            datamodule.test_dataloader(),
            args.device,
            os.path.join(args.log_path, 'model_comparison_reconstructions.png'),
            datamodule.labels,
        )

    metrics = run_metrics(model, datamodule.test_dataloader(), args.device)
    metrics_file = os.path.join(args.log_path, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        for metric_name, metric_value in metrics.items():
            f.write(f'{metric_name}: {metric_value}\n')

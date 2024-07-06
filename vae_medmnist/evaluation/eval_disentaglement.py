import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from vae_medmnist.dataloader.medmnist_datamodule import MedMNISTDataModule
from vae_medmnist.models.beta_vae_v1 import BetaVAE  # Assuming your model definition is in beta_vae_v1.py
from medmnist.dataset import TissueMNIST, ChestMNIST

def load_model(checkpoint_path, hparams_file=None):
    model = BetaVAE.load_from_checkpoint(checkpoint_path, hparams_file=hparams_file)
    model.eval()
    return model

def visualize_latent_space_traversal(model, data, traversal_range=(-3, 3), steps=10):
    sample_data = data[0].unsqueeze(0)  # Select a single data sample
    traversal_values = np.linspace(traversal_range[0], traversal_range[1], steps)

    for i in range(model.latent_dim):
        fig, axes = plt.subplots(1, steps, figsize=(15, 3))
        for j, val in enumerate(traversal_values):
            with torch.no_grad():
                latent_sample = model.encoder(sample_data)
                latent_sample[0, i] = val
                reconstructed = model.decoder(latent_sample)
            axes[j].imshow(reconstructed.squeeze().numpy(), cmap='gray')
            axes[j].axis('off')
        plt.show()

def compute_metrics(model, dataloader):
    # Placeholder for computing Beta-VAE score, MIG, SAP, etc.
    # You'll need to define how to compute these metrics based on your specific setup
    metrics = {
        'beta_vae_score': 0.0,
        'mig': 0.0,
        'sap': 0.0,
    }
    return metrics

def main(args):
    model = load_model(args.checkpoint_path, args.hparams_file)
    
    # Load dataset based on hparams
    hparams = model.hparams
    if hparams['dataset'] == 'tissuemnist':
        datasetclass = TissueMNIST
        datamodule = MedMNISTDataModule(datasetclass, batch_size=10)
    elif isinstance(hparams['datasets'], list):
        datamodule = AccumulatedMedMNIST(hparams['datasets'], batch_size=10)
    else:
        raise ValueError(f"Unknown dataset: {hparams['datasets']}")
    
    datamodule.setup(stage='test')
    test_data = datamodule.test_dataloader()

    # Perform latent space traversal visualization
    data_iter = iter(test_data)
    sample_batch = next(data_iter)
    visualize_latent_space_traversal(model, sample_batch[0])

    # Compute metrics
    metrics = compute_metrics(model, test_data)
    print(metrics)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default="results/checkpoints/beta_vae/betavae-best-checkpoint-v24.ckpt", help='Path to the model checkpoint')
    parser.add_argument('--hparams_file', type=str, default="results/version_32/hparams.yaml", help='Path to the hyperparameters file')
    args = parser.parse_args()

    main(args)

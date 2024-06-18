# vae-medmnist

## Table of Contents

- [vae-medmnist](#vae-medmnist)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Features](#features)
    - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Quick-Start](#quick-start)
    - [Running the VAE Model](#running-the-vae-model)
    - [Evaluating the Model](#evaluating-the-model)
  - [Configuration](#configuration)
    - [Configuration File Structure](#configuration-file-structure)
    - [Viewing Available Configuration Options](#viewing-available-configuration-options)
  - [Results](#results)
    - [Version 1](#version-1)
      - [Experiment Details](#experiment-details)
      - [Performance Metrics](#performance-metrics)
        - [Training Loss](#training-loss)
        - [Validation Loss](#validation-loss)
        - [KL Divergence](#kl-divergence)
        - [Visual Outputs](#visual-outputs)
  - [Contributing](#contributing)
  - [License](#license)
  - [Citations](#citations)

## Introduction

This repository contains the implementation of a Variational Autoencoder (VAE) tailored for the [MedMNIST](https://medmnist.com/) dataset, utilizing PyTorch and PyTorch Lightning frameworks. Developed during the Deep Generative Models course (SoSe 24) at TU Darmstadt, this project explores the application of VAEs across various MedMNIST dataset modalities, with extensions into conditional VAEs and disentanglement techniques.

### Features

- **Variational Autoencoder Architecture**: Implements a VAE with customizable encoder and decoder components.
- **Conditional VAEs and Disentanglement**: Supports experiments with conditional VAEs and disentanglement to explore different generative model capabilities. _(#NotImplemented)_
- **PyTorch Lightning Integration**: Leverages PyTorch Lightning for scalable and efficient model training and evaluation.
- **Comprehensive Evaluation and Visualization**: Includes scripts for detailed evaluation metrics and generation of visual outputs to assess model performance.

### Project Structure

- `vae_medmnist/`: Contains the installable package.
  - `models/`: Contains the VAE model definitions including encoder and decoder modules.
  - `datamodules/`: Data handling modules for loading and preprocessing the MedMNIST dataset.
  - `evaluation/`: Scripts for model evaluation, including loss metrics and image generation.
- `config/`: Configuration files for model training parameters and settings.
- `results/`: Folder for storing results.
- `scripts/`: Scripts that were used for testing and exploring.

For a detailed guide on installation, configuration, and usage, please refer to the corresponding sections below.

## Installation

To install the necessary components for running this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repository/vae-medmnist.git
   cd vae-medmnist
   ```

2. Install Git Large File Storage (LFS):

   ```bash
   git lfs install
   git lfs pull
   ```

3. Install the required Python packages: I strongly recommend using [Rye](https://rye.astral.sh/guide/installation/#installing-rye) to install the project as the dependencies were handled through that, else you might run into incompatible versions. **Make sure you install the correct packages for torch manually using the [PyTorch website](https://pytorch.org/)**

   ```bash
   rye sync
   source .venv/bin/activate
   ```

These steps will set up the environment with all the necessary dependencies and data required to run the project.

## Quick-Start

### Running the VAE Model

To quickly start using the VAE model for training and evaluation, follow these [install](#installation) instructions above first.

**Run the Training:**
To start training the model, use the command-line interface provided in the `resnet_vae.py` script. You need to specify the [configuration file](#configuration) which includes all the necessary parameters:

```bash
python vae_medmnist/models/resnet_vae.py --config config/config.yaml
```

**Monitor Training:**
Training progress can be monitored via the logs generated in the specified log directory, which is set in your configuration file.

### Evaluating the Model

After training, you can evaluate the model and generate visual outputs using the `evaluation.py` script, it will put is in the same directory as your `log_path`:

```bash
python vae_medmnist/evaluation/evaluation.py --log_path results/logs/training_logs/version_{X}
```

## Configuration

The configuration of the VAE model and its training process is managed through YAML files. These files allow you to specify various parameters that control the behavior of the model and the training process.

### Configuration File Structure

A typical configuration file includes the following parameters:

- `dataset`: Specifies the dataset to use. For example, `tissuemnist`.
- `max_epochs`: Defines the maximum number of training epochs.
- `checkpoint_dir`: Directory to save the model checkpoints.
- `log_dir`: Directory to store logs generated during training.
- `batch_size`: Number of samples per batch.

An example configuration file (`config/vae_v1.yml`) is shown below:

```yaml
dataset: tissuemnist
max_epochs: 10
checkpoint_dir: results/checkpoints/resnet_vae
log_dir: results/logs
batch_size: 32
```

### Viewing Available Configuration Options

To view all available configuration options for the model, you can use the help functionality provided by the `argparse` module in the `resnet_vae.py` script. Run the following command to see the options:

```bash
python vae_medmnist/models/resnet_vae.py --help
```

Similarly, for evaluation-related configurations, refer to the `evaluation.py` script:

```bash
python vae_medmnist/evaluation/evaluation.py --help
```

These commands will display all the command-line arguments that can be set in the configuration files or passed directly via the command line when running the scripts.

## Results

The results section provides insights into the performance and output of the experiments conducted using different configuration files.

### Version 1

#### Experiment Details

The experiment was conducted using the tissuemnist dataset with the following [parameters](/results/v1/hparams.yamlvae_v1.yml):

```yaml
batch_size: 32
checkpoint_dir: results/checkpoints/resnet_vae
config: null
data_dir: .
dataset: tissuemnist
enc_out_dim: 512
first_conv: false
input_height: 28
kl_coeff: 0.1
latent_dim: 256
log_dir: results/logs
lr: 0.0001
max_epochs: 10
maxpool1: false
num_workers: 8
```

#### Performance Metrics

Throughout the experiment, the training and validation losses were carefully monitored. Below are the plots generated from these metrics, which can be found in the logs directory as specified in the configuration file.

##### Training Loss

![Training Loss Plot](results/v1/train_losses.png)  
The training loss decreased steadily, indicating that the model was learning effectively over the epochs.

##### Validation Loss

![Validation Loss Plot](results/v1/val_losses.png)  
The validation loss mirrored the training loss, suggesting that the model was generalizing well to unseen data.

##### KL Divergence

![KL Divergence Plot](results/v1/loss_comparison.png)  
The KL divergence plot shows how the model's latent space regularization evolved during training.

##### Visual Outputs

The model's ability to generate and reconstruct images was also evaluated. The visual assessment of the model's performance is demonstrated through:

Generated Images: Sample images generated by the model are displayed below.
![Generated Images](results/v1/generated_images.png)

Reconstructions: Below are comparisons between original images and their reconstructions.
![Reconstructions](results/v1/reconstructions.png)

## Contributing

[Pascal Kraus](https://github.com/Spycner)

## License

## Citations

Some parts of this project were largely inspired by the [Lightning Bolts](https://lightning.ai/docs/pytorch/stable/ecosystem/bolts.html) library. The ResNet architecture for the model was modified, updated and extended to handle the MedMNIST dataset.

For examples using other datasets, you might want to view the following colab: [VAE tutorial](https://colab.research.google.com/drive/1_yGmk8ahWhDs23U4mpplBFa-39fsEJoT?usp=sharing#scrollTo=MvBo844ZHQhF)

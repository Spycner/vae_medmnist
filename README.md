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
    - [Version 2](#version-2)
    - [Version 3](#version-3)
  - [Contributing](#contributing)
  - [Citations](#citations)

## Introduction

This repository contains the implementation of a Variational Autoencoder (VAE) tailored for the [MedMNIST](https://medmnist.com/) dataset, utilizing PyTorch and PyTorch Lightning frameworks. Developed during the Deep Generative Models course (SoSe 24) at TU Darmstadt, this project explores the application of VAEs across various MedMNIST dataset modalities, with extensions into conditional VAEs and disentanglement techniques.

### Features

- **Variational Autoencoder Architecture**: Implements a VAE with customizable encoder and decoder components.
- **Conditional VAEs**: Supports experiments with conditional VAEs to explore different generative model capabilities.
- **Disentanglement**: Explores the capabilities of VAEs in disentangling latent representations of different modalities. _(#NotImplemented)_
- **PyTorch Lightning Integration**: Leverages PyTorch Lightning for scalable and efficient model training and evaluation.
- **Comprehensive Evaluation and Visualization**: Includes scripts for detailed evaluation metrics and generation of visual outputs to assess model performance.

### Project Structure

- `vae_medmnist/`: Contains the installable package.
  - `models/`: Contains the VAE model definitions including encoder and decoder modules.
  - `dataloader/`: Data handling modules for loading and preprocessing the MedMNIST dataset. Or a custom dataloader for the combined datasets.
  - `evaluation/`: Scripts for model evaluation, including loss metrics and image generation.
- `config/`: Configuration files for model training parameters and settings.
- `results/`: Folder for storing results.
- `scripts/`: Scripts that were used for testing and exploring.

For a detailed guide on installation, configuration, and usage, please refer to the corresponding sections below.

## Installation

To install the necessary components for running this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Spycner/vae-medmnist.git
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
To start training the model, use the command-line interface provided in the models script. You need to specify the [configuration file](#configuration) which includes all the necessary parameters:

```bash
python vae_medmnist/models/vae.py --config config/config.yaml
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

To view all available configuration options for the model, you can use the help functionality provided by the `argparse` module in the models script. Run the following command to see the options:

```bash
python vae_medmnist/models/{}vae.py --help
```

Similarly, for evaluation-related configurations, refer to the `evaluation.py` script:

```bash
python vae_medmnist/evaluation/evaluation.py --help
```

These commands will display all the command-line arguments that can be set in the configuration files or passed directly via the command line when running the scripts.

## Results

The results section provides insights into the performance and output of the experiments conducted using different configuration files.

### [Version 1](/results/v1/description.md)

### [Version 2](/results/v2/description.md)

### [Version 3](/results/v3/description.md)

## Contributing

[Pascal Kraus](https://github.com/Spycner)

## Citations

Yang, J., Shi, R., Wei, D. et al. MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Sci Data 10, 41 (2023). <https://doi.org/10.1038/s41597-022-01721-8>

Some parts of this project were largely inspired by the [Lightning Bolts](https://lightning.ai/docs/pytorch/stable/ecosystem/bolts.html) library. The ResNet architecture for the model was modified, updated and extended to handle the MedMNIST dataset.

For examples using other datasets, you might want to view the following colab: [VAE tutorial](https://colab.research.google.com/drive/1_yGmk8ahWhDs23U4mpplBFa-39fsEJoT?usp=sharing#scrollTo=MvBo844ZHQhF)

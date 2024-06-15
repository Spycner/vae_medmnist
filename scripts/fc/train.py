import argparse
import logging
import os
import yaml

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torch

from vae_medmnist.models.fc_vae import FcVAE
from vae_medmnist.models.dataloader import FcMedMNISTDataModule


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser(description="FcVAE MedMNIST Training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["log_dir"])

    vae_model = FcVAE(
        hidden_dims=config["hidden_dims"],
        latent_dim=config["latent_dim"],
    )
    data_module = FcMedMNISTDataModule(
        data_flag=config["data_flag"], batch_size=config["batch_size"]
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=3,
        verbose=True,
        mode="min",
    )

    checkpoint_dir = config["checkpoint_dir"]

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename=f"vae-{config['data_flag']}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        mode="min",
        save_weights_only=True,
    )

    logger = CSVLogger(config["log_dir"], name="fcvae_training")

    torch.set_float32_matmul_precision("high")  # or 'medium' depending on your needs
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
    )

    logging.info("Starting training...")
    try:
        trainer.fit(vae_model, datamodule=data_module)
    except Exception as e:
        logging.error(f"Training failed: {e}")
    logging.info("Training finished successfully.")


if __name__ == "__main__":
    main()

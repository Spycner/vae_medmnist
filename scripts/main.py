import argparse
import logging
import os
import yaml

from medmnist import INFO
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from vae_medmnist.models.vae import VAE
from vae_medmnist.models.dataloader import MedMNISTDataModule


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
    parser = argparse.ArgumentParser(description="VAE MedMNIST Training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    setup_logging(config["log_dir"])

    in_channels = INFO[config["data_flag"]]["n_channels"]

    vae_model = VAE(input_channels=in_channels, latent_dim=config["latent_dim"])

    data_module = MedMNISTDataModule(
        data_flag=config["data_flag"], batch_size=config["batch_size"]
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config["checkpoint_dir"],
        filename=f"vae-{config['data_flag']}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback],
    )

    logging.info("Starting training...")
    try:
        trainer.fit(vae_model, datamodule=data_module)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise e
    logging.info("Training finished successfully.")


if __name__ == "__main__":
    main()

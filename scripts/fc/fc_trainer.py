import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
import torch

from vae_medmnist.models.fc_vae import model
from vae_medmnist.models.dataloader import FcMedMNISTDataModule

data_module = FcMedMNISTDataModule(data_flag="tissuemnist", batch_size=128)

# Initialize the trainer and fit the model
trainer = Trainer(max_epochs=10)
trainer.fit(model, data_module)

model.eval()
data_loader = data_module.val_dataloader()
batch = next(iter(data_loader))
x, _ = batch
x = x.to(model.device)

with torch.no_grad():
    recon_x, _, _ = model(x)


# Plot original and reconstructed images
def plot_images(original, reconstructed, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].cpu().numpy().reshape(28, 28), cmap="gray")
        ax.axis("off")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].cpu().numpy().reshape(28, 28), cmap="gray")
        ax.axis("off")
    plt.savefig("fc_reconstruction.png")


plot_images(x, recon_x)

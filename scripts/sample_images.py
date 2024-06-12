# Function to sample from latent space and generate images
def sample_and_generate_images(model, n_images, latent_dim):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_images, latent_dim).to(model.device)
        generated_images = model.decode(z).cpu()
    return generated_images


# Sample and generate images
n_samples = 10
generated_images = sample_and_generate_images(
    vae_model, n_samples, vae_model.fc_mu.out_features
)

print("Generated Images:")
plot_images(generated_images, 1, n_samples)

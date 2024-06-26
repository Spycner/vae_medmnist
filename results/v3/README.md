# Version 3

## Experiment Details

This experiment was conducted using the tissuemnist dataset with the following parameters (see [hparams.yaml](hparams.yaml) for full details):

```yaml
batch_size: 512
checkpoint_dir: results/v3
config: null
datasets:
- tissuemnist
- octmnist
- organamnist
hidden_channels: null
img_size: 28
input_channels: 1
kl_coeff: 0.00025
latent_dim: 128
log_dir: results/logs
lr: 0.005
max_epochs: 100
model: cvae
num_classes: 23
num_workers: 8
```

## Performance Metrics

The training process was monitored using various metrics. The plots generated from these metrics can be found in the logs directory.

### Training Loss

![Training Loss Plot](train_losses.png)  
The training loss decreased steadily, indicating that the model was learning effectively over the epochs.

### Validation Loss

![Validation Loss Plot](val_losses.png)  
The validation loss mirrored the training loss, suggesting that the model was generalizing well to unseen data.

### Train vs. Val Loss

![Train vs. Val Loss Plot](loss_comparison.png)  
The train vs. val loss plot shows how the model's loss evolved during training.

## Visual Outputs

The model's performance was also evaluated visually through generated and reconstructed images.

### Generated Images

![Generated Images](generated_images.png)
These are sample images generated by the model after training.

### Reconstructions

![Reconstructions](reconstructions.png)
This image compares original input images with their reconstructions by the model.

## Conclusion

The Version 3 experiment shows promising results in terms of both quantitative metrics and qualitative visual outputs. The model demonstrates good learning capabilities and generalization, as evidenced by the loss curves and the quality of generated and reconstructed images.

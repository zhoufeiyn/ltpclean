import diffusers
import torch.nn as nn
class SDXLVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = diffusers.models.AutoencoderKL.from_pretrained('stabilityai/sdxl-vae')

    def encode(self, x):
        return self.model.encode(x).latent_dist

    def decode(self, latent):
        return self.model.decode(latent).sample





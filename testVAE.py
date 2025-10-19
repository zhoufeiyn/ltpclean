from models.vae.sdvae import SDVAE
from config.configTrain import *
import torch
from algorithm import Algorithm
import os
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

def get_img_data(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def get_web_img(img):
    # img.shape = [c, h, w] 3,256,256
    img_3ch = np.transpose(img, (1,2,0)) # [h, w, c]
    img_3ch = np.clip(img_3ch*0.5+0.5, 0, 1)
    img_3ch = (img_3ch*255.0).astype(np.uint8)
    return img_3ch

device = "cuda"
model = Algorithm().to(device)
vae = SDVAE().to(device)# my vae
custom_vae_path = vae_model
if custom_vae_path and os.path.exists(custom_vae_path):
    print(f"📥 load your own vae ckpt: {custom_vae_path}")
    vae_state_dict = torch.load(custom_vae_path, map_location=device)
    vae.load_state_dict(vae_state_dict['network_state_dict'], strict=False)
    print("✅ your vae ckpt loaded successfully！")
else:
    print("ℹ️ use default pre-trained vae ckpt")

model.vae = vae # model vae

batch_data = {}

obs = get_img_data(test_img_path1)
latent_model = model.vae.encode(obs.to(model.device))
latent_model = latent_model.sample()
decoded_obs_model = model.vae.decode(latent_model)
decoded_obs_array_model=Image.fromarray(get_web_img(decoded_obs_model[0].cpu().numpy()))
decoded_obs_model.save("decode_model.png")


latent_vae = vae.encode(obs.to(model.device))
latent_vae = latent_vae.sample()
decoded_obs_vae = model.vae.decode(latent_vae)
decoded_obs_array_vae=Image.fromarray(get_web_img(decoded_obs_vae[0].cpu().numpy()))
decoded_obs_model.save("decode_vae.png")
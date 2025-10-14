from models.vae.sdxlvae import SDXLVAE
import os
import numpy as np
from PIL import Image
from infer_test import get_img_data,get_web_img

out_dir='decode_test/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

data_path = 'datatrain/'
image_paths = []
for root, dirs, files in os.walk(data_path):
    for file in files:  # 遍历files列表，不是root
        if file.lower().endswith('.png'):
            file_path = os.path.join(root, file)
            image_paths.append(file_path)

device ="cuda:0"
vae = SDXLVAE().to(device)

for p in image_paths:
    img = get_img_data(p).to(device)
    latent = vae.encode(img)
    latent = latent.sample()
    decode = vae.decode(latent)
    decoded_img_array = get_web_img(decode[0].cpu().numpy())
    decoded_img = Image.fromarray(decoded_img_array)
    decoded_img.save(f"{out_dir}{os.path.basename(p)}")

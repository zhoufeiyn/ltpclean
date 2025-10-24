from torchvision.transforms import InterpolationMode

from models.vae.sdvae import SDVAE
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import config.configTrain as cfg
def get_img_data(img_path):
    img = Image.open(img_path).convert('RGB')
    transform =  transforms.Compose([
                # 按宽度等比例缩放，保持比例不变形
                transforms.Lambda(
                    lambda img: transforms.functional.resize(
                        img,
                        size=(int(img.height * (256 / img.width)), 256),  # → 256×192
                        interpolation=InterpolationMode.NEAREST
                    )
                ),
                # 居中补上下边，使高宽都为256
                transforms.Lambda(
                    lambda img: transforms.functional.pad(
                        img,
                        padding=(0, (256 - img.height) // 2, 0, 256 - img.height - (256 - img.height) // 2),
                        fill=(107, 140, 255)
                    )
                ),
                # 转Tensor并归一化到[-1,1]
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def get_web_img(img):
    # img.shape = [c, h, w] 3,256,256
    img_3ch = np.transpose(img, (1,2,0)) # [h, w, c]
    img_3ch = np.clip(img_3ch*0.5+0.5, 0, 1)
    img_3ch = (img_3ch*255.0).astype(np.uint8)
    img_3ch = img_3ch[32:-32, :, :]
    return img_3ch
def decode():
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
  vae = SDVAE().to(device)
  custom_vae_path = cfg.vae_model
  if custom_vae_path and os.path.exists(custom_vae_path):
      print(f"📥 load your own vae ckpt: {custom_vae_path}")
      custom_state_dict = torch.load(custom_vae_path, map_location=device)
      vae.load_state_dict(custom_state_dict['network_state_dict'], strict=False)
      print("✅ your vae ckpt loaded successfully！")
  else:
      print("ℹ️ use default pre-trained vae ckpt")
  vae.eval()
  with torch.no_grad():
      for p in image_paths:
          img = get_img_data(p).to(device)
          latent = vae.encode(img)
          latent = latent.sample()
          decode = vae.decode(latent)
          decoded_img_array = get_web_img(decode[0].cpu().numpy())
          decoded_img = Image.fromarray(decoded_img_array)
          decoded_img.save(f"{out_dir}{os.path.basename(p)}")

if __name__=='__main__':
  decode()

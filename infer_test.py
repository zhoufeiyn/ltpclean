import argparse
import os
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import imageio

def get_jave_7action(key):
    if key == "r":
        action = 1
    elif key == "rj":
        action = 2
    elif key == "l":
        action = 3
    elif key == "lj":
        action = 4
    elif key == "j":
        action = 5
    elif key == "f":
        action = 6
    else:
        action = 0
    return [action]


def get_action_sequence(actions):
    ret = []
    for a in actions:
        assert a in ['l', 'r', 'f', 'j', 'lj', 'rj',
                     'n'], f"action {a} should be in [l->left r->right j->jump f->fire lj->left jump rj->right jump n->null]"
        ret.extend(get_jave_7action(a))
    return ret





def image_to_numpy_array(filepath):
    img = Image.open(filepath)
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return img_array





def get_img_data(img_path, device):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1]
    ])
    img = transform(img)
    img.unsqueeze(0)
    print(f"int_img shape:{img.size()}")
    return img


def init_simulator(model, batch):
    obs = batch["observations"][0]
    latent = model.vae.encode(obs.reshape(-1,3,256,256))
    latent = latent.sample() * 0.18215
    latent = latent.reshape(4,32,32)
    init_z =model.df_model.init_df_modelmodel(latent)
    return init_z

def get_web_img(img):
    # img.shape = [c, h, w] 3,256,256
    img_3ch = np.transpose(img, (1,2,0)) # [h, w, c]
    img_3ch = np.clip(img_3ch*0.5+0.5, 0, 1)
    img_3ch = (img_3ch*255.0).astype(np.uint8)
    return img_3ch

def model_test(img_path='eval_data/demo1.png', actions=['r','r','r','r'], model=None, device='cuda',sample_step =4,epochs=0):
    """测试训练好的模型"""
    
    # 检查输入参数
    if model is None:
        print("❌ Error: model is None")
        return
    
    if not os.path.exists(img_path):
        print(f"❌ Error: Test image not found: {img_path}")
        return
    try:
        img_list=[]
        batch_data={}
        batch_data['observations']=get_img_data(img_path, device) #(1,3, 256,256)
        with torch.no_grad():
            zeta = init_simulator(model,batch_data) #(1,32,32,32)
        img_list.append(get_web_img(batch_data['observations'][0].cpu().numpy()))
        actions=get_action_sequence(actions)
        for a in actions:
            with torch.no_grad():
                a = torch.tensor([a],device=device).long()
                zeta, obs = model.df_model.step(zeta, a.float(), sample_step)
                obs = model.vae.decode(obs / 0.18215)
            img_list.append(get_web_img(obs[0].cpu().numpy()))
            
        if not os.path.isdir('./output/'):
            os.makedirs('./output/')
        imageio.mimsave(f'./output/output_{epochs}.gif', img_list, duration=0.2)
        print("✅ output.gif saved in ./output/")
        
    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()





import argparse
import os
from PIL import Image
import cv2
import torch
import numpy as np
import imageio
from model import get_model, get_data, get_web_img
import configTrain as cfg

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

def preprocess_img(img):
    target_size = (128, 128)
    cropped_image = cv2.resize(img[:, :, :3], target_size) 
    final_image = np.expand_dims(np.expand_dims(cropped_image.transpose(2, 0, 1), axis=0), axis=0)
    final_image = torch.tensor(final_image, dtype=torch.float32, device=model.device)
    batch_data = {'observations': final_image}
    return batch_data



def image_to_numpy_array(filepath):
    img = Image.open(filepath)
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return img_array


obs_shape = [3, 128, 128]


def get_img_data(img_path, device):
    img = Image.open(img_path).convert('RGB').resize((128, 128))
    img = np.array(img)
    img_data = img[np.newaxis, ...]  # (1, 128,128,3)

    cur_img = torch.tensor(img_data, dtype=torch.float32, device=device)

    cur_img = cur_img.permute(0, 3, 1, 2)
    cur_img = cur_img.reshape(1, obs_shape[0], obs_shape[1], obs_shape[2])
    cur_img = cur_img / 255.0
    return cur_img


def init_simulator(model, batch):
    obs = batch["observations"]
    with torch.no_grad():
        wm_env = model.init_wm(obs)
        return obs, wm_env



def model_test(img_path='eval_data/demo1.png', actions=['r','r','r','r'], model=None, device='cuda',sample_step =4,epochs=0):
    """测试训练好的模型"""
    
    # 检查输入参数
    if model is None:
        print("❌ Error: model is None")
        return
    
    if not os.path.exists(img_path):
        print(f"❌ Error: Test image not found: {img_path}")
        return
    
    # print(f"🧪 Testing model with image: {img_path}")
    # print(f"    actions: {actions}")
    # print(f"    sample_step: {sample_step}")
    
    try:
        img_list=[]
        batch_data={}
        batch_data['observations']=get_img_data(img_path, device)
        with torch.no_grad():
            obs,zeta = init_simulator(model,batch_data)
        img_list.append(get_web_img(obs[0].cpu().numpy()))
        actions=get_action_sequence(actions)
        for a in actions:
            with torch.no_grad():
                obs, zeta = model.real_time_infer(zeta, torch.tensor([a]).long(), sample_step)
            img_list.append(get_web_img(obs[0].cpu().numpy()))
            
        if not os.path.isdir('./output/'):
            os.makedirs('./output/')
        imageio.mimsave(f'./output/output_{epochs}.gif', img_list, duration=0.2)
        print("✅ output.gif saved in ./output/")
        
    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()




if __name__ == "__main__":
    model = get_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_test(
        img_path='eval_data/demo1.png',
        actions=['r','r','r','r'],
        model=model,
        device=device,
        sample_step=cfg.sample_step
    )
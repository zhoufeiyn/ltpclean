import argparse
import os
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import imageio
from algorithm import Algorithm
from config.configTrain import *
from models.vae.sdvae import SDVAE
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

def init_simulator(model, batch):
    obs = batch["observations"]

    latent_dist = model.vae.encode(obs.to(model.device))

    latent = latent_dist.sample()
    decoded_obs = model.vae.decode(latent)

    latent_for_df = latent * 0.18215
    latent_for_df = latent_for_df.reshape(4, 32, 32)
    init_z = model.df_model.init_df_model(latent_for_df)
    
    return init_z, decoded_obs

def get_web_img(img):
    # img.shape = [c, h, w] 3,256,256
    img_3ch = np.transpose(img, (1,2,0)) # [h, w, c]
    img_3ch = np.clip(img_3ch*0.5+0.5, 0, 1)
    img_3ch = (img_3ch*255.0).astype(np.uint8)
    return img_3ch

def model_test(img_path='eval_data/demo1.png', actions=['r'], model=None, device='cuda',sample_step =4,name='infer',epoch=None,output_dir='output'):
    """测试训练好的模型"""
    
    # 检查输入参数
    if model is None:
        print("❌ Error: model is None")
        return
    
    if not os.path.exists(img_path):
        print(f"❌ Error: Test image not found: {img_path}")
        return
    
    # 保存当前模型状态
    was_training = model.training
    vae_was_training = model.vae.training if hasattr(model, 'vae') and model.vae is not None else None
    
    try:
        # 设置为评估模式
        model.eval()
        img_list=[]
        batch_data={}
        batch_data['observations']=get_img_data(img_path) #(1,3, 256,256)
        with torch.no_grad():
            zeta,obs_start = init_simulator(model,batch_data) #(1,32,32,32)
        img_list.append(get_web_img(obs_start[0].cpu().numpy()))
        actions=get_action_sequence(actions)
        for a in actions:
            with torch.no_grad():
                a = torch.tensor([a],device=device).long()
                zeta, obs = model.df_model.step(zeta, a.float(), sample_step)
                obs = model.vae.decode(obs / 0.18215)
            img_list.append(get_web_img(obs[0].cpu().numpy()))
            
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if epoch:
            if not os.path.exists(f'{output_dir}/epoch{epoch}'):
                os.makedirs(f'{output_dir}/epoch{epoch}')
            imageio.mimsave(f'{output_dir}/epoch{epoch}/{name}.gif', img_list, duration=0.2)
            print(f"✅ output.gif saved in {output_dir}/epoch{epoch}/")
        else:
            imageio.mimsave(f'{output_dir}/{name}.gif', img_list, duration=0.2)
            print(f"✅ output.gif saved in {output_dir}/")

    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复模型原始状态
        if was_training:
            model.train()
        # 恢复 VAE 的原始状态
        if hasattr(model, 'vae') and model.vae is not None and vae_was_training is not None:
            if vae_was_training:
                model.vae.train()
            else:
                model.vae.eval()

def parse_comma_separated_list(value):
    return value.split(',')
def arg():
    parser = argparse.ArgumentParser(description="Direct inference of Playable Game Generation")

    parser.add_argument('-i', "--img", type=str, required=True, help="The initial screen of the game")
    parser.add_argument('-a', "--actions", required=True, type=parse_comma_separated_list,
                        help="action sequences\n l->left r->right j->jump f->fire lj->left jump rj->right jump n->null")
    parser.add_argument('-s', "--sample_step", default=20, type=int, help="diffusion sample step")

    args = parser.parse_args()
    return args

if __name__ =="__main__":
    args = arg()
    sample_step = args.sample_step
    model = Algorithm(model_name,device)
    vae = SDVAE()
    model.vae = vae

    state_dict = torch.load(os.path.join("ckpt",model_path),map_location=device,weights_only=False)
    model.load_state_dict(state_dict["network_state_dict"],strict=False)
    model.eval().to(device)

    model_test(args.img,args.actions,model,device,sample_step,f'{args.img[-9:-4]}_test',epoch=None,output_dir='output')
    # python infer_test.py -i 'eval_data/demo1.png' -a r,r,r,r,r,r







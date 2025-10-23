import importlib
import os.path as osp
import numpy as np
import torch
from config.Config_VAE import (Config)
from config.configTrain import *
from algorithm import Algorithm
import random
from einops import rearrange
import time
import cv2

def read_model(model_name, model_path, action_space, device='cpu'):
    model = Algorithm(model_name,device)
    state_dict = torch.load(
        osp.join("ckpt", model_path),
        map_location=torch.device(device),weights_only=False
    )
    model.load_state_dict(state_dict['network_state_dict'],strict=False)
    model.eval().to(device)
    return model

def read_file(data_file, data_type="java"):
    print("read_file", data_type)
    data_with_nan = np.genfromtxt(data_file, delimiter=',')
    np_data = data_with_nan[:, ~np.isnan(data_with_nan).any(axis=0)]
    data = np_data[:, Config.data_start_end[0]:Config.data_start_end[1]]
    return data

def split_data_mario(epi_data, horizon_len, device):
    episode_len, _ = epi_data.shape
    assert episode_len % horizon_len == 0
    data_tensor = torch.tensor(epi_data, dtype=torch.float32, device=device)
    cur_img = data_tensor[:, :-1].reshape(episode_len, Config.resolution, Config.resolution, 3)
    cur_img = cur_img.permute(0, 3, 1, 2)
    cur_img = cur_img.reshape(episode_len, 3, Config.resolution, Config.resolution)
    cur_img = cur_img / 255.0
    cur_action = data_tensor[:, -1].reshape(episode_len, 1).long()
    dict_data = {
        "epi_len": episode_len,
        "cur_img": cur_img,
        "cur_act_int": cur_action,
    }
    return dict_data

def process_npdata(np_data, horizon_len, start_idx, device):
    file_data_len = np_data.shape[0]
    end_idx = start_idx + horizon_len
    if end_idx > file_data_len:
        end_idx = file_data_len
        start_idx = end_idx - horizon_len
    epi_data = np_data[start_idx:end_idx]
    dict_data = split_data_mario(epi_data, horizon_len, device)
    batch = {}
    for k, v in dict_data.items():
        if torch.is_tensor(v):
            batch[k] = v
    batch_data = data_formater(batch, horizon_len)
    batch_data["start_idx"] = start_idx
    return batch_data

def data_formater(data_dict, epi_len):
    batch = {}
    ori_obs = data_dict["cur_img"]
    batch["observations"] = rearrange(ori_obs, '(b t) c h w -> b t c h w', t=epi_len)
    batch["cur_actions"] = rearrange(data_dict["cur_act_int"], '(b t) e -> b t e', t=epi_len, e=1)
    return batch

def init_simulator(model, batch):
    obs = batch["observations"]
    obs = rearrange(obs, 'b t c h w -> (b t) c h w')
    with torch.no_grad():
        init_zeta = model.init_wm(obs[0:1])
        return obs, init_zeta

def get_web_img(img):
    # img.shape = [c, h, w]
    img_3ch = np.transpose(img, (1,2,0)) # [h, w, c]
    img_3ch = np.clip(img_3ch, 0, 1)
    img_3ch = cv2.resize(img_3ch, (300, 300), interpolation=cv2.INTER_LINEAR)
    # img_3ch = cv2.resize(img_3ch, (25, 25), interpolation=cv2.INTER_LINEAR)
    img_3ch = (img_3ch*255.0).astype(np.uint8)
    return img_3ch



# np_data = dataloader.read_file(file_path, data_type)

def get_data(if_random=False):
    start_time = time.time()
    start_idx = 0
    if if_random:
        start_idx = random.randint(0, np_data.shape[0])
    batch_data = process_npdata(np_data, 1, start_idx, device)
    return batch_data


    """map key(s) to action based on SMB dataset encoding:
    A=128(jump), up=64(climb), left=32, B=16(run/fire), 
    start=8, right=4, down=2, select=1
    
    Args:
        key: str or list of str - single key or list of pressed keys
    
    Examples:
        map_Key_to_Action("r") -> 4 (right)
        map_Key_to_Action(["r", "f"]) -> 20 (right + B = running right)
        map_Key_to_Action(["r", "f", "j"]) -> 148 (right + B + A = running jump right)
    """
    # 如果输入是单个键，转换为列表
    if isinstance(key, str):
        keys = [key]
    else:
        keys = key

    action = 0

    # 遍历所有按下的键，累加动作值
    for k in keys:
        if k == "r" or k == "right" or k == "→":
            action += 4  # right
        elif k == "l" or k == "left" or k == "←":
            action += 32  # left
        elif k == "j" or k == "a":
            action += 128  # A (jump)
        elif k == "up" or k == "↑":
            action += 64  # up (climb)
        elif k == "f" or k == "b":
            action += 16  # B (run/fire)
        elif k == "s":
            action += 8  # start
        elif k == "d" or k == "down" or k == "↓":
            action += 2  # down
        elif k == "enter":
            action += 1  # select

    return action


if __name__ == "__main__":
    model = read_model(model_name, model_path, action_space, device)

    np_data = read_file(file_path, data_type)

    start_time = time.time()
    start_idx = random.randint(0, np_data.shape[0])
    start_idx = 0
    batch_data = process_npdata(np_data, 1, start_idx, device)
    end_time = time.time()
    print(f"process_npdata cost time: {end_time - start_time:.4f} s")
    start_time = time.time()
    data = batch_data["observations"]

    obs, wm = init_simulator(model, batch_data)
    img = get_web_img(obs[0].cpu().numpy())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./eval_data/init.jpg', img)
    # print(img.shape)


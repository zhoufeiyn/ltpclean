"""User Model Global Config"""
model_name = 'df_z32_c1_dit_n11_mario_km_tanh_ldm'
action_space = 7
model_path = "model_epoch10000_20251012_07.pth" #"model.pth"
device = 'cuda:0'

"""Local Web"""
file_path= '../eval_data/0-frameArray.txt'
data_type='java'
SEQ_LEN = 100
out_dir: str = "./output"
data_path: str = "./datatrain"
ckpt_path: str = "./ckpt"

"""Train Config"""
img_size = 256
img_channel = 3
base_ch: int = 64          # 减少基础通道数以适应GPU内存
num_actions: int = 46
num_frames: int = 16
frame_interval: int = 16

data_save_epoch: int = 10  # loss数据print和保存至log日志的间隔 \log
gif_save_epoch: int = 100  # gif保存间隔 \output
best_save_interval: int = 50000  # 最佳模型保存间隔（大于num个epoch,且超过最小改善幅度，保存一次最佳模型）
min_improvement: float = 0.15  # 最小改善幅度（15%）

batch_size: int = 1        # 单张图像过拟合
epochs: int = 2000          # 测试epoch数量



sample_step: int = 20
test_img_path: str = "./eval_data/demo1.png"
actions = ['rj','rj','rj','rj','rj','rj','rj','rj']

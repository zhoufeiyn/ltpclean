"""User Model Global Config"""
model_name = 'df_z32_c1_dit_n11_mario_km_tanh_ldm'
action_space = 7
model_path = "model_epoch10000_20251012_07.pth" #"model.pth"
device = 'cuda:0'
vae_model = 'ckpt/VAE/vae_epoch10_20251025_06.pth'

"""Resume Training Config"""
resume_training = True  # 是否继续训练
resume_checkpoint_path = "ckpt/model_epoch8000_20251025_20.pth"  # 继续训练的checkpoint路径，例如: "ckpt/model_epoch100_20251018_19.pth"

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

# # Large dataset train
# num_frames: int = 8
# frame_interval: int = 2
# loss_log_iter: int = 20  # loss数据print和保存至log日志的间隔 \log
# # gif_save_iter: int = 400
# gif_save_epoch: int = 8  # avgloss和gif保存间隔 \output
# checkpoint_save_epoch: int = 15  # checkpoint保存间隔
# min_improvement: float = 0.15  # 最小改善幅度（15%）
# batch_size: int = 30        # 单张图像过拟合
# epochs: int = 64          # 测试epoch数量

# sample_step: int = 20
# test_img_path: str = "./eval_data/demo.png"
# test_img_path1: str = "./eval_data/demo1.png"
# test_img_path2: str = "./eval_data/demo2.png"
# test_img_path3: str = "./eval_data/demo3.png"
# actions = ['rj','rj','rj','rj','rj','rj']
# actions1 = ['r','r','r','r','r','r']
# actions2 = ['rj','rj','rj','rj','rj','rj']

# small dataset train
num_frames: int = 6
frame_interval: int = 6
loss_log_iter: int = 20  # loss数据print和保存至log日志的间隔 \log
# gif_save_iter: int = 400
gif_save_epoch: int = 500  # avgloss和gif保存间隔 \output
checkpoint_save_epoch: int = 1000  # checkpoint保存间隔
min_improvement: float = 0.15  # 最小改善幅度（15%）
batch_size: int = 2        # 单张图像过拟合
epochs: int = 10000          # 测试epoch数量

sample_step: int = 20
test_img_path1: str = "./eval_data/demo4.png"

actions1 = ['r','r','r','r','r','r']
actions2 = ['rj','rj','rj','rj','rj','rj']

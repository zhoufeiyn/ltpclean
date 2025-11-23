"""User Model Global Config"""
model_name = 'df_z32_c1_dit_n11_mario_km_tanh_ldm'
train_sample = 1
out_dir: str = "./output"
# data_path: str = "./datatrain/"
data_path = "./mario_data/mariodata/"
ckpt_path: str = "./ckpt"
model_path = "model_epoch155_20251116_07.pth" # infer或者load pretrain权重时候用
device = 'cuda:0'
# vae_model = '/content/drive/MyDrive/my_models/1025sdxl/vae_epoch10_20251025_06.pth'
vae_model = './ckpt/VAE/vae_epoch6_20251112_03.pth'

"""Resume Training Config"""
resume_training = True  # 是否继续训练
resume_checkpoint_path = "./ckpt/model_epoch55_20251116_07.pth"

"""Local Web"""
file_path= './eval_data/0-frameArray1.txt'
data_type='java'


"""Train Config"""
img_size = 128
img_channel = 3
base_ch: int = 64          # 减少基础通道数以适应GPU内存
num_workers_folders=12
num_workers = 12
gradient_accumulation_steps: int = 3  # 梯度累积步数，用于模拟更大的batch size
scale_factor: float = 0.7064

# Large dataset train
num_frames: int = 15
frame_interval: int = 4
loss_log_iter: int = 20  # loss数据print和保存至log日志的间隔 \log
# gif_save_iter: int = 400
gif_save_epoch: int = 1  # avgloss和gif保存间隔 \output
checkpoint_save_epoch: int = 5  # checkpoint保存间隔
min_improvement: float = 0.15  # 最小改善幅度（15%）
batch_size: int = 32
epochs: int = 75          # 测试epoch数量

sample_step: int = 20

test_img_path1: str = "./eval_data/demo1.png"
test_img_path2: str = "./eval_data/demo2.png"
test_img_path3: str = "./eval_data/demo3.png"
test_img_path4: str = "./eval_data/demo4.png"

actions1 = ['r','r','r','r','r','r','r','r','r']
actions2 = ['rj','rj','rj','rj','rj','rj','rj','rj','rj']

# # small dataset train
# num_frames: int = 12
# frame_interval: int = 12
# loss_log_iter: int = 50  # loss数据print和保存至log日志的间隔 \log
# # gif_save_iter: int = 400
# gif_save_epoch: int = 500  # avgloss和gif保存间隔 \output
# checkpoint_save_epoch: int = 1000  # checkpoint保存间隔
# min_improvement: float = 0.15  # 最小改善幅度（15%）
# batch_size: int = 1        # 单张图像过拟合
# epochs: int = 5000          # 测试epoch数量
#
# sample_step: int = 20
# test_img_path1: str = "./eval_data/demo1.png"
# test_img_path2: str = "./eval_data/demo22.png"
# test_img_path3: str = "./eval_data/demo33.png"
# actions1 = ['r','r','r','r','r','r','r','r','r','r']
# # actions2 = ['rj','rj','rj','rj','rj','rj']
# actions1 = ['r','r','r','r','r','r','r','r','rj','rj','rj','j','j','j','j','j','j','j','j','j','n','n','n','n','n','n','n','n','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r']
# actions2 = ['n','n','j','j','j','j','j','j','j','j','j','j','j','j','j','j','j','j','j','j','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n','n']
# actions3 = ['r','r','r','n','n','n','n','n','n','n', 'n','n','n','n','n','n','n','n','n','n', 'n','n','n','n','n','n','n','n','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r']
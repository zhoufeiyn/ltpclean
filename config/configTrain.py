"""User Model Global Config"""
model_name = 'df_z32_c1_dit_n11_mario_km_tanh_ldm'
train_sample = 1
model_path = "model_epoch10000_20251012_07.pth" #"model.pth"
device = 'cuda:0'
vae_model = '/content/drive/MyDrive/my_models/1025sdxl/vae_epoch10_20251025_06.pth'

"""Resume Training Config"""
resume_training = True  # 是否继续训练
resume_checkpoint_path = "/content/drive/MyDrive/my_models/1026largeDATA_df/model_epoch60_20251028_06.pth"  # 继续训练的checkpoint路径，例如: "ckpt/model_epoch100_20251018_19.pth"

"""Local Web"""
file_path= '../eval_data/0-frameArray.txt'
out_dir: str = "./output"
data_path: str = "/content/drive/MyDrive/mario_data/"
# data_path = "./datatrain/"
ckpt_path: str = "./ckpt"

"""Train Config"""
img_size = 256
img_channel = 3
base_ch: int = 64          # 减少基础通道数以适应GPU内存
num_workers_folders=8
num_workers = 32
gradient_accumulation_steps: int = 4  # 梯度累积步数，用于模拟更大的batch size


# Large dataset train
num_frames: int = 12
frame_interval: int = 4
loss_log_iter: int = 10  # loss数据print和保存至log日志的间隔 \log
# gif_save_iter: int = 400
gif_save_epoch: int = 5  # avgloss和gif保存间隔 \output
checkpoint_save_epoch: int = 5  # checkpoint保存间隔
min_improvement: float = 0.15  # 最小改善幅度（15%）
batch_size: int = 24
epochs: int = 130          # 测试epoch数量

sample_step: int = 20

test_img_path1: str = "./eval_data/demo1.png"
test_img_path2: str = "./eval_data/demo2.png"
test_img_path3: str = "./eval_data/demo3.png"

actions1 = ['r','r','r','r','r','r','r','r','r']
actions2 = ['rj','rj','rj','rj','rj','rj','rj','rj','rj']

# # small dataset train
# num_frames: int = 4
# frame_interval: int = 1
# loss_log_iter: int = 20  # loss数据print和保存至log日志的间隔 \log
# # gif_save_iter: int = 400
# gif_save_epoch: int = 500  # avgloss和gif保存间隔 \output
# checkpoint_save_epoch: int = 1000  # checkpoint保存间隔
# min_improvement: float = 0.15  # 最小改善幅度（15%）
# batch_size: int = 4        # 单张图像过拟合
# epochs: int = 15000          # 测试epoch数量
#
# sample_step: int = 20
# test_img_path1: str = "./eval_data/demo1.png"
# test_img_path2: str = "./eval_data/demo2.png"
# actions1 = ['r','r','r']
# actions2 = ['rj','rj','rj','rj','rj','rj']

"""Local Web"""
"""sd 1.5 512*512 VAE"""

out_dir: str = "./output/VAE"
data_path: str = "./mario_data"
ckpt_path: str = "./ckpt/VAE"
model_path: str = ""


"""Train Config"""
lr = 1e-4
img_size = 256
img_channel = 3
latent_ch: int = 4
latent_size: int = 32


loss_log_iter: int = 20  # loss数据print和保存至log日志的间隔 \log

img_save_epoch: int = 6  # avgloss和gif保存间隔 \output

checkpoint_save_epoch: int = 6  # checkpoint保存间隔


batch_size: int = 64
epochs: int = 30          # 测试epoch数量


test_img_path: str = "./eval_data/vae"
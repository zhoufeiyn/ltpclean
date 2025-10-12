# 0920 update: try to overfit level1-1 in one directory

from typing import Optional
import re
from models.vae.sdxlvae import SDXLVAE
from algorithm import Algorithm
import torch

from torch.utils.data import Dataset
from torchvision import transforms
import config.configTrain as cfg

import matplotlib.pyplot as plt
from PIL import Image

import os
from datetime import datetime
from infer_test import model_test
import logging



device: str = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Logging Setup
# -----------------------------
def setup_logging():
    """设置日志记录"""
    # 创建logs目录
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H")
    log_filename = f"training_log_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"init log: {log_path}")
    return logger, log_path

def save_loss_curve(loss_history, data_save_epoch, save_path="output"):
    """保存损失曲线图"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 创建损失曲线图
    plt.figure(figsize=(10, 6))
    x_epochs = [(i + 1) * data_save_epoch for i in range(len(loss_history))]
    plt.plot(x_epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
    plt.title('Training Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H")
    loss_curve_path = os.path.join(save_path, f"loss_curve_{timestamp}.png")
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Loss curve saved to: {loss_curve_path}")
    return loss_curve_path

# -----------------------------
# Model Saving Function
# -----------------------------
def save_model(model, epochs, final_loss,path=cfg.ckpt_path):
    """保存训练好的模型到ckpt目录"""    

    if not os.path.exists(path):
        os.makedirs(path)

    
    # 生成文件名（包含时间戳和epoch信息）
    timestamp = datetime.now().strftime("%Y%m%d_%H")
    model_filename = f"model_epoch{epochs}_{timestamp}.pth"
    model_path = os.path.join(path, model_filename)
    
    # 准备保存的数据
    save_data = {
        'network_state_dict': model.state_dict(),
        'epochs': epochs,
        'loss': final_loss,
        'model_name': cfg.model_name,
        'batch_size': cfg.batch_size,
        'num_frames': cfg.num_frames,
    }
    
    # 保存模型
    try:
        torch.save(save_data, model_path)
        print(f"✅ save model to {model_path}")

    except Exception as e:
        print(f"❌ save model failed: {e}")

def save_best_checkpoint(model, epoch,  best_loss, is_best=False,path=cfg.ckpt_path):
    """保存检查点（定期保存和最佳模型保存）"""


    if not os.path.exists(path):
        os.makedirs(path)
    
    # 保存最新模型
    if is_best:
        latest_path = os.path.join(path, "best_model.pth")
        checkpoint_data = {
            'network_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': best_loss,
            'model_name': cfg.model_name,
            'batch_size': cfg.batch_size,
            'num_frames': cfg.num_frames
        }
    
        torch.save(checkpoint_data, latest_path)
        print(f"🏆 save best model: epoch: {epoch},best loss = {best_loss:.6f}")
    

# -----------------------------
# Custom Dataset for Mario Data
# -----------------------------

class MarioDataset(Dataset):
    """load mario dataset __init__ action and img paths,
     __getitem__  will return image and corresponding action"""
    """up to date: 2025-09-20 only load all frames in one directory,
     return array ofimages and actions"""
    def __init__(self, data_path: str, image_size):
        self.data_path = data_path
        self.image_size = image_size
        self.image_files = [] # image files path (xxx.png)
        self.actions = [] # action (0-255)
        self._load_data()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # [0, 1]
            transforms.Normalize(0.5, 0.5),  # [-1, 1]
        ])
    def _load_data(self):
        """load all png files and corresponding actions"""
        print(f" data path is scanning: {self.data_path}")
        if not os.path.exists(self.data_path): 
            print(f"❌ data path not found: {self.data_path}")
            return
        
        total_files = 0
        valid_files = 0

        for root, dirs, files in os.walk(self.data_path):
            if root == self.data_path:
                continue
            for file in files:
                if file.lower().endswith('.png'):
                    total_files += 1
                    file_path = os.path.join(root, file)
                    
                    # 尝试从文件名提取动作
                    action = self._extract_action_from_filename(file)
                    if action is not None:
                        self.image_files.append(file_path)
                        self.actions.append(action)
                        valid_files += 1
                    else:
                        print(f"⚠️ can't extract action from filename: {file}")
    
    def _map_action_to_playgenaction(self, action: int) -> int:
        """map action to playgenaction"""
        # 255 to 7
        if action == 20: # right + B = 4+16=running right
            return 1
        elif action == 148: # right + B + A = 4+16+128=running jump right
            return 2
        elif action == 48: # left + B = 32+16=running left
            return 3
        elif action == 176: # left + B + A = 32+16+128=running jump left
            return 4
        elif action == 128: # A = 128=jump
            return 5
        elif action == 16: # B = 16=fire or run
            return 6
        elif action == 0: # null
            return 0

            
        
    def _extract_action_from_filename(self, filename: str) -> Optional[int]:
        """extract action from filename"""
        # 文件名格式: Rafael_dp2a9j4i_e6_1-1_f1000_a20_2019-04-13_20-13-16.win.png
        pattern = r'_a(\d+)_'
        match = re.search(pattern, filename)
        if match:
            action = int(match.group(1))
            action_mapped = self._map_action_to_playgenaction(action)
            return action_mapped
        return None
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """get the data sample of the specified index"""
        if idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_files)}")
        
        # 加载图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # 获取动作（如果有的话）
        action = self.actions[idx] if idx < len(self.actions) else 0
        
        return image, action


def build_video_sequence(dataset, start_idx, end_idx):
    """build video sequence in one batch from dataset, return [1, num_frames, ch, h, w]"""

    # 存储一个视频序列的数据
    video_images = []  # 存储当前视频的num_frames帧图像
    video_actions = []  # 存储当前视频的num_frames个动作
    video_nonterminals = []  # 存储当前视频的num_frames个nonterminals

    # 构建当前视频序列
    for frame_idx in range(start_idx, end_idx):
        image, action = dataset[frame_idx]
        video_images.append(image)  # image shape: [3, 128, 128]
        video_actions.append(action)  # action是整数，直接使用
        video_nonterminals.append(True)  # 先都默认True
    # 转换为tensor并组织成目标格式
    # [num_frames, channels, h, w] = [num_frames, 3, 128, 128]
    images_tensor = torch.stack(video_images, dim=0)  # [num_frames, 3, 128, 128]
    images_tensor = images_tensor.unsqueeze(0)  # [1, num_frames, 3, 128, 128]

    # [batch_size, num_frames, action_dim] = [1, num_frames, 1]
    actions_tensor = torch.tensor(video_actions, dtype=torch.long)  # [num_frames]
    actions_tensor = actions_tensor.unsqueeze(0).unsqueeze(-1)  # [1, num_frames, 1]

    # [batch_size, num_frames] = [1, num_frames]
    nonterminals_tensor = torch.tensor(video_nonterminals, dtype=torch.bool)  # [num_frames]
    nonterminals_tensor = nonterminals_tensor.unsqueeze(0)  # [1, num_frames]

    # 返回tensor而不是列表
    return images_tensor, actions_tensor, nonterminals_tensor

def vae_encode(batch_data_images, vae_model, device, scale_factor=0.18215):
    """vae encode the images"""
    # 将图像编码到潜在空间: [batch_size, num_frames, 3, 128, 128] -> [batch_size, num_frames, 4, 32, 32]
    with torch.no_grad():
        batch_size_videos, num_frames, channels, h, w = batch_data_images.shape
        # 重塑为 [batch_size * num_frames, 3, 128, 128] 进行批量编码
        images_flat = batch_data_images.reshape(-1, channels, h, w).to(device)
        
        # VAE编码
        if vae_model is not None:
            latent_dist = vae_model.encode(images_flat) # [batch_size * num_frames, 3, 128, 128]
            latent_images = latent_dist.sample()  # 采样潜在表示 [batch_size * num_frames, 4, 32, 32]


            latent_images = latent_images * scale_factor
            # print(f"   Using scale factor: {Config.scale_factor}")
            
            # 重塑回 [batch_size, num_frames, 4, 32, 32]
            latent_images = latent_images.reshape(batch_size_videos, num_frames, 4, 32, 32) #[batch_size, num_frames, 4, 32, 32]
        else:
            print("⚠️ Cannot find VAE model, use original image")
            # 如果没有VAE，直接使用原始图像，但需要调整形状
            latent_images = images_flat.reshape(batch_size_videos, num_frames, channels, h, w)
            print(f"   Using original image shape: {latent_images.shape}")
        
        # 更新batch_data[0]为编码后的潜在表示，保持在GPU上
        return latent_images

def train():
    # 初始化日志记录
    logger, log_path = setup_logging()
    
    device_obj = torch.device(device)
    dataset = MarioDataset(cfg.data_path, cfg.img_size)

    # video sequence parameters
    num_frames = cfg.num_frames
    batch_size = cfg.batch_size


    model_name = cfg.model_name

    best_save_interval = cfg.best_save_interval
    data_save_epoch = cfg.data_save_epoch
    gif_save_epoch = cfg.gif_save_epoch
    
    # 使用Algorithm类加载完整的预训练模型（包含VAE和Diffusion）
    model = Algorithm(model_name, device_obj)
    
    # 加载预训练checkpoint
    checkpoint_path = os.path.join(cfg.ckpt_path, cfg.model_path)
    if os.path.exists(checkpoint_path):
        print(f"📥 load pretrained checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(state_dict['network_state_dict'], strict=False)
        print("✅ Checkpoint loaded successfully！")

    else:
        print(f"⚠️ Checkpoint not found: {checkpoint_path},use random initialized model")
    model = model.to(device_obj)
    # model.eval()  # 设置为评估模式，但允许训练
    
    # 获取VAE和Diffusion模型
    vae = SDXLVAE().to(device_obj)
    model.vae = vae
    diffusion_model = model.df_model

    
    if vae is not None:
        vae.eval()
        for param in vae.parameters(): # freeze VAE parameters
            param.requires_grad = False
        print("✅ VAE already loaded，VAE parameters has been frozen")
    else:
        print("⚠️ Cannot find VAE model")
    epochs, batch_size = cfg.epochs, cfg.batch_size
    

    opt = diffusion_model.configure_optimizers_gpt()
    

    print("---1. start training----")
    print("---2. load dataset---")
    total_samples = len(dataset)
    # 检查是否有足够的数据
    if total_samples < num_frames:
        print(f"❌ dataset not enough: need at least {num_frames} samples, but only {total_samples} samples")
        return
    # 计算可以创建多少个完整的视频序列
    num_videos = total_samples // num_frames
    print(f"dataset loaded: {total_samples} samples, construct {num_videos} complete video sequences, each video has {num_frames} frames, construct {num_videos//batch_size} batches, each batch has {batch_size} videos")
    
    # 初始化最佳损失跟踪
    best_loss = float('inf')
    min_improvement = cfg.min_improvement  # 最小改善幅度
    final_avg_loss = 0  # 用于保存最终的avg_loss
    
    # 初始化损失历史记录
    loss_history = []  
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        avg_loss = 0

        # 遍历所有视频序列
        for i in range(0, total_samples, batch_size*num_frames):
            
            batch_images = []
            batch_actions = []
            batch_nonterminals = []
            
            # 检查是否有足够的数据构建完整批次
            if i + batch_size*num_frames > total_samples:
                print(f"⚠️ jump to next batch: need {batch_size*num_frames} samples, but only {total_samples - i} samples left")
                break
            
            for batch_idx in range(batch_size):
                start_idx = i + batch_idx*num_frames
                end_idx = start_idx + num_frames
                
                # 确保不超出数据集边界
                if end_idx > total_samples:
                    print(f"⚠️ jump to next batch: start_idx={start_idx}, end_idx={end_idx}, total_samples={total_samples}")
                    break
                    
                video_images, video_actions, video_nonterminals = build_video_sequence(dataset, start_idx, end_idx)
                
                # 添加到批次列表中
                batch_images.append(video_images)
                batch_actions.append(video_actions)
                batch_nonterminals.append(video_nonterminals)
            
            # 拼接成batch_tensor: [batch_size, num_frames, c, h, w]
            batch_data = [
                torch.cat(batch_images, dim=0).to(device_obj),
                torch.cat(batch_actions, dim=0).to(device_obj),
                torch.cat(batch_nonterminals, dim=0).to(device_obj)
            ]

           
            batch_data[0] = vae_encode(batch_data[0], vae, device_obj)
  
            
            # 扩展batch_size: [1, num_frames, channels, h, w] -> [16, num_frames, channels, h, w]
            batch_data[0] = batch_data[0].repeat(32, 1, 1, 1, 1)

            
            # 同步扩展actions和nonterminals
            batch_data[1] = batch_data[1].repeat(32, 1, 1)  # actions: [1, num_frames, 1] -> [16, num_frames, 1]
            batch_data[2] = batch_data[2].repeat(32, 1)     # nonterminals: [1, num_frames] -> [16, num_frames]

            # 训练步骤
            try:
                out_dict = diffusion_model.training_step(batch_data)
                loss = out_dict["loss"]  # 用loss还是original_loss??
                
                # 反向传播
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # if batch_count % 1 == 0:
                #     print(f"   Batch {batch_count}, Loss: {loss.item():.6f}") # print loss in every 1 batch
                
            except Exception as e:
                print(f"   ❌ error in training step: {e}")
                print(f"   batch_data shapes:")
                print(f"     images: {batch_data[0].shape}")
                print(f"     actions: {batch_data[1].shape}")
                print(f"     nonterminals: {batch_data[2].shape}")
                raise e        
        
        # 计算每个epoch的平均损失并记录
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            # scheduler.step(avg_loss)
            final_avg_loss = avg_loss  # 更新最终的avg_loss
            
            # 每 cfg.data_save_epoch 个epoch打印一次损失并记录到历史
            if (epoch+1) % data_save_epoch == 0:
                loss_history.append(avg_loss)  # 只记录打印的损失值
                loss_message = f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}"
                
                logger.info(loss_message)

            # 检查是否是最佳模型，如果是，且epoch> best_save_interval，则保存最佳模型
                is_best = avg_loss < best_loss
                
                if is_best:
                    # 立即更新最佳损失
                    improvement = (best_loss - avg_loss) / best_loss if best_loss != float('inf') else 1.0
                    best_loss = avg_loss
                    best_message = f"This is the new best loss(improvement: {improvement:.2%})"
                    
                    logger.info(best_message)
                    
                    # 检查是否在保存间隔内且有显著改善
                    if (epoch + 1) >= best_save_interval and improvement >= min_improvement:
                        save_best_checkpoint(model, epoch + 1, best_loss, is_best=True, path=cfg.ckpt_path)
                        save_best_message = f"save best model in {cfg.ckpt_path}(improvement: {improvement:.2%})"
                        
                        logger.info(save_best_message)
        
        # 每gif_save_epoch个epoch run一次test,保存 gif
        if (epoch+1) % gif_save_epoch == 0:
            model_test(cfg.test_img_path, cfg.actions, model, device_obj, cfg.sample_step,epoch+1)

    
    completion_message = "Training completed!"
    print(completion_message)
    logger.info(completion_message)
    

    # 训练完成后保存最终模型
    if epochs >= 1000 and final_avg_loss > 0:
        save_message = "💾 save final training model..."
        print(save_message)
        logger.info(save_message)
        
        save_model(model, epochs, final_avg_loss, path=cfg.ckpt_path)
        
        # 记录训练统计信息
        stats_message = f"📊 training statistics: total epochs: {epochs}, best loss: {best_loss:.6f}, final loss: {final_avg_loss:.6f}, total batches: {batch_count * epochs}"
        print(f"📊 training statistics:")
        print(f"    total epochs: {epochs}")
        print(f"    best loss: {best_loss:.6f}")
        print(f"    final loss: {final_avg_loss:.6f}")
        print(f"    total batches: {batch_count * epochs}")
        logger.info(stats_message)
        
        # 训练完成后进行测试
        model_test(cfg.test_img_path, cfg.actions, model, device_obj, cfg.sample_step,epochs)
    
    # 保存最终损失曲线到output目录
    if len(loss_history) > 0:
        final_loss_curve_path = save_loss_curve(loss_history, data_save_epoch, save_path="output")
        logger.info(f"Final loss curve saved to: {final_loss_curve_path}")
    
    # 记录日志文件路径
    final_log_message = f"log path: {log_path}"
    print(final_log_message)
    logger.info(final_log_message)


    


if __name__ == "__main__":
    train()
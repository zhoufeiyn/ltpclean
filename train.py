# 0920 update: try to overfit level1-1 in one directory

from models.vae.sdxlvae import SDXLVAE
from algorithm import Algorithm
import torch
import config.configTrain as cfg
import matplotlib.pyplot as plt
import os
from datetime import datetime
from infer_test import model_test
import logging
# 导入数据加载模块
from dataLoad import MarioDataset, build_video_sequence_batch



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
    # 使用多进程数据加载优化
    dataset = MarioDataset(cfg.data_path, cfg.img_size, num_workers=8)

    # video sequence parameters
    num_frames = cfg.num_frames
    frame_interval = cfg.frame_interval
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
    num_videos = (total_samples-num_frames) // frame_interval + 1
    print(f"dataset loaded: {total_samples} samples, construct {num_videos} complete video sequences, "
        f"each video has {num_frames} frames, construct {(num_videos + batch_size - 1) // batch_size } batches, the batch size is {batch_size}")

    # 初始化最佳损失跟踪
    best_loss = float('inf')
    min_improvement = cfg.min_improvement  # 最小改善幅度
    final_avg_loss = 0  # 用于保存最终的avg_loss

    # 初始化损失历史记录
    loss_history = []

    # 预计算所有有效的视频序列起始位置
    valid_starts = []
    for start in range(0, total_samples - num_frames + 1, frame_interval):
        valid_starts.append(start)
    
    # 按batch_size分组处理
    num_valid_videos = len(valid_starts)
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        avg_loss = 0

        # 按batch处理 - 优化版本
        for batch_start in range(0, num_valid_videos, batch_size):
            batch_end = min(batch_start + batch_size, num_valid_videos)
            current_batch_size = batch_end - batch_start
            
            # 获取当前batch的起始索引
            current_start_indices = valid_starts[batch_start:batch_end]
            
            # 批量构建视频序列
            batch_images, batch_actions, batch_nonterminals = build_video_sequence_batch(
                dataset, current_start_indices, num_frames
            )
            
            # 如果batch不满，用最后一个视频复制补齐
            if current_batch_size < batch_size:
                last_video_images = batch_images[-1]
                last_video_actions = batch_actions[-1]
                last_video_nonterminals = batch_nonterminals[-1]
                
                for _ in range(batch_size - current_batch_size):
                    batch_images.append(last_video_images)
                    batch_actions.append(last_video_actions)
                    batch_nonterminals.append(last_video_nonterminals)

            # 拼接成batch_tensor
            batch_data = [
                torch.cat(batch_images, dim=0).to(device_obj),
                torch.cat(batch_actions, dim=0).to(device_obj),
                torch.cat(batch_nonterminals, dim=0).to(device_obj)
            ]

           
            batch_data[0] = vae_encode(batch_data[0], vae, device_obj)
            
            # 扩展batch_size: [1, num_frames, channels, h, w] -> [16, num_frames, channels, h, w]
            batch_data[0] = batch_data[0].repeat(16, 1, 1, 1, 1)
            
            
            # 同步扩展actions和nonterminals
            batch_data[1] = batch_data[1].repeat(16, 1, 1)  # actions: [1, num_frames, 1] -> [16, num_frames, 1]
            batch_data[2] = batch_data[2].repeat(16, 1)     # nonterminals: [1, num_frames] -> [16, num_frames]

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
    if epochs >= 200 and final_avg_loss > 0:
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
# 0920 update: try to overfit level1-1 in one directory
# 更新: 添加了权重保存和继续训练功能

"""
使用说明:
1. 正常训练: 在 config/configTrain.py 中设置 resume_training = False
   - 会加载预训练模型 (cfg.model_path)

2. 继续训练: 在 config/configTrain.py 中设置:
   - resume_training = True
   - resume_checkpoint_path = "ckpt/model_epoch100_20251018_19.pth"  # 指定要加载的checkpoint路径
   - 会优先加载继续训练的checkpoint，如果失败则回退到预训练模型

权重加载优先级:
1. 继续训练checkpoint (包含模型权重+优化器状态+训练信息)
2. 预训练模型 (仅包含模型权重)
3. 随机初始化模型

保存的模型包含:
- 模型权重 (network_state_dict)
- 优化器状态 (optimizer_state_dict)
- 训练信息 (epochs, loss, model_name等)

自动保存:
- 定期checkpoint: model_epoch{epoch}_{timestamp}.pth
"""

from models.vae.sdvae import SDVAE
from algorithm import Algorithm
import torch
import config.configTrain as cfg
import matplotlib.pyplot as plt
import os
from datetime import datetime
from infer_test import model_test
import logging
import random
# 导入数据加载模块
from dataloader.dataLoad import MarioDataset, build_video_sequence_batch

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
def save_model(model, optimizer, epochs, final_loss, path=cfg.ckpt_path):
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
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'loss': final_loss,
        'model_name': cfg.model_name,
        'batch_size': cfg.batch_size,
        'num_frames': cfg.num_frames,
        'timestamp': timestamp,
    }

    # 保存模型
    try:
        torch.save(save_data, model_path)
        print(f"✅ save model to {model_path}")

    except Exception as e:
        print(f"❌ save model failed: {e}")


def load_model(model, optimizer, checkpoint_path, device_obj):
    """加载模型权重和优化器状态"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')

    try:
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)

        # 加载模型权重
        model.load_state_dict(checkpoint['network_state_dict'], strict=False)
        print("✅ Model weights loaded successfully!")

        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state loaded successfully!")
        else:
            print("⚠️ No optimizer state found in checkpoint")

        # 获取训练信息
        start_epoch = checkpoint.get('epochs', 0)
        best_loss = checkpoint.get('loss', float('inf'))

        print(f"📊 Loaded checkpoint info:")
        print(f"   - Epoch: {start_epoch}")
        print(f"   - Loss: {best_loss:.6f}")
        print(f"   - Model: {checkpoint.get('model_name', 'Unknown')}")

        return start_epoch, best_loss

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return 0, float('inf')


def vae_encode(batch_data_images, vae_model, device, scale_factor=0.18215):
    """vae encode the images"""
    # 将图像编码到潜在空间: [batch_size, num_frames, 3, 128, 128] -> [batch_size, num_frames, 4, 32, 32]
    with torch.no_grad():
        batch_size_videos, num_frames, channels, h, w = batch_data_images.shape
        # 重塑为 [batch_size * num_frames, 3, 128, 128] 进行批量编码
        images_flat = batch_data_images.reshape(-1, channels, h, w).to(device)

        # VAE编码
        if vae_model is not None:
            latent_dist = vae_model.encode(images_flat)  # [batch_size * num_frames, 3, 128, 128]
            latent_images = latent_dist.sample()  # 采样潜在表示 [batch_size * num_frames, 4, 32, 32]

            latent_images = latent_images * scale_factor
            # print(f"   Using scale factor: {Config.scale_factor}")

            # 重塑回 [batch_size, num_frames, 4, 32, 32]
            latent_images = latent_images.reshape(batch_size_videos, num_frames, 4, 32,
                                                  32)  # [batch_size, num_frames, 4, 32, 32]
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

    loss_log_iter = cfg.loss_log_iter
    # gif_save_iter = cfg.gif_save_iter
    gif_save_epoch = cfg.gif_save_epoch
    checkpoint_save_epoch = cfg.checkpoint_save_epoch

    # 使用Algorithm类加载完整的预训练模型（包含VAE和Diffusion）
    model = Algorithm(model_name, device_obj)
    model = model.to(device_obj)


    # 初始化训练状态
    start_epoch = 0
    best_loss = float('inf')

    # 检查是否需要继续训练 - 优先加载继续训练的checkpoint
    if cfg.resume_training and cfg.resume_checkpoint_path:
        print(f"🔄 Resuming training from checkpoint: {cfg.resume_checkpoint_path}")
        start_epoch, best_loss = load_model(model, opt, cfg.resume_checkpoint_path, device_obj)
        if start_epoch > 0:
            print(f"✅ Resuming training from epoch {start_epoch}")
        else:
            print("⚠️ Failed to load resume checkpoint, falling back to pretrained model")
            # 如果继续训练加载失败，回退到预训练模型
            checkpoint_path = os.path.join(cfg.ckpt_path, cfg.model_path)
            if os.path.exists(checkpoint_path):
                print(f"📥 Loading diffusion forcing pretrained checkpoint: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
                model.load_state_dict(state_dict['network_state_dict'], strict=False)
                print("✅ diffusion forcing Pretrained checkpoint loaded successfully!")
            else:
                print(
                    f"⚠️ diffusion forcing pretrained checkpoint not found: {checkpoint_path}, using random initialized model")
    else:
        # 没有设置继续训练，加载预训练模型
        checkpoint_path = os.path.join(cfg.ckpt_path, cfg.model_path)
        if os.path.exists(checkpoint_path):
            print(f"📥 Loading diffusion forcing pretrained checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
            model.load_state_dict(state_dict['network_state_dict'], strict=False)
            print("✅ diffusion forcing pretrained checkpoint loaded successfully!")
        else:
            print(
                f"⚠️ diffusion forcing pretrained checkpoint not found: {checkpoint_path}, using random initialized model")
        print("🆕 Starting fresh training")

    # 获取VAE和Diffusion模型
    vae = SDVAE().to(device_obj)
    diffusion_model = model.df_model

    opt = diffusion_model.configure_optimizers_gpt()
    # 初始化优化器 - 使用简单的AdamW
    # opt = torch.optim.AdamW(diffusion_model.parameters(), lr=1e-4, weight_decay=1e-5)
    # 加载您自己训练的VAE权重
    custom_vae_path = cfg.vae_model
    if custom_vae_path and os.path.exists(custom_vae_path):
        print(f"📥 load your own vae ckpt: {custom_vae_path}")
        custom_state_dict = torch.load(custom_vae_path, map_location=device_obj)
        vae.load_state_dict(custom_state_dict['network_state_dict'], strict=False)
        print("✅ your vae ckpt loaded successfully！")
    else:
        print("ℹ️ use default pre-trained vae ckpt")

    if vae is not None:
        vae.eval()
        for param in vae.parameters():  # freeze VAE parameters
            param.requires_grad = False
        print("✅ VAE already loaded，VAE parameters has been frozen")
    else:
        print("⚠️ Cannot find VAE model")
    epochs, batch_size = cfg.epochs, cfg.batch_size

    print("---1. start training----")
    print("---2. load dataset---")
    total_samples = len(dataset)
    # 检查是否有足够的数据
    if total_samples < num_frames:
        print(f"❌ dataset not enough: need at least {num_frames} samples, but only {total_samples} samples")
        return
    # 计算可以创建多少个完整的视频序列
    num_videos = (total_samples - num_frames) // frame_interval + 1
    print(f"dataset loaded: {total_samples} samples, construct {num_videos} complete video sequences, "
          f"each video has {num_frames} frames, construct {(num_videos + batch_size - 1) // batch_size} batches, the batch size is {batch_size}")

    # 初始化损失历史记录
    loss_history = []
    final_avg_loss = 0  # 用于保存最终的avg_loss

    # 预计算所有有效的视频序列起始位置,间隔一个frame_interval取一个video sequence, 最终剩下不足一个video的扔掉
    valid_starts = []
    for start in range(0, total_samples - num_frames + 1, frame_interval):
        valid_starts.append(start)

    # 按batch_size分组处理
    num_valid_videos = len(valid_starts)

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        batch_count = 0

        # 🔥 每个epoch开始时shuffle视频序列顺序
        shuffled_valid_starts = valid_starts.copy()
        random.shuffle(shuffled_valid_starts)

        # 按batch处理 - 优化版本
        for batch_start in range(0, num_valid_videos, batch_size):
            batch_end = min(batch_start + batch_size, num_valid_videos)
            current_batch_size = batch_end - batch_start

            # 获取当前batch的起始索引（现在是shuffled的）
            current_start_indices = shuffled_valid_starts[batch_start:batch_end]

            # 批量构建视频序列
            batch_images, batch_actions, batch_nonterminals = build_video_sequence_batch(dataset, current_start_indices, num_frames)

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

            # # for small dataset 扩展batch_size: [b, num_frames, channels, h, w] -> [b*16, num_frames, channels, h, w]
            batch_data[0] = batch_data[0].repeat(156, 1, 1, 1, 1)

            # 同步扩展actions和nonterminals
            batch_data[1] = batch_data[1].repeat(156, 1, 1)  # actions: [1, num_frames, 1] -> [16, num_frames, 1]
            batch_data[2] = batch_data[2].repeat(156, 1)  # nonterminals: [1, num_frames] -> [16, num_frames]

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

            # 查看batch里的loss和 gif
            if batch_count % loss_log_iter == 0:
                batch_loss = loss.item()
                loss_message = f"Epoch {epoch + 1}/{epochs}, in batch: {batch_count},  Loss: {batch_loss:.6f}"
                logger.info(loss_message)

        # 一个epoch
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            # scheduler.step(avg_loss)
            final_avg_loss = avg_loss  # 更新最终的avg_loss
            # 每 1 个epoch打印一次损失并记录到历史
            loss_history.append(avg_loss)  # 只记录打印的损失值
            loss_message = f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}"
            logger.info(loss_message)
            # 检查是否是最佳模型，如果是，则保存最佳模型
            is_best = avg_loss < best_loss
            if is_best:
                # 立即更新最佳损失
                improvement = (best_loss - avg_loss) / best_loss if best_loss != float('inf') else 1.0
                best_loss = avg_loss
                best_message = f"This is the new best loss(improvement: {improvement:.2%})"
                logger.info(best_message)

        # 每gif_save_epoch个epoch run一次test,保存 gif
        if (epoch + 1) % gif_save_epoch == 0:
            # 确保output目录存在

            model_test(cfg.test_img_path1, cfg.actions1, model, vae, device_obj, cfg.sample_step,
                       f'{cfg.test_img_path1[-9:-4]}_epoch{epoch + 1}_r', epoch=epoch + 1, output_dir='output')
            # model_test(cfg.test_img_path1, cfg.actions2, model, vae, device_obj, cfg.sample_step,
            #            f'{cfg.test_img_path1[-9:-4]}_epoch{epoch + 1}_rj', epoch=epoch + 1, output_dir='output')

        # 每checkpoint_save_epoch个epoch保存一次checkpoint
        if (epoch + 1) % checkpoint_save_epoch == 0:
            current_loss = avg_loss if batch_count > 0 else 0
            save_model(model, opt, epoch + 1, current_loss, path=cfg.ckpt_path)
            checkpoint_message = f"💾 Checkpoint saved at epoch {epoch + 1}"
            logger.info(checkpoint_message)

    completion_message = "Training completed!"
    print(completion_message)
    logger.info(completion_message)

    # 训练完成后保存最终模型
    if epochs >= 1 and final_avg_loss > 0:
        save_message = "💾 save final training model..."
        print(save_message)
        logger.info(save_message)

        save_model(model, opt, epochs, final_avg_loss, path=cfg.ckpt_path)

        # 记录训练统计信息
        stats_message = f"📊 training statistics: total epochs: {epochs}, best loss: {best_loss:.6f}, final loss: {final_avg_loss:.6f}, total batches: {batch_count * epochs}"
        print(f"📊 training statistics:")
        print(f"    total epochs: {epochs}")
        print(f"    best loss: {best_loss:.6f}")
        print(f"    final loss: {final_avg_loss:.6f}")
        print(f"    total batches: {batch_count * epochs}")
        logger.info(stats_message)

        # 训练完成后进行测试
        model_test(cfg.test_img_path1, cfg.actions1, model, vae, device_obj, cfg.sample_step,
                   f'{cfg.test_img_path1[-9:-4]}_result_{epochs}_r', epoch='result', output_dir='output')
        # model_test(cfg.test_img_path1, cfg.actions2, model, vae, device_obj, cfg.sample_step,
        #            f'{cfg.test_img_path1[-9:-4]}_result_{epochs}_rj', epoch='result', output_dir='output')

    # 保存最终损失曲线到output目录
    if len(loss_history) > 0:
        final_loss_curve_path = save_loss_curve(loss_history, 1, save_path="output")
        logger.info(f"Final loss curve saved to: {final_loss_curve_path}")

    # 记录日志文件路径
    final_log_message = f"log path: {log_path}"
    print(final_log_message)
    logger.info(final_log_message)


if __name__ == "__main__":
    train()
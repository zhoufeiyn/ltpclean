from models.vae.sdvae import SDVAE
import torch
import config.configVAE as cfg
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from datetime import datetime

from utils.dataLoad import MarioDataset, build_img_batch
from train import setup_logging, save_loss_curve

device: str = "cuda" if torch.cuda.is_available() else "cpu"

def save_model_with_optimizer(model, optimizer, scheduler, epochs, final_loss, best_loss, loss_history, path=cfg.ckpt_path):
    """保存训练好的模型到ckpt目录，包含优化器和调度器状态"""
    if not os.path.exists(path):
        os.makedirs(path)

    # 生成文件名（包含时间戳和epoch信息）
    timestamp = datetime.now().strftime("%Y%m%d_%H")
    model_filename = f"vae_epoch{epochs}_{timestamp}.pth"
    model_path = os.path.join(path, model_filename)

    # 准备保存的数据
    save_data = {
        'network_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epochs,
        'loss': final_loss,
        'best_loss': best_loss,
        'loss_history': loss_history,
        'model_name': 'SDVAE',
        'batch_size': cfg.batch_size,
    }

    # 保存模型
    try:
        torch.save(save_data, model_path)
        print(f"✅ VAE model saved to {model_path}")

    except Exception as e:
        print(f"❌ Save VAE model failed: {e}")


def validate_model(model, val_dataset, device_obj, batch_size):
    """验证模型性能"""
    model.eval()
    val_loss = 0
    val_count = 0

    with torch.no_grad():
        for idx in range(0, len(val_dataset), batch_size):
            batch_img = build_img_batch(val_dataset, idx, batch_size)

            try:
                # VAE前向传播
                encoded = model.encode(batch_img)
                latent = encoded.sample()
                decode_img = model.decode(latent)

                # 只使用L1重建损失
                loss = F.l1_loss(decode_img, batch_img)

                val_loss += loss.item()
                val_count += 1

            except Exception as e:
                print(f"❌ Error in validation: {e}")
                continue

    model.train()
    return val_loss / val_count if val_count > 0 else float('inf')

def vae_test(img_path, model, device_obj, e=None, out_dir='output/VAE' ):
    """测试VAE模型的编码解码效果"""
    import os
    import glob
    from PIL import Image
    import torchvision.transforms as transforms
    import torch.nn.functional as F

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 确保输出目录存在
    output_dir = out_dir+f"/epoch{e}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件
    if os.path.isfile(img_path):
        img_files = [img_path]
    else:
        img_files = glob.glob(os.path.join(img_path, "*.png")) + glob.glob(os.path.join(img_path, "*.jpg"))
    
    if not img_files:
        print(f"❌ No images found in {img_path}")
        return
    
    model.eval()
    total_loss = 0
    num_images = 0
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    with torch.no_grad():
        for img_file in img_files[:10]:  # 限制测试图片数量
            try:
                # 加载和预处理图像
                img = Image.open(img_file).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device_obj)
                
                # VAE编码解码
                encoded = model.encode(img_tensor)
                latent = encoded.sample()
                decoded = model.decode(latent)
                
                # 计算重建损失
                loss = F.l1_loss(decoded, img_tensor)
                total_loss += loss.item()
                num_images += 1
                
                # 保存原始图像和重建图像
                img_name = os.path.splitext(os.path.basename(img_file))[0]
                
                # 转换为可保存的格式
                original_img = (img_tensor[0].cpu() * 0.5 + 0.5).clamp(0, 1)
                reconstructed_img = (decoded[0].cpu() * 0.5 + 0.5).clamp(0, 1)
                
                # 保存图像
                transforms.ToPILImage()(original_img).save(os.path.join(output_dir, f"{img_name}_original.png"))
                transforms.ToPILImage()(reconstructed_img).save(os.path.join(output_dir, f"{img_name}_reconstructed.png"))
                
            except Exception as ex:
                print(f"❌ Error processing {img_file}: {ex}")
                continue
    
    if num_images > 0:
        avg_loss = total_loss / num_images
        print(f"✅ VAE test completed. Average reconstruction loss: {avg_loss:.6f}")
        print(f"📁 Test images saved to: {output_dir}")
    else:
        print("❌ No images were successfully processed")
    
    model.train()  # 恢复训练模式


def train():
    logger, log_path = setup_logging()

    device_obj = torch.device(device)
    # 使用多进程数据加载优化
    dataset = MarioDataset(cfg.data_path, cfg.img_size, num_workers=8)
    model = SDVAE().to(device_obj)
    
    # 分割训练集和验证集
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)  # 80%用于训练
    val_size = total_samples - train_size   # 20%用于验证
    
    # # 创建随机索引分割（确保训练集和验证集都包含不同时间段的图片）
    import random
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 创建训练和验证数据集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"📊 Dataset split: {train_size} training samples, {val_size} validation samples")

    epochs = cfg.epochs
    loss_log_iter = cfg.loss_log_iter
    img_save_epoch = cfg.img_save_epoch
    batch_size = cfg.batch_size
    ckpt_save_epoch = cfg.checkpoint_save_epoch

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, verbose=True)
    
    # 检查是否有预训练检查点
    start_epoch = 0
    best_loss = float('inf')
    loss_history = []
    
    # 查找最新的检查点
    checkpoint_files = []
    if os.path.exists(cfg.ckpt_path):
        checkpoint_files = [f for f in os.listdir(cfg.ckpt_path) if f.endswith('.pth')]
    
    if checkpoint_files:
        # 按文件名排序，获取最新的检查点
        checkpoint_files.sort()
        latest_checkpoint = os.path.join(cfg.ckpt_path, checkpoint_files[-1])
        
        try:
            print(f"📥 Loading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device_obj, weights_only=False)
            
            model.load_state_dict(checkpoint['network_state_dict'])
            opt.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
            scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))
            start_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('best_loss', float('inf'))
            loss_history = checkpoint.get('loss_history', [])
            
            print(f"✅ Checkpoint loaded successfully! Starting from epoch {start_epoch + 1}")
            print(f"📊 Previous best loss: {best_loss:.6f}")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("🔄 Starting training from scratch...")
    
    final_avg_loss = 0  # 用于保存最终的avg_loss



    for e in range(start_epoch, epochs):
        total_loss = 0
        batch_count = 0
        for idx in range(0, train_size, batch_size):
            batch_img = build_img_batch(train_dataset, idx, batch_size)
            try:
                # VAE前向传播
                encoded = model.encode(batch_img)
                latent = encoded.sample()
                decode_img = model.decode(latent)
                
                # 只使用L1重建损失
                loss = F.l1_loss(decode_img, batch_img)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                batch_count += 1
            except Exception as e:
                print(f"   ❌ error in training step: {e}")
                print(f"    batch_data shapes: {batch_img.shape}")
                raise e

            if batch_count % loss_log_iter ==0:
                batch_loss = loss.item()
                loss_message = f"Epoch {e + 1}/{epochs}, in batch: {batch_count},  Loss: {batch_loss:.6f}"
                logger.info(loss_message)

        # 一个epoch
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            
            # 验证模型
            val_loss = validate_model(model, val_dataset, device_obj, batch_size)
            
            scheduler.step(val_loss)  # 使用验证损失更新学习率
            final_avg_loss = avg_loss  # 更新最终的avg_loss
            
            # 每 1 个epoch打印一次损失并记录到历史
            loss_history.append(avg_loss)  # 只记录打印的损失值
            loss_message = f"Epoch {e + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}"
            logger.info(loss_message)
            
            # 检查是否是最佳模型（基于验证损失）
            is_best = val_loss < best_loss
            if is_best:
                # 立即更新最佳损失
                improvement = (best_loss - val_loss) / best_loss if best_loss != float('inf') else 1.0
                best_loss = val_loss
                best_message = f"This is the new best validation loss(improvement: {improvement:.2%})"
                logger.info(best_message)

        if (e + 1) % img_save_epoch == 0:
            vae_test(cfg.test_img_path,model,device_obj,e,cfg.out_dir)

        if (e + 1) % ckpt_save_epoch == 0:
            current_loss = avg_loss if batch_count > 0 else 0
            save_model_with_optimizer(model, opt, scheduler, e + 1, current_loss, best_loss, loss_history, path=cfg.ckpt_path)
            checkpoint_message = f"💾 Checkpoint saved at epoch {e + 1}"
            logger.info(checkpoint_message)

    completion_message = "Training completed!"
    print(completion_message)
    logger.info(completion_message)
    if epochs >= 1 and final_avg_loss > 0:
        save_message = "💾 save final training model..."
        print(save_message)
        logger.info(save_message)

        save_model_with_optimizer(model, opt, scheduler, epochs, final_avg_loss, best_loss, loss_history, path=cfg.ckpt_path)

        # 记录训练统计信息
        stats_message = f"📊 training statistics: total epochs: {epochs}, best loss: {best_loss:.6f}, final loss: {final_avg_loss:.6f}, batches per epoch: {batch_count}"
        print(f"📊 training statistics:")
        print(f"    total epochs: {epochs}")
        print(f"    best loss: {best_loss:.6f}")
        print(f"    final loss: {final_avg_loss:.6f}")
        print(f"    batches per epoch: {batch_count}")
        logger.info(stats_message)

        vae_test(cfg.test_img_path,model,device_obj,'result',cfg.out_dir)

        if len(loss_history) > 0:
            final_loss_curve_path = save_loss_curve(loss_history, 1, save_path=cfg.out_dir)
            logger.info(f"Final loss curve saved to: {final_loss_curve_path}")

        # 记录日志文件路径
        final_log_message = f"log path: {log_path}"
        print(final_log_message)
        logger.info(final_log_message)

if __name__ == "__main__":
    train()
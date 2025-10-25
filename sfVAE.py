from torchvision.transforms import InterpolationMode

from models.vae.sdvae import SDVAE
import torch
import config.configVAE as cfg
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from datetime import datetime

from dataloader.dataLoad import MarioDataset
from train import setup_logging, save_loss_curve

def build_img_batch_from_indices(dataset, indices):
    """根据索引列表构建图片批次"""
    batch_images = []
    for idx in indices:
        image, _, _ = dataset[idx]
        batch_images.append(image)
    return torch.stack(batch_images, dim=0)

device: str = "cuda" if torch.cuda.is_available() else "cpu"

def save_model_with_optimizer(model, optimizer, epochs, final_loss, best_loss, loss_history, path=cfg.ckpt_path):
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

def infer_test(img):
    device_obj = torch.device(device)
    model = SDVAE()
    
    # 只有当 model_path 不为空时才拼接路径
    if cfg.model_path:
        ckpt_path = os.path.join(cfg.ckpt_path, cfg.model_path)
    else:
        ckpt_path = cfg.ckpt_path
    
    if os.path.exists(ckpt_path):
        print(f"📥 load pretrained checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(state_dict['network_state_dict'], strict=False)
        print("ckpt loaded successfully")
    else:
        print(f"⚠️ Checkpoint not found: {ckpt_path}, use initialized model")
    
    model = model.to(device_obj)
    vae_test(img, model, device_obj, 'infer')





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
    output_dir = out_dir+f"/epoch{e+1}"
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
        transforms.Resize((256, 256),interpolation=InterpolationMode.NEAREST),
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


def estimate_scaling_factor():
    target_std = 1.0

    device_obj = torch.device(device)
    # 使用多进程数据加载优化
    dataset = MarioDataset(cfg.data_path, cfg.img_size, num_workers=8)
    model = SDVAE().to(device_obj)
    
    # 加载训练好的模型
    if cfg.model_path:
        ckpt_path = os.path.join(cfg.ckpt_path, cfg.model_path)
    else:
        ckpt_path = cfg.ckpt_path
    
    if os.path.exists(ckpt_path):
        print(f"📥 Loading pretrained checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(state_dict['network_state_dict'], strict=False)
        print("✅ Checkpoint loaded successfully")
    else:
        print(f"⚠️ Checkpoint not found: {ckpt_path}, using initialized model")
    
    model.eval()
    std_list = []
    
    # 使用全部数据进行统计
    total_samples = len(dataset)
    batch_size = cfg.batch_size
    
    print(f"📊 Computing scaling factor using {total_samples} samples")
    print(f"📊 Batch size: {batch_size}")

    with torch.no_grad():
        for batch_idx in range(0, total_samples, batch_size):
            # 构建批次索引
            end_idx = min(batch_idx + batch_size, total_samples)
            batch_indices = list(range(batch_idx, end_idx))
            
            # 构建批次数据
            batch_img = build_img_batch_from_indices(dataset, batch_indices).to(device_obj)
            
            # VAE前向传播
            encoded = model.encode(batch_img).sample()
            std_list.append(encoded.std().item())
            
            if batch_idx % (batch_size * 10) == 0:
                print(f"Processed {batch_idx}/{total_samples} samples")
    
    avg_std = sum(std_list) / len(std_list)
    scaling = target_std / avg_std
    
    print(f"📊 Computed scaling factor:")
    print(f"   Average latent std: {avg_std:.6f}")
    print(f"   Target std: {target_std:.6f}")
    print(f"   Recommended scaling factor: {scaling:.6f}")
    
    return scaling




if __name__ == "__main__":
    estimate_scaling_factor()
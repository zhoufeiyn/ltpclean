# 数据加载模块 - 优化的大数据集处理
# 包含MarioDataset类和相关的视频序列构建函数

from typing import Optional
import re
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from concurrent.futures import ProcessPoolExecutor


class MarioDataset(Dataset):
    """load mario dataset __init__ action and img paths,
     __getitem__  will return image and corresponding action"""
    """up to date: 2025-09-20 only load all frames in one directory,
     return array ofimages and actions"""
    def __init__(self, data_path: str, image_size, num_workers=4):
        self.data_path = data_path
        self.image_size = image_size
        self.num_workers = num_workers
        self.image_files = [] # image files path (xxx.png)
        self.actions = [] # action (0-255)
        self._load_data()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # [0, 1]
            transforms.Normalize(0.5, 0.5),  # [-1, 1]
        ])
        
    def _load_data(self):
        """load all png files and corresponding actions - optimized for large datasets"""
        print(f" data path is scanning: {self.data_path}")
        if not os.path.exists(self.data_path): 
            print(f"❌ data path not found: {self.data_path}")
            return
        
        # 使用多进程扫描文件
        import multiprocessing as mp
        
        # 收集所有子目录
        subdirs = []
        for root, dirs, files in os.walk(self.data_path):
            if root != self.data_path and files:  # 跳过根目录，只处理有文件的子目录
                subdirs.append(root)
        
        print(f"Found {len(subdirs)} subdirectories to scan")
        
        # 并行处理每个子目录
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(MarioDataset._scan_directory, subdir) for subdir in subdirs]
            
            for future in futures:
                files, actions = future.result()
                self.image_files.extend(files)
                self.actions.extend(actions)
        
        print(f"✅ Loaded {len(self.image_files)} valid images")
    
    @staticmethod
    def _scan_directory(directory):
        """扫描单个目录，返回文件路径和动作"""
        files = []
        actions = []
        
        for file in os.listdir(directory):
            if file.lower().endswith('.png'):
                file_path = os.path.join(directory, file)
                action = MarioDataset._extract_action_from_filename_static(file)
                if action is not None:
                    files.append(file_path)
                    actions.append(action)
        
        return files, actions
    
    @staticmethod
    def _extract_action_from_filename_static(filename: str) -> Optional[int]:
        """静态方法版本的动作提取函数"""
        pattern = r'_a(\d+)_'
        match = re.search(pattern, filename)
        if match:
            action = int(match.group(1))
            action_mapped = MarioDataset._map_action_to_playgenaction_static(action)
            return action_mapped
        return None
    
    @staticmethod
    def _map_action_to_playgenaction_static(action: int) -> int:
        """静态方法版本的动作映射函数"""
        if action == 20:
            return 1
        elif action == 148:
            return 2
        elif action == 48:
            return 3
        elif action == 176:
            return 4
        elif action == 128:
            return 5
        elif action == 16:
            return 6
        elif action == 0:
            return 0
        else:
            return 0
    

        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """get the data sample of the specified index - optimized for large datasets"""
        if idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_files)}")
        
        # 加载图像 - 使用更高效的图像加载
        image_path = self.image_files[idx]
        try:
            # 使用PIL的优化选项
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个默认的黑色图像
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # 获取动作
        action = self.actions[idx] if idx < len(self.actions) else 0
        
        return image, action


def build_video_sequence_batch(dataset, start_indices, num_frames):
    """批量构建视频序列，优化大数据集处理"""
    batch_images = []
    batch_actions = []
    batch_nonterminals = []
    
    # 批量获取数据
    for start_idx in start_indices:
        end_idx = start_idx + num_frames
        
        # 构建单个视频序列
        video_images = []
        video_actions = []
        video_nonterminals = []
        
        for frame_idx in range(start_idx, end_idx):
            image, action = dataset[frame_idx]
            video_images.append(image)
            video_actions.append(action)
            video_nonterminals.append(True)
        
        # 转换为tensor
        images_tensor = torch.stack(video_images, dim=0).unsqueeze(0)  # [b, num_frames, 3, 128, 128]
        actions_tensor = torch.tensor(video_actions, dtype=torch.long).unsqueeze(0).unsqueeze(-1)  # [b, num_frames, 1]
        nonterminals_tensor = torch.tensor(video_nonterminals, dtype=torch.bool).unsqueeze(0)  # [b, num_frames]
        
        batch_images.append(images_tensor)
        batch_actions.append(actions_tensor)
        batch_nonterminals.append(nonterminals_tensor)
    
    return batch_images, batch_actions, batch_nonterminals


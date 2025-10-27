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

from torchvision.transforms import InterpolationMode


class MarioDataset(Dataset):
    """load mario dataset __init__ action and img paths,
     __getitem__  will return image and corresponding action"""
    """up to date: 2025-09-20 only load all frames in one directory,
     return array ofimages and actions"""
    # def __init__(self, data_path: str, image_size, num_workers=4, train_sample=1,num_frames=12):
    def __init__(self, cfg):
        self.data_path = cfg.data_path
        self.image_size = cfg.img_size
        self.num_workers_folders = cfg.num_workers_folders
        self.train_sample = cfg.train_sample
        self.num_frames = cfg.num_frames
        self.frame_interval = cfg.frame_interval  # 添加 frame_interval 参数
        self.image_files = [] # image files path (xxx.png)
        self.actions = [] # action (0-255)
        self.nonterminals = []
        self._load_data()
        image_size = cfg.img_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),interpolation=InterpolationMode.NEAREST),
            # transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # [0, 1]
            transforms.Normalize(0.5, 0.5),  # [-1, 1]
        ])
        
        # 预计算有效的视频序列起始位置（间隔 frame_interval）
        self.valid_starts = []
        total_samples = len(self.image_files)
        for start in range(0, total_samples - self.num_frames + 1, self.frame_interval):
            self.valid_starts.append(start)
        
        print(f"📊 valid video sequences: {len(self.valid_starts)} (interval {self.frame_interval} samples)")
        
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
        
        # 并行处理每个子目录，每个子目录内已按帧号排序
        with ProcessPoolExecutor(max_workers=self.num_workers_folders) as executor:
            futures = [executor.submit(MarioDataset._scan_directory, subdir,self.train_sample) for subdir in subdirs]
            
            for future in futures:
                files, actions, nonterminals = future.result()
                self.image_files.extend(files)
                self.actions.extend(actions)
                self.nonterminals.extend(nonterminals)
        print(f"✅ Loaded {len(self.image_files)} valid images from {len(subdirs)} levels")
    
    @staticmethod
    def _scan_directory(directory,train_sample):
        """扫描单个目录，返回文件路径和动作，按帧号排序"""

        file_data = []  # 存储(file_path, action, nonterminal, frame_num)的列表
        
        # 首先收集所有文件并按帧号排序
        all_files = []
        for file in os.listdir(directory):
            if file.lower().endswith('.png'):
                file_path = os.path.join(directory, file)
                action, nonterminal = MarioDataset._extract_action_nonterminal_from_filename_static(file)
                frame_num = MarioDataset._extract_frame_number_from_filename_static(file)
                if action is not None and frame_num is not None:
                    all_files.append((file_path, action, nonterminal, frame_num))
        
        # 按帧号排序
        all_files.sort(key=lambda x: x[3])
        
        # 按顺序进行跳帧处理（基于实际帧号差值）
        # 逻辑：只保留与上一个添加帧的差值 > train_sample 的帧
        # - nt=0（结束帧）总是添加
        # - nt=1时，如果当前帧与上个已添加帧的帧号差值 > train_sample，才添加
        # 这样自然跳过密集帧，保留稀疏帧
        last_added_frame_num = None  # 记录上一个添加的帧号
        
        for file_path, action, nonterminal, frame_num in all_files:
            # nt=0（游戏结束帧）总是添加
            if not nonterminal:
                file_data.append((file_path, action, nonterminal, frame_num))
                last_added_frame_num = frame_num
            # nt=1（游戏进行中）
            elif last_added_frame_num is None:
                # 第一帧，直接添加
                file_data.append((file_path, action, nonterminal, frame_num))
                last_added_frame_num = frame_num
            elif frame_num - last_added_frame_num > train_sample:
                # 帧号差值>train_sample，添加
                file_data.append((file_path, action, nonterminal, frame_num))
                last_added_frame_num = frame_num
            # 否则跳过（帧号差值<=train_sample且nt=1）
        
        # 将最后一帧的nonterminal设置为False（游戏结束）
        if file_data:
            last_item = file_data[-1]
            file_data[-1] = (last_item[0], last_item[1], False, last_item[3])
        
        # 分离数据
        files = [item[0] for item in file_data]
        actions = [item[1] for item in file_data]
        nonterminals = [item[2] for item in file_data]
        
        return files, actions, nonterminals
    
    @staticmethod
    def _extract_frame_number_from_filename_static(filename: str) -> Optional[int]:
        """从文件名中提取帧号"""
        pattern = r'_f(\d+)_'
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
        return None
    
    @staticmethod
    def _extract_action_nonterminal_from_filename_static(filename: str) -> Optional[int]:
        """静态方法版本的动作提取函数"""
        pattern1 = r'_a(\d+)_'
        pattern2 = r'_nt(\d+)'  # 匹配nt后面的数字（后面可能是下划线或点号）
        match1 = re.search(pattern1, filename)
        match2 = re.search(pattern2, filename)
        if match1:
            action_mapped = int(match1.group(1))
            # action_mapped = MarioDataset._map_action_to_playgenaction_static(action)
        else:
            action_mapped = None
        
        # if match2:
        #     nonterminal = int(match2.group(1))  # 修改：group(1)而不是group(2)
        #     nonterminal = nonterminal == 1
        # else:
        #     nonterminal = False
        if match2:
            nonterminal = True
        return action_mapped, nonterminal


    # @staticmethod
    # def _map_action_to_playgenaction_static(action: int) -> int:
    #     """静态方法版本的动作映射函数
    #     映射规则：
    #     - 0/45: 无动作或未识别
    #     - 1: 右移 (r)
    #     - 2: 向右跳 (rj)
    #     - 3: 左移 (l)
    #     - 4: 向左跳 (lj)
    #     - 5: 原地跳 (j)
    #     - 6: 加速或下蹲 (b 或 bd)
    #     - 7: 加速向右下 (brd)
    #     - 8: 加速向左下 (bld)
    #     """
    #     if action == 0: # 无动作
    #         return 0
    #     if action == 2:
    #         return 1  #
    #     elif action == 148:
    #         return 2
    #     elif action == 48:
    #         return 3
    #     elif action == 176:
    #         return 4
    #     elif action == 144:
    #         return 5
    #     elif action in (16, 18):
    #         return 6
    #     elif action == 22:
    #         return 7
    #     elif action == 50:
    #         return 8
    #     else:
    #         return 45

    def __len__(self):
        """返回有效的视频序列数量（不是原始样本数量）"""
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        """get the data sample of the specified index - optimized for large datasets
        注意：idx 是 valid_starts 中的索引，不是原始样本索引
        """
        if idx >= len(self.valid_starts):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.valid_starts)}")

        # 从 valid_starts 中获取真实的起始索引
        start_idx = self.valid_starts[idx]
        end_idx = start_idx + self.num_frames

        # 构建单个视频序列
        video_images = []
        video_actions = []
        video_nonterminals = []

        for cur_idx in range(start_idx, end_idx):
            # 加载图像 - 使用更高效的图像加载
            image_path = self.image_files[cur_idx]
            try:
                # 使用PIL的优化选项
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # 返回一个默认的黑色图像
                image = torch.zeros(3, self.image_size, self.image_size)

            # 获取动作
            action = self.actions[cur_idx] if cur_idx < len(self.actions) else 0
            nonterminal = self.nonterminals[cur_idx] if cur_idx < len(self.nonterminals) else False
            video_images.append(image)
            video_actions.append(action)
            video_nonterminals.append(nonterminal)

        # 转换为tensor
        images_tensor = torch.stack(video_images, dim=0)  # [num_frames, 3, 128, 128]
        actions_tensor = torch.tensor(video_actions, dtype=torch.long).unsqueeze(-1)  # [num_frames, 1]
        nonterminals_tensor = torch.tensor(video_nonterminals, dtype=torch.bool)  # [num_frames]

        return images_tensor, actions_tensor, nonterminals_tensor


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

            image, action, nonterminal = dataset[frame_idx]
            video_images.append(image)
            video_actions.append(action)
            video_nonterminals.append(nonterminal)
        
        # 转换为tensor
        images_tensor = torch.stack(video_images, dim=0).unsqueeze(0)  # [b, num_frames, 3, 128, 128]
        actions_tensor = torch.tensor(video_actions, dtype=torch.long).unsqueeze(0).unsqueeze(-1)  # [b, num_frames, 1]
        nonterminals_tensor = torch.tensor(video_nonterminals, dtype=torch.bool).unsqueeze(0)  # [b, num_frames]
        
        batch_images.append(images_tensor)
        batch_actions.append(actions_tensor)
        batch_nonterminals.append(nonterminals_tensor)
    
    return batch_images, batch_actions, batch_nonterminals


def build_img_batch(dataset, start_indices,batch_size):
    """批量构建图片训练VAE"""
    batch_images = []
    # 批量获取数据
    for idx in range(batch_size):
        image, _, _= dataset[start_indices+idx]
        batch_images.append(image)
    # 转换为tensor
    images_tensor = torch.stack(batch_images, dim=0)  # [b, 3, 256, 256]
    return images_tensor

# 0920 update: try to overfit level1-1 in one directory


import torch

import config.configTrain as cfg


# 导入数据加载模块
from dataloader.dataLoad import MarioDataset, build_video_sequence_batch

device: str = "cuda" if torch.cuda.is_available() else "cpu"



def train():

    device_obj = torch.device(device)
    # 使用多进程数据加载优化
    dataset = MarioDataset(cfg.data_path, cfg.img_size, num_workers=8)

    # video sequence parameters
    num_frames = 8
    frame_interval = 8

    epochs, batch_size = 1, 1


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



    # 预计算所有有效的视频序列起始位置,间隔一个frame_interval取一个video sequence, 最终剩下不足一个video的扔掉
    valid_starts = []
    for start in range(0, total_samples - num_frames + 1, frame_interval):
        valid_starts.append(start)

    # 按batch_size分组处理
    num_valid_videos = len(valid_starts)

    for epoch in range(epochs):


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
            print(batch_data[1],batch_data[2])


            # # 扩展batch_size: [b, num_frames, channels, h, w] -> [b*16, num_frames, channels, h, w]
            # batch_data[0] = batch_data[0].repeat(32, 1, 1, 1, 1)
            # # 同步扩展actions和nonterminals
            # batch_data[1] = batch_data[1].repeat(32, 1, 1)  # actions: [1, num_frames, 1] -> [16, num_frames, 1]
            # batch_data[2] = batch_data[2].repeat(32, 1)  # nonterminals: [1, num_frames] -> [16, num_frames]




if __name__ == "__main__":
    train()
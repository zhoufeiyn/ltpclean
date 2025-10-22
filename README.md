# <div align="center">

# Learning to Play

</div>



![Model Structure](./statics/structure.png)

This project implements a game generation model based on VAE and Diffusion DiT for generating playable game content.

> **Note**: This project is based on the [Playable Game Generation](https://github.com/GreatX3/Playable-Game-Generation) framework by GreatX3.

# 🏭 Environment Setup

Set up the environment, and install dependencies:
```
pip install -r requirements.txt
```
Before starting, you need to set the running device in the configuration file. The default is `device = "cuda:0"`.

# 🚀 Model Training

## VAE Training
Use `trainVAE.py` to train the VAE model:
```
python trainVAE.py
```

## Diffusion DiT Training
Use `train.py` to train the Diffusion DiT model:
```
python train.py
```

# 🔮 Model Inference

Use `infer_test.py` for model inference:
```
python infer_test.py -i 'eval_data/demo1.png' -a r,r,r,r,r,r
```

## Parameters:
- `-i`: Input image path (e.g., `eval_data/demo1.png`)
- `-a`: Action sequence for each frame, separated by commas

## Available Actions:
- `l`: Move left
- `r`: Move right  
- `j`: Jump
- `f`: Fire
- `lj`: Left jump
- `rj`: Right jump
- `n`: No action (null)

After inference is complete, the generated game content will be saved in the output directory.

# 📋 Project Status

![Current Progress](./statics/current.png)

This project is currently under active development. The following components are being implemented:

## 🚧 In Progress

1. **PPO Policy AI Agent Training**: Implementing Proximal Policy Optimization algorithms for intelligent agent behavior learning and large-scale data collection
2. **VAE Model Enhancement**: Optimizing Variational Autoencoder architecture to improve image encoding and decoding quality
3. **Large-scale Diffusion Model Training**: Loading extensive datasets to train robust diffusion models for high-quality game content generation

## 🔄 Development Pipeline

- **Data Collection**: Automated gameplay data gathering through AI agent interactions
- **Model Optimization**: Continuous improvement of VAE and Diffusion DiT architectures
- **Performance Evaluation**: Comprehensive testing and validation of generated content quality

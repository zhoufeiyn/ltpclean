import re

class ConfigDF():
    use_df_config = True

    def get_params(self, model_name):
        if self.use_dit:
            pattern = r'(?:_z(\d+))|(?:_c(\d+))|(?:_n(\d+))'
        else:
            pattern = r'(?:_z(\d+))|(?:_c(\d+))'
        matches = re.findall(pattern, model_name)
        numbers = []
        for match in matches:
            for num in match:
                if num:
                    numbers.append(int(num))
        return tuple(numbers)

    def _set_default_yaml_params(self):
        """设置YAML文件中的所有默认参数"""
        # 基础参数
        self.debug = False
        self.lr = 1e-4
        self.x_shape = [4, 32, 32]
        self.z_shape = [32, 32, 32]
        self.frame_stack = 1
        self.data_mean = 0.5
        self.data_std = 0.5
        self.external_cond_dim = 1
        self.context_frames = 4
        self.weight_decay = 0.002
        self.warmup_steps = 5000
        self.gt_first_frame = 0.0
        self.gt_cond_prob = 0.0
        self.uncertainty_scale = 1
        self.chunk_size = 1
        self.calc_crps_sum = False
        self.learnable_init_z = True
        self.optimizer_beta = [0.9, 0.99]
        
        # 扩散模型参数
        self.network_size = 48
        self.beta_schedule = 'sigmoid'
        self.objective = 'pred_v'
        self.use_snr = True
        self.use_cum_snr = True
        self.snr_clip = 5.0
        self.cum_snr_decay = 0.4096000000000001
        self.timesteps = 1000
        self.self_condition = False
        self.ddim_sampling_eta = 0.0
        self.p2_loss_weight_gamma = 0
        self.p2_loss_weight_k = 1
        self.schedule_fn_kwargs = {}
        self.sampling_timesteps = 4
        self.mask_unet = False
        self.num_gru_layers = 0 #0
        self.num_mlp_layers = 0
        self.return_all_timesteps = False
        self.clip_noise = 6
        self.compute_fid_lpips = False
        self._name = 'df_video_dmlab'

    def _apply_model_overrides(self):
        """根据模型名称应用特定的参数覆盖"""
        # 保持原有的训练参数设置
        self.is_training = True
        self.wd = self.weight_decay  # 保持向后兼容
        self.sequence_len = 48
        
        # 固定参数
        self.world_model_action_num = 46
        self.player_num = 1
        
        # 根据模型标志应用特定覆盖
        if self.use_mc:
            # MC模型特定参数覆盖
            self.z_shape = [32, 64, 64]
            self.cum_snr_decay = 0.96
            self.network_size = 64
            self.frame_stack = 8
            self.context_frames = self.frame_stack
            self.external_cond_dim = self.frame_stack

    def __init__(self, model_name=None):
        # 1. 先设置所有YAML参数的默认值
        self._set_default_yaml_params()
        
        # 2. 设置DIT名称字典
        self.dit_name_dict = {
            0: 'I64_S_2', 1: 'I64_S_4', 2: 'I64_S_8',
            3: 'I64_B_2', 4: 'I64_B_4', 5: 'I64_B_8',
            6: 'I64_L_2', 7: 'I64_L_4', 8: 'I64_L_8',
            9: 'I16_S_1', 10: 'I16_S_2', 11: 'I32_S_2',
            12: 'I32_B_2',
        }

        # 3. 处理模型名称
        if model_name is None:
            model_name = 'df_z32_c1_dit_n11_mario_km_tanh_ldm'

        # print("model name:", model_name)
        self.model_name = model_name
        
        # 4. 解析模型名称中的标志
        self.use_dit = 'dit' in model_name
        self.use_mario = 'mario' in model_name
        self.use_km = 'km' in model_name
        self.use_tanh = 'tanh' in model_name
        self.use_lrelu = 'lrelu' in model_name
        self.use_mc = 'mc' in model_name

        # 5. 从模型名称解析参数
        params = self.get_params(model_name)
        self.zeta = params[0]
        self.context_len = params[1]
        if self.use_dit:
            self.dit_name = self.dit_name_dict[params[2]]

        # 6. 应用模型特定的参数覆盖
        self._apply_model_overrides()




from functools import partial
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from einops import rearrange, repeat
from .gru import Conv2dGRUCell
import math
from .utils import default, exists, cast_tuple, divisible_by
from .gru import Conv2dGRUCell

from .sin_emb import SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb

from .dit_models import DiT_models


class DiT(nn.Module):
    def __init__(
        self,
        dim=32,
        model_name=None, # Config_DF.py self.dit_name
        channels=3,  # 4
        out_dim=None, # 32
        z_cond_dim=None, # 32
        external_cond_dim=None,
        num_gru_layers=False,
        self_condition=False,
        config=None,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_cond=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
    ):
        super().__init__()

        # determine dimensions
        self.config = config
        self.channels = channels # 4
        self.self_condition = self_condition
        self.z_cond_dim = z_cond_dim # 32
        self.external_cond_dim = external_cond_dim # 1
        self.dit = DiT_models[model_name](in_channels=channels + z_cond_dim) # in_channels = 36
        self.hidden_dim =self.dit.hidden_size
        # original code time embeddings
        time_emb_dim = self.hidden_dim // 2
        external_cond_emb_dim = self.hidden_dim // 2
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_cond

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_cond)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, time_emb_dim), nn.GELU(), nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.external_cond_emb = nn.Sequential(
            nn.Embedding(self.config.world_model_action_num, external_cond_emb_dim//self.external_cond_dim),
            nn.Flatten()
        )

        self.out_dim = out_dim
        self.final_conv = nn.Conv2d(channels + z_cond_dim, self.out_dim, 1)

        self.tanh_layer = nn.Sequential(
            nn.Conv2d(channels + z_cond_dim,channels + z_cond_dim,1),
            nn.Tanh(),
        )
        if self.config.use_lrelu:
            self.lrelu_layer = nn.Sequential(
            nn.Conv2d(channels + z_cond_dim,channels + z_cond_dim,1),
            nn.LeakyReLU(),
        )

        # GRU layer for state transition
        self.num_gru_layers = num_gru_layers
        if num_gru_layers > 1:
            raise NotImplementedError("num_gru_layers > 1 is not implemented yet for TransitionUnet.")
        self.gru = Conv2dGRUCell(z_cond_dim, z_cond_dim) if num_gru_layers else None

    def forward(self, x, time, z_cond, external_cond=None, x_self_cond=None):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        if self.z_cond_dim:
            x = torch.cat((z_cond, x), dim=1)


        # original code concantenate emb, external_cond_emb
        emb = self.time_mlp(time) # (batch_size, hidden_size//2=192)
        if self.external_cond_dim:
            if external_cond is None:
                external_cond_emb = torch.zeros((emb.shape[0], self.external_cond_dim)).to(emb)
            else:
                external_cond_emb = self.external_cond_emb(external_cond.long()) # (batch_size, hidden_size//2=192)
            emb = torch.cat([emb, external_cond_emb], -1)# (batch_size, hidden_size=384)


        x = self.dit(x,emb)

        if self.config.use_tanh:
            x = self.tanh_layer(x)
        elif self.config.use_lrelu:
            x = self.lrelu_layer(x)

        z_next = self.final_conv(x)
        if self.num_gru_layers:
            z_next = self.gru(z_next, z_cond)
        return z_next

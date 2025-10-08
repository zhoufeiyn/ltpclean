from functools import partial
import torch
import torch.nn as nn

from .utils import default, exists, cast_tuple, divisible_by

from .sin_emb import SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb

from .dit_models import DiT_models


class DiT(nn.Module):
    def __init__(
        self,
        dim=32, #x_shape[-1]
        model_name=None, # Config_DF.py self.dit_name
        channels=3,  # 4
        out_dim=None, # 32
        z_cond_dim=None, # 32
        external_cond_dim=None,
        self_condition=False,
        config=None,

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
        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta) # t:(batch,)-> t:(batch,dim)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(dim, time_emb_dim), nn.GELU(), nn.Linear(time_emb_dim, time_emb_dim)
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



    def forward(self, x, time, z_cond, external_cond=None, x_self_cond=None):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        if self.z_cond_dim:
            x = torch.cat((z_cond, x), dim=1)  # (batch,36,32,32)


        # original code concantenate emb, external_cond_emb
        emb = self.time_mlp(time) # (batch_size, hidden_size//2=192)
        if self.external_cond_dim:
            if external_cond is None:
                external_cond_emb = torch.zeros((emb.shape[0], self.external_cond_dim)).to(emb)
            else:
                external_cond_emb = self.external_cond_emb(external_cond.long()) # (batch_size, hidden_size//2=192)
            emb = torch.cat([emb, external_cond_emb], -1)# (batch_size, hidden_size=384)


        x = self.dit(x,emb)

        if self.config.use_tanh: # True
            x = self.tanh_layer(x) # (batch,36,32,32)

        elif self.config.use_lrelu:
            x = self.lrelu_layer(x)

        z_next = self.final_conv(x) # (batch, *z.shape)
        return z_next

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

from .denoiser import Denoiser


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 20
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig):
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)

    def set_sigmas(self,denoiser: Denoiser,cfg: DiffusionSamplerConfig):
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)

    @torch.no_grad()
    def sample_next_obs(self, obs: Tensor, act: Tensor, zeta=None):
        device = obs.device
        b, t, c, h, w = obs.size()
        obs = obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        x = torch.randn(b, c, h, w, device=device)
        trajectory = [x]
        denoise_traj = []

        if self.denoiser.cfg.use_zeta:
            zeta = zeta.detach()

        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * self.cfg.s_noise
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
            if self.denoiser.cfg.use_zeta:
                next_zeta,denoised = self.denoiser.denoise_with_zeta(x,sigma,zeta,act)
            else:
                denoised = self.denoiser.denoise(x, sigma, obs, act)
            # denoised = self.denoiser.denoise(x, sigma, obs, act)
            denoise_traj.append(denoised)
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                if self.denoiser.cfg.use_zeta:
                    next_zeta, denoised_2 = self.denoiser.denoise_with_zeta(x_2, next_sigma * s_in, zeta, act)
                else:
                    denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
                # denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, obs, act)
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
            trajectory.append(x)
        if self.denoiser.cfg.use_zeta:
            return x, denoise_traj, next_zeta
        else:
            return x, denoise_traj


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))

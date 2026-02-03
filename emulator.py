import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@torch.jit.script
def lin_log_optimized(x: torch.Tensor, threshold: float = 20.0) -> torch.Tensor:
    """
    Converts linear brightness to logarithmic brightness with a linear segment
    near zero to avoid -inf.
    """
    f = (1.0 / threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x * f, torch.log(x))
    rounding = 1e8
    return torch.round(y * rounding) / rounding


@torch.jit.script
def compute_event_map_optimized(
    diff_frame: torch.Tensor, pos_thres: torch.Tensor, neg_thres: torch.Tensor
):
    """
    Computes the number of positive and negative events.
    """
    pos_frame = F.relu(diff_frame)
    neg_frame = F.relu(-diff_frame)

    pos_evts_frame = (pos_frame / pos_thres).floor()
    neg_evts_frame = (neg_frame / neg_thres).floor()

    return pos_evts_frame, neg_evts_frame


class EventEmulator:
    def __init__(
        self,
        num_envs: int,  # [新增] 环境数量
        img_shape: tuple[int, int],  # [新增] 图像尺寸 (H, W)
        pos_thres: float = 0.2,
        neg_thres: float = 0.2,
        sigma_thres: float = 0.03,
        cutoff_hz: float = 30.0,
        device: str = "cuda",
    ):
        """
        Args:
            num_envs: Number of environments.
            img_shape: Image shape (Height, Width).
            pos_thres: Nominal positive contrast threshold.
            neg_thres: Nominal negative contrast threshold.
            sigma_thres: Standard deviation for threshold noise.
            cutoff_hz: Cutoff frequency for the low-pass filter.
            device: Compute device.
        """
        self.device = device
        self.pos_thres_nominal = pos_thres
        self.neg_thres_nominal = neg_thres
        self.sigma_thres = sigma_thres
        self.cutoff_hz = cutoff_hz

        # 1. 组合出完整的 Shape: (N, H, W)
        full_shape = (num_envs, img_shape[0], img_shape[1])

        # 2. 直接初始化状态张量 (不再设为 None)
        self.base_log_frame = torch.zeros(full_shape, device=self.device, dtype=torch.float32)
        self.lp_log_frame = torch.zeros(full_shape, device=self.device, dtype=torch.float32)

        # 3. 初始化阈值图
        self.pos_thres_map: torch.Tensor | None = None
        self.neg_thres_map: torch.Tensor | None = None
        self._init_thresholds(full_shape)

    def reset_indices(
        self,
        init_frames: np.ndarray | torch.Tensor,
        env_ids: torch.Tensor | list[int],
        randomize_thresholds: bool = False,
    ):
        """
        Partial reset for vectorized environments.
        Only resets the state for the specified environment indices.

        Args:
            init_frames: The current observation frames for ALL environments (B, H, W),
                         or at least containing the data for env_ids.
            env_ids: Indices of the environments to reset.
            randomize_thresholds: If True, regenerates sensor noise for these envs
                                  (Domain Randomization). If False, keeps existing
                                  sensor noise (Simulation of same sensor).
        """
        if self.base_log_frame is None:
            raise RuntimeError("Emulator not initialized. Call reset() first.")

        # Ensure env_ids is a tensor
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Guard against empty resets
        if env_ids.numel() == 0:
            return

        # Handle input frames
        input_tensor = torch.as_tensor(init_frames, dtype=torch.float32, device=self.device)

        target_frames = input_tensor

        # Re-calculate initial log state for these environments
        log_frame = lin_log_optimized(target_frames)

        # Update states at specific indices
        self.lp_log_frame[env_ids] = log_frame
        self.base_log_frame[env_ids] = log_frame

        # Optional: Regenerate thresholds for domain randomization
        if randomize_thresholds and self.sigma_thres > 0:
            # We need to generate noise with shape (N_reset, H, W)
            shape_sub = (len(env_ids), *self.pos_thres_map.shape[1:])

            new_pos = torch.normal(
                self.pos_thres_nominal, self.sigma_thres, size=shape_sub, device=self.device
            ).clamp(min=0.01)

            new_neg = torch.normal(
                self.neg_thres_nominal, self.sigma_thres, size=shape_sub, device=self.device
            ).clamp(min=0.01)

            self.pos_thres_map[env_ids] = new_pos
            self.neg_thres_map[env_ids] = new_neg

    def _init_thresholds(self, shape):
        """Helper to initialize threshold maps."""
        if self.sigma_thres > 0:
            self.pos_thres_map = torch.normal(
                self.pos_thres_nominal, self.sigma_thres, size=shape, device=self.device
            ).clamp(min=0.01)
            self.neg_thres_map = torch.normal(
                self.neg_thres_nominal, self.sigma_thres, size=shape, device=self.device
            ).clamp(min=0.01)
        else:
            self.pos_thres_map = torch.full(shape, self.pos_thres_nominal, device=self.device)
            self.neg_thres_map = torch.full(shape, self.neg_thres_nominal, device=self.device)

    def generate_events(
        self,
        new_frame: np.ndarray | torch.Tensor,
        delta_time_s: float = 0.033,
        use_beta_noise: bool = True,
        beta_concentration: float = 10.0,
    ) -> torch.Tensor:
        """
        Generates event frame with temporal information.

        The temporal value is computed as n/(n+1) where n is the event count,
        which approximates the normalized timestamp of the last event.
        Beta distribution noise is added to prevent uniform temporal patterns.

        Args:
            new_frame: Input frame (batch, H, W).
            delta_time_s: Time delta since last frame.
            use_beta_noise: If True, adds beta distribution noise to temporal values.
            beta_concentration: Concentration parameter for beta distribution.
                Higher values mean less noise (tighter around mean).

        Returns:
            event_frame: Shape (batch, 2, H, W) where:
                - channel 0: positive polarity temporal values
                - channel 1: negative polarity temporal values
                Values are in [0, 1), where 0 means no event.
        """
        if self.base_log_frame is None:
            raise RuntimeError("Emulator not initialized. Call reset() first.")

        input_tensor = torch.as_tensor(new_frame, dtype=torch.float32, device=self.device)
        log_new_frame = lin_log_optimized(input_tensor)

        # LPF
        if self.cutoff_hz > 0:
            tau = 1.0 / (math.pi * 2.0 * self.cutoff_hz)
            eps = min(delta_time_s / tau, 1.0)
            self.lp_log_frame = (1 - eps) * self.lp_log_frame + eps * log_new_frame
        else:
            self.lp_log_frame = log_new_frame

        # Difference
        diff_frame = self.lp_log_frame - self.base_log_frame

        # Thresholding -> event count maps
        pos_evts_frame, neg_evts_frame = compute_event_map_optimized(
            diff_frame, self.pos_thres_map, self.neg_thres_map
        )

        # Update Base Frame
        if self.pos_thres_map is not None:
            self.base_log_frame += pos_evts_frame * self.pos_thres_map
            self.base_log_frame -= neg_evts_frame * self.neg_thres_map

        # Convert count to temporal value: n/(n+1)
        # This approximates normalized timestamp where higher count -> closer to 1
        pos_temporal = pos_evts_frame / (pos_evts_frame + 1.0)
        neg_temporal = neg_evts_frame / (neg_evts_frame + 1.0)

        if use_beta_noise:
            # Add beta distribution noise to avoid uniform temporal patterns
            # Beta(α, β) has mean = α/(α+β), so we set:
            #   α = base_value * concentration
            #   β = (1 - base_value) * concentration
            # This centers the noise around the original value

            pos_mask = pos_evts_frame > 0
            neg_mask = neg_evts_frame > 0

            # Positive polarity
            alpha_pos = (pos_temporal * beta_concentration).clamp(min=0.1)
            beta_pos = ((1.0 - pos_temporal) * beta_concentration).clamp(min=0.1)
            pos_noisy = torch.distributions.Beta(alpha_pos, beta_pos).sample()
            pos_temporal = torch.where(pos_mask, pos_noisy, pos_temporal)

            # Negative polarity
            alpha_neg = (neg_temporal * beta_concentration).clamp(min=0.1)
            beta_neg = ((1.0 - neg_temporal) * beta_concentration).clamp(min=0.1)
            neg_noisy = torch.distributions.Beta(alpha_neg, beta_neg).sample()
            neg_temporal = torch.where(neg_mask, neg_noisy, neg_temporal)

        # Stack to (batch, 2, H, W)
        event_frame = torch.stack(
            [pos_temporal.to(dtype=torch.float32), neg_temporal.to(dtype=torch.float32)],
            dim=1,
        )

        return event_frame

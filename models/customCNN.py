import time

import gym
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
        file originale messo nei modelli di Sistema unreal
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        # Load pretrained ResNet18
        resnet18 = models.resnet18(pretrained=True)

        # Modify the first conv layer: 3 channels -> 2 channels
        # Copy weights from R (index 0) and B (index 2) channels
        original_conv1 = resnet18.conv1
        new_conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )

        # Copy pretrained weights: R (0) -> channel 0, B (2) -> channel 1
        with th.no_grad():
            new_conv1.weight[:, 0, :, :] = original_conv1.weight[:, 0, :, :]  # R -> pos
            new_conv1.weight[:, 1, :, :] = original_conv1.weight[:, 2, :, :]  # B -> neg
            if original_conv1.bias is not None:
                new_conv1.bias.copy_(original_conv1.bias)

        resnet18.conv1 = new_conv1

        # Remove the final FC layer
        self.resnet = th.nn.Sequential(*(list(resnet18.children()))[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = True

        # Compute shape by doing one forward pass
        with th.no_grad():
            print(observation_space.shape)
            dims = self.resnet(
                th.as_tensor(np.zeros(observation_space.shape)).float()
            ).shape

            self.n_flatten = int(np.prod(dims))
            print(self.n_flatten)

        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(self.n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # No preprocessing needed - event_frame is already in [0, 1) range
        b, s, c, w, h = observations.shape
        x0 = observations.view(-1, c, w, h)

        x1 = self.resnet(x0)
        x1 = x1.view(b, s, -1)
        x2 = torch.flatten(x1, start_dim=1)
        x3 = self.linear(x2)
        return x3


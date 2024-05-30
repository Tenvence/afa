import json
import os.path
import pathlib

import torch
import torch.nn as nn
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.transformer_2d import Transformer2DModel
from safetensors.torch import save_file, load_file


class Aggregator(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_size: int,
            num_experts: int,
            num_layers: int,
            num_attn_heads: int,
            cross_attn_dim: int,
            temb_channels: int,
    ):
        super().__init__()

        self.config = {
            'in_channels': in_channels,
            'hidden_size': hidden_size,
            'num_experts': num_experts,
            'num_layers': num_layers,
            'num_attn_heads': num_attn_heads,
            'cross_attn_dim': cross_attn_dim,
            'temb_channels': temb_channels,
        }

        self.conv_in = nn.Conv2d(in_channels=in_channels * num_experts, out_channels=hidden_size, kernel_size=1)
        self.res_block = ResnetBlock2D(in_channels=hidden_size, temb_channels=temb_channels)
        self.transformer = Transformer2DModel(
            num_attention_heads=num_attn_heads,
            attention_head_dim=hidden_size // num_attn_heads,
            in_channels=hidden_size,
            out_channels=None,
            cross_attention_dim=cross_attn_dim,
            num_layers=num_layers,
        )
        self.conv_out = nn.Conv2d(hidden_size, num_experts, kernel_size=1, bias=False)

        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.weight)

    def forward(
            self,
            features: torch.FloatTensor,
            temb: torch.FloatTensor,
            encoder_hidden_states: torch.Tensor,
    ):
        batch_size, num_models, num_channels, h, w = features.shape
        hidden_states = features.reshape(batch_size, num_models * num_channels, h, w)

        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.res_block(hidden_states, temb)
        hidden_states = self.transformer(hidden_states, encoder_hidden_states).sample
        hidden_states = self.conv_out(hidden_states)

        attn_map = torch.softmax(hidden_states, dim=1)

        aggregated_features = features * attn_map[:, :, None, :, :]
        aggregated_features = torch.sum(aggregated_features, dim=1)

        return aggregated_features, attn_map

    def save_pretrained(self, saved_dictionary):
        pathlib.Path(saved_dictionary).mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), os.path.join(saved_dictionary, 'aggregator.safetensors'))
        json.dump(self.config, open(os.path.join(saved_dictionary, 'config.json'), 'w'))

    @classmethod
    def load_pretrained(cls, saved_dictionary):
        config = json.load(open(os.path.join(saved_dictionary, 'config.json'), 'r'))
        aggregator = cls(**config)
        state_dict = load_file(os.path.join(saved_dictionary, 'aggregator.safetensors'))
        aggregator.load_state_dict(state_dict)
        return aggregator

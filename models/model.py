import json
import os.path
from typing import List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as func
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
)
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import SchedulerMixin
from transformers import CLIPTextModel, CLIPTokenizer

from .modules import (
    UNet2DConditionAggregatorModel,
    StableDiffusionAggregatorPipeline,
    Aggregator
)


class Model(nn.Module):
    def __init__(
            self,
            aggregators: List[Aggregator],
            vae: AutoencoderKL,
            text_encoders: List[CLIPTextModel],
            tokenizer: CLIPTokenizer,
            unets: List[UNet2DConditionAggregatorModel],
            train_scheduler: SchedulerMixin,
            test_scheduler: SchedulerMixin,
    ):
        super().__init__()
        self.aggregators = nn.ModuleList(aggregators)
        self.vae = vae
        self.text_encoders = nn.ModuleList(text_encoders)
        self.tokenizer = tokenizer
        self.unets = nn.ModuleList(unets)
        self.train_scheduler = train_scheduler
        self.test_scheduler = test_scheduler

        self.config = {
            'num_models': len(unets),
            'num_aggregators': len(aggregators),
        }

    def train_forward(self, pixel_values: torch.FloatTensor, input_ids: torch.LongTensor):
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)

        num_train_timesteps = self.train_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, num_train_timesteps, (latents.shape[0],), device=latents.device).long()
        noisy_latents = self.train_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = []
        for text_encoder in self.text_encoders:
            encoder_hidden_states.append(text_encoder(input_ids)[0])
        encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)

        noise_pred = self.unets[0](
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            additional_unets=self.unets[1:],
            aggregators=self.aggregators,
            aggregate_from=self.aggregate_from,
        ).sample

        loss = func.mse_loss(noise_pred.float(), noise.float(), reduction='mean')

        return loss

    def test_forward(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> StableDiffusionPipelineOutput:
        pipeline = StableDiffusionAggregatorPipeline(
            vae=self.vae,
            text_encoder=self.text_encoders[0],
            tokenizer=self.tokenizer,
            unet=self.unets[0],
            scheduler=self.test_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        return pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            additional_unets=self.unets[1:],
            additional_text_encoders=self.text_encoders[1:],
            aggregators=self.aggregators,
        )

    @staticmethod
    def load_sd_model(model_path: str):
        if not os.path.exists(model_path):
            return None

        if os.path.isdir(model_path):
            model = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None)
        else:
            model = StableDiffusionPipeline.from_single_file(model_path, load_safety_checker=False)

        return model

    @classmethod
    def load_init(
            cls,
            model_paths: List[str],
            st_model_path: str,
            hidden_size: int,
            num_layers: int,
            num_attn_heads: int,
    ):
        st_model = cls.load_sd_model(st_model_path)

        vae = st_model.vae
        tokenizer = st_model.tokenizer

        unets, text_encoders = [], []
        for mp in model_paths:
            base_model = cls.load_sd_model(mp)

            unet = base_model.unet
            unet = UNet2DConditionAggregatorModel.from_unet_2d_condition_model(unet)
            unet.enable_xformers_memory_efficient_attention()
            unets.append(unet)

            text_encoders.append(base_model.text_encoder)

        train_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        test_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        agg_in_channels = [unets[0].conv_in.out_channels]  # conv_in
        for block_idx, block in enumerate(unets[0].down_blocks):
            num_layers = len(block.resnets) + (0 if block.downsamplers is None else 1)
            for _ in range(num_layers):
                agg_in_channels.append(unets[0].config.block_out_channels[block_idx])  # down_blocks
        agg_in_channels.append(unets[0].config.block_out_channels[-1])  # mid_block
        for block_idx, block in enumerate(unets[0].up_blocks):
            for _ in range(len(block.resnets)):
                agg_in_channels.append(unets[0].config.block_out_channels[-(block_idx + 1)])  # up_blocks
        agg_in_channels[-1] = 4  # conv_out

        aggregators = [Aggregator(
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_experts=len(unets),
            num_layers=num_layers,
            num_attn_heads=num_attn_heads,
            cross_attn_dim=text_encoders[0].config.hidden_size,
            temb_channels=unets[0].config.block_out_channels[0] * 4,
        ) for in_channels in agg_in_channels]

        for aggregator in aggregators:
            aggregator.transformer.enable_xformers_memory_efficient_attention()

        return cls(
            aggregators,
            vae,
            text_encoders,
            tokenizer,
            unets,
            train_scheduler,
            test_scheduler,
        )

    def save_pretrained(self, directory):
        StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoders[0],
            tokenizer=self.tokenizer,
            unet=self.unets[0],
            scheduler=self.test_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).save_pretrained(directory, safe_serialization=True)

        for i, unet in enumerate(self.unets[1:]):
            unet.save_pretrained(os.path.join(directory, f'additional_unet_{i}'), safe_serialization=True)
        for i, text_encoder in enumerate(self.text_encoders[1:]):
            text_encoder.save_pretrained(
                os.path.join(directory, f'additional_text_encoder_{i}'),
                safe_serialization=True,
            )

        for i, aggregator in enumerate(self.aggregators):
            aggregator.save_pretrained(os.path.join(directory, f'aggregator_{i}'))

        json.dump(self.config, open(os.path.join(directory, 'model_config.json'), 'w'))

    @classmethod
    def load_pretrained(cls, directory, device, dtype):
        config = json.load(open(os.path.join(directory, 'model_config.json'), 'r'))

        sd = StableDiffusionPipeline.from_pretrained(
            directory,
            safety_checker=None,
            feature_extractor=None,
        ).to(torch_device=device, torch_dtype=dtype)

        additional_unets = [
            UNet2DConditionModel.from_pretrained(directory, subfolder=f'additional_unet_{i}')
            for i in range(config['num_models'] - 1)
        ]
        unets = [sd.unet] + additional_unets

        additional_text_encoders = [
            CLIPTextModel.from_pretrained(directory, subfolder=f'additional_text_encoder_{i}').to(device, dtype)
            for i in range(config['num_models'] - 1)
        ]
        text_encoders = [sd.text_encoder] + additional_text_encoders

        for idx in range(len(unets)):
            unets[idx] = UNet2DConditionAggregatorModel.from_unet_2d_condition_model(unets[idx]).to(device, dtype)
            unets[idx].enable_xformers_memory_efficient_attention()

        aggregators = [
            Aggregator.load_pretrained(os.path.join(directory, f'aggregator_{i}')).to(device, dtype)
            for i in range(config['num_aggregators'])
        ]

        for aggregator in aggregators:
            aggregator.transformer.enable_xformers_memory_efficient_attention()

        train_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        test_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        return Model(
            aggregators=aggregators,
            vae=sd.vae,
            text_encoders=text_encoders,
            tokenizer=sd.tokenizer,
            unets=unets,
            train_scheduler=train_scheduler,
            test_scheduler=test_scheduler,
        )

import argparse
import math
import time

import accelerate
import torch

from models import Model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--images_saved_path', type=str)
    parser.add_argument('--attn_maps_saved_path', type=str)

    parser.add_argument('--negative_prompt', type=str, default='\'\'')
    parser.add_argument('--num_images_per_prompt', type=int, default=4)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def pad_prompts(prompts):  # pad to average the allocated prompts for each process
    accelerator = accelerate.Accelerator()
    num_processes = accelerator.num_processes

    num_prompts_per_process = math.ceil(len(prompts) / num_processes)
    num_padding = num_prompts_per_process * num_processes - len(prompts)

    padded_prompts = [{'global_idx': idx, 'prompt': p, 'is_padded': False} for idx, p in enumerate(prompts)]
    padded_prompts += [{'global_idx': 0, 'prompt': prompts[0], 'is_padded': True} for _ in range(num_padding)]

    return padded_prompts


@torch.no_grad()
def main():
    args = parse_args()

    model_path = args.model_path
    prompt = args.prompt
    images_saved_path = args.images_saved_path
    attn_maps_saved_path = args.attn_maps_saved_path

    negative_prompt = args.negative_prompt
    num_images_per_prompt = args.num_images_per_prompt
    height = args.height
    width = args.width
    guidance_scale = args.guidance_scale
    num_inference_steps = args.num_inference_steps
    seed = args.seed

    accelerator = accelerate.Accelerator()
    device = accelerator.device

    model = Model.load_pretrained(model_path, device, torch.float16)

    accelerator = accelerate.Accelerator()
    generator = torch.Generator('cuda').manual_seed(seed)

    st = time.time()
    output = model.test_forward(
        prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    gen_images = output.images
    attn_maps = output.inference_attn_maps
    process_idx = accelerator.process_index
    t = time.time() - st
    print(f'inference, process idx: {process_idx}, {t:.1f}s')

    for image_idx, image in enumerate(gen_images):
        image.save(images_saved_path)
    torch.save(attn_maps, attn_maps_saved_path)


if __name__ == '__main__':
    main()

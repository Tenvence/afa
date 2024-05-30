import argparse
import os
import random
import time

import PIL.Image as Image
import accelerate
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from datasets import load_dataset
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from models import Model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_base_path', type=str)
    parser.add_argument('--model_files', type=str, help='file names of models, split by dot')
    parser.add_argument('--st_model_file', type=str, default='stable-diffusion-v1-5')
    parser.add_argument('--dataset_json_file', type=str, default='/JourneyDB/journeydb_valid_data.json')
    parser.add_argument('--num_data', type=int, default=10000)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--drop_text_rate', type=float, default=0.1)

    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--aggregator_hidden_size', type=int, default=128)
    parser.add_argument('--aggregator_num_layers', type=int, default=1)
    parser.add_argument('--aggregator_num_attn_heads', type=int, default=8)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_scheduler', type=str, default='constant')
    parser.add_argument('--lr_warmup_steps', type=int, default=100)
    parser.add_argument('--max_train_epochs', type=int, default=10)

    args = parser.parse_args()
    return args


def get_dataset(dataset_json_file, tokenizer, resolution, num_data, drop_text_rate=0.1):
    img_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = load_dataset('json', data_files=dataset_json_file, split="train").select(range(0, num_data))
    dataset = dataset.select_columns(['image_file', 'text'])

    def batch_transform(batch):
        images = [Image.open(f).convert('RGB') for f in batch['image_file']]
        pixel_values = torch.stack([img_transforms(img) for img in images], dim=0)
        text = [''] * len(batch['text']) if random.random() < drop_text_rate else batch['text']
        input_ids = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")['input_ids']
        return {'pixel_values': pixel_values, 'input_ids': input_ids}

    dataset.set_transform(batch_transform)

    return dataset


def main():
    args = parse_args()

    model_base_path = args.model_base_path
    model_files = args.model_files
    st_model_file = args.st_model_file
    dataset_json_file = args.dataset_json_file
    num_data = args.num_data
    model_save_path = args.model_saved_path
    drop_text_rate = args.drop_text_rate

    resolution = args.resolution
    batch_size = args.batch_size
    num_workers = args.num_workers

    aggregator_hidden_size = args.aggregator_hidden_size
    aggregator_num_layers = args.aggregator_num_layers
    aggregator_num_attn_heads = args.aggregator_num_attn_heads

    lr = args.lr
    weight_decay = args.weight_decay
    lr_scheduler = args.lr_scheduler
    lr_warmup_steps = args.lr_warmup_steps
    max_train_epochs = args.max_train_epochs

    model_paths = [os.path.join(model_base_path, f) for f in model_files.split(',')]
    st_model_path = os.path.join(model_base_path, st_model_file)

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    weight_dtype = torch.float16

    accelerator.print(args)

    st = time.time()
    model = Model.load_init(
        model_paths=model_paths,
        st_model_path=st_model_path,
        hidden_size=aggregator_hidden_size,
        num_layers=aggregator_num_layers,
        num_attn_heads=aggregator_num_attn_heads,
    )
    num_params = sum(param.numel() for param in model.aggregators.parameters())
    accelerator.print(
        f'Loaded all components using {time.time() - st:.1f}s. '
        f'The number of trainable parameters (only aggregators) is {num_params / 1e6:.3f}M.'
    )

    model.vae.requires_grad_(False)
    model.text_encoders.requires_grad_(False)
    model.unets.requires_grad_(False)
    model.aggregators.requires_grad_(True)

    dataset = get_dataset(dataset_json_file, model.tokenizer, resolution, num_data, drop_text_rate)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    optimizer = optim.AdamW(model.aggregators.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)

    for epoch_idx in range(max_train_epochs):
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)

        for batch in dataloader:
            model.train()

            pixel_values = batch['pixel_values'].to(device, weight_dtype)
            input_ids = batch['input_ids'].to(device)

            loss = model.train_forward(pixel_values, input_ids)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)

            progress_bar.set_description(f'TRAIN, EP: {epoch_idx + 1}/{max_train_epochs}, L: {loss.item():.5f}')

    if accelerator.is_local_main_process:
        model.save_pretrained(model_save_path)
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()

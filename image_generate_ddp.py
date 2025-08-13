# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained Diffusion model using DDP.
"""
import math
import torch
import torch.distributed as dist
from torchvision.utils import save_image
import argparse

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

from arc2face import CLIPTextModelWrapper, project_face_embs

import os
import cv2
import torch
import time
from diffusers import LCMScheduler
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np

import PIL
import requests
from typing import Union

from concurrent.futures import ThreadPoolExecutor

def save_image_thread(image, path):
    image.save(path)

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    encoder = CLIPTextModelWrapper.from_pretrained(
        args.model_path, subfolder="encoder", torch_dtype=torch.float16
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.model_path, subfolder="arc2face", torch_dtype=torch.float16
    )

    pipeline = StableDiffusionPipeline.from_pretrained(
            args.base_model_path,
            text_encoder=encoder,
            unet=unet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)

    dist.barrier()
    
    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // n)

    time_start = time.time()

    total = 0
    for i in range(iterations):
        
        print(iterations, i, total+rank)

        id_emb = torch.load(f'{args.id_pt_path}/{total+rank:06d}.pt').to(torch.float16).to(device)
        id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
        id_emb = project_face_embs(pipeline, id_emb)    # pass through the encoder
        images_1 = pipeline(prompt_embeds=id_emb[:25, :], num_inference_steps=args.num_sampling_steps, guidance_scale=3.0, num_images_per_prompt=1).images
        images_2 = pipeline(prompt_embeds=id_emb[25:, :], num_inference_steps=args.num_sampling_steps, guidance_scale=3.0, num_images_per_prompt=1).images
        images = images_1 + images_2

        # 创建输出目录
        output_dir = os.path.join(args.out_dir, f"{total + rank:06d}")
        os.makedirs(output_dir, exist_ok=True)

        # 使用 ThreadPoolExecutor 保存图片
        with ThreadPoolExecutor(max_workers=10) as executor:
            paths = [os.path.join(output_dir, f"{k}.jpg") for k in range(len(images))]
            executor.map(save_image_thread, images, paths)
        
        total += global_batch_size
        time_end = time.time()
        time_sum = time_end - time_start
        print(rank, time_sum)

    time_end = time.time()
    time_sum = time_end - time_start
    print(rank, time_sum)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-fid-samples", type=int, default=10000)
    parser.add_argument("--num-sampling-steps", type=int, default=25)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--base_model_path", type=str, default="ckpt/stable-diffusion-v1-5")
    parser.add_argument("--model_path", type=str, default="ckpt/arc2face")
    parser.add_argument("--id_pt_path", type=str, default="data/id")
    parser.add_argument("--out_dir", type=str, default="output/image")
    args = parser.parse_args()
    main(args)

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 image_generate_ddp.py

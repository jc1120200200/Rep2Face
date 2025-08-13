# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained IRDM using DDP.
"""
import math
import time
import torch
import torch
import torch.distributed as dist
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils.download import find_model
from model.model_mlp import get_model_base, get_model_large
from model.model_frc import FaceRaceClassifier
import argparse
from datetime import datetime

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
    model = get_model_large().to(device)
    # Load a custom IRDM checkpoint from train.py:
    ckpt_path = args.diffusion_ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=False)

    # Load Face-race model
    model_frc = FaceRaceClassifier().to(device)
    frc_ckpt_path = args.frc_ckpt
    frc_state_dict = find_model(frc_ckpt_path)
    model_frc.load_state_dict(frc_state_dict["model"])
    model_frc.eval()
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // n)

    total = 0
    id_common = None
    id_common_gpu = None
    for i in range(iterations):

        print(iterations, i)
        start_time = time.time()

        for j in range(args.num_one_id_samples):
            # Create sampling noise:
            noise_id = torch.randn(args.batch_size, 512, device=device)
            x = noise_id

            if j == 0:
                # Sample ids:
                samples = diffusion.ddim_sample_loop(
                    model.forward, x.shape, x, None, id_common_gpu, model_frc, clip_denoised=False, progress=True, device=device
                ).unsqueeze(dim=1)
                ids = samples

            else:
                # Sample ids:
                samples = diffusion.ddim_sample_loop(
                    model.forward, x.shape, x, ids, id_common_gpu, model_frc, clip_denoised=False, progress=True, device=device
                ).unsqueeze(dim=1)
                ids = torch.cat((ids, samples), dim=1)

        if id_common == None:
            id_common = torch.mean(ids.cpu().detach(), dim=1, keepdim=True)
            id_common_gpu = id_common.clone().requires_grad_(True).to(device)
        else:
            id_common = torch.cat((id_common, torch.mean(ids.cpu().detach(), dim=1, keepdim=True)), dim=0)
            del id_common_gpu
            torch.cuda.empty_cache()
            id_common_gpu = id_common.clone().requires_grad_(True).to(device)

        # # Save ids:
        for k, id in enumerate(ids):
            index = k * dist.get_world_size() + rank + total
            torch.save(id.cpu().detach(), f"output/id/{index:06d}.pt")
        total += global_batch_size

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"程序执行时间: {execution_time:.2f} 秒")

        with open("output/time.txt","a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {rank} {i} {execution_time:.2f} 秒 \n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-one-id-samples", type=int, default=50)
    parser.add_argument("--num-fid-samples", type=int, default=10000)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--diffusion-ckpt", type=str, default="ckpt/irdm/face_id.pt")
    parser.add_argument("--frc-ckpt", type=str, default="ckpt/irdm/face_race.pt")
    args = parser.parse_args()
    main(args)

# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 image_generate_ddp.py

import os
import time

# Stagger process startup to avoid filesystem contention
if "LOCAL_RANK" in os.environ:
    time.sleep(int(os.environ["LOCAL_RANK"]) * 3)


# Command to actually run this (L40)
# srun --pty --partition=gpu-l40 bash
# CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 train_distributed.py
# batch size 7, accum 3 (63 effective size)

# For 3090s
# srun --pty --partition=gpu-normal bash
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_distributed.py
# batchsize 4, accum 4 (64)

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.optim import Adam
from diffusion import Diffusion
from dataloader import load_dataloaders
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

ACCUMULATION_STEPS = 3
RESUME_CHECKPOINT = None # Set to None to train from scratch


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    local_rank = setup()
    is_main = local_rank == 0  # only rank 0 logs/saves

    scaler = GradScaler()
    model = Diffusion(timesteps=100, batch_size=7, device=local_rank)
    optimizer = Adam(model.model.parameters(), lr=1e-4)

    # Pass local_rank so dataloader can use DistributedSampler
    dataloader, sampler = load_dataloaders(distributed=True)
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        if is_main:
            print(f"Loading checkpoint from {RESUME_CHECKPOINT}")
        
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=f"cuda:{local_rank}", weights_only=False)
        
        # Single-GPU checkpoint has no 'module.' prefix, loads directly
        model.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(local_rank)
        
        start_epoch = checkpoint["epoch"] + 1
        if is_main:
            print(f"Resuming from epoch {start_epoch}")
    
    # Wrap with DDP AFTER loading checkpoint
    model.model = DDP(model.model, device_ids=[local_rank])    
    
    epochs = int(1e10)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # important for proper shuffling
        
        # Only show progress bar on main process
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True, disable=not is_main)
        accumulated_loss = 0.0
        
        for step, batch in enumerate(pbar):
            batch = batch.to(local_rank)
            t = torch.randint(0, model.timesteps, (batch.shape[0],), device=local_rank).long()

            with autocast(dtype=torch.float16):
                loss = model.p_losses(batch, t)

            loss /= ACCUMULATION_STEPS
            accumulated_loss += loss.item()
            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if is_main:
                    pbar.set_postfix({"loss": accumulated_loss})
                accumulated_loss = 0.0
            elif is_main:
                pbar.set_postfix({"loss": accumulated_loss})

        # Handle leftover gradients
        if (step + 1) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Save only on main process
        if is_main:
            checkpoint = {
                "model_state_dict": model.model.module.state_dict(),  # .module unwraps DDP
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, f"checkpoint_epoch{epoch}_1k_l40_dist.pt")
            print(f"Saved checkpoint at epoch {epoch}")

    cleanup()

if __name__ == "__main__":
    main()
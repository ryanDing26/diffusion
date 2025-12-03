import torch
from tqdm import tqdm
from torch.optim import Adam
from diffusion import Diffusion
from dataloader import load_dataloaders
from torch.cuda.amp import GradScaler, autocast

ACCUMULATION_STEPS = 8

scaler = GradScaler()
dataloader = load_dataloaders()
model = Diffusion(timesteps=100, batch_size=8) # batch size 16, channels 3, image size 256
optimizer = Adam(model.model.parameters(), lr=1e-4)
epochs = 1e10 # this will essentially keep running forever

# steps_per_epoch = len(dataloader) // model.batch_size

for epoch in range(int(epochs)):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    accumulated_loss = 0.0
    for step, batch in enumerate(pbar):
        batch = batch.to(model.device) # [16, 3, 256, 256] batch
        # sample t ~ U(0, T) for each sample
        t = torch.randint(0, model.timesteps, (batch.shape[0],), device=model.device).long()

        # calculates predicted vs actual noise and scale to accumulation
        with autocast(dtype=torch.float16):
            loss = model.p_losses(batch, t)

        loss /= ACCUMULATION_STEPS
        accumulated_loss += loss.item()
        scaler.scale(loss).backward()

        if (step + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # update tqdm bar text
            pbar.set_postfix({"loss": accumulated_loss})
            accumulated_loss = 0.0
        else:
            pbar.set_postfix({"loss": accumulated_loss})

        # save checkpoint every 20k effective steps (in our case: 1.28 million data points)
        # Note: I commented this out as 20k steps is actually more than an epoch - Ryan
        # if step % (20000 * ACCUMULATION_STEPS) == 0:
        #     checkpoint = {
        #         "model_state_dict": model.model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "epoch": epoch,
        #         "step": step,
        #     }
        #     torch.save(checkpoint, f"checkpoint_epoch{epoch}_step{step}.pt")
        #     print(f"Saved checkpoint at epoch {epoch}, step {step}")
    
    if (step + 1) % ACCUMULATION_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    checkpoint = {
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, f"checkpoint_epoch{epoch}_1k_l40.pt")
    print(f"Saved checkpoint at epoch {epoch}")
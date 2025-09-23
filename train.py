# sae_deterministic_lottery_monitor.py
import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------
# Model (W: [d_in, k], atoms are columns)
# ---------------------------

class SoftThreshold(nn.Module):
    def __init__(self, lam: float): super().__init__(); self.lam = float(lam)
    def forward(self, x): return torch.sign(x) * F.relu(x.abs() - self.lam)

class SAE(nn.Module):
    """
    Sparse AE with tied weights and tied bias:
      - Encode:  z = S_lambda(x @ W + b_e)      (W: [d_in, k], b_e: [k])
      - Decode:  x̂ = z @ W^T + b_d,  b_d = - W @ b_e   (derived each forward)
    Atoms are the columns of W; we keep columns unit-norm.
    """
    def __init__(self, d_in: int, k: int, lam: float = 0.1):
        super().__init__()
        self.d_in, self.k = d_in, k
        self.W   = nn.Parameter(torch.randn(d_in, k) / math.sqrt(d_in))
        self.b_e = nn.Parameter(torch.zeros(k))
        self.act = SoftThreshold(lam)
        with torch.no_grad(): self.renorm_atoms_columns()

    def encode(self, x):  # x: [B, d_in] -> z: [B, k]
        pre = x @ self.W + self.b_e
        return self.act(pre)

    def decode(self, z):  # z: [B, k] -> x̂: [B, d_in]
        b_d = -(self.W @ self.b_e)  # tied/derived
        return z @ self.W.t() + b_d

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z

    @torch.no_grad()
    def renorm_atoms_columns(self, eps: float = 1e-8):
        # Ensure each atom (column) has unit norm
        coln = self.W.norm(dim=0, keepdim=True).clamp_min(eps)  # [1, k]
        self.W.div_(coln)

    @torch.no_grad()
    def lottery_reinit_deterministic_cols(self, idx: torch.LongTensor, optimizer=None, seed: int | None = None):
        """
        Deterministic lottery reinit on ALL ranks:
          1) rank0 broadcasts (idx, seed)
          2) every rank generates identical random unit columns for W[:, idx]
          3) b_e[idx] gets small random noise
          4) zero optimizer state slices for those columns/elements
        """
        if idx is None or idx.numel() == 0:
            if dist.is_initialized(): dist.barrier()
            return

        device = self.W.device
        # ---- sync idx + seed (tiny metadata) ----
        if dist.is_initialized():
            if dist.get_rank() == 0:
                seed = int(torch.empty((), dtype=torch.int64).random_().item() if seed is None else seed)
                payload = [idx.detach().cpu(), seed]
            else:
                payload = [torch.empty_like(idx.detach().cpu()), 0]
            dist.broadcast_object_list(payload, src=0)
            idx_cpu, seed = payload
            idx = idx_cpu.to(device)
        else:
            seed = int(torch.empty((), dtype=torch.int64).random_().item() if seed is None else seed)

        # ---- generate identical atoms on every rank ----
        d_in, K = self.W.size(0), idx.numel()
        g = torch.Generator(device=device).manual_seed(seed)
        atoms = torch.randn(d_in, K, generator=g, device=device)
        atoms = atoms / atoms.norm(dim=0, keepdim=True).clamp_min(1e-8)

        # ---- write columns + small lottery bias; enforce unit norms ----
        self.W[:, idx] = atoms
        self.b_e[idx].normal_(mean=0.0, std=1e-3)
        # (columns already unit, but keep strict)
        self.renorm_atoms_columns()

        # ---- zero optimizer state ONLY for touched slices ----
        if optimizer is not None:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    st = optimizer.state.get(p, None)
                    if not st: continue
                    if p is self.W:
                        for v in list(st.values()):
                            if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape == self.W.shape:
                                v[:, idx].zero_()
                    elif p is self.b_e:
                        for v in list(st.values()):
                            if isinstance(v, torch.Tensor) and v.ndim == 1 and v.shape == self.b_e.shape:
                                v[idx].zero_()

        if dist.is_initialized(): dist.barrier()

# ---------------------------
# Firing-rate monitor + dead selection (patience/cap/cooldown)
# ---------------------------

@torch.no_grad()
def epoch_firing_rate(model: SAE, loader: DataLoader, device, tau: float = 1e-6):
    model.eval()
    k = model.k
    count = torch.zeros(k, device=device)
    total = 0
    for x in loader:
        x = x.to(device)
        _, z = model(x)
        count += (z.abs() > tau).float().sum(dim=0)
        total += z.size(0)
    return (count / max(total, 1)).clamp_(0, 1)

def pick_dead(
    firing_rate: torch.Tensor,
    thr: float = 0.01,
    patience: int = 2,
    max_frac: float = 0.02,
    dead_epochs: torch.Tensor | None = None,
    cooldown: torch.Tensor | None = None,
):
    """
    Hysteresis + cap + cooldown.
    Returns (chosen_idx, dead_epochs, cooldown).
    """
    device = firing_rate.device
    if dead_epochs is None or cooldown is None:
        dead_epochs = torch.zeros_like(firing_rate, dtype=torch.int64, device=device)
        cooldown    = torch.zeros_like(dead_epochs)

    active = firing_rate >= thr
    newly_dead = ~active & (cooldown == 0)
    dead_epochs[newly_dead] += 1
    dead_epochs[active] = 0

    candidates = (dead_epochs >= patience).nonzero(as_tuple=False).flatten()
    if candidates.numel() == 0:
        cooldown[cooldown > 0] -= 1
        return candidates, dead_epochs, cooldown

    kmax = max(1, int(max_frac * firing_rate.numel()))
    chosen = candidates[torch.randperm(candidates.numel(), device=device)[:kmax]]
    cooldown[chosen] = 2
    cooldown[cooldown > 0] -= 1
    return chosen, dead_epochs, cooldown

# ---------------------------
# Toy data (replace with your dataset)
# ---------------------------

class ToyGaussian(Dataset):
    def __init__(self, n=50000, d=128, centers=5, seed=0):
        g = torch.Generator().manual_seed(seed)
        means = torch.randn(centers, d, generator=g)
        xs = []
        for _ in range(n):
            c = int(torch.randint(0, centers, (1,), generator=g))
            xs.append(means[c] + 0.3 * torch.randn(d, generator=g))
        self.data = torch.stack(xs)
    def __len__(self): return self.data.size(0)
    def __getitem__(self, i): return self.data[i]

# ---------------------------
# DDP helpers
# ---------------------------

def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank if torch.cuda.is_available() else 0)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return rank, world, local_rank
    return 0, 1, 0

def ddp_cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ---------------------------
# Training
# ---------------------------

def main():
    rank, world, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_master = (rank == 0)

    d_in, k = 128, 1024
    lam = 0.1

    train_set = ToyGaussian(n=20000, d=d_in, centers=6, seed=123)
    val_set   = ToyGaussian(n=4000,  d=d_in, centers=6, seed=456)

    train_sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    val_sampler   = DistributedSampler(val_set,   num_replicas=world, rank=rank, shuffle=False) if world > 1 else None

    train_loader = DataLoader(train_set, batch_size=256, sampler=train_sampler, shuffle=(train_sampler is None), drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=512, sampler=val_sampler,   shuffle=False)

    model = SAE(d_in, k, lam=lam).to(device)
    wrapped = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False) if world > 1 else model
    opt = torch.optim.Adam(wrapped.parameters(), lr=2e-3)

    # Track deadness across epochs (lives on device)
    dead_epochs = torch.zeros(k, dtype=torch.int64, device=device)
    cooldown    = torch.zeros_like(dead_epochs)

    epochs = 10
    warmup_epochs = 1
    for ep in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        wrapped.train()
        for x in train_loader:
            x = x.to(device)
            xhat, z = wrapped(x)
            recon = F.mse_loss(xhat, x, reduction='mean')
            l1 = z.abs().mean()
            loss = recon + 1e-3 * l1

            opt.zero_grad()
            loss.backward()
            opt.step()

            # keep atoms unit-norm
            (wrapped.module if isinstance(wrapped, DDP) else wrapped).renorm_atoms_columns()

        # ---- End of epoch: monitor + deterministic lottery reinit ----
        torch.cuda.synchronize(device) if torch.cuda.is_available() else None

        core = wrapped.module if isinstance(wrapped, DDP) else wrapped
        fr = epoch_firing_rate(core, val_loader, device, tau=1e-6)
        to_reset, dead_epochs, cooldown = pick_dead(
            fr, thr=0.01, patience=2, max_frac=0.02, dead_epochs=dead_epochs, cooldown=cooldown
        )

        if ep > warmup_epochs and to_reset.numel() > 0:
            core.lottery_reinit_deterministic_cols(idx=to_reset, optimizer=opt)
            # Optional: reset counters for those units
            dead_epochs[to_reset] = 0

        if is_master:
            print(f"[Epoch {ep}] loss≈{loss.item():.4f} | FR<1%: {(fr<0.01).sum().item()} | reinit {to_reset.numel()} units")

        if dist.is_initialized():
            dist.barrier()

    ddp_cleanup()

if __name__ == "__main__":
    main()



"""Train hallucination detection model."""

import logging
import math
import os
import random
from contextlib import nullcontext
from logging.handlers import RotatingFileHandler

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from datasets import HallucinationDataset
from models import ConformerClassifier


config = {
    "seed": 0,
    "output_dir": "results/gemma-3n-E2B-it",
    "save_dir": "results/gemma-3n-E2B-it/save",
    "train_data": "None",
    "valid_data": "None",
    "test_data": "None",
    "num_epochs": 200,
    "batch_size": 12,
    "grad_accumulation_factor": 1,
    "num_workers": 4,
    "lr": 0.005,
    "weight_decay": 0.01,
    "max_grad_norm": 5.0,
    "use_amp": False,
    "improvement_threshold": 0.0025,
    "annealing_factor": 0.9,
    "patience": 0,
    "test_only": False,
    "input_dim": 8192,
    "num_classes": 3,
    "model_kwargs": {
        "hidden_size": 8192,
        "num_layers": 2,
        "num_heads": 4,
        "ffn_expansion": 4.0,
        "conv_kernel": 31,
        "dropout_p": 0.1,
        "max_len": 2048,
        "num_channels": 31,
    },
}
config["save_dir"] = os.path.join(config["output_dir"], "save")


def setup_logger(log_dir, log_filename="train.log", max_bytes=1 * 1024 * 1024):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if already configured
    if not logger.hasHandlers():
        # Console handler (logs to terminal)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)

        # Rotating file handler (logs to file)
        file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=0)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def main(argv):
    del argv

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    is_distributed = "LOCAL_RANK" in os.environ

    # Handle DDP
    if is_distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Set seed
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # Set device
    if is_distributed and dist.get_backend() != "nccl":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(rank)

    # Load data
    train_dataset = HallucinationDataset(config["train_data"])
    valid_dataset = HallucinationDataset(config["valid_data"])
    test_dataset = HallucinationDataset(config["test_data"])

    if is_distributed:
        dist.barrier()

    # Define model
    model = ConformerClassifier(
        config["input_dim"],
        config["num_classes"],
        **config["model_kwargs"],
    ).to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["annealing_factor"],
        patience=config["patience"],
        threshold=config["improvement_threshold"],
        min_lr=1e-6,
    )

    # Define scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])

    # Define loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize counters
    epoch = None
    start_epoch = 0
    optimizer_steps = 0

    # Define logger
    logger = setup_logger(config["output_dir"])

    # Load checkpoint
    checkpoint_paths = []
    if os.path.exists(config["save_dir"]):
        checkpoint_paths = sorted(
            [x for x in os.listdir(config["save_dir"]) if x.startswith("epoch")]
        )
    if checkpoint_paths:
        checkpoint_path = os.path.join(config["save_dir"], checkpoint_paths[-1])
        checkpoint = torch.load(checkpoint_path)
        torch.random.set_rng_state(checkpoint["rng_state"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer_steps = checkpoint["optimizer_steps"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if rank == 0:
            logger.info(f"Loaded checkpoint {checkpoint_path}")

    # Handle DDP
    if is_distributed:
        wrapped_model = DDP(model)
    else:
        wrapped_model = model

    if rank == 0:
        num_parameters = round(sum([x.numel() for x in model.parameters()]) / 1e6)
        logger.info(f"Total parameters: {num_parameters}M")
        num_trainable_parameters = round(
            sum([x.numel() for x in model.parameters() if x.requires_grad]) / 1e6
        )
        logger.info(f"Trainable parameters: {num_trainable_parameters}M")

    # Training loop
    if not config["test_only"]:
        for epoch in range(start_epoch, config["num_epochs"]):
            # Define dataloader
            sampler = WeightedRandomSampler(
                weights=train_dataset.sample_weights,
                num_samples=int(
                    math.ceil(len(train_dataset) / world_size)
                ),  # length of "epoch" on this rank
                replacement=True,
                generator=torch.Generator().manual_seed(
                    config["seed"] + 1000 * rank + epoch
                ),
            )
            dataloader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                sampler=sampler,
                num_workers=config["num_workers"],
                collate_fn=HallucinationDataset.collate,
                pin_memory=(device.type == "cuda"),
                drop_last=True,
            )
            # if is_distributed and isinstance(train_dataloader.sampler, DistributedSampler):
            #    train_dataloader.sampler.set_epoch(epoch)

            train_avg_loss = 0.0
            wrapped_model.train()
            if is_distributed:
                dist.barrier()
            with tqdm(dataloader) as progress_bar:
                for batch_idx, batch in enumerate(progress_bar):
                    should_step = (batch_idx + 1) % config[
                        "grad_accumulation_factor"
                    ] == 0
                    with (
                        wrapped_model.no_sync()
                        if hasattr(wrapped_model, "no_sync") and not should_step
                        else nullcontext()
                    ):
                        with torch.autocast(
                            device.type,
                            dtype=(
                                torch.bfloat16
                                if device.type == "cpu"
                                else torch.float16
                            ),
                            enabled=config["use_amp"],
                        ):
                            inputs, targets, lengths = [x.to(device) for x in batch]
                            logits = wrapped_model(inputs, lengths=lengths)
                            loss = loss_fn(logits, targets)

                        if should_step:
                            scaler.scale(
                                loss / config["grad_accumulation_factor"]
                            ).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                config["max_grad_norm"],
                            )
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            optimizer_steps += 1
                    train_avg_loss += (loss.item() - train_avg_loss) / (batch_idx + 1)
                    if rank == 0:
                        progress_bar.set_postfix(
                            epoch=epoch,
                            optimizer_steps=optimizer_steps,
                            train_loss=train_avg_loss,
                        )

            # Validation
            if is_distributed:
                dist.barrier()

            # Build dataloader
            dataloader = DataLoader(
                valid_dataset,
                collate_fn=HallucinationDataset.collate,
                pin_memory=(device.type == "cuda"),
            )

            valid_avg_loss = 0.0
            correct_total = 0
            seen_total = 0
            correct_per_class = torch.zeros(
                config["num_classes"], device=device, dtype=torch.long
            )
            seen_per_class = torch.zeros(
                config["num_classes"], device=device, dtype=torch.long
            )
            model.eval()
            with tqdm(dataloader) as progress_bar:
                for sample_idx, batch in enumerate(progress_bar):
                    with torch.no_grad():
                        inputs, targets, lengths = [x.to(device) for x in batch]
                        logits = wrapped_model(inputs, lengths=lengths)
                        loss = loss_fn(logits, targets)

                    valid_avg_loss += loss.item() / len(dataloader)

                    # Predictions
                    preds = logits.argmax(dim=1)

                    # Overall
                    correct_batch = (preds == targets).sum()
                    correct_total += correct_batch.item()
                    seen_total += targets.numel()

                    # Per-class: bincount by true label
                    seen_per_class += torch.bincount(
                        targets, minlength=config["num_classes"]
                    )
                    correct_mask = preds == targets
                    if correct_mask.any():
                        correct_per_class += torch.bincount(
                            targets[correct_mask], minlength=config["num_classes"]
                        )

                    if rank == 0:
                        progress_bar.set_postfix(
                            epoch=epoch,
                            valid_loss=valid_avg_loss,
                        )

                # End of validation
                valid_acc = 100 * correct_total / max(1, seen_total)
                per_class_acc = (
                    100
                    * correct_per_class.float()
                    / seen_per_class.clamp_min(1).float()
                ).tolist()
                macro_acc = sum(per_class_acc) / len(per_class_acc)

                # Save checkpoint
                if rank == 0:
                    logger.info(
                        f"epoch={epoch}, "
                        f"lr={optimizer.param_groups[0]['lr']}, "
                        f"optimizer_steps={optimizer_steps}, "
                        f"train_loss={train_avg_loss:.4f}, "
                        f"valid_loss={valid_avg_loss:.4f}, "
                        f"valid_acc={valid_acc:.2f}, "
                        f"valid_macro_acc={macro_acc:.2f}, "
                        f"valid_per_class_acc={[f'{a:.2f}' for a in per_class_acc]}"
                    )
                    checkpoint = {}
                    checkpoint["rng_state"] = torch.random.get_rng_state()
                    checkpoint["epoch"] = epoch
                    checkpoint["optimizer_steps"] = optimizer_steps
                    checkpoint["model"] = model.state_dict()
                    checkpoint["optimizer"] = optimizer.state_dict()
                    checkpoint["scaler"] = scaler.state_dict()
                    os.makedirs(config["save_dir"], exist_ok=True)
                    checkpoint_path = os.path.join(
                        config["save_dir"],
                        f"epoch={str(epoch).zfill(3)}_valid={valid_avg_loss:.2f}.pt",
                    )
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint {checkpoint_path}")
                if is_distributed:
                    dist.barrier()

    # Test
    if is_distributed:
        dist.barrier()

    # Build dataloader
    dataloader = DataLoader(
        test_dataset,
        collate_fn=HallucinationDataset.collate,
        pin_memory=(device.type == "cuda"),
    )

    test_avg_loss = 0.0
    correct_total = 0
    seen_total = 0
    correct_per_class = torch.zeros(
        config["num_classes"], device=device, dtype=torch.long
    )
    seen_per_class = torch.zeros(config["num_classes"], device=device, dtype=torch.long)
    model.eval()
    with tqdm(dataloader) as progress_bar:
        for sample_idx, batch in enumerate(progress_bar):
            with torch.no_grad():
                inputs, targets, lengths = [x.to(device) for x in batch]
                logits = wrapped_model(inputs, lengths=lengths)
                loss = loss_fn(logits, targets)

            test_avg_loss += loss.item() / len(dataloader)

            # Predictions
            preds = logits.argmax(dim=1)

            # Overall
            correct_batch = (preds == targets).sum()
            correct_total += correct_batch.item()
            seen_total += targets.numel()

            # Per-class: bincount by true label
            seen_per_class += torch.bincount(targets, minlength=config["num_classes"])
            correct_mask = preds == targets
            if correct_mask.any():
                correct_per_class += torch.bincount(
                    targets[correct_mask], minlength=config["num_classes"]
                )

            if rank == 0:
                progress_bar.set_postfix(
                    epoch=epoch if epoch is not None else start_epoch,
                    test_loss=test_avg_loss,
                )

        # End of validation
        test_acc = 100 * correct_total / max(1, seen_total)
        per_class_acc = (
            100 * correct_per_class.float() / seen_per_class.clamp_min(1).float()
        ).tolist()
        macro_acc = sum(per_class_acc) / len(per_class_acc)

        # Save checkpoint
        if rank == 0:
            logger.info(
                f"epoch={epoch if epoch is not None else start_epoch}, "
                f"optimizer_steps={optimizer_steps}, "
                f"test_acc={test_acc:.2f}, "
                f"test_macro_acc={macro_acc:.2f}, "
                f"test_per_class_acc={[f'{a:.2f}' for a in per_class_acc]}"
            )
        if is_distributed:
            dist.barrier()

    # Clean-up
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(None)

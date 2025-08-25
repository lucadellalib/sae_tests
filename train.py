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

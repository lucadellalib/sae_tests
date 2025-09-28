#!/usr/bin/env python3
"""
Multi-thread writer version:
- Loads many .safetensors activations in parallel (ThreadPoolExecutor).
- Accumulates until target_count, shuffles, casts dtype, and chunks into groups.
- Serializes each group to .safetensors BYTES on the producer side.
- Dispatches bytes to one of N writer threads by shard ownership:
    owner = shard_idx % num_writers
  so that each thread writes disjoint shards (no tar corruption).
- Writer threads write sequentially to tar files with fixed mtime for reproducibility.
"""

import argparse
import io
import os
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- threading-based writers ----
import threading
import queue as queue_mod


# -------------- Utilities --------------

def parse_out_pattern(pattern: str) -> Tuple[str, int, str]:
    """
    Parse patterns like 'shard-{000000..}.tar' into (prefix, width, suffix).
    Width=6 here. Suffix preserves extension.
    If the pattern doesn't contain braces, we append a 6-digit counter.
    """
    if "{000" in pattern and "..}" in pattern:
        pre, rest = pattern.split("{", 1)
        digits = rest.split("..}")[0]
        width = len(digits)
        suf = rest.split("..}", 1)[1]
        return pre, width, suf
    else:
        stem, ext = os.path.splitext(pattern)
        return stem + "-", 6, ext or ".tar"


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def write_safetensors_to_bytes(tensor: torch.Tensor, key: str = "activations") -> bytes:
    """
    Save a single tensor into .safetensors bytes via a temp file
    (robust across safetensors versions).
    """
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=True) as tmp:
        safetensors.torch.save_file({key: tensor}, tmp.name)
        tmp.flush()
        tmp.seek(0)
        data = tmp.read()
    return data


# -------------- Writer worker (threaded) --------------

def writer_worker_thread(out_dir: str,
                         out_pattern: str,
                         q: "queue_mod.Queue[tuple[int,int,bytes] | None]",
                         stop_event: threading.Event,
                         shard_groups: int):
    """
    Thread target.
    Receives tuples (shard_idx, group_idx, data_bytes) and writes into tar shards.
    Uses a dedicated queue so that each thread owns disjoint shard indices and
    therefore never clashes on a tar file.
    """
    ensure_dir(out_dir)
    prefix, width_digits, suffix = parse_out_pattern(out_pattern)

    current_shard = -1
    groups_in_shard = 0
    tar = None

    def open_new_shard(idx: int) -> tarfile.TarFile:
        out_name = f"{prefix}{idx:0{width_digits}d}{suffix}"
        out_path = os.path.join(out_dir, out_name)
        return tarfile.open(out_path, mode="w")  # uncompressed tar

    try:
        while True:
            try:
                item = q.get(timeout=0.2)
            except queue_mod.Empty:
                if stop_event.is_set():
                    break
                continue

            if item is None:  # sentinel => stop this thread
                break

            shard_idx, group_idx, data_bytes = item

            # roll shard if needed
            if shard_idx != current_shard:
                if tar is not None:
                    tar.close()
                current_shard = shard_idx
                groups_in_shard = 0
                tar = open_new_shard(current_shard)

            key = f"{shard_idx:06d}-{group_idx:06d}"
            info = tarfile.TarInfo(name=f"{key}.safetensors")
            info.size = len(data_bytes)
            info.mtime = 0  # deterministic/reproducible
            tar.addfile(tarinfo=info, fileobj=io.BytesIO(data_bytes))
            groups_in_shard += 1

            if groups_in_shard >= shard_groups:
                tar.close()
                tar = None
                current_shard = -1
                groups_in_shard = 0
    finally:
        if tar is not None:
            tar.close()


# -------------- Reader / Accumulator --------------

def load_one_path(path: str, layer_idx: int = 30, remove_prompt: bool = True) -> torch.Tensor:
    """
    Simplified loader tailored to your files:
    - Optionally rewrites the filename to the selected layer suffix.
    - Opens `hidden_states` and optionally removes a leading prompt slice.
    Returns a 2D tensor (N, width).
    """
    if layer_idx is not None:
        path = path.replace(".safetensors", f".l{layer_idx}.safetensors")
    with safetensors.safe_open(path, framework="pt") as f:
        hidden_states = f.get_tensor("hidden_states")
        if remove_prompt and "prompt_length" in f.keys():
            prompt_length = int(f.get_tensor("prompt_length").item())
            hidden_states = hidden_states[prompt_length:]

    if hidden_states.ndim != 2:
        raise ValueError(f"Expected 2D activations, got shape {tuple(hidden_states.shape)} from {path}")
    return hidden_states.cpu().contiguous()


def accumulate_until(
    paths: List[str],
    start_idx: int,
    target_count: int,
    num_readers: int,
) -> Tuple[torch.Tensor, int]:
    """
    Load from paths[start_idx:] in parallel until we collect >= target_count rows.
    Returns (tensor[N>=target_count, width], next_path_index).
    If paths are exhausted, returns whatever was collected (possibly < target_count).
    """
    acc: List[torch.Tensor] = []
    total = 0
    i = start_idx

    with ThreadPoolExecutor(max_workers=num_readers) as ex:
        futures = {}
        # Kick off initial batch
        while i < len(paths) and len(futures) < num_readers:
            futures[ex.submit(load_one_path, paths[i])] = i
            i += 1

        with tqdm(total=target_count, unit="activations", desc="Accumulating", leave=False) as pbar:
            while futures:
                done = next(as_completed(futures))
                idx = futures.pop(done)
                try:
                    arr = done.result()
                except Exception as e:
                    print(f"[WARN] Failed to load {paths[idx]}: {e}", file=sys.stderr)
                    arr = None

                if arr is not None:
                    acc.append(arr)
                    total += arr.shape[0]
                    pbar.update(arr.shape[0] if total <= target_count
                                else max(0, target_count - (total - arr.shape[0])))

                # Queue more if needed
                if total < target_count and i < len(paths):
                    futures[ex.submit(load_one_path, paths[i])] = i
                    i += 1

                # Stop early once we have enough and drained futures
                if total >= target_count and not futures:
                    break

    if not acc:
        return torch.empty((0, 8192), dtype=torch.float32), i  # width unknown; pick 8192 default

    big = torch.cat(acc, dim=0)
    return big, i


def shuffle_inplace(t: torch.Tensor, seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    perm = torch.randperm(t.shape[0], generator=g)
    return t[perm]


# -------------- Main orchestrator --------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--list", type=str, required=True, help="Text file with one safetensors path per line.")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for shards.")
    ap.add_argument("--out-pattern", type=str, default="shard-{000000..}.tar", help="Output shard name pattern.")
    ap.add_argument("--group-size", type=int, default=1024, help="Activations per sample/group.")
    ap.add_argument("--shard-groups", type=int, default=100, help="Max groups (samples) per shard.")
    ap.add_argument("--target-count", type=int, default=1_000_000, help="Accumulation size before shuffle/write.")
    ap.add_argument("--num-readers", type=int, default=16, help="Parallel reader threads for loading inputs.")
    ap.add_argument("--num-writers", type=int, default=4, help="Number of writer threads (partitioned by shard).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for shuffling.")
    ap.add_argument("--resume-index", type=int, default=0, help="Start index in the paths list (resume).")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"], help="Output dtype.")
    args = ap.parse_args()

    # Read input list
    with open(args.list, "r") as f:
        paths = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    if args.resume_index >= len(paths):
        print("Nothing to do: resume-index beyond input list.", file=sys.stderr)
        return

    ensure_dir(args.out_dir)

    # --- writer threads setup ---
    stop_evt = threading.Event()
    num_writers = max(1, int(args.num_writers))
    writer_queues: list[queue_mod.Queue] = [queue_mod.Queue(maxsize=128) for _ in range(num_writers)]
    writer_threads: list[threading.Thread] = []

    for i in range(num_writers):
        t = threading.Thread(
            target=writer_worker_thread,
            args=(args.out_dir, args.out_pattern, writer_queues[i], stop_evt, args.shard_groups),
            daemon=True,
        )
        t.start()
        writer_threads.append(t)

    def dispatch(shard_idx: int, group_idx: int, data_bytes: bytes):
        """Route packet to owner queue based on shard index partitioning."""
        owner = shard_idx % num_writers
        writer_queues[owner].put((shard_idx, group_idx, data_bytes))

    next_path_idx = args.resume_index
    shard_index = 0
    group_counter = 0

    total_loaded = 0
    total_emitted_groups = 0

    # dtype map
    _dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    out_dtype = _dtype_map[args.dtype]

    try:
        while next_path_idx < len(paths):
            # 1) Accumulate a big chunk
            big, next_path_idx = accumulate_until(
                paths=paths,
                start_idx=next_path_idx,
                target_count=args.target_count,
                num_readers=args.num_readers,
            )
            if big.numel() == 0:
                break

            total_loaded += big.shape[0]

            # 2) Shuffle
            big = shuffle_inplace(big, seed=args.seed + shard_index)  # vary by cycle

            # 3) Cast dtype (saves space if fp16/bf16)
            big = big.to(out_dtype)

            # 4) Slice into groups and enqueue to writers (serialized to bytes here)
            width = big.shape[1]
            n_groups = big.shape[0] // args.group_size
            usable = n_groups * args.group_size
            if usable == 0:
                print(f"[INFO] Remainder < group-size ({big.shape[0]}), carrying over.", file=sys.stderr)

            groups_tensor = big[:usable].reshape(n_groups, args.group_size, width)

            # Emit groups
            for gi in range(n_groups):
                group = groups_tensor[gi].contiguous()
                data_bytes = write_safetensors_to_bytes(group, key="activations")

                shard_idx = shard_index + (total_emitted_groups // args.shard_groups)
                group_idx = group_counter % args.shard_groups

                dispatch(shard_idx, group_idx, data_bytes)

                group_counter += 1
                total_emitted_groups += 1

            # Advance shard_index to next shard boundary
            shard_index = (total_emitted_groups // args.shard_groups)

            # 5) Handle remainder (carry to next cycle)
            rem = big.shape[0] - usable
            if rem > 0 and next_path_idx < len(paths):
                carry = big[usable:].contiguous()
                with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
                    safetensors.torch.save_file({"activations": carry}, tmp.name)
                    temp_path = tmp.name
                # Insert the temp path so accumulate_until will pick it up first
                paths.insert(next_path_idx, temp_path)

        # done producing; fall through to shutdown
    finally:
        # Signal writers to finish; send sentinel to each queue
        stop_evt.set()
        for q in writer_queues:
            q.put(None)
        for t in writer_threads:
            t.join(timeout=60)

        print(f"[DONE] Loaded activations: {total_loaded:,d} | Emitted groups: {total_emitted_groups:,d}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
Convert a (possibly remote) list of .safetensors activation files into WebDataset shards.

Pipeline (pipelined & parallel where it matters):
1) Read + load many .safetensors files in parallel (threads).
2) Accumulate activations in RAM until we reach --target-count (default: 1_000_000).
3) Shuffle the activations in-memory.
4) Chunk into groups of --group-size (default: 1024) activations.
5) Stream groups to a writer process pool which packs them into tar shards (WebDataset-compatible):
   each sample is one .safetensors file inside the tar (no JSON sidecar by default).
6) Repeat until input list is exhausted. Any remainder < target-count is processed at the end.

Notes
-----
- Memory math: 1,000,000 x 8192 x 2 bytes (float16) ~= 16 GiB. Ensure you have RAM, or reduce --target-count.
- "WebDataset format" here means shard tar files compatible with `webdataset` library consumption.

The out-pattern determines how many shards at most you'll produce; if you exceed, the index continues.
"""

import io
import os
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue
from pathlib import Path
from typing import List, Optional, Tuple

import safetensors
import safetensors.torch
import torch
from absl import app, flags
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "list",
    None,
    "Text file with one safetensors path per line (local/remote).",
)
flags.DEFINE_string(
    "out_dir",
    None,
    "Output directory for shards.",
)
flags.DEFINE_string(
    "out_pattern",
    "shard-{000000..}.tar",
    "Output shard name pattern.",
)
flags.DEFINE_integer(
    "group_size",
    1024,
    "Number of activations per sample/group.",
)
flags.DEFINE_integer(
    "shard_groups",
    100,
    "Max groups per shard (controls shard size).",
)
flags.DEFINE_integer(
    "target_count",
    1_000_000,
    "Activations to accumulate before each shuffle/write cycle.",
)
flags.DEFINE_integer(
    "num_readers",
    16,
    "Parallel reader threads for downloading+loading.",
)
flags.DEFINE_integer(
    "num_writers",
    4,
    "Parallel writer processes for tar creation.",
)
flags.DEFINE_integer(
    "seed",
    0,
    "RNG seed for shuffling.",
)
flags.DEFINE_integer(
    "resume_index",
    0,
    "Start index in the paths list (resume).",
)


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
    Save a single tensor into .safetensors bytes. We use a NamedTemporaryFile to avoid
    relying on a rarely-used bytes API; this is robust across safetensors versions.
    """
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=True) as tmp:
        safetensors.torch.save_file({key: tensor}, tmp.name)  # write to temp file
        tmp.flush()
        tmp.seek(0)
        data = tmp.read()
    return data


# -------------- Writer worker --------------


@dataclass
class GroupPacket:
    shard_index: int
    group_index: int
    tensor: torch.Tensor  # shape (group_size, width)


def writer_worker(
    out_dir: str, out_pattern: str, queue: Queue, stop_event: Event, shard_groups: int
):
    """
    Receives GroupPacket objects and writes them into tar shards.
    Each group becomes one .safetensors member named '{global_idx:09d}.safetensors'.
    """
    ensure_dir(out_dir)
    prefix, width, suffix = parse_out_pattern(out_pattern)

    current_shard = -1
    groups_in_shard = 0
    tar: Optional[tarfile.TarFile] = None

    def open_new_shard(idx: int) -> tarfile.TarFile:
        out_name = f"{prefix}{idx:0{width}d}{suffix}"
        out_path = os.path.join(out_dir, out_name)
        return tarfile.open(out_path, mode="w")  # uncompressed tar

    try:
        while not (stop_event.is_set() and queue.empty()):
            try:
                pkt: GroupPacket = queue.get(timeout=0.2)
            except Exception:
                continue
            # Roll shard if needed
            if pkt.shard_index != current_shard:
                if tar is not None:
                    tar.close()
                current_shard = pkt.shard_index
                groups_in_shard = 0
                tar = open_new_shard(current_shard)

            # Serialize this group to bytes
            data_bytes = write_safetensors_to_bytes(pkt.tensor, key="activations")
            # Build tar member
            key = f"{pkt.shard_index:06d}-{pkt.group_index:06d}"
            info = tarfile.TarInfo(name=f"{key}.safetensors")
            info.size = len(data_bytes)
            info.mtime = 0  # stable
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


def load_one_path(
    path: str, layer_idx: int = 30, remove_prompt: bool = True
) -> torch.Tensor:
    if layer_idx is not None:
        path = path.replace(".safetensors", f".l{layer_idx}.safetensors")
    with safetensors.safe_open(path, framework="pt") as f:
        hidden_states = f.get_tensor("hidden_states")
        if remove_prompt:
            prompt_length = f.get_tensor("prompt_length").item()
            hidden_states = hidden_states[prompt_length:]
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
    If paths are exhausted, returns whatever was collected.
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

        with tqdm(total=target_count, unit="activations", desc="Accumulating") as pbar:
            while futures:
                done, _ = as_completed(futures, timeout=None).__next__(), None
                # Collect this one
                idx = futures.pop(done)
                try:
                    arr = done.result()
                except Exception as e:
                    print(f"[WARN] Failed to load {paths[idx]}: {e}")
                    arr = None
                if arr is not None:
                    acc.append(arr)
                    total += arr.shape[0]
                    pbar.update(
                        arr.shape[0]
                        if total <= target_count
                        else max(0, target_count - (total - arr.shape[0]))
                    )

                # If we have enough, we can stop early (but let current in-flight finish)
                if total >= target_count:
                    # We won't queue more, but drain existing futures
                    pass
                else:
                    # Submit a new one if available
                    if i < len(paths):
                        futures[ex.submit(load_one_path, paths[i])] = i
                        i += 1

                # Break condition: we have enough and no pending futures we care for
                if total >= target_count and not futures:
                    break

            # Drain any extra completed futures (we didn't queue new after reaching target)
            for fut, idx in list(futures.items()):
                try:
                    _ = fut.result()
                except Exception:
                    pass

    if not acc:
        print(f"[WARN] acc is empty!")
        return (
            torch.empty((0, 8192), dtype=torch.float16),
            i,
        )  # width/dtype as you prefer

    big = torch.cat(acc, dim=0)
    return big, i


def shuffle_inplace(t: torch.Tensor, seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    perm = torch.randperm(t.shape[0], generator=g)
    return t[perm]


# -------------- Main orchestrator --------------


def main(argv):
    del argv

    # Read path list
    with open(FLAGS.list, "r") as f:
        paths = [
            ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")
        ]

    if FLAGS.resume_index >= len(paths):
        print("Nothing to do: resume-index beyond input list.")
        return

    ensure_dir(FLAGS.out_dir)

    # Start writer pool
    stop_event = Event()
    queue: Queue = Queue(maxsize=FLAGS.num_writers * 4)

    # We assign shard indexes monotonically as we emit groups.
    # Each cycle (target_count) produces C = floor(N / group_size) groups.
    # We pack <= shard_groups groups per tar.
    writer_procs: List[Process] = []
    for _ in range(FLAGS.num_writers):
        p = Process(
            target=writer_worker,
            args=(
                FLAGS.out_dir,
                FLAGS.out_pattern,
                queue,
                stop_event,
                FLAGS.shard_groups,
            ),
            daemon=True,
        )
        p.start()
        writer_procs.append(p)

    next_path_idx = FLAGS.resume_index
    shard_index = 0
    group_counter = 0

    total_loaded = 0
    total_emitted_groups = 0

    try:
        while next_path_idx < len(paths):
            # 1) Accumulate a big chunk
            big, next_path_idx = accumulate_until(
                paths=paths,
                start_idx=next_path_idx,
                target_count=FLAGS.target_count,
                num_readers=FLAGS.num_readers,
            )
            if big.numel() == 0:
                break

            total_loaded += big.shape[0]

            # 2) Shuffle
            big = shuffle_inplace(big, seed=FLAGS.seed + shard_index)  # vary by cycle

            # 3) Maybe cast dtype (saves space if float16/bfloat16 is acceptable)
            assert big.dtype == torch.float16

            # 4) Slice into groups and enqueue to writers
            n_groups = big.shape[0] // FLAGS.group_size
            usable = n_groups * FLAGS.group_size
            if usable == 0:
                print(
                    f"[INFO] Remainder < group-size ({big.shape[0]}), carrying over to finalization."
                )
            groups_tensor = big[:usable].view(n_groups, FLAGS.group_size, -1)

            # Emit groups
            for gi in range(n_groups):
                pkt = GroupPacket(
                    shard_index=shard_index
                    + (total_emitted_groups // FLAGS.shard_groups),
                    group_index=group_counter % FLAGS.shard_groups,
                    tensor=groups_tensor[gi].clone(),  # ensure independent storage
                )
                queue.put(pkt)
                group_counter += 1
                total_emitted_groups += 1

            # Advance shard_index to next shard boundary
            shard_index = total_emitted_groups // FLAGS.shard_groups

            # 5) Handle remainder (carry it over to the next cycle)
            rem = big.shape[0] - usable
            if rem > 0:
                carry = big[usable:].clone()
            else:
                carry = None

            # If there is a carry, we will prepend it to the next accumulation by a small hack:
            if carry is not None and next_path_idx < len(paths):
                # Temporarily write to a NamedTemporaryFile and insert at front via an artificial path list.
                with tempfile.NamedTemporaryFile(
                    suffix=".safetensors", delete=False
                ) as tmp:
                    safetensors.torch.save_file({"activations": carry}, tmp.name)
                    temp_path = tmp.name
                # Insert the temp path so accumulate_until will pick it up first
                paths.insert(next_path_idx, temp_path)

        # Finalization: if any paths were exhausted but we still might have a small leftover after last loop,
        # the above carry mechanism should have flushed it into the last cycle. Nothing else to do.

    finally:
        # Signal writers to stop after draining the queue
        stop_event.set()
        # Drain queue
        queue.close()
        queue.join_thread()
        # Join writers
        for p in writer_procs:
            p.join(timeout=10)
        print(
            f"[DONE] Loaded activations: {total_loaded:,d} | Emitted groups: {total_emitted_groups:,d}"
        )


if __name__ == "__main__":
    app.run(main)

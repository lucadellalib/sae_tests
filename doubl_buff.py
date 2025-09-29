import threading, queue
import torch
from torch.utils.data import IterableDataset
import safetensors
import safetensors.torch
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------- low-level I/O ----------

def _load_activations(path, layer_idx=30, remove_prompt=True, dtype=torch.float16):
    if layer_idx is not None:
        path = path.replace(".safetensors", f".l{layer_idx}.safetensors")
    with safetensors.safe_open(path, framework="pt") as f:
        hs = f.get_tensor("hidden_states")
        if remove_prompt and "prompt_length" in f.keys():
            pl = int(f.get_tensor("prompt_length").item())
            hs = hs[pl:]
    return hs.to(dtype).cpu().contiguous()


def _accumulate_buffer_ddp(paths, start_idx, target_count, num_readers, dtype, rank, world):
    """Fill >= target_count rows using thread pool; stride file indices by world across ranks."""
    N = len(paths)
    if N == 0:
        return torch.empty((0, 8192), dtype=dtype), start_idx
    acc, total = [], 0

    def _next_i(i):  # stride by world with wrap-around
        return (i + world) % N

    i = (start_idx + rank) % N
    with ThreadPoolExecutor(max_workers=num_readers) as ex:
        futures = {}
        while len(futures) < num_readers and total < target_count:
            futures[ex.submit(_load_activations, paths[i], dtype=dtype)] = i
            i = _next_i(i)
        while futures and total < target_count:
            done = next(as_completed(futures))
            idx_sub = futures.pop(done)
            try:
                arr = done.result()
            except Exception as e:
                print(f"[WARN] failed {paths[idx_sub]}: {e}")
                arr = None
            if arr is not None:
                acc.append(arr)
                total += arr.shape[0]
            if total < target_count:
                futures[ex.submit(_load_activations, paths[i], dtype=dtype)] = i
                i = _next_i(i)

    if not acc:
        return torch.empty((0, 8192), dtype=dtype), i
    return torch.cat(acc, dim=0), i  # next index to continue from


# ---------- dataset (double buffered) ----------

class InfiniteBufferedDataset(IterableDataset):
    """
    Double-buffered infinite streaming dataset (DDP-aware, checkpointable):
      - Loader thread builds big activation buffers in parallel and enqueues them (buffer_q).
      - Batcher thread shuffles each buffer, slices into fixed-size batches, enqueues batches (batch_q).
      - Iterator yields from batch_q while the loader prepares the next buffer in the background.

    Checkpointable state: resume_idx (unstrided path index), epoch counter, RNG state.
    """

    def __init__(self,
                 paths,
                 batch_size=1024,
                 buffer_size=1_000_000,
                 dtype=torch.float16,
                 shuffle_seed=0,
                 num_reader_threads=8,
                 layer_idx=30,
                 remove_prompt=True,
                 # queues
                 buffer_queue_size=2,   # how many full big buffers can wait (double-buffering=2)
                 batch_queue_size=256,  # how many ready batches can wait
                 # DDP
                 rank=None,
                 world_size=None,
                 # checkpoint
                 resume_idx=0,
                 resume_epoch=0,
                 rng_state_bytes: bytes | None = None):
        super().__init__()
        self.paths = list(paths)
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.dtype = dtype
        self.shuffle_seed = int(shuffle_seed)
        self.num_reader_threads = int(num_reader_threads)
        self.layer_idx = layer_idx
        self.remove_prompt = remove_prompt

        # Detect DDP if not provided
        if rank is None or world_size is None:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
                    world_size = dist.get_world_size()
            except Exception:
                pass
        self.rank = int(rank) if rank is not None else 0
        self.world = int(world_size) if world_size is not None else 1

        # State (checkpointable)
        self._resume_idx = int(resume_idx)
        self._epoch = int(resume_epoch)
        self._rng_state_bytes = rng_state_bytes

        # Queues
        self.buffer_q_size = int(buffer_queue_size)
        self.batch_q_size = int(batch_queue_size)

        # State lock
        self._state_lock = threading.Lock()

        # Bind loader params without touching globals in multi-rank
        def _loader(path, layer_idx=self.layer_idx, remove_prompt=self.remove_prompt, dtype=self.dtype):
            return _load_activations(path, layer_idx=layer_idx, remove_prompt=remove_prompt, dtype=dtype)
        global _load_activations
        _load_activations = _loader  # safe per-process

    # ---- Checkpoint API ----
    def get_state(self):
        with self._state_lock:
            return {
                "resume_idx": self._resume_idx,
                "epoch": self._epoch,
                "rng_state": (bytes(self._rng_state_bytes) if self._rng_state_bytes is not None else None),
                "rank": self.rank,
                "world_size": self.world,
            }

    def set_state(self, state: dict):
        with self._state_lock:
            self._resume_idx = int(state.get("resume_idx", 0))
            self._epoch = int(state.get("epoch", 0))
            self._rng_state_bytes = state.get("rng_state", None)

    # ---- Iterator ----
    def __iter__(self):
        buffer_q: "queue.Queue[tuple[torch.Tensor,int,int]]" = queue.Queue(maxsize=self.buffer_q_size)
        batch_q:  "queue.Queue[tuple[torch.Tensor,int,int] | tuple[str,int,int,bytes]]" = queue.Queue(maxsize=self.batch_q_size)
        stop_event = threading.Event()

        # RNG
        rng = torch.Generator()
        if self._rng_state_bytes is not None:
            rng.set_state(torch.tensor(list(self._rng_state_bytes), dtype=torch.uint8))
        else:
            rng.manual_seed(self.shuffle_seed + self.rank * 9973 + self._epoch)

        # ---- Stage 1: Loader thread (fills big buffers) ----
        def loader_thread(start_idx: int, epoch0: int):
            idx = start_idx
            local_epoch = epoch0
            while not stop_event.is_set():
                buf, idx = _accumulate_buffer_ddp(
                    self.paths, idx, self.buffer_size,
                    self.num_reader_threads, self.dtype,
                    self.rank, self.world
                )
                if buf.numel() == 0:
                    # If paths missing/empty: spin lightly
                    continue
                buffer_q.put((buf, idx, local_epoch))
                # When we wrap around the file list naturally, epoch will get bumped by batcher (after buffer use)

        # ---- Stage 2: Batcher thread (shuffles, slices, enqueues batches) ----
        def batcher_thread():
            local_epoch = self._epoch
            local_rng = rng
            while not stop_event.is_set():
                buf, idx_after_buf, epoch_of_buf = buffer_q.get()
                # Send a "state" ping before we permute so resume persists idx/epoch and RNG
                batch_q.put(("state", idx_after_buf, epoch_of_buf, bytes(local_rng.get_state().tolist())))

                # Shuffle the buffer (rank-specific RNG)
                perm = torch.randperm(buf.shape[0], generator=local_rng)
                buf = buf[perm]

                # Emit full batches; drop remainder
                n_batches = buf.shape[0] // self.batch_size
                if n_batches > 0:
                    batches = buf[:n_batches * self.batch_size].reshape(
                        n_batches, self.batch_size, buf.shape[1]
                    )
                    for b in batches:
                        batch_q.put((b, idx_after_buf, epoch_of_buf))

                # Advance epoch salt for next buffer’s permutation
                local_epoch = epoch_of_buf + 1
                local_rng.manual_seed(self.shuffle_seed + self.rank * 9973 + local_epoch)

        # launch threads
        lt = threading.Thread(target=loader_thread, args=(self._resume_idx, self._epoch), daemon=True)
        bt = threading.Thread(target=batcher_thread, daemon=True)
        lt.start()
        bt.start()

        try:
            while True:
                item = batch_q.get()
                if isinstance(item, tuple) and item and item[0] == "state":
                    _, idx, ep, rng_bytes = item
                    with self._state_lock:
                        self._resume_idx = idx
                        self._epoch = ep
                        self._rng_state_bytes = rng_bytes
                    continue
                batch, idx, ep = item  # actual data batch
                with self._state_lock:
                    self._resume_idx = idx
                    self._epoch = ep
                yield batch
        finally:
            stop_event.set()
            lt.join()
            bt.join()

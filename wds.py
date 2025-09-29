import threading, queue
import torch
from torch.utils.data import IterableDataset
import safetensors
import safetensors.torch
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    """Fill a buffer with >= target_count rows; stride file indices by world across ranks."""
    N = len(paths)
    if N == 0:
        return torch.empty((0, 8192), dtype=dtype), start_idx
    acc, total = [], 0

    def _next_i(i):
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
    return torch.cat(acc, dim=0), i  # 'i' is the next index to continue from (already strided)


class InfiniteBufferedDataset(IterableDataset):
    """
    DDP-aware infinite streaming dataset with checkpointable state.
    - Per-rank sharded file stream (stride by world size)
    - Parallel file reads fill a big buffer (~buffer_size activations)
    - Buffer-level global shuffle, then emit full batches (drop remainder)
    - Exposes get_state()/set_state() for (resume_idx, epoch, rng_state)
    """

    def __init__(self,
                 paths,
                 batch_size=1024,
                 buffer_size=1_000_000,
                 dtype=torch.float16,
                 shuffle_seed=0,
                 num_reader_threads=8,
                 queue_size=256,
                 layer_idx=30,
                 remove_prompt=True,
                 rank=None,
                 world_size=None,
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
        self.queue_size = int(queue_size)
        self.layer_idx = layer_idx
        self.remove_prompt = remove_prompt

        # Detect DDP rank/world if not provided
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
        self._resume_idx = int(resume_idx)   # next path index *in unstrided space*
        self._epoch = int(resume_epoch)
        self._rng_state_bytes = rng_state_bytes  # torch.Generator serialized state (or None)

        # Thread-safety for state updates
        self._state_lock = threading.Lock()

        # Bind loader parameters to the loader function (no globals mutated)
        def _loader(path, layer_idx=self.layer_idx, remove_prompt=self.remove_prompt, dtype=self.dtype):
            return _load_activations(path, layer_idx=layer_idx, remove_prompt=remove_prompt, dtype=dtype)
        # Replace the callable used inside the pool
        global _load_activations
        _load_activations = _loader  # safe per-process

    # ---- Checkpoint API ----
    def get_state(self):
        with self._state_lock:
            return {
                "resume_idx": self._resume_idx,
                "epoch": self._epoch,
                "rng_state": bytes(self._rng_state_bytes) if self._rng_state_bytes is not None else None,
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
        q: "queue.Queue[tuple[torch.Tensor,int,int] | tuple[str,int,int,bytes]]" = queue.Queue(maxsize=self.queue_size)
        stop_event = threading.Event()

        rng = torch.Generator()
        if self._rng_state_bytes is not None:
            rng.set_state(torch.tensor(list(self._rng_state_bytes), dtype=torch.uint8))
        else:
            rng.manual_seed(self.shuffle_seed + self.rank * 9973 + self._epoch)

        def producer(start_idx: int, epoch0: int):
            idx = start_idx
            local_epoch = epoch0
            while not stop_event.is_set():
                buf, idx = _accumulate_buffer_ddp(
                    self.paths, idx, self.buffer_size,
                    self.num_reader_threads, self.dtype,
                    self.rank, self.world
                )
                if buf.numel() == 0:
                    # no data; spin
                    continue

                # BEFORE shuffling, emit a state ping with current RNG state
                q.put(("state", idx, local_epoch, bytes(rng.get_state().tolist())))

                # Shuffle the buffer with rank-specific RNG
                perm = torch.randperm(buf.shape[0], generator=rng)
                buf = buf[perm]

                # Emit full batches; drop remainder
                n_batches = buf.shape[0] // self.batch_size
                if n_batches > 0:
                    batches = buf[:n_batches * self.batch_size].reshape(
                        n_batches, self.batch_size, buf.shape[1]
                    )
                    for b in batches:
                        q.put((b, idx, local_epoch))

                # Advance epoch salt so next buffer gets a different permutation basis
                local_epoch += 1
                rng.manual_seed(self.shuffle_seed + self.rank * 9973 + local_epoch)

        t = threading.Thread(target=producer, args=(self._resume_idx, self._epoch), daemon=True)
        t.start()

        try:
            while True:
                item = q.get()
                # State update messages
                if isinstance(item, tuple) and item and item[0] == "state":
                    _, idx, ep, rng_bytes = item
                    with self._state_lock:
                        self._resume_idx = idx
                        self._epoch = ep
                        self._rng_state_bytes = rng_bytes
                    continue
                # Batch message
                batch, idx, ep = item  # type: ignore[assignment]
                with self._state_lock:
                    self._resume_idx = idx
                    self._epoch = ep
                yield batch
        finally:
            stop_event.set()
            t.join()

def encode(self, x: torch.Tensor) -> torch.Tensor:
    z = (x - self.b) @ self.W
    z = torch.relu(z)
    if self.topk is not None and 0 < self.topk < z.shape[1]:
        # old:
        # vals, idx = torch.topk(z, self.topk, dim=1)
        # mask = torch.zeros_like(z).scatter(1, idx, 1.0)
        # z = z * mask
        # new (ghost grads):
        z = topk_with_ghost(z, self.topk, ghost_k=8, gamma=0.1, tau=0.5, mode="nextk")
    return z



import torch

class TopKGhost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, k: int, ghost_k: int, gamma: float, tau: float, mode: str):
        """
        z: pre-activation after ReLU, shape [B, Kfeat]
        k: keep this many active features
        ghost_k: number of 'next-best' features to receive ghost grad (only for mode='nextk')
        gamma: ghost grad scale (0..1)
        tau: temperature for softmax mode
        mode: 'nextk' or 'softmax'
        """
        B, Kfeat = z.shape
        vals, idx = torch.topk(z, k, dim=1)
        mask = torch.zeros_like(z).scatter(1, idx, 1.0)
        y = z * mask  # hard Top-K in forward

        ctx.save_for_backward(z, mask, idx)
        ctx.k = k
        ctx.ghost_k = ghost_k
        ctx.gamma = gamma
        ctx.tau = tau
        ctx.mode = mode
        return y

    @staticmethod
    def backward(ctx, grad_out):
        z, mask, idx_top = ctx.saved_tensors
        k, ghost_k, gamma, tau, mode = ctx.k, ctx.ghost_k, ctx.gamma, ctx.tau, ctx.mode

        # Base: Top-K get full gradient
        grad_in = grad_out * mask

        if gamma > 0:
            if mode == "nextk":
                # give scaled gradient to next ghost_k by preactivation
                # compute ranks by sorting descending
                vals_all, idx_all = torch.sort(z, dim=1, descending=True)
                if ghost_k > 0:
                    ghost_idx = idx_all[:, k:k+ghost_k]  # [B, ghost_k]
                    ghost_mask = torch.zeros_like(z).scatter(1, ghost_idx, 1.0)
                    grad_in = grad_in + gamma * (grad_out * ghost_mask)
            elif mode == "softmax":
                # distribute a small fraction to everyone proportional to softmax
                w = torch.softmax(z / max(tau, 1e-6), dim=1)
                # but remove the Top-K (already have full grad) to avoid double counting
                w = w * (1.0 - mask)
                # normalize ghost mass to 1 per row then scale by gamma
                denom = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
                w = (w / denom)
                grad_in = grad_in + gamma * (grad_out * w)
        return grad_in, None, None, None, None, None


def topk_with_ghost(z, k: int, *, ghost_k: int = 8, gamma: float = 0.1, tau: float = 0.5, mode: str = "nextk"):
    """
    Convenience wrapper. z: [B, Kfeat], already ReLU'ed.
    Returns: masked codes with hard Top-K forward, ghosted gradients in backward.
    """
    return TopKGhost.apply(z, k, ghost_k, gamma, tau, mode)




# sae_streaming_all.py
# Python 3.10+, PyTorch 2.x

import math, random
from typing import Callable, Iterable, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader


# =========================
# 0) Fixed MSN scalar (unit mean-squared norm)
# =========================
class FixedMSNScalar:
    def __init__(self):
        self.s: Optional[torch.Tensor] = None  # device-matched scalar

    @torch.no_grad()
    def fit_from_stream(self, activations_iter: Iterable[torch.Tensor], device=None, max_rows: int = 500_000):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total, count = 0.0, 0
        for X in activations_iter:
            X = X.to(device, non_blocking=True)
            B, d = X.shape
            total += (X.pow(2).sum(dim=1) / float(d)).sum().item()
            count += B
            if count >= max_rows:
                break
        if count == 0:
            raise RuntimeError("No activations to fit MSN scalar.")
        mean_msn = total / count
        self.s = torch.tensor(mean_msn, device=device).sqrt().clamp_min(1e-12)

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        assert self.s is not None, "Call fit_from_stream() first."
        return X / self.s

    @torch.no_grad()
    def inverse_transform(self, Xn: torch.Tensor) -> torch.Tensor:
        assert self.s is not None, "Call fit_from_stream() first."
        return Xn * self.s


# =========================
# 1) Geometric median (Weiszfeld)
# =========================
@torch.no_grad()
def geometric_median(X: torch.Tensor, iters: int = 50, eps: float = 1e-8, tol: float = 1e-6) -> torch.Tensor:
    y = X.mean(dim=0)
    for _ in range(iters):
        diff = X - y
        dist = diff.norm(dim=1).clamp_min(eps)
        w = 1.0 / dist
        y_next = (w[:, None] * X).sum(dim=0) / w.sum()
        if (y_next - y).norm() < tol:
            y = y_next
            break
        y = y_next
    return y


# =========================
# 2) Tied SAE (W, b)
# =========================
class SAE(nn.Module):
    """
    Tied-bias, tied-weights SAE:
      encode(x) = ReLU((x - b) @ W - theta?)   (we keep it simple: no theta by default)
      decode(c) = c @ W^T + b
    """
    def __init__(self, d_in: int, n_features: int, sparsity_mode: str = "l1+topk", l1_coeff: float = 1e-3, topk: Optional[int] = 32, unit_norm_dict: bool = True):
        super().__init__()
        self.d, self.k = d_in, n_features
        self.sparsity_mode = sparsity_mode.lower()
        assert self.sparsity_mode in {"l1", "topk", "l1+topk"}
        self.l1_coeff = l1_coeff
        self.topk = topk
        self.unit_norm_dict = unit_norm_dict

        self.W = nn.Parameter(torch.randn(self.d, self.k) * (1.0 / math.sqrt(self.d)))
        self.b = nn.Parameter(torch.zeros(self.d))

    @torch.no_grad()
    def enforce_unit_norm(self):
        if not self.unit_norm_dict:
            return
        col_norms = self.W.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W.div_(col_norms)

    @torch.no_grad()
    def init_bias_from_stream(self, activations_iter: Iterable[torch.Tensor], device: torch.device, max_rows: int = 200_000, method: str = "geomedian"):
        rows, ncol = [], 0
        for X in activations_iter:
            X = X.to(device, non_blocking=True)
            take = min(X.shape[0], max_rows - ncol)
            if take > 0:
                rows.append(X[:take])
                ncol += take
            if ncol >= max_rows:
                break
        if ncol == 0:
            raise RuntimeError("No samples collected to initialize bias.")
        Xcat = torch.cat(rows, dim=0)
        b = geometric_median(Xcat) if method == "geomedian" else Xcat.mean(dim=0)
        self.b.copy_(b)

    def _apply_topk(self, C: torch.Tensor) -> torch.Tensor:
        if self.topk is None or self.topk <= 0 or self.topk >= C.size(1):
            return C
        vals, idx = torch.topk(C, self.topk, dim=1)
        mask = torch.zeros_like(C).scatter(1, idx, 1.0)
        return C * mask

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        C = torch.relu((X - self.b) @ self.W)
        if self.sparsity_mode in {"topk", "l1+topk"}:
            C = self._apply_topk(C)
        return C

    def decode(self, C: torch.Tensor) -> torch.Tensor:
        return C @ self.W.T + self.b

    def forward(self, X: torch.Tensor):
        C = self.encode(X)
        Xh = self.decode(C)
        return Xh, C

    def loss(self, X: torch.Tensor):
        Xh, C = self.forward(X)
        mse = F.mse_loss(Xh, X)
        l1 = C.abs().mean()
        if self.sparsity_mode == "l1":
            loss = mse + self.l1_coeff * l1
        elif self.sparsity_mode == "topk":
            loss = mse
        else:
            loss = mse + self.l1_coeff * l1
        return loss, {"mse": mse.detach(), "l1": l1.detach()}


# =========================
# 3) Streaming activation pipeline (IterableDataset → DataLoader → ShuffleBuffer)
# =========================
class ActivationStreamDataset(IterableDataset):
    """
    Streams activation *rows* [m, d] per yield, one utterance at a time.
    You provide:
      - hook_fn(model, batch) -> activations [B, T, d] (or [T, d] if B=1)
      - select_mask_fn(batch) -> bool mask [B, T] (True = keep row)  (e.g., text steps only)
      - scaler: apply MSN normalization here so the trainer sees normalized data
    """
    def __init__(self, base_dataset, model, hook_fn: Callable[[nn.Module, Dict[str, Any]], torch.Tensor], select_mask_fn: Callable[[Dict[str, Any]], torch.Tensor], scaler: Optional[FixedMSNScalar] = None, rows_per_chunk: int = 2048, device: str = "cuda"):
        super().__init__()
        self.base_dataset = base_dataset
        self.model = model
        self.hook_fn = hook_fn
        self.select_mask_fn = select_mask_fn
        self.scaler = scaler
        self.rows_per_chunk = rows_per_chunk
        self.device = torch.device(device)

    def __iter__(self):
        loader = DataLoader(self.base_dataset, batch_size=1, shuffle=False, num_workers=0)
        for batch in loader:
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            acts_BTd = self.hook_fn(self.model, batch)          # float32 [B, T, d]
            mask_BT = self.select_mask_fn(batch).to(acts_BTd.device)  # bool [B, T]
            X = acts_BTd[mask_BT]                               # [N_rows, d]
            if X.numel() == 0:
                continue
            if self.scaler is not None:
                X = self.scaler.transform(X)
            for i in range(0, X.shape[0], self.rows_per_chunk):
                yield X[i:i+self.rows_per_chunk].cpu()


class ShuffleBuffer:
    """
    Fixed-capacity shuffle buffer stored on CPU (float16 by default).
    Uses append-then-reservoir replacement; sample() returns random rows.
    """
    def __init__(self, d: int, capacity: int = 500_000, dtype=torch.float16, pin_memory: bool = True):
        self.d = d
        self.capacity = capacity
        self.dtype = dtype
        self.buf = torch.empty((capacity, d), dtype=dtype, pin_memory=pin_memory)
        self.size = 0
        self._rng = random.Random(123)

    def push(self, X_chunk: torch.Tensor):
        X_chunk = X_chunk.to(self.buf.dtype)
        n = X_chunk.shape[0]
        if n == 0:
            return
        space = self.capacity - self.size
        start = 0
        if space > 0:
            take = min(space, n)
            self.buf[self.size:self.size+take].copy_(X_chunk[:take])
            self.size += take
            start = take
        for i in range(start, n):
            j = self._rng.randrange(0, self.capacity)
            self.buf[j].copy_(X_chunk[i])

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, self.size, (batch_size,), dtype=torch.long)
        return self.buf[idx]


@torch.no_grad()
def fill_buffer_until(buffer: ShuffleBuffer, stream_loader: Iterable[torch.Tensor], min_rows: int):
    it = iter(stream_loader)
    while buffer.size < min_rows:
        try:
            X = next(it)
        except StopIteration:
            it = iter(stream_loader)
            X = next(it)
        buffer.push(X)


def streaming_batches(stream_loader: Iterable[torch.Tensor], buffer: ShuffleBuffer, batch_size: int):
    it = iter(stream_loader)
    while True:
        while not buffer.can_sample(batch_size):
            try:
                X = next(it)
            except StopIteration:
                it = iter(stream_loader)
                X = next(it)
            buffer.push(X)
        yield buffer.sample(batch_size)


# =========================
# 4) Metrics & Validation
# =========================
@torch.no_grad()
def l0_count(C: torch.Tensor) -> float:
    return float((C != 0).sum(dim=1).float().mean())

@torch.no_grad()
def l0_frac(C: torch.Tensor) -> float:
    return float((C != 0).float().mean())

@torch.no_grad()
def validate(sae: SAE, val_iter: Iterable[torch.Tensor], device=None):
    sae.eval()
    device = device or next(sae.parameters()).device
    tot_loss = tot_mse = tot_l1 = tot_l0 = tot_l0f = 0.0
    nb = 0
    for X in val_iter:
        X = X.to(device, non_blocking=True).to(torch.float32)
        loss, logs = sae.loss(X)
        _, C = sae(X)
        tot_loss += loss.item()
        tot_mse  += logs["mse"].item()
        tot_l1   += logs["l1"].item()
        tot_l0   += l0_count(C)
        tot_l0f  += l0_frac(C)
        nb += 1
    if nb == 0:
        return None
    return {"loss": tot_loss/nb, "mse": tot_mse/nb, "l1": tot_l1/nb, "L0": tot_l0/nb, "L0_frac": tot_l0f/nb}


# =========================
# 5) Trainer (streaming, RAM-safe)
# =========================
def train_sae_streaming(
    sae: SAE,
    stream_loader: Iterable[torch.Tensor],  # yields [m, d] CPU chunks (normalized)
    *,
    epochs: int = 5,
    steps_per_epoch: int = 1000,
    batch_size: int = 512,
    lr: float = 2e-3,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
    buffer_capacity: int = 500_000,
    warmup_rows: int = 100_000,
    log_every: int = 100,
    val_iter: Optional[Iterable[torch.Tensor]] = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    # Peek one chunk to infer d
    tmp_iter = iter(stream_loader)
    first = next(tmp_iter)  # [m, d]
    d = first.shape[1]
    buffer = ShuffleBuffer(d=d, capacity=buffer_capacity, dtype=torch.float16, pin_memory=True)
    buffer.push(first)
    fill_buffer_until(buffer, stream_loader, min_rows=warmup_rows)

    batch_iter = streaming_batches(stream_loader, buffer, batch_size)

    for ep in range(1, epochs + 1):
        sae.train()
        running = {"loss": 0.0, "mse": 0.0, "l1": 0.0, "spars": 0.0}
        for step in range(1, steps_per_epoch + 1):
            X = next(batch_iter).to(device, non_blocking=True).to(torch.float32)
            loss, logs = sae.loss(X)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(sae.parameters(), grad_clip)
            opt.step()

            with torch.no_grad():
                _, C = sae(X)
                running["loss"] += loss.item()
                running["mse"]  += logs["mse"].item()
                running["l1"]   += logs["l1"].item()
                running["spars"]+= float((C != 0).float().mean())

            if step % 50 == 0:
                sae.enforce_unit_norm()

            if step % log_every == 0:
                denom = log_every
                print(f"[ep {ep} step {step}] loss={running['loss']/denom:.5f} | mse={running['mse']/denom:.5f} | "
                      f"l1={running['l1']/denom:.5f} | spars(frac)={running['spars']/denom:.4f}")
                running = {"loss": 0.0, "mse": 0.0, "l1": 0.0, "spars": 0.0}

        sae.enforce_unit_norm()

        if val_iter is not None:
            logs = validate(sae, val_iter, device=device)
            if logs:
                print(f"[val ep {ep}] loss={logs['loss']:.5f} | mse={logs['mse']:.5f} | l1={logs['l1']:.5f} | "
                      f"L0={logs['L0']:.2f} | L0_frac={logs['L0_frac']:.4f}")
        else:
            print(f"[epoch {ep} done] (no val)")

    return sae


# =========================
# 6) EXAMPLE STUBS you should replace with your model specifics
# =========================

# -- Your hook: run model and return activations [B, T, d] at the chosen layer --
def hook_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    """
    Replace with your forward pass + cached hidden states.
    For demo, we synthesize activations (drop this in real use).
    """
    B = 1
    T = int(batch["length"])
    d = model["d"]
    # fake "last-layer residuals"
    U = model["U"].to(next(iter(batch.values())).device)
    z = torch.randn(T, U.shape[1], device=U.device)
    acts = z @ U.T  # [T, d]
    return acts.unsqueeze(0)  # [B, T, d]

# -- Your selection mask: keep text steps only (drop speech prefix) --
def select_mask_fn(batch: Dict[str, Any]) -> torch.Tensor:
    """
    Return bool mask [B, T] where True = keep row (text time-steps).
    Here we assume batch has 'speech_len' and 'length' (T). Keep indices >= speech_len.
    """
    B = 1
    T = int(batch["length"])
    speech_len = int(batch["speech_len"])
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, speech_len:] = True
    return mask


# =========================
# 7) Demo wiring (toy dataset). Replace with your utterance dataset.
# =========================
class ToyUtteranceDataset(IterableDataset):
    def __init__(self, n_utts=2000, T_min=64, T_max=256, speech_frac=0.3):
        super().__init__()
        self.items = []
        rng = random.Random(0)
        for _ in range(n_utts):
            T = rng.randint(T_min, T_max)
            speech_len = int(speech_frac * T)
            self.items.append({"length": T, "speech_len": speech_len})
    def __iter__(self):
        for itm in self.items:
            yield itm

def make_stream(dataset, model, scaler=None, rows_per_chunk=4096, device="cuda"):
    ds = ActivationStreamDataset(
        base_dataset=dataset,
        model=model,
        hook_fn=hook_fn,
        select_mask_fn=select_mask_fn,   # text-only positions
        scaler=scaler,                   # apply MSN here so trainer sees normalized data
        rows_per_chunk=rows_per_chunk,
        device=device,
    )
    # DataLoader over IterableDataset returns already-yielded tensors
    return DataLoader(ds, batch_size=None, num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True)

@torch.no_grad()
def iter_for_scalar(stream_loader, max_rows=200_000):
    """Flatten chunks into a bounded row iterator for fitting the MSN scalar or bias init."""
    collected = 0
    for X in stream_loader:
        n = X.shape[0]
        take = min(n, max_rows - collected)
        if take > 0:
            yield X[:take]
            collected += take
        if collected >= max_rows:
            break

# =========================
# 8) Main
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Your model placeholder (replace with the real model & real hook) ---
    d = 1024
    k = 4096
    model_stub = {"d": d, "U": torch.randn(d, 64)}  # demo only

    # --- Datasets (utterance-wise) ---
    train_utts = ToyUtteranceDataset(n_utts=3000)
    val_utts   = ToyUtteranceDataset(n_utts=300,  T_min=64, T_max=256)

    # --- Build streaming loaders (utterance → chunks of normalized rows) ---
    # First pass with scaler=None to fit MSN on RAW acts; then rebuild with scaler applied.
    pre_stream = make_stream(train_utts, model_stub, scaler=None, device=device)

    # Fit MSN scalar on generation-time (text) activations only
    scaler = FixedMSNScalar()
    scaler.fit_from_stream(iter_for_scalar(pre_stream, max_rows=500_000), device=device, max_rows=500_000)
    print("MSN scalar s =", float(scaler.s))

    # Rebuild loaders with normalization applied
    train_stream = make_stream(train_utts, model_stub, scaler=scaler, device=device)
    val_stream   = make_stream(val_utts,   model_stub, scaler=scaler, device=device)

    # SAE
    sae = SAE(d_in=d, n_features=k, sparsity_mode="l1+topk", l1_coeff=5e-4, topk=32, unit_norm_dict=True)

    # Geometric-median bias init on normalized activations (text steps only)
    with torch.no_grad():
        sae.enforce_unit_norm()
        sae.init_bias_from_stream(iter_for_scalar(train_stream, max_rows=200_000), device=device, max_rows=200_000, method="geomedian")

    # Train (streaming, RAM-safe). Validation logs L0 & L0_frac.
    trained = train_sae_streaming(
        sae,
        stream_loader=train_stream,
        epochs=5,
        steps_per_epoch=1000,
        batch_size=512,
        lr=2e-3,
        grad_clip=1.0,
        device=device,
        buffer_capacity=500_000,
        warmup_rows=100_000,
        log_every=100,
        val_iter=iter_for_scalar(val_stream, max_rows=20_000),  # small bounded iterator for val
    )

    # Quick sanity: sample a chunk and compute sparsity
    with torch.no_grad():
        sample = next(iter(train_stream)).to(device).to(torch.float32)
        _, C = trained(sample)
        print("Sample L0(count):", float((C != 0).sum(dim=1).float().mean()), " | L0_frac:", float((C != 0).float().mean()))

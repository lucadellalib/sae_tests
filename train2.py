# sae_all_in_one_with_val.py
# Python 3.10+, PyTorch 2.x

import math
from typing import Callable, Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 0) Fixed MSN scalar (unit mean-squared norm)
# =========================
class FixedMSNScalar:
    def __init__(self):
        self.s: Optional[torch.Tensor] = None

    @torch.no_grad()
    def fit_from_stream(self, activations_iter: Iterable[torch.Tensor], device=None, max_samples: int = 500_000):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total = 0.0
        count = 0
        for x in activations_iter:
            x = x.to(device, non_blocking=True)
            B, d = x.shape
            total += (x.pow(2).sum(dim=1) / float(d)).sum().item()
            count += B
            if count >= max_samples:
                break
        if count == 0:
            raise RuntimeError("No activations to fit MSN scalar.")
        mean_msn = total / count
        self.s = torch.tensor(mean_msn, device=device).sqrt().clamp_min(1e-12)

    @torch.no_grad()
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.s is not None, "Call fit_from_stream() first."
        return x / self.s

    @torch.no_grad()
    def inverse_transform(self, x_norm: torch.Tensor) -> torch.Tensor:
        assert self.s is not None, "Call fit_from_stream() first."
        return x_norm * self.s

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
# 2) Tied Sparse Autoencoder (W, b)
# =========================
class SAE(nn.Module):
    """
    Tied-bias, tied-weights SAE:
        encode(x) = ReLU((x - b) @ W - theta)  # theta optional thresholds (>=0)
        decode(c) = c @ W^T + b
    """
    def __init__(
        self,
        d_in: int,
        n_features: int,
        sparsity_mode: str = "l1+topk",  # "l1", "topk", "l1+topk"
        l1_coeff: float = 1e-3,
        topk: Optional[int] = 32,
        unit_norm_dict: bool = True,
        use_thresholds: bool = False,
    ):
        super().__init__()
        self.d, self.k = d_in, n_features
        self.sparsity_mode = sparsity_mode.lower()
        assert self.sparsity_mode in {"l1", "topk", "l1+topk"}
        self.l1_coeff = l1_coeff
        self.topk = topk
        self.unit_norm_dict = unit_norm_dict

        self.W = nn.Parameter(torch.randn(self.d, self.k) * (1.0 / math.sqrt(self.d)))
        self.b = nn.Parameter(torch.zeros(self.d))
        self.theta = nn.Parameter(torch.zeros(self.k)) if use_thresholds else None

    @torch.no_grad()
    def enforce_unit_norm(self):
        if not self.unit_norm_dict:
            return
        col_norms = self.W.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W.div_(col_norms)

    @torch.no_grad()
    def init_bias_from_stream(
        self,
        activations_iter: Iterable[torch.Tensor],
        device: torch.device,
        max_samples: int = 200_000,
        method: str = "geomedian",  # or "mean"
    ):
        samples, ncollected = [], 0
        for x in activations_iter:
            x = x.to(device, non_blocking=True)
            n = x.shape[0]
            take = min(n, max_samples - ncollected)
            if take > 0:
                samples.append(x[:take])
                ncollected += take
            if ncollected >= max_samples:
                break
        if ncollected == 0:
            raise RuntimeError("No samples collected to initialize bias.")
        X = torch.cat(samples, dim=0)
        b = geometric_median(X) if method == "geomedian" else X.mean(dim=0)
        self.b.copy_(b)

    def _apply_topk(self, codes: torch.Tensor) -> torch.Tensor:
        if self.topk is None or self.topk <= 0 or self.topk >= codes.size(1):
            return codes
        vals, idx = torch.topk(codes, self.topk, dim=1)
        mask = torch.zeros_like(codes).scatter(1, idx, 1.0)
        return codes * mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.b) @ self.W
        if self.theta is not None:
            z = z - self.theta.clamp_min(0.0)
        z = F.relu(z)
        if self.sparsity_mode in {"topk", "l1+topk"}:
            z = self._apply_topk(z)
        return z

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.W.T + self.b

    def forward(self, x: torch.Tensor):
        c = self.encode(x)
        x_hat = self.decode(c)
        return x_hat, c

    def loss(self, x: torch.Tensor):
        x_hat, c = self.forward(x)
        mse = F.mse_loss(x_hat, x)
        l1 = c.abs().mean()
        if self.sparsity_mode == "l1":
            loss = mse + self.l1_coeff * l1
        elif self.sparsity_mode == "topk":
            loss = mse
        else:
            loss = mse + self.l1_coeff * l1
        logs = {"mse": mse.detach(), "l1": l1.detach()}
        return loss, logs

# =========================
# 3) Rare-feature resampler
# =========================
class RareFeatureResampler:
    def __init__(
        self,
        sae: SAE,
        base_iter_fn: Callable[[], Iterable[torch.Tensor]],  # normalized (B, d)
        batch_size: int,
        pool_mult: int = 4,
        tau: float = 0.0,
        alpha: float = 0.5,
        ema_beta: float = 0.99,
        device: Optional[torch.device] = None,
    ):
        self.sae = sae
        self.base_iter_fn = base_iter_fn
        self.batch_size = batch_size
        self.pool_mult = pool_mult
        self.tau = tau
        self.alpha = alpha
        self.ema_beta = ema_beta
        self.device = device or next(sae.parameters()).device
        self.fire_rate = torch.full((sae.k,), 1e-6, device=self.device)
        self._raw_iter = None

    def _ensure_iter(self):
        if self._raw_iter is None:
            self._raw_iter = iter(self.base_iter_fn())

    @torch.no_grad()
    def _next_raw(self):
        self._ensure_iter()
        try:
            return next(self._raw_iter)
        except StopIteration:
            self._raw_iter = iter(self.base_iter_fn())
            return next(self._raw_iter)

    @torch.no_grad()
    def step(self) -> torch.Tensor:
        pool = []
        need = self.pool_mult * self.batch_size
        while len(pool) < need:
            pool.append(self._next_raw())
        X = torch.cat(pool, dim=0).to(self.device, non_blocking=True)  # (P, d)
        _, C = self.sae.forward(X)
        active = (C > self.tau)
        fr = active.float().mean(dim=0)
        self.fire_rate.mul_(self.ema_beta).add_(fr * (1.0 - self.ema_beta))
        eps = 1e-6
        w = (self.fire_rate + eps).pow(-self.alpha)              # (k,)
        scores = (active.float() * w.unsqueeze(0)).sum(dim=1)    # (P,)
        idx = torch.topk(scores, k=self.batch_size, dim=0).indices
        return X[idx]

# =========================
# 4) Metrics & Validation
# =========================
@torch.no_grad()
def l0_count(codes: torch.Tensor) -> float:
    """Avg number of nonzero features per sample (L0)."""
    return float((codes != 0).sum(dim=1).float().mean())

@torch.no_grad()
def l0_fraction(codes: torch.Tensor) -> float:
    """Avg fraction of active features per sample (L0/k)."""
    return float((codes != 0).float().mean())

@torch.no_grad()
def validate(sae: SAE, val_iter: Iterable[torch.Tensor], device: Optional[torch.device] = None):
    sae.eval()
    device = device or next(sae.parameters()).device
    total_loss = total_mse = total_l1 = total_l0 = total_l0frac = 0.0
    nb = 0
    for x in val_iter:
        x = x.to(device, non_blocking=True)
        loss, logs = sae.loss(x)
        _, c = sae.forward(x)
        total_loss += loss.item()
        total_mse  += logs["mse"].item()
        total_l1   += logs["l1"].item()
        total_l0   += l0_count(c)
        total_l0frac += l0_fraction(c)
        nb += 1
    if nb == 0:
        return {}
    return {
        "loss": total_loss/nb,
        "mse": total_mse/nb,
        "l1": total_l1/nb,
        "L0": total_l0/nb,
        "L0_frac": total_l0frac/nb,
    }

# =========================
# 5) Training (with resampling + validation)
# =========================
def train_with_resampling(
    sae: SAE,
    base_iter_fn: Callable[[], Iterable[torch.Tensor]],   # normalized train stream
    val_iter_fn: Optional[Callable[[], Iterable[torch.Tensor]]] = None,  # normalized val stream
    *,
    epochs: int = 5,
    steps_per_epoch: int = 1000,
    batch_size: int = 512,
    lr: float = 2e-3,
    grad_clip: Optional[float] = 1.0,
    pool_mult: int = 4,
    tau: float = 0.0,
    alpha: float = 0.5,
    ema_beta: float = 0.99,
    renorm_every: int = 50,
    log_every: int = 100,
    device: Optional[torch.device] = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    resampler = RareFeatureResampler(
        sae, base_iter_fn, batch_size=batch_size, pool_mult=pool_mult,
        tau=tau, alpha=alpha, ema_beta=ema_beta, device=device
    )

    for ep in range(1, epochs + 1):
        sae.train()
        running = {"loss": 0.0, "mse": 0.0, "l1": 0.0, "spars": 0.0}
        for step in range(1, steps_per_epoch + 1):
            x = resampler.step()
            loss, logs = sae.loss(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(sae.parameters(), grad_clip)
            opt.step()

            with torch.no_grad():
                _, c = sae.forward(x)
                running["loss"] += loss.item()
                running["mse"]  += logs["mse"].item()
                running["l1"]   += logs["l1"].item()
                running["spars"]+= float((c != 0).float().mean())

            if renorm_every and (step % renorm_every == 0):
                sae.enforce_unit_norm()

            if log_every and (step % log_every == 0):
                denom = log_every
                print(f"[ep {ep} step {step}] "
                      f"loss={running['loss']/denom:.5f} | "
                      f"mse={running['mse']/denom:.5f} | "
                      f"l1={running['l1']/denom:.5f} | "
                      f"spars(frac)={running['spars']/denom:.4f}")
                running = {"loss": 0.0, "mse": 0.0, "l1": 0.0, "spars": 0.0}

        sae.enforce_unit_norm()

        # ---- validation with L0 metrics ----
        if val_iter_fn is not None:
            val_logs = validate(sae, val_iter_fn(), device=device)
            if val_logs:
                print(f"[val ep {ep}] loss={val_logs['loss']:.5f} | mse={val_logs['mse']:.5f} | "
                      f"l1={val_logs['l1']:.5f} | L0={val_logs['L0']:.2f} | L0_frac={val_logs['L0_frac']:.4f}")
        else:
            print(f"[epoch {ep} done] (no val)")

    return sae

# =========================
# 6) Eval (MSE in normalized space)
# =========================
@torch.inference_mode()
def eval_recon_mse(sae: SAE, activations_iter: Iterable[torch.Tensor], device: Optional[torch.device] = None) -> float:
    device = device or next(sae.parameters()).device
    sae.eval()
    total, n_rows = 0.0, 0
    for x in activations_iter:
        x = x.to(device, non_blocking=True)
        recon, _ = sae(x)
        total += F.mse_loss(recon, x, reduction="sum").item()
        n_rows += x.shape[0]
    return total / max(n_rows, 1)

# =========================
# 7) Example activations (replace with real model hooks)
# =========================
def toy_activation_iterator(n_batches=1000, batch_size=512, d=1024, device="cpu"):
    torch.manual_seed(0)
    U = torch.randn(d, 64, device=device)
    for _ in range(n_batches):
        z = torch.randn(batch_size, 64, device=device)
        x = z @ U.T
        idx = torch.randint(0, d, (batch_size, 5), device=device)
        bumps = torch.zeros_like(x).scatter(1, idx, 3.0 * torch.rand(batch_size, 5, device=device))
        yield (x + bumps).to("cpu")  # emulate CPU loader → GPU training

# =========================
# 8) All together (demo)
# =========================
if __name__ == "__main__":
    d = 1024
    k = 4096
    batch_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Raw streams
    def raw_train_iter():
        return toy_activation_iterator(n_batches=2000, batch_size=batch_size, d=d, device=device)

    def raw_val_iter():
        return toy_activation_iterator(n_batches=200, batch_size=batch_size, d=d, device=device)

    # (A) Fit MSN scalar on raw training activations
    scaler = FixedMSNScalar()
    scaler.fit_from_stream(raw_train_iter(), device=device, max_samples=500_000)
    print("MSN scalar s =", float(scaler.s))

    # (B) Normalized wrappers
    def norm_train_iter():
        for x in raw_train_iter():
            yield scaler.transform(x)

    def norm_val_iter():
        for x in raw_val_iter():
            yield scaler.transform(x)

    # (C) SAE
    sae = SAE(
        d_in=d,
        n_features=k,
        sparsity_mode="l1+topk",
        l1_coeff=5e-4,
        topk=32,
        unit_norm_dict=True,
        use_thresholds=False,
    )

    # (D) Geometric-median bias init (on normalized acts)
    with torch.no_grad():
        sae.enforce_unit_norm()
        sae.init_bias_from_stream(
            activations_iter=norm_train_iter(),
            device=device,
            max_samples=200_000,
            method="geomedian",
        )

    # (E) Train + validate (validation logs L0 and L0_frac)
    trained = train_with_resampling(
        sae,
        base_iter_fn=norm_train_iter,
        val_iter_fn=norm_val_iter,
        epochs=5,
        steps_per_epoch=1000,
        batch_size=batch_size,
        lr=2e-3,
        grad_clip=1.0,
        pool_mult=4,
        tau=0.0,
        alpha=0.5,
        ema_beta=0.99,
        renorm_every=50,
        log_every=100,
        device=device,
    )

    # (F) Quick eval (MSE in normalized space)
    mse = eval_recon_mse(trained, activations_iter=(x for x in norm_val_iter()), device=device)
    print(f"Eval recon MSE (normalized space): {mse:.6f}")

    # (G) Example: codes + optional raw-space recon
    with torch.no_grad():
        batch_raw = next(raw_val_iter())
        x_norm = scaler.transform(batch_raw).to(device)
        recon_norm, codes = trained(x_norm)
        recon_raw = scaler.inverse_transform(recon_norm)
        print("Val sample L0(count):", l0_count(codes), " | L0_frac:", l0_fraction(codes))

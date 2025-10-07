from typing import Tuple, Union, Sequence, List, Optional
import numpy as np
import torch


class SingleProbeSignedSteererSparse:
    """Single-stream sparse steerer with one logistic probe and a separate scaling method.

    Flow per step:
      1) Build rolling window features on the selected subset (mean 'strength' or 'counts').
      2) Score the probe: p = sigmoid(w @ x + b).
      3) If p >= threshold, compute multipliers with `compute_multipliers(...)` and
         apply only on PRESENT selected codes of the current frame.

    Args:
      vocab_size: Global code id range [0, vocab_size).
      selected_idx: Global code ids to track & steer (r_sel,).
      weights: Probe weights (r_sel,) or (vocab_size,) — full will be sliced to selected_idx.
      intercept: Probe bias (scalar).
      threshold: Probability threshold to trigger intervention.
      mode: 'strength' or 'counts' (feature definition).
      window_frames: Rolling window size W in frames.
      eta: Step strength for scaling (larger = stronger change).
      m_min: Lower clamp for multipliers.
      m_max: Upper clamp for multipliers.
      device: Torch device or None for CPU.
      dtype: Floating dtype.
    """

    def __init__(self,
                 vocab_size: int,
                 selected_idx: Union[Sequence[int], np.ndarray, torch.Tensor],
                 weights: Union[np.ndarray, torch.Tensor],
                 intercept: float,
                 threshold: float,
                 *,
                 mode: str = "strength",
                 window_frames: int = 10,
                 eta: float = 0.3,
                 m_min: float = 0.2,
                 m_max: float = 3.0,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: torch.dtype = torch.float32):
        self.vocab_size = int(vocab_size)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.win = int(window_frames); assert self.win > 0
        assert mode in ("strength", "counts")
        self.mode = mode

        sel = torch.as_tensor(selected_idx, device=self.device, dtype=torch.long).reshape(-1)
        self.sel = torch.sort(sel).values
        self.r_sel = int(self.sel.numel())

        w = torch.as_tensor(weights, dtype=self.dtype, device=self.device).reshape(-1)
        if w.numel() == self.vocab_size:
            w = w[self.sel]  # slice to selected
        assert w.numel() == self.r_sel, "weights must align to selected_idx"
        self.w = w.contiguous()

        self.intercept = torch.tensor(float(intercept), dtype=self.dtype, device=self.device)
        self.threshold = float(threshold)

        self.eta = float(eta)
        self.m_min = float(m_min)
        self.m_max = float(m_max)

        # rolling state on selected subset (single stream)
        self.ring_strength = torch.zeros(self.win, self.r_sel, dtype=self.dtype, device=self.device)
        self.ring_hits     = torch.zeros(self.win, self.r_sel, dtype=self.dtype, device=self.device)
        self.sum_strength  = torch.zeros(self.r_sel, dtype=self.dtype, device=self.device)
        self.count_hits    = torch.zeros(self.r_sel, dtype=self.dtype, device=self.device)
        self.n_frames      = torch.tensor(0, dtype=torch.int32, device=self.device)
        self.t_ptr         = 0

    def reset(self) -> None:
        """Reset internal buffers."""
        self.ring_strength.zero_(); self.ring_hits.zero_()
        self.sum_strength.zero_();  self.count_hits.zero_()
        self.n_frames.zero_();      self.t_ptr = 0

    @torch.no_grad()
    def update(self,
               frame: Tuple[Union[np.ndarray, torch.Tensor, Sequence[int]],
                            Union[np.ndarray, torch.Tensor, Sequence[float]]]
               ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float, List[int]]:
        """Process one sparse frame and optionally steer it.

        Args:
          frame: (idxs, vals) for the current timestep.

        Returns:
          (idxs_out, vals_out), prob, changed_ids
        """
        idxs_t, vals_t = frame
        dev, dt = self.device, self.dtype
        ring_pos = int(self.t_ptr % self.win)

        # map current sparse -> selected subset
        idxs = torch.as_tensor(idxs_t, device=dev, dtype=torch.long).reshape(-1)
        vals = torch.as_tensor(vals_t, device=dev, dtype=dt).reshape(-1)
        pos_est = torch.searchsorted(self.sel, idxs)
        valid = (pos_est < self.r_sel) & (self.sel[pos_est] == idxs)
        pos_sel = pos_est[valid]
        vals_sel = vals[valid]

        cur_vals = torch.zeros(self.r_sel, dtype=dt, device=dev)
        if pos_sel.numel():
            cur_vals.index_add_(0, pos_sel, vals_sel)
        present = torch.zeros(self.r_sel, dtype=torch.bool, device=dev)
        if pos_sel.numel():
            present[pos_sel] = True

        # ring update
        old_vals = self.ring_strength[ring_pos]
        old_hits = self.ring_hits[ring_pos].bool()
        self.sum_strength += (cur_vals - old_vals)
        self.count_hits   += (present.float() - old_hits.float())
        self.ring_strength[ring_pos] = cur_vals
        self.ring_hits[ring_pos]     = present.float()
        if self.n_frames.item() < self.win:
            self.n_frames += 1

        # features and probability
        denom = torch.clamp(self.n_frames.to(dt), min=1.0)
        feats = (self.sum_strength / denom) if self.mode == "strength" else (self.count_hits / denom)
        p = float(torch.sigmoid(self.w @ feats + self.intercept).item())

        # not triggered or nothing present → return as-is
        if p < self.threshold or pos_sel.numel() == 0:
            self.t_ptr += 1
            return (idxs, vals), p, []

        # compute per-code multipliers and apply on present entries
        m = self.compute_multipliers(present=present, prob=p)  # (r_sel,)
        cur_new = cur_vals.clone()
        cur_new[present] = cur_new[present] * m[present]

        # keep features consistent (strength mode)
        if self.mode == "strength":
            delta = cur_vals - cur_new
            self.sum_strength -= delta
            self.ring_strength[ring_pos] -= delta

        # write back to original sparse vectors (only selected entries)
        vals_out = vals.clone()
        changed_mask = (cur_new[pos_sel] != vals_sel)
        changed_ids = self.sel[pos_sel[changed_mask]].tolist() if changed_mask.any() else []
        if changed_ids:
            vals_out[valid] = cur_new[pos_sel]

        self.t_ptr += 1
        return (idxs, vals_out), p, [int(i) for i in changed_ids]

    # ------------------------- scaling method -------------------------

    def compute_multipliers(self, present: torch.Tensor, prob: float) -> torch.Tensor:
        """Compute per-code multipliers for the current frame (signed & normalized).

        Multiplier rule:
            m_j = clamp( exp( -eta * gamma * w_j / Z ), m_min, m_max )

        Where:
          - gamma = normalized exceedance = ((prob - threshold) / (1 - threshold)) clipped to [0, 1]
          - Z = max(|w_j|) over PRESENT selected codes (avoids over-steering)
          - positive w_j → m_j < 1 (attenuate), negative w_j → m_j > 1 (boost)

        Args:
          present: Boolean mask over selected codes that are present in the current frame.
          prob: Current probe probability p (scalar).

        Returns:
          Tensor of shape (r_sel,) with per-code multipliers.
        """
        if prob <= self.threshold or not bool(present.any()):
            return torch.ones(self.r_sel, dtype=self.dtype, device=self.device)

        eps = torch.finfo(self.dtype).eps
        gamma = max(0.0, min(1.0, (prob - self.threshold) / max(1.0 - self.threshold, float(eps))))

        w_pres = self.w[present]
        z = float(w_pres.abs().max().item()) if w_pres.numel() else 1.0
        z = max(z, 1e-8)

        m = torch.exp(-self.eta * gamma * (self.w / z))
        return torch.clamp(m, self.m_min, self.m_max)

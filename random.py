from typing import Optional, Tuple, List, Iterable, Union, Literal
import numpy as np
import torch

# --- type aliases (lowercase) ---
trigger_mode_t = Literal["any", "max", "sum"]
agg_mode_t     = Literal["prod", "min", "logprod"]
intervene_t    = Literal["attenuation", "softcap", "gd"]


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


class BatchedRuntimeSteererVec:
    """
    Batched multi-probe streaming steerer with a fully vectorized tensor ring buffer.

    Modes
    -----
    - 'attenuation': shrink top-L positive contributors by alpha (union across triggered probes)
    - 'softcap'   : Hill saturator per-probe on positive contributions, aggregated across probes
    - 'gd'        : projected gradient step on *current-frame* codes to reduce undesired logits

    Features
    --------
    - Windowed features over selected atoms (union across probes), either:
        * 'strength': per-frame magnitudes (summed across window then / n_frames)
        * 'counts'  : per-frame presence 0/1 (summed then / n_frames)
    - Fully-vectorized ring buffer: (batch, window, atoms), O(batch·atoms) per step + O(nnz) scatter
    - Multi-probe: weights (probes, atoms), intercepts (probes,), thresholds (probes,)

    Notes
    -----
    - For 'counts', attenuation/GD won’t change the feature this frame (presence is binary),
      but they still change returned magnitudes (if you use them downstream).
    """

    def __init__(self,
                 batch_size: int,
                 selected_atoms: Union[np.ndarray, torch.Tensor],  # (atoms,)
                 weights: Union[np.ndarray, torch.Tensor],         # (atoms,) or (probes, atoms)
                 intercepts: Union[float, np.ndarray, torch.Tensor],
                 thresholds: Union[float, np.ndarray, torch.Tensor],
                 *,
                 mode: str = "strength",            # 'strength' or 'counts'
                 window_frames: int = 100,
                 # attenuation:
                 alpha: float = 0.5,
                 l_max: int = 16,
                 risk_proportional: bool = True,
                 topk_per_probe: Optional[int] = None,
                 # trigger & softcap:
                 trigger_mode: trigger_mode_t = "any",
                 intervene_mode: intervene_t = "attenuation",
                 softcap_q: float = 0.95,
                 softcap_lam: float = 0.75,
                 softcap_p: float = 2.0,
                 aggregate: agg_mode_t = "prod",
                 use_guard: bool = False,
                 guard_beta: float = 1.0,
                 # caps (per-atom per-frame value caps; only used in 'strength' mode):
                 caps: Optional[Union[np.ndarray, torch.Tensor]] = None,  # (atoms,)
                 # GD settings:
                 gd_lr: float = 0.2,
                 gd_steps: int = 1,
                 gd_topk: Optional[int] = None,       # restrict to top-|grad| present atoms
                 gd_project_nonneg: bool = True,
                 gd_use_only_triggered: bool = True,
                 # torch settings:
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize a batched, vectorized runtime steerer.

        Parameters
        ----------
        batch_size : int
            Number of parallel streams processed at once.
        selected_atoms : (atoms,) array-like of int
            Shared, **sorted** vocabulary of atom IDs to track/steer (union across probes).
            Only atoms listed here are aggregated and modifiable at runtime.
        weights : (atoms,) or (probes, atoms) array-like of float
            Probe weight vector(s) aligned to `selected_atoms`.
            Positive weights are interpreted as **harmful**, negatives as **protective**.
        intercepts : float or (probes,) array-like of float
            Logistic bias(es) for each probe. Scalar is broadcast to all probes.
        thresholds : float or (probes,) array-like of float
            Probability threshold(s) used for trigger logic. Scalar broadcasts to all probes.
        mode : {'strength', 'counts'}, default 'strength'
            Determines how features are computed over the window:
              - 'strength': window-average of per-frame magnitudes (sum / n_frames).
              - 'counts'  : window-average presence rate (0/1 per frame).
            Use the same mode you used to train the probes.
        window_frames : int, default 100
            Rolling window length in frames for tail feature computation.
            Larger → smoother/laggy; smaller → more reactive.
        alpha : float in (0,1], default 0.5
            Attenuation factor for `intervene_mode='attenuation'`. new_value = alpha * old_value.
        l_max : int, default 16
            Max atoms per triggered probe to act on (used by attenuation and selection union).
            With `risk_proportional=True`, the actual per-probe L scales with (p - thr).
        risk_proportional : bool, default True
            If True, scale the number of acted-on atoms per probe with exceedance over threshold.
        topk_per_probe : Optional[int], default None
            Additional cap on per-probe footprint when selecting by contribution.
            If provided, keep at most this many atoms per triggered probe.
        trigger_mode : {'any','max','sum'}, default 'any'
            Policy to decide if a stream triggers intervention:
              - 'any': trigger if **any** probe p >= threshold_p.
              - 'max': trigger if max(prob) >= max(thresholds).
              - 'sum': trigger if sum(max(prob - thr, 0) over probes) > 0.
        intervene_mode : {'attenuation','softcap','gd'}, default 'attenuation'
            Runtime action applied to the **current frame**:
              - 'attenuation': uniform shrink (alpha) on the union of top contributors.
              - 'softcap'   : smooth saturation via Hill caps, aggregated over probes.
              - 'gd'        : projected gradient descent step on codes to reduce logits.
        softcap_q : float in (0,1), default 0.95
            Per-probe quantile used to estimate the cap c from positive contributions.
            Higher q → gentler (larger cap).
        softcap_lam : float > 0, default 0.75
            Scale multiplier for the cap: c = lam * quantile_q(positives).
            Lower lam → stronger clamping.
        softcap_p : float >= 1, default 2.0
            Hill exponent controlling softcap sharpness. 1 = very soft, 2–4 = sharper.
        aggregate : {'prod','min','logprod'}, default 'prod'
            How to combine per-probe multipliers into one per-atom multiplier:
              - 'prod'    : softer, order-independent.
              - 'min'     : most aggressive (take strongest shrink).
              - 'logprod' : like 'prod' but numerically stable for many probes.
        use_guard : bool, default False
            If True, skip shrinking atoms whose **protective** mass outweighs harmful:
            allow only if sum(relu(w*x)) > guard_beta * sum(relu(-w*x)).
        guard_beta : float >= 0, default 1.0
            Threshold factor for the protective guard (see `use_guard`).
        caps : Optional[(atoms,) array-like of float], default None
            Per-atom **per-frame** safe caps (only used in 'strength' mode) applied after intervention.
            Typical source: percentiles from negative-tail windows.
        gd_lr : float, default 0.2
            Step size for `intervene_mode='gd'`.
        gd_steps : int, default 1
            Number of GD iterations on the current frame (1–3 usually enough).
        gd_topk : Optional[int], default None
            If set, restrict GD updates to the top-|grad| **present** atoms per stream.
        gd_project_nonneg : bool, default True
            If True, clamp codes to >= 0 after each GD step (ReLU-like).
        gd_use_only_triggered : bool, default True
            If True, sum gradients **only** from probes at/over threshold on that stream.
        device : Optional[str or torch.device], default None
            Torch device for internal tensors (e.g., 'cuda', 'cpu'). If None, uses CPU.
        dtype : torch.dtype, default torch.float32
            Floating dtype for internal tensors and math.
        """
        # core sizes / device
        self.bsz = int(batch_size)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype  = dtype
        self.win = int(window_frames); assert self.win > 0

        # selected atoms (keep *sorted* for fast mapping)
        sel = self._to_long(selected_atoms).to(self.device)
        sel_sorted, perm = torch.sort(sel)
        self.sel = sel_sorted                      # (atoms,)
        self.n_atoms = int(self.sel.numel())

        # weights aligned to selected_atoms (reorder to match our sorted sel)
        w_in = self._to_float(weights).to(self.device, self.dtype)
        if w_in.dim() == 1:
            w_in = w_in.unsqueeze(0)               # (1, atoms_original)
        assert w_in.shape[1] == perm.numel(), "weights second dim must equal selected_atoms length"
        w_in = w_in[:, perm]                       # align to sorted sel
        self.w_mat = w_in.contiguous()             # (probes, atoms)
        self.n_probes = int(self.w_mat.size(0))
        self.w_pos = torch.clamp(self.w_mat, min=0.0)  # positive (harmful) parts

        # intercepts & thresholds
        ints = self._to_float(intercepts).reshape(-1).to(self.device, self.dtype)
        self.intercepts = (ints if ints.numel() == self.n_probes else ints.expand(self.n_probes)).contiguous()
        thrs = self._to_float(thresholds).reshape(-1).to(self.device, self.dtype)
        self.thresholds = (thrs if thrs.numel() == self.n_probes else thrs.expand(self.n_probes)).contiguous()

        # features / policy
        assert mode in ("strength", "counts")
        self.mode = mode
        self.alpha = float(alpha)
        self.l_max = int(l_max)
        self.risk_proportional = bool(risk_proportional)
        self.topk_per_probe = topk_per_probe
        self.trigger_mode = trigger_mode
        self.intervene_mode = intervene_mode
        self.softcap_q = float(softcap_q)
        self.softcap_lam = float(softcap_lam)
        self.softcap_p = float(softcap_p)
        self.aggregate = aggregate
        self.use_guard = bool(use_guard)
        self.guard_beta = float(guard_beta)

        # caps (optional, strength mode)
        self.caps = None
        if caps is not None:
            c = self._to_float(caps).reshape(-1).to(self.device, self.dtype)
            assert int(c.numel()) == self.n_atoms
            self.caps = c

        # gd settings
        self.gd_lr = float(gd_lr)
        self.gd_steps = int(gd_steps)
        self.gd_topk = None if gd_topk is None else int(gd_topk)
        self.gd_project_nonneg = bool(gd_project_nonneg)
        self.gd_use_only_triggered = bool(gd_use_only_triggered)

        # ring buffers and rolling sums
        self.ring_strength = torch.zeros(self.bsz, self.win, self.n_atoms, dtype=self.dtype, device=self.device)
        self.ring_hits     = torch.zeros(self.bsz, self.win, self.n_atoms, dtype=self.dtype, device=self.device)
        self.sum_strength  = torch.zeros(self.bsz, self.n_atoms, dtype=self.dtype, device=self.device)
        self.count_hits    = torch.zeros(self.bsz, self.n_atoms, dtype=self.dtype, device=self.device)
        self.n_frames      = torch.zeros(self.bsz, dtype=torch.int32, device=self.device)
        self.t_ptr         = 0  # ring pointer

    # -------------------- public API --------------------

    def reset(self):
        self.ring_strength.zero_(); self.ring_hits.zero_()
        self.sum_strength.zero_();  self.count_hits.zero_()
        self.n_frames.zero_();      self.t_ptr = 0

    @torch.no_grad()
    def update(self,
               batch_frames: List[Tuple[Union[np.ndarray, torch.Tensor, Iterable[int]],
                                         Union[np.ndarray, torch.Tensor, Iterable[float]]]]
               ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, List[List[int]]]:
        """
        Process one step for a batch of streams.

        Parameters
        ----------
        batch_frames : list of length batch_size
            Each element is a tuple (idxs_b, vals_b) for stream b:
              - idxs_b: 1D int tensor/array of active atom IDs for the **current frame**
              - vals_b: 1D float tensor/array of per-atom magnitudes (same length as idxs_b)

        Returns
        -------
        frames_out : list of length batch_size
            Per stream output (idxs_out, vals_out) after applying the intervention to the current frame.
        p_combined : torch.Tensor, shape (batch_size,)
            Combined risk per stream for logging (e.g., max over probe probs).
        changed_lists : list of list[int]
            Per stream list of atom IDs whose values changed on this frame.
        """
        assert len(batch_frames) == self.bsz
        device, dtype = self.device, self.dtype
        ring_pos = self.t_ptr % self.win

        # --- 1) build current per-stream vectors (bsz, atoms) via vectorized scatter ---
        cur_vals = torch.zeros(self.bsz, self.n_atoms, dtype=dtype, device=device)
        present  = torch.zeros(self.bsz, self.n_atoms, dtype=torch.bool, device=device)

        flat_b, flat_pos, flat_v = [], [], []
        per_stream_meta = []  # [(idxs_orig, valid_mask, pos_sel, vals_orig), ...]
        for b, (idxs_t, vals_t) in enumerate(batch_frames):
            idxs = self._to_long(idxs_t).to(device)
            vals = self._to_float(vals_t).to(device, dtype)
            pos_est = torch.searchsorted(self.sel, idxs)                # (nb,)
            valid = (pos_est < self.n_atoms) & (self.sel[pos_est] == idxs)
            pos_sel = pos_est[valid]
            vals_sel = vals[valid]
            if pos_sel.numel() > 0:
                flat_b.append(torch.full_like(pos_sel, b, dtype=torch.long))
                flat_pos.append(pos_sel)
                flat_v.append(vals_sel)
            per_stream_meta.append((idxs, valid, pos_sel, vals))

        if flat_b:
            fb = torch.cat(flat_b); fp = torch.cat(flat_pos); fv = torch.cat(flat_v)
            cur_vals.index_put_((fb, fp), fv, accumulate=True)
            hit_acc = torch.zeros_like(cur_vals)
            hit_acc.index_put_((fb, fp), torch.ones_like(fv, dtype=dtype), accumulate=True)
            present = hit_acc > 0

        # --- 2) ring roll: subtract outgoing, add incoming (vectorized) ---
        old_vals = self.ring_strength[:, ring_pos, :]
        old_hits = self.ring_hits[:, ring_pos, :].bool()

        self.sum_strength += (cur_vals - old_vals)
        self.count_hits   += (present.float() - old_hits.float())

        self.ring_strength[:, ring_pos, :] = cur_vals
        self.ring_hits[:, ring_pos, :]     = present.float()

        not_full = self.n_frames < self.win
        self.n_frames[not_full] += 1

        # --- 3) features and risks ---
        denom = torch.clamp(self.n_frames.to(dtype).unsqueeze(1), min=1.0)
        feats = (self.sum_strength / denom) if self.mode == "strength" else (self.count_hits / denom)  # (bsz, atoms)

        logits = feats @ self.w_mat.t() + self.intercepts.unsqueeze(0)  # (bsz, probes)
        probs  = _sigmoid(logits)

        trig, p_combined = self._trigger_mask(probs)                    # trig: (bsz,)

        # --- 4) selection union (for attenuation / optional softcap restriction) ---
        union_mask = torch.zeros(self.bsz, self.n_atoms, dtype=torch.bool, device=device)
        if self.intervene_mode in ("attenuation", "softcap") and (self.l_max > 0 or (self.topk_per_probe or 0) > 0):
            contrib = torch.relu(self.w_pos.unsqueeze(0) * feats.unsqueeze(1))  # (bsz, probes, atoms)
            neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
            active_pp = (probs >= self.thresholds.unsqueeze(0)) & trig.unsqueeze(1)  # (bsz, probes)
            masked = torch.where(active_pp.unsqueeze(-1), contrib, neg_inf)          # (bsz, probes, atoms)

            if self.risk_proportional:
                eps = torch.finfo(dtype).eps
                frac = torch.clamp((probs - self.thresholds.unsqueeze(0)) /
                                   torch.clamp(1 - self.thresholds.unsqueeze(0), min=eps), 0, 1)
                l_per_probe = torch.ceil(self.l_max * frac).to(torch.long)           # (bsz, probes)
            else:
                l_per_probe = torch.full_like(probs, self.l_max, dtype=torch.long)
            if self.topk_per_probe is not None:
                l_per_probe = torch.clamp(l_per_probe, max=self.topk_per_probe)

            kcap = int(min(self.n_atoms, int(l_per_probe.max().item()) if l_per_probe.numel() else 0))
            if kcap > 0:
                vals_top, idx_top = torch.topk(masked, k=kcap, dim=2)               # (bsz, probes, kcap)
                ranks = torch.arange(kcap, device=device).view(1, 1, kcap)
                keep = (vals_top > 0) & (ranks < l_per_probe.unsqueeze(-1))         # (bsz, probes, kcap)
                sel_count = torch.zeros(self.bsz, self.n_probes, self.n_atoms, dtype=dtype, device=device)
                sel_count.scatter_add_(2, idx_top, keep.to(dtype))
                union_mask = (sel_count.sum(dim=1) > 0)                             # (bsz, atoms)

        # --- 5) per-atom multipliers if softcap ---
        mult = None  # (bsz, atoms)
        if self.intervene_mode == "softcap":
            eps = torch.finfo(dtype).eps
            contrib_all = torch.relu(self.w_mat.unsqueeze(0) * feats.unsqueeze(1))      # (bsz, probes, atoms)
            act = (probs >= self.thresholds.unsqueeze(0)) & trig.unsqueeze(1)           # (bsz, probes)

            c_cap = torch.zeros(self.bsz, self.n_probes, dtype=dtype, device=device)
            c_masked = torch.where(contrib_all > 0, contrib_all, torch.nan)
            if hasattr(torch, "nanquantile"):
                c_cap = torch.nanquantile(c_masked, self.softcap_q, dim=2)
                c_cap = torch.nan_to_num(c_cap, nan=0.0)
            else:
                pos_counts = (contrib_all > 0).sum(dim=2)
                one_m_q = max(1e-6, 1.0 - self.softcap_q)
                k = torch.clamp((pos_counts.float() * one_m_q).ceil().to(torch.long), min=1)
                neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
                masked2 = torch.where(contrib_all > 0, contrib_all, neg_inf)
                kmax = int(max(1, k.max().item()))
                topk_vals, _ = torch.topk(masked2, k=kmax, dim=2)
                b_ix = torch.arange(self.bsz, device=device).view(self.bsz, 1)
                p_ix = torch.arange(self.n_probes, device=device).view(1, self.n_probes)
                c_cap = topk_vals[b_ix, p_ix, (k - 1).clamp_max(kmax - 1)]
                c_cap = torch.where(pos_counts > 0, c_cap, torch.zeros_like(c_cap))
            c_cap = (self.softcap_lam * c_cap).clamp_min(0)

            cap_p = (c_cap + eps).unsqueeze(-1)
            cpos = (contrib_all + eps)
            if self.softcap_p == 1.0:
                m = cap_p / (cpos + cap_p)
            else:
                cap_p_pow = cap_p ** self.softcap_p
                m = cap_p_pow / (cpos ** self.softcap_p + cap_p_pow)
            m = torch.where(act.unsqueeze(-1), m, torch.ones_like(m))

            if self.aggregate == "min":
                mult = m.min(dim=1).values
            elif self.aggregate == "logprod":
                mult = torch.exp(torch.log(m.clamp_min(eps)).sum(dim=1))
            elif self.aggregate == "prod":
                mult = m.prod(dim=1)
            else:
                raise ValueError("invalid aggregate mode")

            if self.use_guard:
                harm = torch.relu(self.w_mat.unsqueeze(0) * feats.unsqueeze(1)).sum(dim=1)
                prot = torch.relu(-(self.w_mat.unsqueeze(0) * feats.unsqueeze(1))).sum(dim=1)
                allow = harm > (self.guard_beta * prot)
                mult = torch.where(allow, mult, torch.ones_like(mult))
            mult = mult.clamp(0.0, 1.0)

        # --- 6) if gd, compute per-stream new frame vector (bsz, atoms) ---
        cur_vals_new = cur_vals.clone()
        if self.intervene_mode == "gd":
            cur_vals_new = self._gd_step_on_current_frame(
                cur_vals=cur_vals,
                present=present,
                feats=feats,
                probs=probs,
                n_frames=self.n_frames,
                trig_only=self.gd_use_only_triggered,
                lr=self.gd_lr,
                steps=self.gd_steps,
                topk=self.gd_topk,
                caps=self.caps if (self.mode == "strength") else None,
                project_nonneg=self.gd_project_nonneg,
            )

        # --- 7) write back to ragged outputs & track changed atoms (nnz-linear loop) ---
        frames_out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        changed_lists: List[List[int]] = []

        for b, (idxs_orig, valid_b, pos_sel_b, vals_orig) in enumerate(per_stream_meta):
            vals_out = vals_orig.clone()
            changed_atoms: List[int] = []

            if pos_sel_b.numel() > 0 and bool(trig[b].item()):
                if self.intervene_mode == "attenuation":
                    apply_mask = union_mask[b, pos_sel_b] if union_mask.any() \
                                 else torch.zeros_like(pos_sel_b, dtype=torch.bool, device=device)
                    new_sel_vals = vals_orig[valid_b].clone()
                    new_sel_vals[apply_mask] = new_sel_vals[apply_mask] * self.alpha
                    if self.mode == "strength" and self.caps is not None:
                        new_sel_vals = torch.minimum(new_sel_vals, self.caps[pos_sel_b])
                    vals_out[valid_b] = new_sel_vals
                    cur_vals_new[b, pos_sel_b[apply_mask]] = new_sel_vals[apply_mask]
                    changed_mask = apply_mask & (new_sel_vals != vals_orig[valid_b])
                    if changed_mask.any():
                        changed_atoms = self.sel[pos_sel_b[changed_mask]].tolist()

                elif self.intervene_mode == "softcap":
                    mult_b = mult[b] if mult is not None else torch.ones(self.n_atoms, dtype=dtype, device=device)
                    apply_mask = union_mask[b, pos_sel_b] if union_mask.any() \
                                 else torch.ones_like(pos_sel_b, dtype=torch.bool, device=device)
                    new_sel_vals = vals_orig[valid_b].clone()
                    scale = mult_b[pos_sel_b]
                    new_sel_vals[apply_mask] = new_sel_vals[apply_mask] * scale[apply_mask]
                    if self.mode == "strength" and self.caps is not None:
                        new_sel_vals = torch.minimum(new_sel_vals, self.caps[pos_sel_b])
                    vals_out[valid_b] = new_sel_vals
                    cur_vals_new[b, pos_sel_b] = torch.where(apply_mask, new_sel_vals, cur_vals_new[b, pos_sel_b])
                    changed_mask = apply_mask & (new_sel_vals != vals_orig[valid_b])
                    if changed_mask.any():
                        changed_atoms = self.sel[pos_sel_b[changed_mask]].tolist()

                else:  # 'gd'
                    new_sel_vals = cur_vals_new[b, pos_sel_b]
                    if self.mode == "strength" and self.caps is not None:
                        new_sel_vals = torch.minimum(new_sel_vals, self.caps[pos_sel_b])
                    changed_mask = (new_sel_vals != vals_orig[valid_b])
                    vals_out[valid_b] = new_sel_vals
                    if changed_mask.any():
                        changed_atoms = self.sel[pos_sel_b[changed_mask]].tolist()

            frames_out.append((idxs_orig.to(device), vals_out.to(device, dtype)))
            changed_lists.append([int(a) for a in changed_atoms])

        # --- 8) adjust ring/sums for strength features (vectorized) ---
        if self.intervene_mode in ("attenuation", "softcap", "gd"):
            delta = cur_vals - cur_vals_new
            if self.mode == "strength":
                self.sum_strength -= delta
                self.ring_strength[:, ring_pos, :] -= delta

        self.t_ptr += 1
        return frames_out, p_combined, changed_lists

    # -------------------- gd internals --------------------

    @torch.no_grad()
    def _gd_step_on_current_frame(self,
                                  cur_vals: torch.Tensor,      # (bsz, atoms)
                                  present: torch.Tensor,       # (bsz, atoms) bool
                                  feats: torch.Tensor,         # (bsz, atoms)
                                  probs: torch.Tensor,         # (bsz, probes)
                                  n_frames: torch.Tensor,      # (bsz,)
                                  *,
                                  trig_only: bool,
                                  lr: float,
                                  steps: int,
                                  topk: Optional[int],
                                  caps: Optional[torch.Tensor],
                                  project_nonneg: bool) -> torch.Tensor:
        """
        Apply projected GD on the current frame codes to push undesired logits down.

        Loss
        ----
        BCE with target 0 on triggered probes:
            dL/dz = sigmoid(z) = probs
            dL/dX = probs @ w_mat
            dX/d(cur_vals) = 1 / n_frames (only on present atoms)

        Returns
        -------
        torch.Tensor
            New current-frame values (bsz, atoms) after GD and projection.
        """
        bsz, atoms = cur_vals.shape
        device, dtype = cur_vals.device, cur_vals.dtype
        new_vals = cur_vals.clone()
        inv_n = (1.0 / torch.clamp(n_frames.to(dtype), min=1.0)).unsqueeze(1)  # (bsz,1)

        if trig_only:
            mask_p = (probs >= self.thresholds.unsqueeze(0)).to(dtype)  # (bsz, probes)
        else:
            mask_p = torch.ones_like(probs, dtype=dtype)

        for _ in range(max(1, steps)):
            grad_logits = mask_p * probs                                 # (bsz, probes)  (dL/dz)
            grad_feats  = grad_logits @ self.w_mat                        # (bsz, atoms)  (dL/dX)
            grad_vals   = grad_feats * inv_n                              # (bsz, atoms)  chain to current-frame values

            if topk is not None and topk > 0:
                k = min(topk, atoms)
                masked = torch.where(present, grad_vals.abs(), torch.tensor(float("-inf"), device=device, dtype=dtype))
                vals, idx = torch.topk(masked, k=k, dim=1)
                keep = vals > 0
                rows = torch.arange(bsz, device=device).unsqueeze(1).expand_as(idx)
                sel = torch.zeros_like(present, dtype=torch.bool)
                sel[rows[keep], idx[keep]] = True
                grad_vals = torch.where(sel, grad_vals, torch.zeros_like(grad_vals))
            else:
                grad_vals = torch.where(present, grad_vals, torch.zeros_like(grad_vals))

            new_vals = new_vals - lr * grad_vals
            if project_nonneg:
                new_vals = torch.clamp(new_vals, min=0.0)
            if caps is not None:
                new_vals = torch.minimum(new_vals, caps.unsqueeze(0))

        return new_vals

    # -------------------- helpers --------------------

    def _trigger_mask(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        thr = self.thresholds
        over = (probs >= thr.unsqueeze(0))
        if self.trigger_mode == "any":
            trig = over.any(dim=1)
            p_comb = probs.max(dim=1).values
        elif self.trigger_mode == "max":
            trig = (probs.max(dim=1).values >= thr.max())
            p_comb = probs.max(dim=1).values
        elif self.trigger_mode == "sum":
            excess = torch.clamp(probs - thr.unsqueeze(0), min=0)
            trig = (excess.sum(dim=1) > 0)
            p_comb = probs.max(dim=1).values
        else:
            raise ValueError("invalid trigger_mode")
        return trig, p_comb

    def _to_long(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=torch.long)
        arr = np.asarray(list(x) if not isinstance(x, np.ndarray) else x).ravel().astype(np.int64)
        return torch.from_numpy(arr)

    def _to_float(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=self.dtype)
        arr = np.asarray(list(x) if not isinstance(x, np.ndarray) else x).ravel().astype(np.float32)
        return torch.from_numpy(arr)


# --- helper to align per-probe (sel_p, w_p) to a shared union (all lowercase) ---
def align_probes_to_union(sel_list, w_list, device="cpu", dtype=torch.float32):
    """
    Build a shared union of atoms and align each probe's weights to it.

    Parameters
    ----------
    sel_list : list of 1D arrays/LongTensors
        Per-probe atom ID lists (any order).
    w_list : list of 1D arrays/Tensors
        Per-probe weight vectors, aligned elementwise to the corresponding `sel_list` entry.
    device : str or torch.device, default 'cpu'
        Device to place outputs.
    dtype : torch.dtype, default torch.float32
        Floating dtype for the weight matrix.

    Returns
    -------
    sel_union : LongTensor, shape (atoms,)
        Sorted union of all provided atom IDs.
    w_mat : Tensor, shape (probes, atoms)
        Weight matrix with rows aligned to `sel_union` (zeros where a probe has no weight).
    """
    sels = [torch.as_tensor(s, dtype=torch.long, device=device).reshape(-1) for s in sel_list]
    ws   = [torch.as_tensor(w, dtype=dtype, device=device).reshape(-1) for w in w_list]
    assert all(s.numel() == w.numel() for s, w in zip(sels, ws))

    sel_union = torch.unique(torch.cat(sels)).sort().values
    atoms = sel_union.numel()
    probes = len(sels)
    w_mat = torch.zeros(probes, atoms, dtype=dtype, device=device)

    for p, (sp, wp) in enumerate(zip(sels, ws)):
        sp_sorted, perm = torch.sort(sp)
        wp_sorted = wp[perm]
        pos = torch.searchsorted(sel_union, sp_sorted)
        ok = (pos < atoms) & (sel_union[pos] == sp_sorted)
        if not bool(ok.all()):
            raise ValueError("atom ids must be integers from the same space.")
        w_mat[p, pos] = wp_sorted
    return sel_union, w_mat

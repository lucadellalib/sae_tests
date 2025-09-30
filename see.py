
"""
sae_probe_toolkit.py

A compact toolkit to:
  1) Build utterance-level sparse features from SAE top-k codes (counts or strengths).
  2) Run L1-logistic CV and perform stability selection across seeds.
  3) Fit a final model, choose a deployment threshold (by target precision), and
     export a JSON config (selected atoms, weights, intercept, threshold, metadata).

Inputs expected
---------------
- Utterances: a Python list of length N; each item is one utterance represented as
  a list of frames. Each frame is a (idxs_t, vals_t) pair:
    idxs_t: 1-D int array of active atom indices (values in [0, V-1])
    vals_t: 1-D float array of the same length with activation magnitudes
  Example: utt = [(np.array([2,5,7]), np.array([0.8,1.2,0.4])), ...]

- V: dictionary size (e.g., ~62000)

Feature modes
-------------
- "counts": per-utterance frequency of each atom (optionally binary presence)
- "strength": per-utterance sum of activation magnitudes per atom
Both optionally length-normalized by number of frames (recommended).

Author: ChatGPT
"""

from __future__ import annotations
import json, time, math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import average_precision_score, precision_recall_curve

# ------------------------- Feature building -------------------------

def _normalize_frame(idxs_t, vals_t):
    """Coerce a frame's indices/values into clean 1-D arrays with matching length."""
    idxs = np.asarray(idxs_t).ravel()
    vals = np.asarray(vals_t).ravel()
    if idxs.dtype.kind not in "iu":
        idxs = idxs.astype(np.int64, copy=False)
    if vals.dtype.kind not in "fc":
        vals = vals.astype(np.float32, copy=False)
    if idxs.shape[0] != vals.shape[0]:
        raise ValueError(f"idxs and vals length mismatch: {idxs.shape[0]} vs {vals.shape[0]}")
    return idxs.astype(np.int32, copy=False), vals.astype(np.float32, copy=False)


def features_counts(utt: List[Tuple[np.ndarray, np.ndarray]], V: int,
                    length_norm: bool=True, binary: bool=False) -> csr_matrix:
    """
    Return 1xV CSR row with per-utterance counts (or binary presence) of atoms.
    - utt: list of frames; each frame is (idxs_t, vals_t)
    - V: vocabulary/dictionary size
    """
    T = len(utt)
    if T == 0:
        return csr_matrix((1, V), dtype=np.float32)

    acc = {}
    for idxs_t, vals_t in utt:
        idxs, _ = _normalize_frame(idxs_t, vals_t)
        if binary:
            for j in idxs:
                acc[int(j)] = 1.0
        else:
            for j in idxs:
                acc[int(j)] = acc.get(int(j), 0.0) + 1.0

    if not acc:
        return csr_matrix((1, V), dtype=np.float32)

    idxs = np.fromiter(acc.keys(), dtype=np.int32)
    data = np.fromiter(acc.values(), dtype=np.float32)
    if length_norm and not binary:
        data /= float(T)

    indptr = np.array([0, idxs.size], dtype=np.int32)
    return csr_matrix((data, idxs, indptr), shape=(1, V), dtype=np.float32)


def features_strength(utt: List[Tuple[np.ndarray, np.ndarray]], V: int,
                      length_norm: bool=True) -> csr_matrix:
    """
    Return 1xV CSR row with per-utterance sum of activation magnitudes per atom.
    - utt: list of frames; each frame is (idxs_t, vals_t)
    - V: vocabulary/dictionary size
    """
    T = len(utt)
    if T == 0:
        return csr_matrix((1, V), dtype=np.float32)

    acc = {}
    for idxs_t, vals_t in utt:
        idxs, vals = _normalize_frame(idxs_t, vals_t)
        for j, v in zip(idxs, vals):
            acc[int(j)] = acc.get(int(j), 0.0) + float(v)

    if not acc:
        return csr_matrix((1, V), dtype=np.float32)

    idxs = np.fromiter(acc.keys(), dtype=np.int32)
    data = np.fromiter(acc.values(), dtype=np.float32)
    if length_norm:
        data /= float(T)

    indptr = np.array([0, idxs.size], dtype=np.int32)
    return csr_matrix((data, idxs, indptr), shape=(1, V), dtype=np.float32)


def build_X(utterances: List[List[Tuple[np.ndarray, np.ndarray]]], V: int,
            mode: str="counts", length_norm: bool=True, binary: bool=False) -> csr_matrix:
    """
    Build [N, V] CSR matrix from a list of utterances.
    mode: "counts" or "strength"
    """
    rows = []
    if mode not in {"counts", "strength"}:
        raise ValueError("mode must be 'counts' or 'strength'")
    for utt in utterances:
        if mode == "counts":
            row = features_counts(utt, V, length_norm=length_norm, binary=binary)
        else:
            row = features_strength(utt, V, length_norm=length_norm)
        rows.append(row)
    return vstack(rows).tocsr() if rows else csr_matrix((0, V), dtype=np.float32)


# ------------------------- CV / Stability / Threshold -------------------------

def _as_csr(X) -> csr_matrix:
    if isinstance(X, list):
        return vstack(X).tocsr()
    if issparse(X):
        return X.tocsr()
    return csr_matrix(X)


def pr_auc_cv(X, y, groups=None, C=0.5, n_splits=5, seed=0, l1_ratio=None):
    """
    Compute mean PR-AUC via CV. Returns mean, std, and list of coef vectors.
    If l1_ratio is None: use pure L1. Else use Elastic-Net with given l1_ratio.
    """
    X = _as_csr(X)
    y = np.asarray(y).ravel().astype(np.int32)
    N, V = X.shape

    if groups is not None:
        cv = GroupKFold(n_splits)
        split_iter = cv.split(np.zeros(N), y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = cv.split(np.zeros(N), y)

    aps, coefs, intercepts = [], [], []
    for tr, te in split_iter:
        if l1_ratio is None:
            clf = LogisticRegression(
                penalty="l1", solver="saga", C=C,
                class_weight="balanced", max_iter=5000, n_jobs=-1
            )
        else:
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
                class_weight="balanced", max_iter=5000, n_jobs=-1
            )
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aps.append(average_precision_score(y[te], p))
        coefs.append(clf.coef_[0].copy())
        intercepts.append(float(clf.intercept_[0]))

    return {
        "pr_auc_mean": float(np.mean(aps)),
        "pr_auc_std": float(np.std(aps)),
        "coef_matrix": np.vstack(coefs) if coefs else np.zeros((0, X.shape[1])),
        "intercepts": np.array(intercepts, dtype=np.float32),
    }


def stability_select(X, y, groups=None, C=0.5, n_splits=5, seeds=range(5),
                     freq_threshold=0.6, l1_ratio=None):
    """
    Run CV across multiple seeds; return selection frequency and avg |coef| per feature.
    """
    X = _as_csr(X)
    V = X.shape[1]
    sel_counts = np.zeros(V, dtype=np.int32)
    mean_abs = np.zeros(V, dtype=np.float64)
    total_runs = 0

    for seed in seeds:
        res = pr_auc_cv(X, y, groups=groups, C=C, n_splits=n_splits, seed=seed, l1_ratio=l1_ratio)
        coef_mat = res["coef_matrix"]  # [n_splits, V]
        if coef_mat.size == 0:
            continue
        nz = (coef_mat != 0)
        sel_counts += nz.sum(axis=0)
        mean_abs += np.abs(coef_mat).sum(axis=0)
        total_runs += coef_mat.shape[0]

    if total_runs == 0:
        raise RuntimeError("No CV runs executed; check inputs.")

    freq = sel_counts / float(total_runs)
    mean_abs /= float(total_runs)

    # stable idx: selected in at least freq_threshold fraction of runs
    stable_idx = np.where(freq >= freq_threshold)[0].astype(np.int32)
    # order by (freq desc, then avg |w| desc)
    order = np.lexsort((-mean_abs[stable_idx], -freq[stable_idx]))
    stable_idx = stable_idx[order]

    return stable_idx, freq.astype(np.float32), mean_abs.astype(np.float32)


def oof_probs(X, y, groups=None, C=0.5, n_splits=5, seed=0, l1_ratio=None, restrict_idx=None):
    """
    Get out-of-fold probabilities for threshold selection.
    If restrict_idx is provided, train and evaluate on X[:, restrict_idx].
    """
    X = _as_csr(X)
    y = np.asarray(y).ravel().astype(np.int32)
    N = X.shape[0]

    if groups is not None:
        cv = GroupKFold(n_splits)
        split_iter = cv.split(np.zeros(N), y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = cv.split(np.zeros(N), y)

    probs = np.zeros(N, dtype=np.float32)
    for tr, te in split_iter:
        Xtr = X[tr]; Xte = X[te]
        if restrict_idx is not None:
            Xtr = Xtr[:, restrict_idx]
            Xte = Xte[:, restrict_idx]

        if l1_ratio is None:
            clf = LogisticRegression(
                penalty="l1", solver="saga", C=C,
                class_weight="balanced", max_iter=5000, n_jobs=-1
            )
        else:
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
                class_weight="balanced", max_iter=5000, n_jobs=-1
            )
        clf.fit(Xtr, y[tr])
        probs[te] = clf.predict_proba(Xte)[:, 1]

    return probs


def choose_threshold_by_precision(y_true, y_prob, precision_target=0.9):
    """
    Given true labels and probabilities, pick the smallest threshold whose precision >= target.
    Returns (threshold, precision, recall).
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # Note: precision_recall_curve returns one more precision/recall than thresholds
    mask = prec[:-1] >= precision_target
    if not np.any(mask):
        # Fall back to threshold maximizing F1
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
        i = int(np.nanargmax(f1))
    else:
        i = int(np.argmax(mask))  # first threshold achieving target precision
    return float(thr[i]), float(prec[i]), float(rec[i])


def fit_final(X, y, restrict_idx=None, C=0.5, l1_ratio=None):
    """
    Fit final logistic regression on all data (optionally restricted to selected features).
    Returns fitted classifier.
    """
    X = _as_csr(X)
    y = np.asarray(y).ravel().astype(np.int32)
    if restrict_idx is not None:
        X = X[:, restrict_idx]

    if l1_ratio is None:
        clf = LogisticRegression(
            penalty="l1", solver="saga", C=C,
            class_weight="balanced", max_iter=5000, n_jobs=-1
        )
    else:
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
            class_weight="balanced", max_iter=5000, n_jobs=-1
        )
    clf.fit(X, y)
    return clf


# ------------------------- Export / Inference helpers -------------------------

@dataclass
class ExportMeta:
    mode: str                     # "counts" or "strength"
    length_norm: bool
    binary: bool
    V: int
    C: float
    l1_ratio: Optional[float]
    freq_threshold: float
    precision_target: float
    pr_auc_cv: float
    pr_auc_cv_std: float
    prevalence: float


def export_probe_json(path: str, clf: LogisticRegression, selected_idx: np.ndarray,
                      threshold: float, precision_at_thr: float, recall_at_thr: float,
                      meta: ExportMeta,
                      selection_frequency: Optional[np.ndarray]=None,
                      mean_abs_weight: Optional[np.ndarray]=None):
    """
    Write a JSON file containing the final probe configuration.
    """
    w = clf.coef_.ravel().tolist()
    intercept = float(clf.intercept_[0]) if np.ndim(clf.intercept_) > 0 else float(clf.intercept_)
    cfg = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "selected_atoms": selected_idx.astype(int).tolist(),
        "weights": w,                 # weights in the (restricted) model's feature order
        "intercept": intercept,
        "threshold": threshold,       # apply sigmoid(w·x + b), compare to this
        "precision_at_threshold": precision_at_thr,
        "recall_at_threshold": recall_at_thr,
        "meta": {
            "mode": meta.mode,
            "length_norm": meta.length_norm,
            "binary": meta.binary,
            "V": int(meta.V),
            "C": float(meta.C),
            "l1_ratio": (None if meta.l1_ratio is None else float(meta.l1_ratio)),
            "freq_threshold": float(meta.freq_threshold),
            "precision_target": float(meta.precision_target),
            "pr_auc_cv_mean": float(meta.pr_auc_cv),
            "pr_auc_cv_std": float(meta.pr_auc_cv_std),
            "prevalence": float(meta.prevalence),
        }
    }
    if selection_frequency is not None:
        cfg["selection_frequency"] = selection_frequency.astype(float).tolist()
    if mean_abs_weight is not None:
        cfg["mean_abs_weight"] = mean_abs_weight.astype(float).tolist()

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


# ------------------------- End-to-end convenience -------------------------

def run_end_to_end(
    utterances: List[List[Tuple[np.ndarray, np.ndarray]]],
    y: np.ndarray,
    V: int,
    mode: str = "counts",          # or "strength"
    length_norm: bool = True,
    binary: bool = False,          # only for mode="counts"
    C: float = 0.5,
    l1_ratio: Optional[float] = None,
    n_splits: int = 5,
    seeds: Iterable[int] = (0,1,2,3,4),
    freq_threshold: float = 0.6,
    precision_target: float = 0.9,
    groups: Optional[np.ndarray] = None,
    export_path: Optional[str] = None,
):
    """
    Build features, do stability selection, fit final model on selected atoms,
    choose threshold (by precision target) using out-of-fold probs, and export JSON.
    Returns dict with keys: X, stable_idx, clf, threshold, precision, recall, export_path.
    """
    # 1) Build X
    X = build_X(utterances, V, mode=mode, length_norm=length_norm, binary=binary)

    # 2) CV metric for reporting
    cv_res = pr_auc_cv(X, y, groups=groups, C=C, n_splits=n_splits, seed=0, l1_ratio=l1_ratio)

    # 3) Stability selection -> pick stable atoms
    stable_idx, sel_freq, mean_abs = stability_select(
        X, y, groups=groups, C=C, n_splits=n_splits, seeds=seeds,
        freq_threshold=freq_threshold, l1_ratio=l1_ratio
    )

    # If nothing passes threshold, fall back to top K by frequency
    if stable_idx.size == 0:
        order = np.argsort(-sel_freq)[: max(50, min(200, X.shape[1]))]
        stable_idx = order

    # 4) Choose threshold on OOF probs using only selected atoms
    probs = oof_probs(X, y, groups=groups, C=C, n_splits=n_splits,
                      seed=0, l1_ratio=l1_ratio, restrict_idx=stable_idx)
    thr, prec, rec = choose_threshold_by_precision(y, probs, precision_target=precision_target)

    # 5) Fit final model restricted to selected atoms
    clf = fit_final(X, y, restrict_idx=stable_idx, C=C, l1_ratio=l1_ratio)

    # 6) Export
    exported = None
    if export_path is not None:
        meta = ExportMeta(
            mode=mode, length_norm=length_norm, binary=binary, V=V, C=C,
            l1_ratio=l1_ratio, freq_threshold=freq_threshold,
            precision_target=precision_target,
            pr_auc_cv=cv_res["pr_auc_mean"], pr_auc_cv_std=cv_res["pr_auc_std"],
            prevalence=float(np.mean(y))
        )
        exported = export_probe_json(
            export_path, clf, stable_idx, thr, prec, rec, meta,
            selection_frequency=sel_freq, mean_abs_weight=mean_abs
        )

    return {
        "X": X,
        "cv": cv_res,
        "stable_idx": stable_idx,
        "clf": clf,
        "threshold": thr,
        "precision_at_thr": prec,
        "recall_at_thr": rec,
        "export_path": exported,
    }


# ------------------------- Minimal top-k helper (optional) -------------------------

def dense_to_frame_lists_topk_np(codes: np.ndarray, k: int = 128):
    """
    Convert a dense [T, V] array of activations to a list of (idxs_t, vals_t),
    keeping only top-k entries per frame (drops exact zeros).
    """
    T, V = codes.shape
    out = []
    kk = min(k, V)
    for t in range(T):
        row = codes[t]
        idxs_k = np.argpartition(row, -kk)[-kk:]
        vals_k = row[idxs_k]
        m = vals_k != 0
        out.append((idxs_k[m].astype(np.int32), vals_k[m].astype(np.float32)))
    return out

# ========================= Runtime Steering Helper =========================

from collections import deque

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class RuntimeSteerer:
    """
    Streaming helper that:
      - Maintains a rolling tail window of the last W frames (sparse frames as (idxs, vals))
      - Computes tail features (counts-rate or strength-rate) restricted to selected atoms
      - Scores risk with a logistic probe (weights/intercept from exported model)
      - If risk >= threshold, clips top-contributing risky atoms on the *current* frame
        using attenuation and optional safe caps
      - Supports fixed or risk-proportional number of atoms to clip
      - Cooldown prevents repeated clipping bursts

    Parameters
    ----------
    selected_atoms : np.ndarray[int]    # indices in [0, V) used by the probe
    weights        : np.ndarray[float]  # weights aligned to selected_atoms (same length R)
    intercept      : float
    threshold      : float              # probability threshold to trigger clipping
    mode           : {'counts', 'strength'}  # feature mode used during training
    window_frames  : int                # tail window size in frames (e.g., 100 for ~2s @50Hz)
    alpha          : float              # attenuation factor in [0,1]; e.g., 0.5
    L_max          : int                # maximum number of atoms to clip per trigger
    risk_proportional : bool            # if True, adapt L based on p
    cooldown_frames  : int              # frames to wait after a trigger before next action
    caps           : Optional[np.ndarray]  # per-atom safe caps for selected atoms (length R) or None
                                           # caps correspond to the *feature space*: for strength, a cap
                                           # on per-frame value; for counts mode, caps are not used.

    Notes
    -----
    - For 'counts' mode, clipping values won’t change counts immediately (presence is binary).
      It may still help downstream if your system re-selects top-k after the modification.
    - For 'strength' mode, attenuation changes magnitudes on the current frame (and we update
      the running sum so features remain consistent).
    """
    def __init__(self,
                 selected_atoms: np.ndarray,
                 weights: np.ndarray,
                 intercept: float,
                 threshold: float,
                 mode: str = "strength",
                 window_frames: int = 100,
                 alpha: float = 0.5,
                 L_max: int = 16,
                 risk_proportional: bool = True,
                 cooldown_frames: int = 80,
                 caps: Optional[np.ndarray] = None):
        self.selected_atoms = np.asarray(selected_atoms, dtype=np.int32)
        self.R = self.selected_atoms.shape[0]
        self.weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        assert self.weights.shape[0] == self.R, "weights must align with selected_atoms"
        self.intercept = float(intercept)
        self.threshold = float(threshold)
        self.mode = mode
        assert mode in ("counts", "strength"), "mode must be 'counts' or 'strength'"
        self.W = int(window_frames)
        self.alpha = float(alpha)
        self.L_max = int(L_max)
        self.risk_proportional = bool(risk_proportional)
        self.cooldown_frames = int(cooldown_frames)
        self.cooldown = 0
        # mapping from atom id -> position in selected set
        self.pos = {int(a): i for i, a in enumerate(self.selected_atoms)}
        # rolling buffer of frames
        self.buf = deque(maxlen=self.W)  # each item: (idxs, vals) AFTER any clipping
        self.n_frames = 0
        # running aggregates restricted to selected atoms
        self.sum_strength = np.zeros(self.R, dtype=np.float32)
        self.count_hits = np.zeros(self.R, dtype=np.float32)
        # positive weights used for contribution ranking
        self.wpos = np.maximum(self.weights, 0.0)
        # caps per selected atom (None means no cap)
        if caps is not None:
            caps = np.asarray(caps, dtype=np.float32).reshape(-1)
            assert caps.shape[0] == self.R, "caps must have length equal to selected_atoms"
        self.caps = caps

    def reset(self):
        self.buf.clear()
        self.n_frames = 0
        self.sum_strength.fill(0.0)
        self.count_hits.fill(0.0)
        self.cooldown = 0

    def _add_frame_to_aggregates(self, idxs: np.ndarray, vals: np.ndarray):
        """Add a frame's contribution to running aggregates (restricted to selected atoms)."""
        for j, v in zip(idxs, vals):
            pj = self.pos.get(int(j), None)
            if pj is None: 
                continue
            self.count_hits[pj] += 1.0
            if self.mode == "strength":
                self.sum_strength[pj] += float(v)

    def _remove_frame_from_aggregates(self, idxs: np.ndarray, vals: np.ndarray):
        """Remove a frame's contribution when it falls out of the window."""
        for j, v in zip(idxs, vals):
            pj = self.pos.get(int(j), None)
            if pj is None: 
                continue
            self.count_hits[pj] -= 1.0
            if self.mode == "strength":
                self.sum_strength[pj] -= float(v)

    def _current_features(self):
        """Return feature vector x (length R) from aggregates (length-normalized)."""
        n = max(1, self.n_frames)
        if self.mode == "strength":
            return (self.sum_strength / float(n)).copy()
        else:  # counts-rate
            return (self.count_hits / float(n)).copy()

    def _score(self, x):
        return float(sigmoid(self.weights @ x + self.intercept))

    def _choose_atoms_to_clip(self, x, p):
        if self.L_max <= 0:
            return np.empty((0,), dtype=np.int32)
        if self.risk_proportional:
            L = int(np.ceil(self.L_max * max(0.0, (p - self.threshold)) / max(1e-8, (1.0 - self.threshold))))
            L = max(0, min(self.L_max, L))
        else:
            L = self.L_max
        if L == 0:
            return np.empty((0,), dtype=np.int32)
        contrib = self.wpos * x  # only positive weights contribute to risk
        order = np.argsort(contrib)[::-1]
        order = order[:L]
        return self.selected_atoms[order]

    def _apply_clipping_to_current(self, idxs, vals, clip_set: set):
        """Apply attenuation and caps to atoms present in current frame if they are in clip_set.
           Returns (modified_vals, changed_atoms) where changed_atoms is a dict {atom_id: (old, new)}.
        """
        changed = {}
        new_vals = vals.copy()
        for t in range(len(idxs)):
            atom = int(idxs[t])
            if atom not in clip_set:
                continue
            old = float(new_vals[t])
            new = old * self.alpha
            # apply cap if provided and in selected set
            pj = self.pos.get(atom, None)
            if pj is not None and self.caps is not None:
                cap = float(self.caps[pj])
                if self.mode == "strength":
                    # cap applies to per-frame value
                    new = min(new, cap)
            new_vals[t] = new
            if new != old:
                changed[atom] = (old, new)
        return new_vals, changed

    def update(self, idxs_t, vals_t):
        """
        Ingest one frame (idxs_t, vals_t), possibly clip present atoms, and return:
          - (idxs_out, vals_out): possibly modified current frame
          - prob: current risk probability
          - clipped_atoms: list of atom ids clipped on this frame
        """
        # normalize inputs
        idxs = np.asarray(idxs_t).ravel().astype(np.int32)
        vals = np.asarray(vals_t).ravel().astype(np.float32)
        if idxs.shape[0] != vals.shape[0]:
            raise ValueError("idxs and vals must have the same length")

        # If buffer full, pop oldest and remove its aggregate contributions
        if self.n_frames == self.W and len(self.buf) == self.W:
            old_idxs, old_vals = self.buf[0]
            self._remove_frame_from_aggregates(old_idxs, old_vals)

        # First, tentatively add the *original* current frame to aggregates and buffer
        self.buf.append((idxs.copy(), vals.copy()))
        if self.n_frames < self.W:
            self.n_frames += 1
        self._add_frame_to_aggregates(idxs, vals)

        # Compute features and risk
        x = self._current_features()
        p = self._score(x)

        clipped_atoms = []
        # decide to clip only if above threshold and not in cooldown
        if p >= self.threshold and self.cooldown == 0:
            to_clip = self._choose_atoms_to_clip(x, p)
            clip_set = set(int(a) for a in to_clip.tolist())

            # Apply clipping to the *current* frame
            new_vals, changed = self._apply_clipping_to_current(idxs, vals, clip_set)

            if changed:
                # Update aggregates to reflect new (reduced) current frame values (strength mode only)
                if self.mode == "strength":
                    for atom, (old, new) in changed.items():
                        pj = self.pos.get(int(atom), None)
                        if pj is not None:
                            self.sum_strength[pj] -= (old - new)
                # Replace the most recent buffer entry with modified values
                self.buf[-1] = (idxs, new_vals)
                vals = new_vals  # return modified

                clipped_atoms = list(changed.keys())
                # start cooldown
                self.cooldown = self.cooldown_frames

        # Cooldown countdown
        if self.cooldown > 0:
            self.cooldown -= 1

        return (idxs, vals), p, clipped_atoms

# --------- Optional: compute safe caps from negative tails (offline) ---------

def compute_caps_from_negatives(neg_windows, selected_atoms, mode="strength", q=0.75):
    """
    Estimate per-atom caps from negative tail windows.
    neg_windows: list of arrays [L, V] (dense) *or* list of lists of frames [(idxs, vals), ...]
    selected_atoms: array of atom ids to compute caps for
    mode: 'strength' or 'counts' (for counts, caps are not used but function returns zeros)
    q: percentile in [0,1]; e.g., 0.75 for 75th percentile

    Returns np.ndarray caps aligned to selected_atoms length.
    """
    selected_atoms = np.asarray(selected_atoms, dtype=np.int32)
    R = selected_atoms.shape[0]
    pos = {int(a): i for i, a in enumerate(selected_atoms)}
    per_atom_values = [[] for _ in range(R)]

    # accumulate per-frame values for each selected atom
    for win in neg_windows:
        if isinstance(win, tuple) or (isinstance(win, list) and len(win) > 0 and isinstance(win[0], tuple)):
            # sparse window: list of (idxs, vals)
            for idxs, vals in win:
                idxs = np.asarray(idxs).ravel().astype(np.int32)
                vals = np.asarray(vals).ravel().astype(np.float32)
                for j, v in zip(idxs, vals):
                    pj = pos.get(int(j), None)
                    if pj is not None:
                        if mode == "strength":
                            per_atom_values[pj].append(float(v))
                        else:
                            per_atom_values[pj].append(1.0)  # presence
        else:
            # dense [L, V] window
            arr = np.asarray(win)
            for pj, a in enumerate(selected_atoms):
                col = arr[:, int(a)]
                if mode == "strength":
                    per_atom_values[pj].extend(col.astype(np.float32).tolist())
                else:
                    per_atom_values[pj].extend((col != 0).astype(np.float32).tolist())

    caps = np.zeros(R, dtype=np.float32)
    for pj in range(R):
        if len(per_atom_values[pj]) == 0:
            caps[pj] = 0.0
        else:
            caps[pj] = float(np.quantile(np.asarray(per_atom_values[pj], dtype=np.float32), q))
    return caps



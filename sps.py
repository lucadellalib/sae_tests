#!/usr/bin/env python3
import os
import torch
from typing import Optional

# ---------- Functionality under test (pure PyTorch) ----------

def save_csr_compact(path: str, x_TK: torch.Tensor) -> None:
    """
    x_TK: 2D [T, K], dense or sparse.
    Saves a compact CSR bundle that is portable across PyTorch versions.
    """
    if not x_TK.is_sparse:
        x_TK = x_TK.to_sparse_coo()
    x_TK = x_TK.coalesce().to_sparse_csr()
    torch.save({
        "crow": x_TK.crow_indices().cpu(),
        "col":  x_TK.col_indices().cpu(),
        "val":  x_TK.values().cpu(),
        "size": tuple(x_TK.size()),
        "dtype": x_TK.dtype,
        "layout": "csr",
    }, path)


class CSRFrames:
    def __init__(self, bundle: dict, device: torch.device | str = "cpu"):
        assert bundle.get("layout") == "csr", "Not a CSR bundle"
        self.crow = bundle["crow"].to(device)
        self.col  = bundle["col"].to(device)
        self.val  = bundle["val"].to(device)
        self.T, self.K = bundle["size"]

    def nnz_range(self, t: int) -> slice:
        a = int(self.crow[t]); b = int(self.crow[t + 1])
        return slice(a, b)

    def frame_coo(self, t: int) -> torch.Tensor:
        """Return frame t as a 1D sparse COO vector of length K."""
        sl = self.nnz_range(t)
        cols = self.col[sl]
        vals = self.val[sl]
        if cols.numel() == 0:
            idx = torch.empty((1, 0), dtype=torch.long, device=self.val.device)
            return torch.sparse_coo_tensor(idx, vals, (self.K,), device=self.val.device)
        idx = cols.unsqueeze(0)  # (1, nnz)
        return torch.sparse_coo_tensor(idx, vals, (self.K,), device=self.val.device).coalesce()

    def frame_dense(self, t: int) -> torch.Tensor:
        """Return frame t as a dense 1D tensor of length K."""
        sl = self.nnz_range(t)
        cols = self.col[sl]
        vals = self.val[sl]
        out = torch.zeros(self.K, dtype=self.val.dtype, device=self.val.device)
        if cols.numel():
            out[cols] = vals
        return out

    def iter_frames(self):
        """Yields (t, cols_view, vals_view) as VIEWS into col/val (zero-copy)."""
        for t in range(self.T):
            sl = self.nnz_range(t)
            yield t, self.col[sl], self.val[sl]

def load_csr_frames(path: str, device: torch.device | str = "cpu") -> CSRFrames:
    bundle = torch.load(path, map_location="cpu")
    return CSRFrames(bundle, device=device)

# ---------- Test helpers ----------

def make_random_sparse_TK(
    T: int, K: int, density: float = 0.2, seed: int = 0,
    force_empty_rows: Optional[list[int]] = None, dtype=torch.float32
) -> torch.Tensor:
    """
    Create a random sparse [T, K] with given density. Optionally force some rows to be empty.
    """
    g = torch.Generator().manual_seed(seed)
    nnz_target = max(0, int(T * K * density))
    if nnz_target == 0:
        # return an all-empty sparse tensor
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=dtype),
            size=(T, K),
        ).coalesce()

    rows = torch.randint(0, T, (nnz_target,), generator=g)
    cols = torch.randint(0, K, (nnz_target,), generator=g)

    if force_empty_rows:
        mask = torch.ones(nnz_target, dtype=torch.bool)
        for r in force_empty_rows:
            mask &= (rows != r)
        rows = rows[mask]
        cols = cols[mask]

    # remove duplicates by coalescing
    idx = torch.stack([rows, cols], dim=0)
    vals = torch.randn(idx.size(1), generator=g, dtype=dtype)
    x = torch.sparse_coo_tensor(idx, vals, size=(T, K)).coalesce()
    return x


def reconstruct_dense_from_frames(frames: CSRFrames) -> torch.Tensor:
    """Build a dense [T, K] by stitching frame_dense() rows."""
    out = torch.zeros((frames.T, frames.K), dtype=frames.val.dtype, device=frames.val.device)
    for t in range(frames.T):
        out[t] = frames.frame_dense(t)
    return out

def compare_dense(a: torch.Tensor, b: torch.Tensor, tol=0.0):
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} vs {b.shape}")
    max_abs = (a - b).abs().max().item()
    if max_abs > tol:
        raise AssertionError(f"Max abs diff {max_abs} > tol {tol}")

# ---------- Main test ----------

def main():
    torch.set_printoptions(edgeitems=3, threshold=20, linewidth=120)

    T, K = 7, 13
    density = 0.25
    path = "tmp_csr_bundle.pt"

    print("Generating random sparse [T,K] with some empty rows …")
    x_sparse = make_random_sparse_TK(T, K, density=density, seed=123, force_empty_rows=[0, 3])
    print("Original (COO) nnz:", x_sparse._nnz())
    # Save a dense view for comparison
    x_dense = x_sparse.to_dense()

    print("\nSaving compact CSR bundle …")
    save_csr_compact(path, x_sparse)
    assert os.path.exists(path), "File was not written"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading CSR bundle on device={device} …")
    frames = load_csr_frames(path, device=device)
    print("Loaded size:", (frames.T, frames.K))

    print("\nChecking per-frame reconstruction …")
    # 1) check dense row equality
    for t in range(T):
        d_ref = x_dense[t].to(device)
        d_rec = frames.frame_dense(t)
        compare_dense(d_ref, d_rec, tol=0.0)
    print("✓ frame_dense(t) matches original dense rows")

    # 2) check sparse 1D per-frame reconstruction
    for t in range(T):
        v = frames.frame_coo(t).coalesce()
        d_from_sparse = torch.zeros(K, dtype=v.values().dtype, device=v.values().device)
        if v.values().numel():
            d_from_sparse[v.indices().squeeze(0)] = v.values()
        compare_dense(x_dense[t].to(device), d_from_sparse, tol=0.0)
    print("✓ frame_coo(t) matches original rows when densified")

    # 3) rebuild the full dense matrix from frames and compare
    rebuilt = reconstruct_dense_from_frames(frames)
    compare_dense(x_dense.to(device), rebuilt, tol=0.0)
    print("✓ reconstruct_dense_from_frames matches original dense")

    # 4) also test saving starting from a DENSE input
    print("\nRe-testing save/load path starting from a DENSE input …")
    save_csr_compact(path, x_dense.cpu())  # dense path
    frames2 = load_csr_frames(path, device=device)
    rebuilt2 = reconstruct_dense_from_frames(frames2)
    compare_dense(x_dense.to(device), rebuilt2, tol=0.0)
    print("✓ dense → CSR → frames roundtrip OK")

    # 5) quick zero-copy iteration demonstration
    print("\nZero-copy per-frame iteration demo (first 3 frames):")
    for t, cols, vals in frames.iter_frames():
        if t >= 3:
            break
        print(f" t={t:>2} | nnz={vals.numel():>2} | cols={cols.tolist()} | vals shape={tuple(vals.shape)}")

    # cleanup
    try:
        os.remove(path)
    except OSError:
        pass

    print("\nAll tests passed ✅")

if __name__ == "__main__":
    main()

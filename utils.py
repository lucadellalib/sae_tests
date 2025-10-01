# ========== Per-frame helpers operating on a CSR tensor ==========

def row_range(csr: torch.Tensor, t: int) -> slice:
    """Return the slice [a:b] into col/val arrays that corresponds to frame t."""
    crow = csr.crow_indices()
    a = int(crow[t])
    b = int(crow[t + 1])
    return slice(a, b)

def frame_cols_vals(csr: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (cols_t, vals_t) for frame t as VIEWS (zero-copy) into CSR storage.
    """
    r = row_range(csr, t)
    return csr.col_indices()[r], csr.values()[r]

def frame_dense(csr: torch.Tensor, t: int) -> torch.Tensor:
    """Materialize frame t as a dense 1D tensor of length K."""
    K = csr.size(1)
    cols, vals = frame_cols_vals(csr, t)
    out = torch.zeros(K, dtype=csr.values().dtype, device=csr.values().device)
    if cols.numel():
        out[cols] = vals
    return out

def frame_coo_1d(csr: torch.Tensor, t: int) -> torch.Tensor:
    """Return frame t as a 1D sparse COO vector of length K."""
    K = csr.size(1)
    cols, vals = frame_cols_vals(csr, t)
    if cols.numel() == 0:
        idx = torch.empty((1, 0), dtype=torch.long, device=vals.device)
        return torch.sparse_coo_tensor(idx, vals.new_empty(0), (K,), device=vals.device)
    idx = cols.unsqueeze(0)  # (1, nnz)
    return torch.sparse_coo_tensor(idx, vals, (K,), device=vals.device).coalesce()

def iter_rows(csr: torch.Tensor) -> Iterator[Tuple[int, torch.Tensor, torch.Tensor]]:
    """
    Iterate over frames yielding (t, cols_t, vals_t) as VIEWS into CSR storage.
    """
    T = csr.size(0)
    crow = csr.crow_indices()
    col  = csr.col_indices()
    val  = csr.values()
    for t in range(T):
        a = int(crow[t]); b = int(crow[t + 1])
        yield t, col[a:b], val[a:b]

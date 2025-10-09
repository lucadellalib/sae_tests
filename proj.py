import torch
import torch.nn.functional as F

@torch.no_grad()
def build_Q_posneg_from_matrix(decoder, weight_matrix):
    """
    Build Q_pos, Q_neg from multiple probes stacked in a weight matrix.

    decoder       : (K, H)
    weight_matrix : (P, K)   # P probes, each row is a probe's weights

    Returns:
        Q_pos : (H, r1)  # risky subspace
        Q_neg : (H, r2)  # protective subspace
    """
    # Aggregate across probes
    w_pos = torch.clamp(weight_matrix, min=0).sum(dim=0)  # (K,)
    w_neg = torch.clamp(-weight_matrix, min=0).sum(dim=0) # (K,)

    Dn = F.normalize(decoder, dim=1)

    Q_pos, Q_neg = None, None
    if w_pos.any():
        Vt_pos = (w_pos.unsqueeze(1) * Dn).T
        Q_pos, _ = torch.linalg.qr(Vt_pos, mode="reduced")

    if w_neg.any():
        Vt_neg = (w_neg.unsqueeze(1) * Dn).T
        Q_neg, _ = torch.linalg.qr(Vt_neg, mode="reduced")

    return Q_pos, Q_neg


@torch.no_grad()
def steer_Q_posneg(h, Q_pos, Q_neg, eta_pos=0.3, eta_neg=0.1):
    """
    h' = h - η_pos * Q_pos(Q_pos^T h) + η_neg * Q_neg(Q_neg^T h)
    """
    single = h.ndim == 1
    if single:
        h = h.unsqueeze(0)

    h_new = h.clone()
    if Q_pos is not None:
        h_new = h_new - eta_pos * (h_new @ Q_pos) @ Q_pos.T
    if Q_neg is not None:
        h_new = h_new + eta_neg * (h_new @ Q_neg) @ Q_neg.T

    return h_new.squeeze(0) if single else h_new

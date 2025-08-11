import os

import matplotlib.pyplot as plt


# ADD after summarize_features(...)
def plot_density_hist(density, out_dir, fname="feature_density_hist.png"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.hist(density, bins=50)
    plt.xlabel("Activation density per feature")
    plt.ylabel("Count")
    plt.title("SAE Feature Activation Density")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()


def plot_topk_bar(per_utt_top, sample_idx, out_dir, fname=None):
    os.makedirs(out_dir, exist_ok=True)
    tops = per_utt_top[sample_idx]
    if not tops:
        return
    fids = [t[0] for t in tops]
    vals = [t[1] for t in tops]
    plt.figure()
    plt.bar([str(x) for x in fids], vals)
    plt.xlabel("Feature ID")
    plt.ylabel("Avg activation")
    plt.title(f"Top-k SAE features — sample {sample_idx}")
    plt.tight_layout()
    if fname is None:
        fname = f"sample_{sample_idx}_topk_bar.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()


def plot_token_feature_heatmap(
    feats, attn_mask, per_utt_top, sample_idx, topm=32, out_dir="plots", fname=None
):
    """
    Show a token × feature heatmap for one sample, restricted to its topM features (by avg activation).
    feats: (B, T, D), attn_mask: (B, T)
    """
    os.makedirs(out_dir, exist_ok=True)
    tops = per_utt_top[sample_idx]
    if not tops:
        return
    # pick topm features among its top list (or fewer if k < topm)
    chosen = [fid for fid, _ in tops[:topm]]

    valid_len = int(attn_mask[sample_idx].sum().item())
    F = feats[sample_idx, :valid_len, :]  # (L, D)
    sub = F[:, chosen].detach().cpu().numpy()  # (L, topm)

    plt.figure()
    plt.imshow(sub.T, aspect="auto")  # features x tokens
    plt.xlabel("Token index")
    plt.ylabel("Feature (topM subset)")
    plt.title(f"Token × Feature Heatmap — sample {sample_idx}")
    plt.tight_layout()
    if fname is None:
        fname = f"sample_{sample_idx}_heatmap.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()


def plot_global_top_feature_frequency(
    per_utt_top, out_dir, topn=20, fname="global_top_feature_freq.png"
):
    """
    Count how often features appear in utterance-level top-k lists and plot the most frequent ones.
    """
    os.makedirs(out_dir, exist_ok=True)
    from collections import Counter

    c = Counter()
    for tops in per_utt_top:
        c.update([fid for fid, _ in tops])
    if not c:
        return
    common = c.most_common(topn)
    fids = [str(k) for k, _ in common]
    counts = [v for _, v in common]

    plt.figure()
    plt.bar(fids, counts)
    plt.xlabel("Feature ID")
    plt.ylabel("Frequency across utterances")
    plt.title(f"Most frequent top-k features (top {topn})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()

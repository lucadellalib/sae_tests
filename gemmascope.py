#!/usr/bin/env python3

"""GemmaScope example."""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from plot_utils import *


def load_fleurs_texts(lang="en_us", split="test", limit=32):
    ds = load_dataset("google/fleurs", lang, split=split)
    # FLEURS fields: "raw_transcription" and "transcription"
    # We'll use normalized "transcription".
    texts = [ex["transcription"] for ex in ds.select(range(min(limit, len(ds))))]
    # texts = ["Would you be able to travel through time using a wormhole?" for ex in ds.select(range(min(limit, len(ds))))]
    return texts


@torch.no_grad()
def get_residual_activations(
    model, tokenizer, texts, layer_idx=12, max_len=1024, device="cuda"
):
    """
    Returns the post-block residual stream for a chosen layer.
    Hugging Face models return hidden_states as:
      hidden_states[0]   -> embeddings
      hidden_states[i+1] -> output after block i (i = 0..n_layers-1)
    This lines up with Gemma Scope's `blocks.{i}.hook_resid_post`.
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    ).to(device)

    out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states  # tuple(len = n_layers + 1)
    resid_post = hs[layer_idx + 1]  # (batch, seq, hidden)
    attn_mask = enc["attention_mask"]
    return resid_post, attn_mask, enc


def load_gemma_scope_sae(
    release="gemma-scope-2b-pt-res",
    sae_id="layer_20/width_16k/average_l0_71",
    device="cuda",
):
    """
    Loads a Gemma Scope SAE on the residual stream at the requested layer.
    Valid layers for 2B canonical res-16k include many mid layers; 10-13 work well.
    """
    sae, cfg, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    return sae


def summarize_features(feats, attn_mask, topk=5):
    """
    feats: (B, T, D_sae) sparse feature activations (ReLU)
    attn_mask: (B, T)
    Returns top-k feature IDs per sample and a global density stat.
    """
    B, T, D = feats.shape
    valid = attn_mask.bool()
    # Flatten over tokens that are valid
    active_tokens = feats[valid]  # (N_tokens, D)
    # Density: fraction of nonzeros per feature
    nnz_per_feature = (active_tokens > 0).sum(dim=0)  # (D,)
    density = (nnz_per_feature.float() / active_tokens.size(0)).cpu().tolist()

    # Per-utterance top-k (by average activation over valid tokens)
    per_utt_top = []
    start = 0
    for b in range(B):
        Lb = valid[b].sum().item()
        if Lb == 0:
            per_utt_top.append([])
            continue
        fb = feats[b, :Lb]  # (Lb, D)
        mean_act = fb.mean(dim=0)  # (D,)
        top_vals, top_idx = torch.topk(mean_act, k=min(topk, D))
        per_utt_top.append(
            [(int(i), float(v)) for i, v in zip(top_idx.cpu(), top_vals.cpu())]
        )
        start += Lb

    return density, per_utt_top


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lang",
        default="en_us",
        help="FLEURS language config (e.g., en_us, fr_fr, hi_in, ...)",
    )
    p.add_argument("--split", default="test")
    p.add_argument(
        "--limit", type=int, default=16, help="How many utterances to process"
    )
    p.add_argument("--model_name", default="google/gemma-2-2b")
    p.add_argument("--sae_release", default="gemma-scope-2b-pt-res")
    p.add_argument("--sae_id", default="layer_20/width_16k/average_l0_71")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="gemma_scope_features.jsonl")
    # In main() -> add args:
    p.add_argument("--plot_dir", default="plots", help="Directory to save plots")
    p.add_argument(
        "--plot_sample_idx",
        type=int,
        default=0,
        help="Which sample to visualize for bar/heatmap",
    )
    p.add_argument(
        "--heatmap_topm",
        type=int,
        default=32,
        help="How many top features to show in the heatmap",
    )
    p.add_argument(
        "--save_density_csv", action="store_true", help="Also dump density as CSV"
    )
    args = p.parse_args()

    device = args.device
    texts = load_fleurs_texts(args.lang, args.split, args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    layer_idx = int(args.sae_id.split("/")[0].split("_")[1])
    resid_post, attn_mask, enc = get_residual_activations(
        model,
        tokenizer,
        texts,
        layer_idx=layer_idx,
        max_len=args.max_len,
        device=device,
    )

    sae = load_gemma_scope_sae(
        release=args.sae_release, sae_id=args.sae_id, device=device
    )

    # SAE encode: returns sparse feature activations (same leading dims allowed)
    # Some SAELens versions accept broadcasted shapes; if not, flatten to (N_tokens, d_in) then reshape back.
    feats = sae.encode(resid_post)  # (B, T, D_sae)

    density, per_utt_top = summarize_features(feats, attn_mask, topk=8)

    # Optional sanity: reconstruction quality & cosine similarity
    recon = sae.decode(feats).to(resid_post.dtype)
    cos = (
        torch.nn.functional.cosine_similarity(
            resid_post[attn_mask.bool()].float(),
            recon[attn_mask.bool()].float(),
            dim=-1,
        )
        .mean()
        .item()
    )

    # Write JSONL with per-sample top features
    with open(args.out, "w") as f:
        for i, (txt, tops) in enumerate(zip(texts, per_utt_top)):
            record = {
                "idx": i,
                "text": txt,
                "top_features": [
                    {"feature_id": fid, "avg_activation": val} for fid, val in tops
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Also dump a small summary
    summary = {
        "lang": args.lang,
        "split": args.split,
        "num_samples": len(texts),
        "model": args.model_name,
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "feature_dim": feats.shape[-1],
        "mean_tokenwise_cosine(resid, recon)": cos,
        "feature_density_mean": float(torch.tensor(density).mean().item()),
        "feature_density_median": float(torch.tensor(density).median().item()),
    }
    print(json.dumps(summary, indent=2))

    # After computing `density, per_utt_top = summarize_features(...)` and `feats`:
    plot_dir = args.plot_dir

    # 1) Density histogram
    plot_density_hist(density, plot_dir)

    # 2) Per-utterance bar chart for selected sample
    plot_topk_bar(per_utt_top, args.plot_sample_idx, plot_dir)

    # 3) Token × feature heatmap on selected sample, zoom to topM features
    plot_token_feature_heatmap(
        feats,
        attn_mask,
        per_utt_top,
        sample_idx=args.plot_sample_idx,
        topm=args.heatmap_topm,
        out_dir=plot_dir,
    )

    # 4) Global feature frequency plot
    plot_global_top_feature_frequency(per_utt_top, plot_dir, topn=20)

    # Optional CSV dump for density
    if args.save_density_csv:
        df = pd.DataFrame({"feature_id": np.arange(len(density)), "density": density})
        df.to_csv(os.path.join(plot_dir, "feature_density.csv"), index=False)


if __name__ == "__main__":
    main()

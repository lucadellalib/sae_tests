#!/usr/bin/env python3

"""Gemma 3n FLEURS ASR."""

import argparse
import math
import os

import jiwer
import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


# ---------------------------
# Helpers
# ---------------------------
def compute_wer(refs, hyps):
    return 100 * float(jiwer.wer(refs, hyps))


def normalize_text(s: str):
    # simple whitespace normalization; customize if you want case/punct rules
    transform = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveEmptyStrings(),
        ]
    )
    return transform(s)


def make_messages_batch(audio_batch, user_prompt: str):
    """
    audio_batch: list of dicts, each like {"array": np.ndarray, "sampling_rate": int, ...}
    Returns a list of conversations (one turn each), ready for apply_chat_template.
    """
    msgs = []
    for x in audio_batch:
        msgs.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": x["array"]},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
        )
    return msgs


# ---------------------------
# Main
# ---------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="google/gemma-3n-E2B-it")
    parser.add_argument(
        "--lang", default="en_us", help="FLEURS language code (e.g., en_us)"
    )
    parser.add_argument(
        "--split", default="test", help="Dataset split (e.g., 'test', 'test[:200]')"
    )
    parser.add_argument(
        "--resample_hz", type=int, default=16000, help="Target sample rate for audio"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prompt", default="Transcribe this audio")
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--output_dir", default="gemma3n_fleurs_en_us")
    args = parser.parse_args()

    # Model & processor
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    # FLEURS (resample to model-preferred rate)
    ds = load_dataset("google/fleurs", args.lang, split=args.split)
    ds = ds.cast_column("audio", Audio(sampling_rate=args.resample_hz))

    arrays = ds[
        "audio"
    ]  # list of dicts: {"array": np.ndarray, "sampling_rate": int, ...}
    refs = ds["transcription"]  # normalized reference text
    uttids = ds["id"] if "id" in ds.features else list(range(len(ds)))

    rows = []
    num_items = len(arrays)
    num_batches = math.ceil(num_items / args.batch_size)

    for b in tqdm(range(num_batches), desc="Batches"):
        start = b * args.batch_size
        end = min((b + 1) * args.batch_size, num_items)

        batch_audio = arrays[start:end]
        batch_refs = refs[start:end]
        batch_ids = uttids[start:end]

        # Build chat messages for this batch (directly with arrays)
        messages = make_messages_batch(batch_audio, args.prompt)

        # Tokenize via chat template
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move tensors to model device (keep integer dtypes intact)
        inputs = inputs.to(model.device, dtype=model.dtype)

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions=False,
        )

        # Store hidden states
        # Shape: [num_altup_inputs, batch_size, num_tokens, hidden_size]
        # (see https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/modeling_gemma3n.py#L1048)
        hidden_states = torch.cat(
            [x[args.layer_idx] for x in outputs.hidden_states], dim=-2
        )
        hidden_states = hidden_states.movedim(0, -2).flatten(start_dim=-2)
        lengths = (
            (outputs.sequences != processor.tokenizer.pad_token_id).sum(dim=1) - 1
        ).long()
        batch_hidden_states = [x[:l] for x, l in zip(hidden_states, lengths)]
        prompt_lengths = (
            (inputs.input_ids != processor.tokenizer.pad_token_id).sum(dim=1)
        ).long()

        # Decode
        batch_texts = processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Strip text prompt
        batch_texts = [
            x.replace(f"user\n\n\n\n\n{args.prompt}\nmodel\n", "") for x in batch_texts
        ]

        # Collect normalized refs/hyps
        for uid, ref, hyp in zip(batch_ids, batch_refs, batch_texts):
            ref = normalize_text(ref)
            hyp = normalize_text(hyp)
            rows.append(
                {
                    "id": uid,
                    "ref": ref,
                    "hyp": hyp,
                    "wer": compute_wer(ref, hyp),
                }
            )

        # Dump hidden states
        for i, uid in enumerate(batch_ids):
            output_dir = os.path.join(
                args.output_dir, "data", f"layer={args.layer_idx}"
            )
            os.makedirs(output_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(output_dir, f"{uid}.npz"),
                token_ids=outputs.sequences[i][: lengths[i] + 1].cpu().numpy(),
                hidden_states=batch_hidden_states[i].float().cpu().numpy(),
                prompt_length=int(prompt_lengths[i].cpu().numpy()),
            )

    # WER
    wer_val = compute_wer(
        [r["ref"] for r in rows],
        [r["hyp"] for r in rows],
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"metadata.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"WER: {wer_val:.4f}")


if __name__ == "__main__":
    main()

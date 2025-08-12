#!/usr/bin/env python3

"""Gemma-3n FLEURS ASR."""

import argparse
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


def compute_cer(refs, hyps):
    return 100 * float(jiwer.cer(refs, hyps))


def normalize_text(s):
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


def make_messages_batch(audio_batch, user_prompt):
    msgs = []
    for x in audio_batch:
        msg = {
            "role": "user",
            "content": [
                {"type": "audio", "audio": x["array"]},
                {"type": "text", "text": user_prompt},
            ],
        }
        msgs.append([msg])
    return msgs


def run_batch(processor, model, args, batch_audio, batch_refs, batch_ids, rows):
    messages = make_messages_batch(batch_audio, args.prompt)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device, dtype=model.dtype)

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_attentions=False,
    )

    hidden_states = torch.cat([torch.stack(x) for x in outputs.hidden_states], dim=-2)
    hidden_states = hidden_states.movedim(1, -2).flatten(start_dim=-2).movedim(0, -1)
    lengths = (
        (outputs.sequences != processor.tokenizer.pad_token_id).sum(dim=1) - 1
    ).long()
    batch_hidden_states = [x[:l] for x, l in zip(hidden_states, lengths)]
    prompt_lengths = (
        (inputs["input_ids"] != processor.tokenizer.pad_token_id).sum(dim=1)
    ).long()

    batch_texts = processor.batch_decode(
        outputs.sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    batch_texts = [
        x.replace(f"user\n\n\n\n\n{args.prompt}\nmodel\n", "") for x in batch_texts
    ]

    # Per-utterance WER + CER
    for uid, ref, hyp in zip(batch_ids, batch_refs, batch_texts):
        ref = normalize_text(ref)
        hyp = normalize_text(hyp)
        rows.append(
            {
                "id": uid,
                "ref": ref,
                "hyp": hyp,
                "wer": compute_wer(ref, hyp),
                "cer": compute_cer(ref, hyp),
            }
        )

    for i, uid in enumerate(batch_ids):
        output_dir = os.path.join(args.output_dir, args.lang, "data")
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(output_dir, f"{uid}.npz"),
            token_ids=outputs.sequences[i][: lengths[i] + 1].cpu().numpy(),
            hidden_states=batch_hidden_states[i].float().cpu().numpy(),
            prompt_length=int(prompt_lengths[i].cpu().numpy()),
        )


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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prompt", default="Transcribe this audio")
    parser.add_argument("--output_dir", default="gemma3n_e2b_fleurs")
    args, *_ = parser.parse_known_args()

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    print("Loaded model!")

    ds = load_dataset("google/fleurs", args.lang, split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=args.resample_hz))
    print("Loaded dataset!")

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=lambda batch: {
            "audio": [b["audio"] for b in batch],
            "ref": [b["transcription"] for b in batch],
            "id": [b["id"] for b in batch],
        },
        num_workers=0,  # keep 0 unless you need multiprocessing
        pin_memory=False,
    )

    rows = []
    for batch in tqdm(dataloader, desc="Batches (DataLoader)"):
        batch_audio = batch["audio"]
        batch_refs = batch["ref"]
        batch_ids = batch["id"]
        run_batch(processor, model, args, batch_audio, batch_refs, batch_ids, rows)

    # Overall WER & CER
    overall_wer = compute_wer([r["ref"] for r in rows], [r["hyp"] for r in rows])
    overall_cer = compute_cer([r["ref"] for r in rows], [r["hyp"] for r in rows])

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, args.lang, "metadata.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"WER: {overall_wer:.4f}")
    print(f"CER: {overall_cer:.4f}")


if __name__ == "__main__":
    main()

"""Gemma 3n FLEURS ASR (ABSL version)."""

import os
import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from absl import app, flags, logging
import jiwer
from tqdm import tqdm

# ---------------------------
# Flags
# ---------------------------
FLAGS = flags.FLAGS
flags.DEFINE_string("model_id", "google/gemma-3n-E2B-it", "Model identifier")
flags.DEFINE_string("lang", "en_us", "FLEURS language code (e.g., en_us)")
flags.DEFINE_string("split", "test", "Dataset split (e.g., 'test', 'test[:200]')")
flags.DEFINE_integer("resample_hz", 16000, "Target sample rate for audio")
flags.DEFINE_integer("batch_size", 4, "Batch size for inference")
flags.DEFINE_integer("max_new_tokens", 512, "Maximum new tokens to generate")
flags.DEFINE_string("prompt", "Transcribe this audio", "Transcription prompt")
flags.DEFINE_integer("layer_idx", -1, "Decoder layer index for hidden-states dump")
flags.DEFINE_string("output_dir", "gemma3n_fleurs_en_us", "Output directory")

# ---------------------------
# Helpers
# ---------------------------
def compute_wer(refs, hyps):
    return 100 * float(jiwer.wer(refs, hyps))

def compute_cer(refs, hyps):
    return 100 * float(jiwer.cer(refs, hyps))

def normalize_text(s: str):
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

@torch.no_grad()
def run_batch(processor, model, batch_audio, batch_refs, batch_ids, rows):
    # Build chat messages for this batch (directly with arrays)
    messages = make_messages_batch(batch_audio, FLAGS.prompt)

    # Tokenize via chat template
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Move tensors to model device (keep integer dtypes intact)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Forward generate with decoder hidden states
    outputs = model.generate(
        **inputs,
        max_new_tokens=FLAGS.max_new_tokens,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_attentions=False,
    )

    # Store decoder hidden states (see modeling_gemma3n.py notes in your original)
    hidden_states = torch.cat([x[FLAGS.layer_idx] for x in outputs.hidden_states], dim=-2)
    hidden_states = hidden_states.movedim(0, -2).flatten(start_dim=-2)
    lengths = ((outputs.sequences != processor.tokenizer.pad_token_id).sum(dim=1) - 1).long()
    batch_hidden_states = [x[:l] for x, l in zip(hidden_states, lengths)]
    prompt_lengths = ((inputs["input_ids"] != processor.tokenizer.pad_token_id).sum(dim=1)).long()

    # Decode (assistant-only thanks to chat template)
    batch_texts = processor.batch_decode(
        outputs.sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    # Guard in case the model ever echoes the prompt header
    batch_texts = [x.replace(f"user\n\n\n\n\n{FLAGS.prompt}\nmodel\n", "") for x in batch_texts]

    # Collect normalized refs/hyps with per-utt WER/CER
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

    # Dump hidden states per utterance
    for i, uid in enumerate(batch_ids):
        out_dir = os.path.join(FLAGS.output_dir, "data", f"layer={FLAGS.layer_idx}")
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, f"{uid}.npz"),
            token_ids=outputs.sequences[i][: lengths[i] + 1].cpu().numpy(),
            hidden_states=batch_hidden_states[i].float().cpu().numpy(),
            prompt_length=int(prompt_lengths[i].cpu().numpy()),
        )

@torch.no_grad()
def main(_argv):
    # Model & processor
    processor = AutoProcessor.from_pretrained(FLAGS.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        FLAGS.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    logging.info("Loaded model %s", FLAGS.model_id)

    # FLEURS (streaming + resample)
    ds = load_dataset("google/fleurs", FLAGS.lang, split=FLAGS.split, streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=FLAGS.resample_hz))

    rows = []
    batch_audio, batch_refs, batch_ids = [], [], []
    pbar = tqdm(desc="Batches (streaming)")

    for ex in ds:
        a = ex["audio"]           # {"array": np.ndarray, "sampling_rate": int, ...}
        r = ex["transcription"]   # reference text
        uid = ex.get("id", None)

        batch_audio.append(a)
        batch_refs.append(r)
        batch_ids.append(uid)

        if len(batch_audio) == FLAGS.batch_size:
            run_batch(processor, model, batch_audio, batch_refs, batch_ids, rows)
            batch_audio.clear(); batch_refs.clear(); batch_ids.clear()
            pbar.update(1)

    # Flush final partial batch
    if batch_audio:
        run_batch(processor, model, batch_audio, batch_refs, batch_ids, rows)
        pbar.update(1)

    pbar.close()

    # Overall WER & CER (micro-avg with jiwer’s corpus functions)
    overall_wer = compute_wer([r["ref"] for r in rows], [r["hyp"] for r in rows])
    overall_cer = compute_cer([r["ref"] for r in rows], [r["hyp"] for r in rows])

    # Save
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    out_csv = os.path.join(FLAGS.output_dir, "metadata.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv}")
    print(f"WER: {overall_wer:.4f}")
    print(f"CER: {overall_cer:.4f}")

if __name__ == "__main__":
    app.run(main)

import os
import torch
import torch.distributed as dist
import webdataset as wds

def decode_safetensors(sample):
    from safetensors.torch import load as st_load
    state = st_load(sample["safetensors"])
    acts = state.get("activations") or next(v for v in state.values() if isinstance(v, torch.Tensor))
    return {"acts": acts}

def make_loader(urls, per_rank_batch_size, global_batches_per_epoch):
    rank = dist.get_rank()
    world = dist.get_world_size()

    # total samples per *global* epoch = global_batches_per_epoch * per_rank_batch_size
    epoch_samples_per_rank = (global_batches_per_epoch * per_rank_batch_size) // world

    ds = (
        wds.WebDataset(wds.ResampledShards(urls), shardshuffle=True)  # infinite
          .shuffle(10000)
          .map(decode_safetensors, handler=wds.warn_and_continue)
          .to_tuple("acts")
          .with_epoch(epoch_samples_per_rank)  # <-- per-rank epoch size in *samples*
          # (WebDataset auto-splits by rank and worker)
    )

    loader = wds.WebLoader(
        ds,
        batch_size=per_rank_batch_size,
        num_workers=4,
        persistent_workers=True,
    ).prefetch(4)

    return loader

def main():
    dist.init_process_group(backend="nccl")  # or "gloo" on CPU
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    urls = "wds_out/shard-*.tar"
    per_rank_batch_size = 8
    global_batches_per_epoch = 1000

    loader = make_loader(urls, per_rank_batch_size, global_batches_per_epoch)

    for epoch in range(1000):
        # Optionally reseed any RNGs here using epoch for determinism
        for step, (batch,) in enumerate(loader):
            # batch shape: [per_rank_batch_size, 1024, 8192]
            # training step...
            pass

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

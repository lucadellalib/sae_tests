"""Datasets."""

import numpy as np
import torch
from torch.utils.data import Dataset


__all__ = ["HallucinationDataset"]


class HallucinationDataset(Dataset):
    """
    Predefined labels in memory; samples are generated on-the-fly.
    classes: 0 -> 1000, 1 -> 500, 2 -> 200
    Each sample is (T, 8192, 31) with T in [200, 250].
    """

    def __init__(
        self,
        counts=(1000, 500, 200),
        H=8192,
        W=31,
        T_range=(200, 250),
        dtype=torch.float32,
    ):
        super().__init__()
        self.counts = tuple(map(int, counts))
        self.labels = np.concatenate(
            [np.full(c, lbl, dtype=np.int64) for lbl, c in enumerate(self.counts)]
        )
        self.H, self.W = int(H), int(W)
        self.T_lo, self.T_hi = int(T_range[0]), int(T_range[1])
        self.dtype = dtype

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        y = int(self.labels[idx])
        # Random T using torch.randint so worker seeds apply
        T = int(torch.randint(self.T_lo, self.T_hi + 1, (1,)).item())
        x = torch.randn((T, self.H, self.W), dtype=self.dtype)
        return x, y

    @property
    def sample_weights(self):
        labels = np.asarray(self.labels)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
        return sample_weights

    @staticmethod
    def collate(batch, pad_value=0.0):
        """
        Pads variable-length (T, H, W) tensors along T using torch.cat/pad_sequence.
        Returns: (B, T_max, H, W), labels, lengths
        """
        xs, ys = zip(*batch)
        lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)

        # pad_sequence expects (seq_len, *dims), so we keep input as (T, H, W)
        padded = torch.nn.utils.rnn.pad_sequence(
            xs, batch_first=True, padding_value=pad_value
        )
        # -> shape (B, T_max, H, W)

        ys = torch.tensor(ys, dtype=torch.long)
        return padded, ys, lengths


# ---- Example usage ----
if __name__ == "__main__":
    counts = (1000, 500, 200)  # labels 0,1,2
    ds = HallucinationDataset(counts=counts, H=8192, W=31, T_range=(200, 250))

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=ds.sample_weights,
        num_samples=len(ds),  # ~1 epoch draws N samples (with replacement)
        replacement=True,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=6,  # any; divisible by #classes helps but not required
        sampler=sampler,  # don't pass shuffle=True
        collate_fn=HallucinationDataset.collate,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # quick sanity check
    from collections import Counter

    xb, yb, L = next(iter(loader))
    print("batch:", xb.shape, "T[min,max]=", int(L.min()), int(L.max()))
    print("labels histogram:", Counter(yb.tolist()))

"""Models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["ConformerClassifier"]


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Returns a boolean mask: True for padding positions."""
    range_ = torch.arange(max_len, device=lengths.device)[None, :]
    return range_ >= lengths[:, None]  # (B, T)


# ---------- Positional Encoding ----------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 10000, dropout_p: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------- Conformer Components ----------
class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, expansion_factor: float = 4.0, dropout_p: float = 0.1):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout_p: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout_p, batch_first=True
        )

    def forward(self, x, key_padding_mask=None):
        out, _ = self.mha(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        return out


class ConvolutionModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, dropout_p: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.ln = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2 * dim, 1)
        self.dw = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(dim)
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        y = self.ln(x).transpose(1, 2)
        y = F.glu(self.pw1(y), dim=1)
        y = self.dw(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.pw2(y).transpose(1, 2)
        return self.dropout(y)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion: float = 4.0,
        conv_kernel: int = 31,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, ffn_expansion, dropout_p)
        self.ffn2 = FeedForwardModule(dim, ffn_expansion, dropout_p)
        self.ln_mha = nn.LayerNorm(dim)
        self.mha = MultiHeadSelfAttention(dim, num_heads, dropout_p)
        self.conv = ConvolutionModule(dim, conv_kernel, dropout_p)
        self.ln_final = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, key_padding_mask=None):
        x = x + 0.5 * self.ffn1(x)
        y = self.ln_mha(x)
        y = self.mha(y, key_padding_mask=key_padding_mask)
        x = x + self.dropout(y)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.ln_final(x)


# ---------- Classifier ----------
class ConformerClassifier(nn.Module):
    """
    If num_channels is not None, expect input shape (B, T, F, C) and apply a learnable
    linear projection over channels (C -> 1) before feeding (B, T, F) to the encoder.
    Otherwise, expect input shape (B, T, F).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ffn_expansion: float = 4.0,
        conv_kernel: int = 31,
        dropout_p: float = 0.1,
        max_len: int = 20000,
        num_channels: int | None = None,
    ):
        super().__init__()
        self.num_channels = num_channels

        # Optional channel reducer: (C -> 1) applied on the last dim if present
        if self.num_channels is not None:
            self.channel_reduce = nn.Linear(self.num_channels, 1, bias=False)
            # Optional: start as average over channels
            nn.init.constant_(self.channel_reduce.weight, 1.0 / self.num_channels)

        self.in_proj = nn.Linear(input_dim, hidden_size)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_size, max_len)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    hidden_size, num_heads, ffn_expansion, conv_kernel, dropout_p
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, lengths=None):
        """
        x: (B, T, F) if num_channels is None
           (B, T, F, C) if num_channels is not None
        lengths: (B,) valid lengths in timesteps
        """
        # Optional channel mixing: (B, T, F, C) -> (B, T, F)
        if self.num_channels is not None:
            assert (
                x.dim() == 4 and x.size(-1) == self.num_channels
            ), f"Expected x shape (B, T, F, {self.num_channels}), got {tuple(x.shape)}"
            x = self.channel_reduce(x).squeeze(-1)  # (B, T, F)

        # Standard Conformer encoder
        x = self.in_proj(x)  # (B, T, H)
        x = self.pos_enc(x)

        mask = None
        if lengths is not None:
            mask = lengths_to_mask(lengths, x.size(1))

        for layer in self.layers:
            x = layer(x, key_padding_mask=mask)

        # Mean pooling (mask-aware)
        if mask is not None:
            valid = (~mask).float()
            pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
                dim=1, keepdim=True
            )
        else:
            pooled = x.mean(dim=1)

        return self.head(pooled)


# ---------- Example ----------
if __name__ == "__main__":
    B, T, H, C = 4, 200, 80, 31
    model = ConformerClassifier(input_dim=H, num_classes=10, num_channels=C)
    x = torch.randn(B, T, H, C)
    lengths = torch.tensor([200, 150, 180, 90])
    out = model(x, lengths)
    print(out.shape)  # (B, num_classes)

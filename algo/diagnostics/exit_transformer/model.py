"""Tiny transformer for exit-decision forward-return prediction."""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: (B, T, D)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class ExitTransformer(nn.Module):
    """Inputs:
        seq:    (B, T, F_seq)   per-minute features
        scalar: (B, F_scalar)   end-of-window context features
       Outputs:
        (B, n_targets) forward log-returns in bps
    """

    def __init__(self, f_seq: int = 3, f_scalar: int = 10,
                 n_targets: int = 4, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 128, dropout: float = 0.1,
                 seq_len: int = 60):
        super().__init__()
        self.proj = nn.Linear(f_seq, d_model)
        self.pos = PositionalEncoding(d_model, max_len=seq_len + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.scalar_proj = nn.Sequential(
            nn.Linear(f_scalar, d_model), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )

    def forward(self, seq: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        # seq: (B, T, F_seq)
        x = self.proj(seq)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos(x)
        h = self.encoder(x)
        seq_pool = h[:, 0]  # cls token
        sc = self.scalar_proj(scalar)
        z = torch.cat([seq_pool, sc], dim=1)
        return self.head(z)


if __name__ == "__main__":
    m = ExitTransformer()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params:,}")
    seq = torch.randn(8, 60, 3)
    sca = torch.randn(8, 10)
    out = m(seq, sca)
    print(f"output: {out.shape}")

"""Train the exit transformer on 60d ETH 1m data.

Time-based split: last 10 days = validation. Saves model weights and
normalisation stats. Run: python -m algo.diagnostics.exit_transformer.train
"""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import (load_bars, build_features, make_dataset,
                  normalize_train_test, SEQ_LEN, TARGETS)
from model import ExitTransformer

ROOT = HERE.parent.parent.parent
DATA = ROOT / "algo" / "state" / "pi_pull_2026_04_30" / "eth_1m_60d.json"
OUT_DIR = ROOT / "algo" / "state" / "exit_transformer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_DAYS = 18      # cutoff before earliest Pi trade (2026-04-13) for fully OOS test
EPOCHS = 30
BATCH = 128
LR = 5e-4
WD = 1e-5
SAMPLE_STRIDE = 3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    bars = load_bars(DATA)
    feats = build_features(bars)
    X_seq, X_scalar, Y, idx = make_dataset(feats, sample_stride=SAMPLE_STRIDE)
    print(f"total samples: {len(X_seq)}")

    # Time-based split: last VAL_DAYS days
    cutoff_ts = bars[-VAL_DAYS * 1440, 0]  # ms
    sample_ts = bars[idx, 0]
    train_mask = sample_ts < cutoff_ts
    val_mask = ~train_mask
    print(f"train: {train_mask.sum()}  val: {val_mask.sum()}")

    Xs_tr, Xc_tr, Y_tr = X_seq[train_mask], X_scalar[train_mask], Y[train_mask]
    Xs_va, Xc_va, Y_va = X_seq[val_mask], X_scalar[val_mask], Y[val_mask]

    Xs_tr, Xc_tr, Xs_va, Xc_va, norm = normalize_train_test(
        Xs_tr, Xc_tr, Xs_va, Xc_va,
    )

    # We also normalize targets (z-score on train) for stable optimisation,
    # storing mean/std so we can de-normalize at inference.
    y_mean = Y_tr.mean(axis=0)
    y_std = Y_tr.std(axis=0) + 1e-6
    Y_tr_n = (Y_tr - y_mean) / y_std
    Y_va_n = (Y_va - y_mean) / y_std
    norm["y_mean"] = y_mean.tolist()
    norm["y_std"] = y_std.tolist()

    train_ds = TensorDataset(torch.from_numpy(Xs_tr), torch.from_numpy(Xc_tr),
                             torch.from_numpy(Y_tr_n))
    val_ds = TensorDataset(torch.from_numpy(Xs_va), torch.from_numpy(Xc_va),
                           torch.from_numpy(Y_va_n))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = ExitTransformer(
        f_seq=3, f_scalar=10, n_targets=len(TARGETS),
        d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1,
        seq_len=SEQ_LEN,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    history = []
    for ep in range(EPOCHS):
        t0 = time.time()
        model.train()
        train_losses = []
        for xs, xc, y in train_dl:
            xs, xc, y = xs.to(device), xc.to(device), y.to(device)
            pred = model(xs, xc)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())
        sched.step()

        model.eval()
        val_losses = []
        per_target_mse = np.zeros(len(TARGETS))
        per_target_n = 0
        bps_mse = np.zeros(len(TARGETS))
        with torch.no_grad():
            for xs, xc, y in val_dl:
                xs, xc, y = xs.to(device), xc.to(device), y.to(device)
                pred = model(xs, xc)
                val_losses.append(loss_fn(pred, y).item())
                # de-normalised MSE (in bps^2)
                p = pred.cpu().numpy() * np.array(y_std) + np.array(y_mean)
                t = y.cpu().numpy() * np.array(y_std) + np.array(y_mean)
                bps_mse += ((p - t) ** 2).sum(axis=0)
                per_target_n += len(p)
        bps_mse /= per_target_n
        bps_rmse = np.sqrt(bps_mse)

        train_l = float(np.mean(train_losses))
        val_l = float(np.mean(val_losses))
        history.append(dict(epoch=ep, train_loss=train_l, val_loss=val_l,
                            val_rmse_bps=bps_rmse.tolist(),
                            elapsed=time.time() - t0))
        print(f"epoch {ep:2d}  train={train_l:.4f}  val={val_l:.4f}  "
              f"val RMSE @h={dict(zip(TARGETS, bps_rmse.round(1).tolist()))} bps  "
              f"({time.time()-t0:.1f}s)")

        if val_l < best_val:
            best_val = val_l
            torch.save({
                "state_dict": model.state_dict(),
                "norm": norm,
                "config": dict(seq_len=SEQ_LEN, targets=TARGETS,
                               horizons=[60, 240, 360, 720, 1440],
                               d_model=64, nhead=4, num_layers=2,
                               dim_ff=128, f_seq=3, f_scalar=10),
            }, OUT_DIR / "exit_model.pt")

    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Saved to {OUT_DIR}/exit_model.pt")


if __name__ == "__main__":
    main()

import os
import json
import time
import math
import argparse
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_processed(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_tr = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_tr = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val = np.load(os.path.join(data_dir, "Y_val.npy"))
    return X_tr, Y_tr, X_val, Y_val


def _cos_sin_mats(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    cos[k,i] = cos(2pi k i / n), sin[k,i] = sin(2pi k i / n)
    """
    k = torch.arange(n, device=device).reshape(n, 1)
    i = torch.arange(n, device=device).reshape(1, n)
    ang = 2.0 * math.pi * k * i / n
    return torch.cos(ang).float(), torch.sin(ang).float()


class RealDFT2D(nn.Module):
    """
    Real-valued 2D DFT/IDFT implemented with cos/sin matrices (no complex autograd).
    This is slower than torch.fft but robust for small grids (60x13).
    """
    def __init__(self, n1: int, n2: int):
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        self.register_buffer("cos1", None, persistent=False)
        self.register_buffer("sin1", None, persistent=False)
        self.register_buffer("cos2", None, persistent=False)
        self.register_buffer("sin2", None, persistent=False)

    def _ensure(self, device: torch.device):
        if self.cos1 is None or self.cos1.device != device:
            c1, s1 = _cos_sin_mats(self.n1, device)
            c2, s2 = _cos_sin_mats(self.n2, device)
            self.cos1, self.sin1, self.cos2, self.sin2 = c1, s1, c2, s2

    def dft(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,C,n1,n2) real
        returns (R,I): (B,C,n1,n2) where R + i I is DFT
        """
        self._ensure(x.device)

        # beta DFT: exp(-i beta) => Rj = x*cos2^T, Ij = -x*sin2^T
        Rj = torch.einsum("lj,bcij->bcil", self.cos2, x)
        Ij = -torch.einsum("lj,bcij->bcil", self.sin2, x)

        # omega DFT: multiply by exp(-i alpha)
        # (Rj+iIj)*(cos1 - i sin1) => R = Rj*cos1 + Ij*sin1 ; I = Ij*cos1 - Rj*sin1
        R = torch.einsum("ki,bcil->bckl", self.cos1, Rj) + torch.einsum("ki,bcil->bckl", self.sin1, Ij)
        I = torch.einsum("ki,bcil->bckl", self.cos1, Ij) - torch.einsum("ki,bcil->bckl", self.sin1, Rj)
        return R, I

    def idft(self, R: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        """
        Inverse DFT using exp(+i·) and 1/n scaling on each axis.
        R,I: (B,C,n1,n2)
        returns x: (B,C,n1,n2) real
        """
        self._ensure(R.device)

        # omega IDFT: (R+iI)*(cos1 + i sin1)
        # Rj = (R*cos1 - I*sin1)/n1 ; Ij = (I*cos1 + R*sin1)/n1
        Rj = (torch.einsum("ki,bckl->bcil", self.cos1, R) - torch.einsum("ki,bckl->bcil", self.sin1, I)) / self.n1
        Ij = (torch.einsum("ki,bckl->bcil", self.cos1, I) + torch.einsum("ki,bckl->bcil", self.sin1, R)) / self.n1

        # beta IDFT: x = (Rj*cos2 - Ij*sin2)/n2
        x = (torch.einsum("lj,bcil->bcij", self.cos2, Rj) - torch.einsum("lj,bcil->bcij", self.sin2, Ij)) / self.n2
        return x


class SpectralConv2d(nn.Module):
    """
    Spectral conv using real-valued DFT representation (R,I).
    """
    def __init__(self, width: int, modes1: int, modes2: int, n1: int, n2: int):
        super().__init__()
        self.width = width
        self.modes1 = min(modes1, n1)
        self.modes2 = min(modes2, n2)
        self.n1 = n1
        self.n2 = n2

        # weights for low modes: (in=width, out=width, m1, m2)
        scale = 1.0 / (width * width)
        self.Wr = nn.Parameter(scale * torch.randn(width, width, self.modes1, self.modes2))
        self.Wi = nn.Parameter(scale * torch.randn(width, width, self.modes1, self.modes2))

        self.dft2 = RealDFT2D(n1, n2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,width,n1,n2)
        """
        B, C, n1, n2 = x.shape
        assert C == self.width and n1 == self.n1 and n2 == self.n2

        R, I = self.dft2.dft(x)  # (B,C,n1,n2)
        Rout = torch.zeros_like(R)
        Iout = torch.zeros_like(I)

        m1, m2 = self.modes1, self.modes2
        Rm = R[:, :, :m1, :m2]
        Im = I[:, :, :m1, :m2]

        # complex multiply + channel mixing:
        # (Rm+iIm) @ (Wr+iWi) => (Rm*Wr - Im*Wi) + i(Rm*Wi + Im*Wr)
        Rout[:, :, :m1, :m2] = torch.einsum("bcxy,coxy->boxy", Rm, self.Wr) - torch.einsum("bcxy,coxy->boxy", Im, self.Wi)
        Iout[:, :, :m1, :m2] = torch.einsum("bcxy,coxy->boxy", Rm, self.Wi) + torch.einsum("bcxy,coxy->boxy", Im, self.Wr)

        out = self.dft2.idft(Rout, Iout)
        return out


class FNO2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int, modes1: int, modes2: int, num_layers: int, dropout: float, n1: int, n2: int):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, modes1, modes2, n1, n2) for _ in range(num_layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(num_layers)])
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(p=dropout)
        self.proj1 = nn.Conv2d(width, width, 1)
        self.proj2 = nn.Conv2d(width, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for s, p in zip(self.spec, self.pw):
            x = self.act(s(x) + p(x))
            x = self.drop(x)
        x = self.act(self.proj1(x))
        return self.proj2(x)


@dataclass
class TrainConfig:
    seed: int = 42
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2
    max_epochs: int = 50
    patience: int = 10
    max_steps: int = 0  # 0 = no limit; >0 = limit steps per epoch (smoke test)
    modes1: int = 12
    modes2: int = 7
    width: int = 8
    num_layers: int = 1
    dropout: float = 0.1
    uq_T: int = 20
    uq_Q: float = 0.95


def train_one(cfg: TrainConfig, data_dir: str, out_dir: str, device: str):
    os.makedirs(out_dir, exist_ok=True)

    X_tr, Y_tr, X_val, Y_val = load_processed(data_dir)
    n1, n2 = X_tr.shape[2], X_tr.shape[3]
    in_ch, out_ch = X_tr.shape[1], Y_tr.shape[1]

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    max_steps = int(getattr(cfg, 'max_steps', 0))
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    dev = torch.device(device)
    model = FNO2d(in_ch, out_ch, cfg.width, cfg.modes1, cfg.modes2, cfg.num_layers, cfg.dropout, n1, n2).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    patience_left = cfg.patience

    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_seen = 0
        step = 0
        for xb, yb in tr_dl:
            xb = xb.to(dev).float()
            yb = yb.to(dev).float()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_seen += xb.size(0)
            step += 1
            if max_steps > 0 and step >= max_steps:
                break
            step += 1
            if max_steps > 0 and step >= max_steps:
                break
        tr_loss /= max(tr_seen, 1)

        model.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            vstep = 0
            for xb, yb in val_dl:
                xb = xb.to(dev).float()
                yb = yb.to(dev).float()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_seen += xb.size(0)
                vstep += 1
                if max_steps > 0 and vstep >= max_steps:
                    break
                vstep += 1
                if max_steps > 0 and vstep >= max_steps:
                    break
        val_loss /= max(val_seen, 1)

        print(f"Epoch {epoch:03d} | train={tr_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_epoch = epoch
            patience_left = cfg.patience
            torch.save(model.state_dict(), os.path.join(out_dir, "best_fno.pth"))
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EarlyStop] epoch={epoch}, best_epoch={best_epoch}, best_val={best_val:.6f}")
                break

    elapsed = time.time() - t0

    with open(os.path.join(out_dir, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    meta = {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "device": device,
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
        "train_shape": list(X_tr.shape),
        "val_shape": list(X_val.shape),
    }
    with open(os.path.join(out_dir, "run_meta_train.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[OK] Saved:", os.path.join(out_dir, "best_fno.pth"))
    print("[OK] Saved:", os.path.join(out_dir, "best_config.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")

    ap.add_argument("--modes1", type=int, default=12)
    ap.add_argument("--modes2", type=int, default=7)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)

    ap.add_argument("--uq_T", type=int, default=20)
    ap.add_argument("--uq_Q", type=float, default=0.95)
    args = ap.parse_args()

    cfg = TrainConfig(
        seed=args.seed, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
        max_epochs=args.max_epochs, patience=args.patience,
        modes1=args.modes1, modes2=args.modes2, width=args.width, num_layers=args.num_layers, dropout=args.dropout,
        uq_T=args.uq_T, uq_Q=args.uq_Q
    )
    if args.quick:
        # Make the run VERY lightweight for a quick smoke test (works on low-resource machines)
        cfg.max_epochs = min(cfg.max_epochs, 3)
        cfg.patience = min(cfg.patience, 2)
        cfg.batch_size = 1
        cfg.width = min(cfg.width, 4)
        cfg.num_layers = 1
        cfg.modes1 = min(cfg.modes1, 4)
        cfg.modes2 = min(cfg.modes2, 4)
        cfg.max_epochs = 1
        cfg.max_steps = 2  # only 2 steps per epoch

    set_seed(cfg.seed)
    train_one(cfg, args.data_dir, args.out_dir, args.device)


if __name__ == "__main__":
    main()

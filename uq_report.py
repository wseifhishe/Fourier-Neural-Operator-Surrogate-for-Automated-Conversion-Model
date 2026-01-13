import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

from train_fno import FNO2d


def load_cfg(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "best_config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, T: int):
    model.train()  # dropout ON
    preds = []
    for _ in range(T):
        preds.append(model(x))
    stack = torch.stack(preds, dim=0)  # (T,B,C,H,W)
    return stack.mean(dim=0), stack.std(dim=0)


def grid_uncertainty(std: torch.Tensor) -> torch.Tensor:
    # std: (B,C,H,W) -> (H,W)
    return std.mean(dim=1).mean(dim=0)


def batched_grid_unc(model: nn.Module, x_np: np.ndarray, device: str, T: int, batch: int = 8) -> torch.Tensor:
    grids = []
    for i in range(0, len(x_np), batch):
        xb = torch.from_numpy(x_np[i:i+batch]).to(device).float()
        _, std = mc_dropout_predict(model, xb, T=T)
        grids.append(grid_uncertainty(std).cpu())
    return torch.stack(grids, dim=0).mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--Q", type=float, default=0.95)
    args = ap.parse_args()

    cfg = load_cfg(args.out_dir)

    X_val = np.load(os.path.join(args.data_dir, "X_val.npy")).astype(np.float32)
    Y_val = np.load(os.path.join(args.data_dir, "Y_val.npy")).astype(np.float32)
    X_test_real = np.load(os.path.join(args.data_dir, "X_test_real.npy")).astype(np.float32)
    X_test_imag = np.load(os.path.join(args.data_dir, "X_test_imag.npy")).astype(np.float32)

    n1, n2 = X_val.shape[2], X_val.shape[3]
    in_ch = X_val.shape[1]
    out_ch = Y_val.shape[1]

    model = FNO2d(
        in_channels=in_ch,
        out_channels=out_ch,
        width=int(cfg["width"]),
        modes1=int(cfg["modes1"]),
        modes2=int(cfg["modes2"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
        n1=n1,
        n2=n2,
    ).to(args.device)

    ckpt_path = os.path.join(args.out_dir, "best_fno.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))

    unc_val = batched_grid_unc(model, X_val, args.device, T=args.T, batch=8)
    thr = float(torch.quantile(unc_val.flatten(), q=args.Q).item())

    unc_real = batched_grid_unc(model, X_test_real, args.device, T=args.T, batch=8)
    unc_imag = batched_grid_unc(model, X_test_imag, args.device, T=args.T, batch=8)

    H, W = unc_val.shape
    os.makedirs(args.out_dir, exist_ok=True)

    out_csv = os.path.join(args.out_dir, "uq_report_grid.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("omega_idx,beta_idx,unc_val,unc_test_real,unc_test_imag,flag_val\n")
        for oi in range(H):
            for bi in range(W):
                uv = float(unc_val[oi, bi].item())
                ur = float(unc_real[oi, bi].item())
                ui = float(unc_imag[oi, bi].item())
                flag = 1 if uv > thr else 0
                f.write(f"{oi},{bi},{uv:.8e},{ur:.8e},{ui:.8e},{flag}\n")

    summary = {
        "T": args.T,
        "Q": args.Q,
        "threshold_from_val": thr,
        "mean_unc_val": float(unc_val.mean().item()),
        "mean_unc_test_real": float(unc_real.mean().item()),
        "mean_unc_test_imag": float(unc_imag.mean().item()),
        "frac_grid_above_thr_val": float((unc_val > thr).float().mean().item()),
    }
    with open(os.path.join(args.out_dir, "uq_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Saved:", out_csv)
    print("[OK] Saved:", os.path.join(args.out_dir, "uq_summary.json"))
    print("[OK] threshold:", thr)


if __name__ == "__main__":
    main()

import os
import argparse
import numpy as np


def augment_phase(H_arr: np.ndarray, phis: np.ndarray) -> np.ndarray:
    """
    H(phi) = Hr*cos(phi) - Hi*sin(phi)

    H_arr: (N, omega, beta, C, 2) [real, imag]
    phis:  (P,)
    return: (N*P, omega, beta, C)
    """
    assert H_arr.ndim == 5 and H_arr.shape[-1] == 2, f"Unexpected shape: {H_arr.shape}"
    N, n_w, n_b, C, _ = H_arr.shape
    Hr = H_arr[..., 0]
    Hi = H_arr[..., 1]

    out = np.empty((N * len(phis), n_w, n_b, C), dtype=np.float32)
    idx = 0
    for phi in phis:
        c, s = np.cos(phi), np.sin(phi)
        out[idx:idx+N] = Hr * c - Hi * s
        idx += N
    return out


def to_fno(A: np.ndarray) -> np.ndarray:
    """(N, omega, beta, C) -> (N, C, omega, beta)"""
    return np.transpose(A, (0, 3, 1, 2)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_phi", type=int, default=40)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    npz = np.load(args.in_npz)
    X_train_tf = npz["X_train_tf"]  # (N_case,60,13,45,2)
    Y_train_tf = npz["Y_train_tf"]  # (N_case,60,13,24,2)
    X_test_tf  = npz["X_test_tf"]
    Y_test_tf  = npz["Y_test_tf"]

    phis = np.linspace(0.0, 2*np.pi, args.n_phi, endpoint=False).astype(np.float32)

    # augment
    X_aug = augment_phase(X_train_tf, phis)  # (N_case*n_phi,60,13,45)
    Y_aug = augment_phase(Y_train_tf, phis)  # (N_case*n_phi,60,13,24)

    N = X_aug.shape[0]
    n_val = max(1, int(N * args.val_ratio))
    perm = rng.permutation(N)
    tr_idx = perm[:-n_val]
    val_idx = perm[-n_val:]

    X_tr = to_fno(X_aug[tr_idx])
    Y_tr = to_fno(Y_aug[tr_idx])
    X_val = to_fno(X_aug[val_idx])
    Y_val = to_fno(Y_aug[val_idx])

    # test real/imag (phi=0, phi=3pi/2)
    phi_real = np.array([0.0], dtype=np.float32)
    phi_imag = np.array([1.5*np.pi], dtype=np.float32)

    X_test_real = to_fno(augment_phase(X_test_tf, phi_real))
    Y_test_real = to_fno(augment_phase(Y_test_tf, phi_real))
    X_test_imag = to_fno(augment_phase(X_test_tf, phi_imag))
    Y_test_imag = to_fno(augment_phase(Y_test_tf, phi_imag))

    # save
    np.save(os.path.join(args.out_dir, "X_train.npy"), X_tr)
    np.save(os.path.join(args.out_dir, "Y_train.npy"), Y_tr)
    np.save(os.path.join(args.out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(args.out_dir, "Y_val.npy"), Y_val)
    np.save(os.path.join(args.out_dir, "X_test_real.npy"), X_test_real)
    np.save(os.path.join(args.out_dir, "Y_test_real.npy"), Y_test_real)
    np.save(os.path.join(args.out_dir, "X_test_imag.npy"), X_test_imag)
    np.save(os.path.join(args.out_dir, "Y_test_imag.npy"), Y_test_imag)

    print("[OK] Saved to", args.out_dir)
    print(" X_train:", X_tr.shape, "Y_train:", Y_tr.shape)
    print(" X_val  :", X_val.shape, "Y_val  :", Y_val.shape)
    print(" X_test_real:", X_test_real.shape, "Y_test_real:", Y_test_real.shape)
    print(" X_test_imag:", X_test_imag.shape, "Y_test_imag:", Y_test_imag.shape)


if __name__ == "__main__":
    main()

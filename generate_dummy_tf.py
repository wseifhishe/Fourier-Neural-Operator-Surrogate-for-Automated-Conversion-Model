import os
import argparse
import numpy as np


def _smooth_basis(n_omega: int, n_beta: int) -> np.ndarray:
    """
    Smooth basis functions on (omega,beta) grid.
    Returns (K, n_omega, n_beta)
    """
    w = np.linspace(0.0, 1.0, n_omega, endpoint=True)
    b = np.linspace(0.0, 1.0, n_beta, endpoint=False)
    W, B = np.meshgrid(w, b, indexing="ij")  # (n_omega, n_beta)

    fw = [
        np.exp(-3.0 * W),
        np.sin(2 * np.pi * W),
        np.cos(2 * np.pi * W),
        np.sin(4 * np.pi * W),
    ]
    gb = [
        np.ones_like(B),
        np.sin(2 * np.pi * B),
        np.cos(2 * np.pi * B),
    ]

    basis = [f * g for f in fw for g in gb]
    return np.stack(basis, axis=0).astype(np.float32)


def _make_complex_tf(rng: np.random.Generator, n_case: int, n_omega: int, n_beta: int, n_chan: int, noise: float):
    """
    Returns complex TF: (n_case, n_omega, n_beta, n_chan) complex64
    """
    basis = _smooth_basis(n_omega, n_beta)  # (K, w, b)
    K = basis.shape[0]

    w_real = rng.normal(0, 1, size=(n_case, n_chan, K)).astype(np.float32)
    w_imag = rng.normal(0, 1, size=(n_case, n_chan, K)).astype(np.float32)

    Hr = np.einsum("nck,kwb->nwbc", w_real, basis)
    Hi = np.einsum("nck,kwb->nwbc", w_imag, basis)

    # higher noise at high frequency
    w = np.linspace(0.0, 1.0, n_omega, endpoint=True).reshape(n_omega, 1, 1)
    noise_scale = noise * (0.3 + 0.7 * w)
    Hr += rng.normal(0, 1, size=Hr.shape).astype(np.float32) * noise_scale
    Hi += rng.normal(0, 1, size=Hi.shape).astype(np.float32) * noise_scale

    # normalize per case
    mag = np.sqrt(Hr**2 + Hi**2).mean(axis=(1, 2, 3), keepdims=True) + 1e-6
    Hr /= mag
    Hi /= mag
    return (Hr + 1j * Hi).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_train_cases", type=int, default=2)
    ap.add_argument("--n_test_cases", type=int, default=1)
    ap.add_argument("--n_omega", type=int, default=60)
    ap.add_argument("--n_beta", type=int, default=13)
    ap.add_argument("--in_channels", type=int, default=45)
    ap.add_argument("--out_channels", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # X TF (input)
    X_train = _make_complex_tf(rng, args.n_train_cases, args.n_omega, args.n_beta, args.in_channels, args.noise)
    X_test  = _make_complex_tf(rng, args.n_test_cases,  args.n_omega, args.n_beta, args.in_channels, args.noise)

    # Correlated Y TF (output) via complex mixing + damping-like gain
    Mr = rng.normal(0, 1, size=(args.out_channels, args.in_channels)).astype(np.float32)
    Mi = rng.normal(0, 1, size=(args.out_channels, args.in_channels)).astype(np.float32)
    M = (Mr + 1j * Mi).astype(np.complex64) / np.sqrt(args.in_channels)

    w = np.linspace(0.0, 1.0, args.n_omega, endpoint=True).reshape(1, args.n_omega, 1, 1)
    gain = (1.0 / (1.0 + 2.0 * (w**2))).astype(np.float32)

    def map_to_Y(Xc: np.ndarray) -> np.ndarray:
        Yc = np.einsum("oi,nwbi->nwbo", M, Xc)  # (n_case,w,b,out)
        Yc = Yc * gain
        Yc += (rng.normal(0, 1, size=Yc.shape).astype(np.float32) * (args.noise * 0.5)) * (1.0 + 0.5 * w)
        return Yc.astype(np.complex64)

    Y_train = map_to_Y(X_train)
    Y_test  = map_to_Y(X_test)

    def pack_ri(Z: np.ndarray) -> np.ndarray:
        return np.stack([Z.real, Z.imag], axis=-1).astype(np.float32)

    out_path = os.path.join(args.out_dir, "dummy_tf.npz")
    np.savez_compressed(
        out_path,
        X_train_tf=pack_ri(X_train),
        Y_train_tf=pack_ri(Y_train),
        X_test_tf=pack_ri(X_test),
        Y_test_tf=pack_ri(Y_test),
    )

    print("[OK] Saved:", out_path)
    print(" X_train_tf:", pack_ri(X_train).shape)
    print(" Y_train_tf:", pack_ri(Y_train).shape)
    print(" X_test_tf :", pack_ri(X_test).shape)
    print(" Y_test_tf :", pack_ri(Y_test).shape)


if __name__ == "__main__":
    main()

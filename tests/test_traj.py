"""Numpy mirror of the TrajBranch math (no torch needed).

Checks:
1. tail_sigmas reproduces the original grid tail (float32, as in
   the node) to grid-inversion precision.
2. capture at step k + resume with the remaining steps lands on the
   uninterrupted trajectory's final state (small float tolerance).
3. capture state at step k equals the full run's state after k steps.
"""
import numpy as np

SHIFT = 3.0
STEPS = 30
K = 12


def grid(steps, shift, f=np.float32):
    t = np.linspace(f(1.0), f(0.0), steps + 1, dtype=f)
    s = f(shift)
    sig = s * t / (1 + (s - 1) * t)
    sig[-1] = f(0.0)
    return sig


def tail(start_sigma, steps, shift, f=np.float32):
    s = f(shift)
    sig = f(start_sigma)
    t_start = sig / (s - (s - 1) * sig)
    t = np.linspace(t_start, f(0.0), steps + 1, dtype=f)
    out = s * t / (1 + (s - 1) * t)
    out[0] = sig
    out[-1] = f(0.0)
    return out


def velocity(x, sigma):
    # a tame curved field
    return np.sin(x) * 0.3 + (1.0 - sigma) * 0.2 + 0.1 * x


def euler(x, sigmas, capture_step=None):
    snap = None
    for i in range(len(sigmas) - 1):
        if capture_step is not None and capture_step == i:
            snap = x.copy()
        s, sn = sigmas[i], sigmas[i + 1]
        denoised = x - s * velocity(x, s)
        x = denoised if sn == 0 else x + velocity(x, s) * (sn - s)
    if capture_step is not None and capture_step == len(sigmas) - 1:
        snap = x.copy()
    return x, snap


def main():
    g = grid(STEPS, SHIFT)
    t = tail(g[K], STEPS - K, SHIFT)
    err_grid = np.max(np.abs(t.astype(np.float64) - g[K:].astype(np.float64)))
    print(f"grid tail max err: {err_grid:.3e}")
    assert err_grid < 1e-6

    rng = np.random.default_rng(42)
    x0 = rng.normal(size=(2, 4, 8, 8)).astype(np.float32)

    full, _ = euler(x0.copy(), g)
    _, snap = euler(x0.copy(), g, capture_step=K)
    mid, _ = euler(x0.copy(), g[:K + 1])  # state after K steps
    assert np.array_equal(snap, mid), "capture state mismatch"
    resumed, _ = euler(snap.copy(), t)
    err = np.max(np.abs(resumed.astype(np.float64) - full.astype(np.float64)))
    print(f"capture+resume vs full: max err {err:.3e}")
    assert err < 1e-4
    print("all checks passed")


if __name__ == "__main__":
    main()

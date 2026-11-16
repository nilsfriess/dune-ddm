#!/usr/bin/env python3
"""Writes the 2D coefficient of the DD30 experiments for ex2 (problem.coefficient = file): a heterogeneous background
with high-contrast meandering channels.

- Background: log10 kappa = 0.5 * clip(g, -2, 2), with g a Gaussian random field with Matern covariance (nu = 1,
  correlation length 0.05 of the domain), normalised to mean 0 and variance 1 on the grid. So kappa lies in [0.1, 10].
- Channels: 12 roughly horizontal channels with kappa = 1e6 and width 1/64 of the domain, centred at heights
  (j + 1/2) / 12 and meandering by a sum of three sines (wavenumbers 1, 2, 3 over the width of the domain, random
  amplitudes and phases). The amplitudes keep neighbouring channels apart by more than 2 elements of a 128^2 mesh, and
  the width makes every channel at least 2 elements wide on it, so that the channels stay connected (DG couples
  elements only through edges) and do not touch on every mesh from 128^2 on.

The field is written on a 2000^2 grid over the domain of ex2, the unit square, in the format of examples/velocity_reader.hh
(raw little-endian float32, sample ix at row iz from the top at index ix * nz + iz), as <out>.bin with <out>.ini.
Needs gstools (e.g. python3 -m venv ~/.venvs/dd30 && ~/.venvs/dd30/bin/pip install gstools matplotlib).

Usage: make_kappa_2d.py [out (default data/kappa_channels_2d)] [--png preview.png] [--kappa-channel 1e6]
(the background does not depend on --kappa-channel, so fields with different contrasts differ only in the channels)
"""
import sys
from pathlib import Path

import gstools as gs
import numpy as np

N = 2000  # not a power of two: the element centres of 2^k x 2^k meshes (k >= 4) never lie on the edge between samples
NU, ELL, BG_SEED = 1.0, 0.05, 1
CHANNELS, WIDTH, KAPPA_CHANNEL, CH_SEED = 12, 1 / 64, 1e6, 3
AMPLITUDE = (0.003, 0.008)  # of each of the three sines


def channel_centres(t, rng):
    """Heights of the channel centrelines at the horizontal positions t (in [0, 1])."""
    out = []
    for j in range(CHANNELS):
        y = np.full_like(t, (j + 0.5) / CHANNELS)
        for k in (1, 2, 3):
            y += rng.uniform(*AMPLITUDE) * np.sin(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi))
        out.append(y)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = lambda name: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None
    png = opt("--png")
    kappa_channel = float(opt("--kappa-channel") or KAPPA_CHANNEL)
    args = [a for a in args if a not in (png, opt("--kappa-channel"))]
    out = Path(args[0] if args else "data/kappa_channels_2d")

    t = (np.arange(N) + 0.5) / N  # sample centres in [0, 1]
    g = gs.SRF(gs.Matern(dim=2, var=1, len_scale=ELL, nu=NU), seed=BG_SEED, mode_no=2000).structured([t, t])
    g = (g - g.mean()) / g.std()
    logk = 0.5 * np.clip(g, -2, 2)  # [ix, iy], iy from the bottom

    centres = channel_centres(t, np.random.default_rng(CH_SEED))
    gaps = [np.min(np.abs(centres[j + 1] - centres[j])) - WIDTH for j in range(CHANNELS - 1)]
    print(f"smallest gap between channels: {min(gaps):.4f} of the domain ({min(gaps) * 128:.2f} elements of a 128^2 mesh)")
    Y = t[None, :]
    for yc in centres:
        logk[np.abs(Y - yc[:, None]) < WIDTH / 2] = np.log10(kappa_channel)
    kappa = (10.0 ** logk).astype("<f4")

    out.parent.mkdir(parents=True, exist_ok=True)
    kappa[:, ::-1].tofile(out.with_suffix(".bin"))  # rows from the top, as velocity_reader.hh expects
    out.with_suffix(".ini").write_text(
        "# Coefficient of the 2D DD30 experiments for ex2 (problem.coefficient = file), written by\n"
        "# scripts/dd30/make_kappa_2d.py (the data file is not in the repository, run the script to create it):\n"
        f"# background 10^(0.5 clip(g, -2, 2)) with g Matern (nu {NU}, correlation length {ELL} of the domain, seed {BG_SEED}),\n"
        f"# {CHANNELS} meandering channels with kappa {kappa_channel:g} and width {WIDTH} of the domain (seed {CH_SEED}).\n"
        "# Raw little-endian float32 over the unit square, sample ix at row iz from the top at index ix * nz + iz.\n"
        f"file = {out.with_suffix('.bin').name}\nnx = {N}\nnz = {N}\ndx = {1 / N}\ndz = {1 / N}\n")
    print(f"wrote {out.with_suffix('.bin')} and {out.with_suffix('.ini')}: kappa in [{kappa.min():.3g}, {kappa.max():.3g}]")

    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(logk.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="log10 kappa")
        fig.savefig(png, dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()

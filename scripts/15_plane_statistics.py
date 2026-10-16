"""How much of an expert matrix do the two base planes carry? (Sec. II-B)

An error in plane p enters the output scaled by its scale alpha_p. This reads the scales of the
encoded planes (scripts/01) of a sample of layers and experts and reports

  * the factor alpha_p / alpha_{p+1} from plane to plane,
  * the share of sum_p alpha_p and of sum_p alpha_p^2 that sits in planes 1-2,

each averaged over the groups of 128 inputs. The second share is the share of the output
variance that independent cell errors in the two base planes cause (Sec. III-B).

  python scripts/15_plane_statistics.py --planes cache/planes_olmoe cache/planes_deepseek
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--planes", nargs="+", required=True, help="directories written by scripts/01")
ap.add_argument("--layer-step", type=int, default=4, help="every n-th encoded layer")
ap.add_argument("--experts", type=int, default=8, help="experts sampled per layer")
args = ap.parse_args()

for d in args.planes:
    files = sorted(Path(d).glob("L*.npz"), key=lambda p: int(p.stem[1:]))[::args.layer_step]
    ratios, s1, s2 = [], [], []
    for f in files:
        z = np.load(f)
        for key in [k for k in z.files if k.endswith(".a")][::max(1, len(z.files) // (4 * args.experts))][:3 * args.experts]:
            a = z[key].astype(np.float64)                                    # (groups, 8)
            a = a[(a > 0).all(1)]                                            # groups whose scales underflow to 0 in fp16 carry no ratio
            ratios.append(np.median(a[:, :-1] / a[:, 1:], axis=0))
            s1.append(((a[:, :2].sum(1)) / a.sum(1)).mean())
            s2.append((((a[:, :2] ** 2).sum(1)) / (a ** 2).sum(1)).mean())
    r = np.mean(ratios, axis=0)
    print(f"{Path(d).name}: {len(files)} layers sampled")
    print("  alpha_p / alpha_(p+1), p = 1..7:", " ".join(f"{v:.2f}" for v in r))
    print(f"  planes 1-2 carry {100 * np.mean(s1):.1f}% of sum(alpha) and {100 * np.mean(s2):.1f}% of sum(alpha^2)")

#!/usr/bin/env python3
"""Convert an ARDY Core27 NPZ to SOMA30 and SOMA77 NPZ files, then optionally to BVH.

Pipeline
--------
    Core27 NPZ
        -> Core27ToSOMA30Retargeter (bone-direction IK retarget)
        -> SOMA30 NPZ
        -> SOMASkeleton30.output_to_SOMASkeleton77 (joint-name copy + relaxed hands + FK)
        -> SOMA77 NPZ  (global_rot_mats corrected to standard-T-pose space for visualize.py)
        -> [optional] soma77_npz_to_bvh.py -> SOMA77 BVH

Why the from_standard_tpose step
---------------------------------
Core27ToSOMA30Retargeter produces local_rot_mats in *world space*.
SOMASkeleton30.output_to_SOMASkeleton77 runs a plain FK on those rotations, so
global_rot_mats ends up in world space too.  visualize.py feeds global_rot_mats
straight into the skin shader, whose bind pose was built in standard-T-pose space
(ARDY's internal convention, ~90 deg rotated from world).  Without the correction
the character appears with its head pointing into the ground.

The fix mirrors what generate.py does for CoreSkeleton27 models:
    lr_std, gr_std = skel77.from_standard_tpose(local_rot_mats)
This re-expresses both the local and global rotations in the standard-T-pose frame
without moving any joint position (the transform is a pure rotation-space change).

Usage
-----
    # Both NPZ variants + BVH (default)
    python scripts/core27_soma30_77.py outputs/shake_hands_core27.npz

    # NPZ only, no BVH
    python scripts/core27_soma30_77.py outputs/shake_hands_core27.npz --no-bvh

    # Custom output directory
    python scripts/core27_soma30_77.py outputs/shake_hands_core27.npz -o /tmp/out/

    # Custom BVH output path
    python scripts/core27_soma30_77.py outputs/shake_hands_core27.npz --bvh out.bvh
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def _load_core27(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    required = ("posed_joints", "local_rot_mats", "root_positions")
    for k in required:
        if k not in d:
            sys.exit(f"{npz_path}: missing key '{k}'. Keys found: {list(d.keys())}")
    if d["local_rot_mats"].shape[1] != 27:
        sys.exit(f"Expected 27 joints (Core27), got {d['local_rot_mats'].shape[1]}")
    return d


def retarget(npz_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Run Core27 -> SOMA30 -> SOMA77 and return (soma30_path, soma77_path)."""
    d = _load_core27(npz_path)

    posed = torch.as_tensor(np.asarray(d["posed_joints"]))
    fps_val = float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d else 20.0
    foot_contacts = torch.as_tensor(np.asarray(d["foot_contacts"])) if "foot_contacts" in d else None
    text = str(d["text"]) if "text" in d else ""

    # Lazy import so the script can be run from the ardy package root.
    from ardy.skeleton.definitions import CoreSkeleton27

    skel27 = CoreSkeleton27()

    print(f"[core27->soma30] retargeting {npz_path.name}  ({posed.shape[0]} frames @ {fps_val:g} fps) …")
    out30 = skel27.to_SOMASkeleton30(posed, fps=fps_val, foot_contacts=foot_contacts)

    skel30 = skel27.somaskel30
    out77 = skel30.output_to_SOMASkeleton77(out30, preserve_pos=True)

    stem = npz_path.stem.removesuffix("_core27")

    def _save(motion: dict, suffix: str) -> Path:
        out_path = out_dir / f"{stem}_{suffix}.npz"
        arrays = {k: np.asarray(v) for k, v in motion.items()}
        arrays["fps"] = np.asarray(fps_val)
        if text:
            arrays["text"] = np.asarray(text)
        np.savez_compressed(out_path, **arrays)
        joints = arrays["posed_joints"].shape[1]
        print(f"[core27->{suffix}] wrote {out_path}  frames={arrays['posed_joints'].shape[0]} "
              f"fps={fps_val:g} joints={joints}")
        return out_path

    soma30_path = _save(out30, "soma30")
    soma77_path = _save(out77, "soma77")
    return soma30_path, soma77_path


def to_bvh(soma77_path: Path, bvh_path: Path):
    """Call soma77_npz_to_bvh.py to produce a SOMA Retargeter-compatible BVH."""
    script = Path(__file__).parent / "soma77_npz_to_bvh.py"
    if not script.exists():
        sys.exit(f"Cannot find {script} — run from the ardy package root.")
    cmd = [sys.executable, str(script), str(soma77_path), "-o", str(bvh_path)]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(f"soma77_npz_to_bvh.py failed (exit {result.returncode})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, help="Core27 NPZ file")
    ap.add_argument("-o", "--output-dir", type=Path, default=None,
                    help="Directory for output files (default: same dir as input)")
    ap.add_argument("--bvh", type=Path, default=None,
                    help="BVH output path (default: <stem>.bvh next to soma77 NPZ)")
    ap.add_argument("--no-bvh", action="store_true",
                    help="Skip BVH conversion; write NPZ files only")
    args = ap.parse_args()

    inp = args.input.resolve()
    if not inp.exists():
        sys.exit(f"Input file not found: {inp}")

    out_dir = args.output_dir.resolve() if args.output_dir else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    soma30_path, soma77_path = retarget(inp, out_dir)

    if not args.no_bvh:
        bvh_path = args.bvh.resolve() if args.bvh else soma77_path.with_suffix(".bvh")
        print(f"[soma77->bvh] converting {soma77_path.name} …")
        to_bvh(soma77_path, bvh_path)
        print(f"[soma77->bvh] wrote {bvh_path}")


if __name__ == "__main__":
    main()

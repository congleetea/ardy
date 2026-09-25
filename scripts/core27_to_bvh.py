#!/usr/bin/env python3
"""Convert an ARDY Core27 motion NPZ to a BVH compatible with SOMA Retargeter.

The output follows the same layout as the BVH files SOMA Retargeter ships with:
  - ROOT Root  : 6 channels, always 0 (dummy node at the world origin)
  - JOINT Hips : 6 channels, position = root_positions in cm, ZYX rotation
  - all others : 3 channels  Zrotation Yrotation Xrotation
  - offsets    : parent-relative rest offsets in centimetres (Y-up)
  - units      : centimetres (soma_retargeter/io/bvh.py multiplies by 0.01)
  - euler seq  : ZYX  i.e.  R = Rz @ Ry @ Rx  (intrinsic, channel order Z Y X)

Usage
-----
    python scripts/core27_to_bvh.py outputs/shake_hands_core27.npz
    python scripts/core27_to_bvh.py outputs/shake_hands_core27.npz -o out.bvh
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------

_ROOT_NAME = "Root"
_HIPS_NAME = "Hips"
_POS_CHANNELS = ["Xposition", "Yposition", "Zposition"]
_ROT_CHANNELS = ["Zrotation", "Yrotation", "Xrotation"]


def _core27_skeleton():
    """Return (names, parent_indices, neutral_cm) from CoreSkeleton27."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from ardy.skeleton.definitions import CoreSkeleton27
        skel = CoreSkeleton27()
        names = list(skel.bone_order_names)
        parents = [int(p) for p in skel.joint_parents]
        neutral = np.asarray(skel.neutral_joints.detach().cpu().numpy(), dtype=np.float64)
    except Exception as e:
        raise SystemExit(f"Cannot import CoreSkeleton27: {e}")

    # Hips is the root (index 0), already at origin in neutral_joints.
    neutral_cm = neutral * 100.0
    return names, parents, neutral_cm


def _parent_relative_offsets(names, parents, neutral_cm):
    """Return dict name -> parent-relative offset in cm."""
    offsets = {}
    for i, name in enumerate(names):
        p = parents[i]
        vec = neutral_cm[i] if p < 0 else neutral_cm[i] - neutral_cm[p]
        offsets[name] = vec
    return offsets


# ---------------------------------------------------------------------------
# BVH hierarchy writer
# ---------------------------------------------------------------------------

def _write_hierarchy(names, parents, offsets_cm):
    children = {i: [] for i in range(len(names))}
    roots = []
    for i, p in enumerate(parents):
        (roots if p < 0 else children[p]).append(i)

    def fmt(v):
        return f"{v:.6g}"

    def emit_joint(i, depth):
        name = names[i]
        pad = "  " * depth
        chans = (_POS_CHANNELS + _ROT_CHANNELS) if name == _HIPS_NAME else _ROT_CHANNELS
        o = offsets_cm[name]
        lines = [
            f"{pad}JOINT {name}",
            f"{pad}{{",
            f"{pad}  OFFSET {fmt(o[0])} {fmt(o[1])} {fmt(o[2])}",
            f"{pad}  CHANNELS {len(chans)} {' '.join(chans)}",
        ]
        if children[i]:
            for c in children[i]:
                lines += emit_joint(c, depth + 1)
        else:
            # leaf: add End Site extending along the bone direction
            d = offsets_cm[name]
            n = np.linalg.norm(d)
            e = (d / n * max(0.4 * n, 2.0)) if n > 1e-8 else np.array([0.0, 5.0, 0.0])
            lines += [
                f"{pad}  End Site",
                f"{pad}  {{",
                f"{pad}    OFFSET {fmt(e[0])} {fmt(e[1])} {fmt(e[2])}",
                f"{pad}  }}",
            ]
        lines.append(f"{pad}}}")
        return lines

    out = [
        "HIERARCHY",
        f"ROOT {_ROOT_NAME}",
        "{",
        "  OFFSET 0 0 0",
        f"  CHANNELS 6 {' '.join(_POS_CHANNELS + _ROT_CHANNELS)}",
    ]
    for r in roots:
        out += emit_joint(r, 1)
    out.append("}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Motion writer
# ---------------------------------------------------------------------------

def _skeleton_nodes(names):
    """Ordered channel list matching the depth-first HIERARCHY emit order."""
    nodes = [{"name": _ROOT_NAME, "channels": _POS_CHANNELS + _ROT_CHANNELS}]
    nodes += [
        {"name": n, "channels": (_POS_CHANNELS + _ROT_CHANNELS) if n == _HIPS_NAME else _ROT_CHANNELS}
        for n in names
    ]
    return nodes


def _build_frames(local_rot, root_positions_cm, names, nodes):
    idx = {n: i for i, n in enumerate(names)}
    T = local_rot.shape[0]
    # Decompose all rotations at once: ZYX means R = Rz @ Ry @ Rx.
    flat = local_rot.reshape(-1, 3, 3)
    eul_zyx = Rotation.from_matrix(flat).as_euler("ZYX", degrees=True)
    eul_zyx = eul_zyx.reshape(T, len(names), 3)  # [..., 0]=Z [..., 1]=Y [..., 2]=X

    frames = []
    for t in range(T):
        values = []
        for nd in nodes:
            chans = nd["channels"]
            if nd["name"] == _ROOT_NAME:
                values.extend([0.0] * len(chans))
                continue
            j = idx[nd["name"]]
            ez, ey, ex = eul_zyx[t, j]
            euler = {"X": ex, "Y": ey, "Z": ez}
            for ch in chans:
                axis = ch[0].upper()
                if ch.endswith("position"):
                    values.append(root_positions_cm[t]["XYZ".index(axis)])
                else:
                    values.append(euler[axis])
        frames.append(" ".join(f"{v:.6g}" for v in values))
    return frames


def write_bvh(out_path, local_rot, root_positions, fps):
    """Write BVH to out_path.

    local_rot      : (T, 27, 3, 3) local rotation matrices
    root_positions : (T, 3) Hips world position in **metres** (Y-up)
    fps            : frame rate
    """
    names, parents, neutral_cm = _core27_skeleton()
    offsets_cm = _parent_relative_offsets(names, parents, neutral_cm)
    nodes = _skeleton_nodes(names)

    root_positions_cm = root_positions * 100.0  # metres -> cm

    hierarchy = _write_hierarchy(names, parents, offsets_cm)
    frames = _build_frames(local_rot, root_positions_cm, names, nodes)

    nb_channels = sum(len(nd["channels"]) for nd in nodes)
    assert len(frames[0].split()) == nb_channels, (
        f"channel count mismatch: expected {nb_channels}, got {len(frames[0].split())}"
    )

    with open(out_path, "w") as fh:
        fh.write(hierarchy.rstrip() + "\n")
        fh.write("MOTION\n")
        fh.write(f"Frames: {len(frames)}\n")
        fh.write(f"Frame Time: {1.0 / fps:.8f}\n")
        fh.write("\n".join(frames) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def convert_file(npz_path: Path, out_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    for key in ("local_rot_mats", "root_positions"):
        if key not in data:
            sys.exit(f"{npz_path}: missing key '{key}'. Keys found: {list(data.keys())}")

    rot = np.asarray(data["local_rot_mats"], dtype=np.float64)
    root = np.asarray(data["root_positions"], dtype=np.float64)
    if rot.ndim == 5:
        print(f"  note: batched npz, converting sample 0 of {rot.shape[0]}")
        rot, root = rot[0], root[0]
    if rot.shape[1] != 27:
        sys.exit(f"Expected 27 joints (Core27), got {rot.shape[1]}")

    fps = float(data["fps"]) if "fps" in data else 20.0
    text = str(data["text"]) if "text" in data else ""

    write_bvh(out_path, rot, root, fps)
    print(f"  {npz_path.name} -> {out_path}  "
          f"({rot.shape[0]} frames @ {fps:g} fps, 27 joints"
          + (f', "{text}"' if text else "") + ")")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help=".npz file or folder of .npz files")
    ap.add_argument("-o", "--output", default=None,
                    help="output .bvh path (or output folder when input is a folder)")
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        files = sorted(inp.glob("*.npz"))
        if not files:
            sys.exit(f"no .npz files in {inp}")
        out_dir = Path(args.output) if args.output else inp
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            convert_file(f, out_dir / (f.stem + ".bvh"))
    else:
        out = Path(args.output) if args.output else inp.with_suffix(".bvh")
        convert_file(inp, out)


if __name__ == "__main__":
    main()

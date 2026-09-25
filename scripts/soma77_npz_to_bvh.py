#!/usr/bin/env python3
"""Convert a SOMA77 motion NPZ into a SOMA77 BVH for SOMA Retargeter.

This is step 3 of the ARDY -> SOMA Retargeter chain::

    ARDY Core27 NPZ -> core27_to_soma77.py --to soma77 -> SOMA77 NPZ -> [this script] -> SOMA77 BVH

Output conventions
------------------
SOMA Retargeter's BVH importer (``soma_retargeter/io/bvh.py``) has two non-standard
rules that dictate the layout written here:

* a joint declaring position channels takes its translation **from the channels**;
  the OFFSET is *replaced*, not added (``wp_convert_frame_animation``);
* a rest rotation stored in a 6-number OFFSET is **dropped** (``construct_skeleton``
  always sets ``rotation = [0, 0, 0]``).

So the OFFSETs must describe the rest skeleton with no rotation, and the channels
must carry the whole pose.  This script builds that rest skeleton from
``SOMASkeleton77.neutral_joints`` — upright, Y-up, with exactly the bone lengths the
NPZ was produced with — which makes the file correct for SOMA Retargeter *and* for
tools that use the standard BVH rule (OFFSET plus channel translation), because the
Hips OFFSET is zero and only the Hips channels carry the pelvis position.

Joint frames — the part that actually matters
---------------------------------------------
ARDY keeps its skeleton in a canonical "standard T-pose" rotation space: the rotations it
writes into the NPZ are *not* expressed in the joint frames of the SOMA BVH files that
SOMA Retargeter ships with (``soma_zero_frame0.bvh``, ``assets/motions/bvh/*``).  Measured
on an ARDY file, the spine bone sits at +Y of its joint frame where the shipped skeleton
puts it at +X, and the arms differ by ~90 deg (a T-pose versus arms-down rest).  Feeding
those frames to the retargeter makes its orientation objective permanently unreachable:
the same pose retargets to 5-33 deg of rotation error from an official sample but 60-171
deg from an ARDY file.

ARDY ships the fix — ``SOMASkeleton77.global_rot_offsets``
(``standard_t_pose_global_offsets_rots.p``, the same matrix its own
``from_standard_tpose`` uses).  Right-multiplying every rotation by that per-joint
constant is a similarity transform::

    global_rot'[t, j] = global_rot[t, j] @ C_j
    local_rot'[t, j]  = C_parent^T @ local_rot[t, j] @ C_j
    offset'[c]        = C_parent^T @ offset[c]

It is exact — it cannot move a single joint — and it reproduces the shipped skeleton:
across the 76 bone directions SOMA77 shares with ``soma_zero_frame0.bvh`` the re-based
directions agree to 0.00 deg median and 2.1 deg max.  ``--keep-ardy-frames`` emits ARDY's
native rotation space instead.

* ``ROOT Root``  : 6 channels, always 0 (it sits at the origin);
* ``JOINT Hips`` : 6 channels, translation = ``root_positions`` (world pelvis, cm),
  rotation = its local rotation;
* every other joint: 3 channels ``Zrotation Yrotation Xrotation`` in degrees,
  decomposed with the BVH convention ``R = Rz @ Ry @ Rx``;
* centimetres (the NPZ is in metres), ``Frame Time = 1 / fps``.

Verified against the source NPZ: forward-kinematics of the written file reproduces
``posed_joints`` to ~1e-5 cm on all 77 joints and frames, and every joint frame matches
the shipped SOMA skeleton's convention.

Usage
-----
    python scripts/soma77_npz_to_bvh.py motion_soma77.npz -o motion.bvh
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_ROOT_NAME = "Root"
_HIPS_NAME = "Hips"
_POS_CHANNELS = ["Xposition", "Yposition", "Zposition"]
_ROT_CHANNELS = ["Zrotation", "Yrotation", "Xrotation"]


def soma77_skeleton():
    """Return (names, parent_indices, neutral) for the 77-joint SOMA skeleton."""
    from ardy.skeleton.definitions import SOMASkeleton77

    skel = SOMASkeleton77()
    names = list(skel.bone_order_names)
    parents = [int(p) for p in skel.joint_parents]  # -1 for the root
    neutral = np.asarray(skel.neutral_joints.detach().cpu().numpy(), dtype=np.float64)
    neutral = neutral - neutral[names.index(_HIPS_NAME)]  # hips at the origin
    return names, parents, neutral


def neutral_offsets(names, parents, neutral):
    """Parent-relative rest offsets in centimetres."""
    return {n: ((neutral[i] if parents[i] < 0 else neutral[i] - neutral[parents[i]]) * 100.0)
            for i, n in enumerate(names)}


def frame_corrections(names):
    """-> {name: C_j}, the change of basis out of ARDY's standard-T-pose rotation space.

    ``C_j`` is ``SOMASkeleton77.global_rot_offsets`` — the same matrix ARDY's own
    ``from_standard_tpose`` applies.  Right-multiplying every rotation by it (and
    transforming the OFFSETs by ``C_parent^T``) cannot move a single joint, and puts every
    joint frame back in the convention of the SOMA BVH files SOMA Retargeter ships with.
    Returns None when the skeleton has no such matrix.
    """
    from ardy.skeleton.definitions import SOMASkeleton77

    skel = SOMASkeleton77()
    offsets = getattr(skel, "global_rot_offsets", None)
    if offsets is None:
        return None
    mats = np.asarray(offsets.detach().cpu().numpy(), dtype=np.float64)
    if mats.shape != (len(names), 3, 3):
        return None
    return {n: mats[i] for i, n in enumerate(names)}


def rebased_offsets(names, parents, neutral, corr):
    """Parent-relative rest offsets (cm) expressed in the re-based joint frames."""
    out = {}
    for i, name in enumerate(names):
        p = parents[i]
        vec = neutral[i] if p < 0 else neutral[i] - neutral[p]
        out[name] = (np.eye(3) if p < 0 else corr[names[p]]).T @ vec * 100.0
    return out


def write_hierarchy(names, parents, offsets):
    """Render the HIERARCHY block of the BVH."""
    children = {i: [] for i in range(len(names))}
    roots = []
    for i, p in enumerate(parents):
        (roots if p < 0 else children[p]).append(i)

    def fmt(v):
        return f"{v:.6g}"

    def emit(i, depth):
        name = names[i]
        pad = "  " * depth
        chans = (_POS_CHANNELS + _ROT_CHANNELS) if name == _HIPS_NAME else _ROT_CHANNELS
        o = offsets[name]
        lines = [f"{pad}JOINT {name}", f"{pad}{{", f"{pad}  OFFSET {fmt(o[0])} {fmt(o[1])} {fmt(o[2])}",
                 f"{pad}  CHANNELS {len(chans)} {' '.join(chans)}"]
        for c in children[i]:
            lines += emit(c, depth + 1)
        lines.append(f"{pad}}}")
        return lines

    out = ["HIERARCHY", f"ROOT {_ROOT_NAME}", "{", "  OFFSET 0 0 0",
           f"  CHANNELS 6 {' '.join(_POS_CHANNELS + _ROT_CHANNELS)}"]
    for r in roots:
        out += emit(r, 2)
    out.append("}")
    return "\n".join(out) + "\n"


def skeleton_nodes(names, parents):
    """Flat node list matching the order in which channels are written."""
    nodes = [{"name": _ROOT_NAME, "channels": _POS_CHANNELS + _ROT_CHANNELS}]
    nodes += [{"name": n, "channels": (_POS_CHANNELS + _ROT_CHANNELS) if n == _HIPS_NAME else _ROT_CHANNELS}
              for n in names]
    return nodes


def load_motion(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    local_rot = np.asarray(data["local_rot_mats"], dtype=np.float64)
    root_pos = np.asarray(data["root_positions"], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).reshape(())) if "fps" in data else 20.0
    if local_rot.shape[1] == 30:
        from ardy.skeleton.definitions import SOMASkeleton30, SOMASkeleton77

        s30, s77 = SOMASkeleton30(), SOMASkeleton77()
        names30, names77 = list(s30.bone_order_names), list(s77.bone_order_names)
        rest = np.asarray(s77.relaxed_hands_rest_pose.detach().cpu().numpy(), dtype=np.float64)
        out = np.repeat(rest[None], local_rot.shape[0], axis=0)
        for i, n in enumerate(names30):
            if n in names77:
                out[:, names77.index(n)] = local_rot[:, i]
        local_rot = out
    elif local_rot.shape[1] != 77:
        raise SystemExit(f"expected 30 or 77 joints in {npz_path}, got {local_rot.shape[1]}")
    return local_rot, root_pos, fps


def build_frames(local_rot, root_pos, names, parents, nodes, scale, corr=None):
    idx = {n: i for i, n in enumerate(names)}
    frames = []
    for t in range(local_rot.shape[0]):
        values = []
        for nd in nodes:
            chans = nd["channels"]
            if nd["name"] == _ROOT_NAME:
                values.extend(0.0 for _ in chans)
                continue
            j = idx[nd["name"]]
            rot = local_rot[t, j]
            if corr is not None:
                p = parents[j]
                parent_corr = np.eye(3) if p < 0 else corr[names[p]]
                rot = parent_corr.T @ rot @ corr[nd["name"]]
            ez, ey, ex = Rotation.from_matrix(rot).as_euler("ZYX", degrees=True)
            euler = {"X": ex, "Y": ey, "Z": ez}
            trans = root_pos[t] * scale if nd["name"] == _HIPS_NAME else None
            for ch in chans:
                axis = ch[0].upper()
                if ch.endswith("position"):
                    values.append(trans["XYZ".index(axis)] if trans is not None else 0.0)
                else:
                    values.append(euler[axis])
        frames.append(" ".join(f"{v:.6g}" for v in values))
    return frames


def write_bvh(out_path, hierarchy, nodes, frames, fps):
    nb_channels = sum(len(nd["channels"]) for nd in nodes)
    assert len(frames[0].split()) == nb_channels, (len(frames[0].split()), nb_channels)
    with open(out_path, "w") as fh:
        fh.write(hierarchy.rstrip() + "\n")
        fh.write("MOTION\n")
        fh.write(f"Frames: {len(frames)}\n")
        fh.write(f"Frame Time: {1.0 / fps:.6g}\n")
        fh.write("\n".join(frames) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, help="SOMA77 (or SOMA30) motion NPZ")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--scale", type=float, default=100.0, help="metres -> output units (100 = cm)")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--keep-ardy-frames", action="store_true",
                    help="write ARDY's standard-T-pose rotation space instead of the SOMA one")
    args = ap.parse_args()

    out = args.output or args.input.with_suffix(".bvh")
    local_rot, root_pos, fps = load_motion(args.input)
    fps = args.fps or fps
    names, parents, neutral = soma77_skeleton()

    corr = None if args.keep_ardy_frames else frame_corrections(names)
    if corr is None:
        if not args.keep_ardy_frames:
            print("warning: no standard-T-pose offsets in this skeleton; writing ARDY's "
                  "standard-T-pose frames")
        offsets = neutral_offsets(names, parents, neutral)
    else:
        offsets = rebased_offsets(names, parents, neutral, corr)

    hierarchy = write_hierarchy(names, parents, offsets)
    nodes = skeleton_nodes(names, parents)
    frames = build_frames(local_rot, root_pos, names, parents, nodes, args.scale, corr)
    write_bvh(out, hierarchy, nodes, frames, fps)
    print(f"{args.input} -> {out}  ({len(frames)} frames @ {fps:g} fps, {len(nodes)} nodes)")


if __name__ == "__main__":
    main()

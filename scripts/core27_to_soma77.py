#!/usr/bin/env python3
"""
Retarget an ARDY Core27 motion NPZ onto the Kimodo SOMA skeleton (SOMA77 by default).

    ARDY Core27 NPZ  ->  SOMA77 NPZ  ->  soma77_npz_to_bvh.py  ->  SOMA77 BVH

Why this exists
---------------
Core27 and SOMA30 have *different bone lengths and different bone counts*
(Core27: 4 spine bones + 1 neck; SOMA30: 3 spine bones + 2 neck; SOMA30 has
longer legs, a shorter torso and similar arms).  A joint-name rename or a
single uniform rescaling therefore cannot work:

* uniform scale by "stature"  -> SOMA legs are too long -> permanent squat
* uniform scale by leg length -> arms/torso are stretched -> hands never reach

So this retargeter works on **bone directions**, not on positions:

    1. every SOMA bone keeps SOMA's own (neutral) bone length,
    2. its direction is copied from the corresponding Core27 bone
       (spine bones are matched by normalised arc length, so 4-vs-3 spine
        bones and 1-vs-2 neck bones are handled generically),
    3. the resulting target skeleton is exactly reachable by SOMA, so the IK
       only has to polish it (and smooth it in time).

Root translation keeps Core27's world XZ trajectory; the root height is
corrected per frame by the hip->toe drop difference of the two skeletons, so
the retargeted feet stay on the ground plane of the source motion.

Output is a Kimodo NPZ (posed_joints + global_rot_mats + local_rot_mats +
root_positions + foot_contacts + fps) that `kimodo_convert` accepts.  Only
numeric fields are written: `kimodo_convert` turns every NPZ key into a torch
tensor, so a `text`/`skeleton` string field makes it fail (`--keep-text` opts
back in).

`--to soma77` additionally expands the result to the 77-joint SOMA skeleton.
That step is *not* a retarget: SOMA30 is a strict subset of SOMA77, so the 30
local rotations are copied by joint name, the 47 remaining joints (finger
chains, toe ends, head end) get the relaxed-hands rest pose, and FK is re-run
on the 77-joint hierarchy - exactly what
`SOMASkeleton30.output_to_SOMASkeleton77` / `kimodo_convert` do internally.

Usage
-----
    python scripts/core27_to_soma77.py walk_core27.npz -o walk_soma77.npz   # soma77 (default)
    python scripts/core27_to_soma77.py walk_core27.npz --to soma30
    python scripts/core27_to_soma77.py walk_core27.npz --to both
    python scripts/soma77_npz_to_bvh.py walk_soma77.npz -o walk.bvh
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------
# Joint layouts
# --------------------------------------------------------------------------

CORE27 = [
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandEnd",
    "RightHandThumb1",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandEnd",
    "LeftHandThumb1",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
]

SOMA30 = [
    "Hips",
    "Spine1", "Spine2", "Chest",
    "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumbEnd", "LeftHandMiddleEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
]

# Spine chains: matched by arc length, not by name.
CORE_SPINE = ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head"]
SOMA_SPINE = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head"]

# soma joint -> (core parent, core child): the direction of that Core27 bone is
# copied onto the SOMA bone ending at `soma joint`.
BONE_SRC = {
    "LeftShoulder": ("Spine3", "LeftShoulder"),
    "LeftArm": ("LeftShoulder", "LeftArm"),
    "LeftForeArm": ("LeftArm", "LeftForeArm"),
    "LeftHand": ("LeftForeArm", "LeftHand"),
    "LeftHandMiddleEnd": ("LeftHand", "LeftHandEnd"),
    "LeftHandThumbEnd": ("LeftHand", "LeftHandThumb1"),
    "RightShoulder": ("Spine3", "RightShoulder"),
    "RightArm": ("RightShoulder", "RightArm"),
    "RightForeArm": ("RightArm", "RightForeArm"),
    "RightHand": ("RightForeArm", "RightHand"),
    "RightHandMiddleEnd": ("RightHand", "RightHandEnd"),
    "RightHandThumbEnd": ("RightHand", "RightHandThumb1"),
    "LeftLeg": ("Hips", "LeftUpLeg"),
    "LeftShin": ("LeftUpLeg", "LeftLeg"),
    "LeftFoot": ("LeftLeg", "LeftFoot"),
    "LeftToeBase": ("LeftFoot", "LeftToeBase"),
    "RightLeg": ("Hips", "RightUpLeg"),
    "RightShin": ("RightUpLeg", "RightLeg"),
    "RightFoot": ("RightLeg", "RightFoot"),
    "RightToeBase": ("RightFoot", "RightToeBase"),
}

# Core27 has no jaw / eyes: keep them at their neutral offset.
NO_SOURCE = ("Jaw", "LeftEye", "RightEye")

# Leaves are not optimized (their rotation drives nothing) but their *position*
# is a constraint on their parent.
NO_SOLVE = NO_SOURCE + (
    "LeftHandThumbEnd", "LeftHandMiddleEnd",
    "RightHandThumbEnd", "RightHandMiddleEnd",
)

WEIGHTS = {
    "LeftHand": 3.0, "RightHand": 3.0,
    "LeftFoot": 4.0, "RightFoot": 4.0,
    "LeftToeBase": 4.0, "RightToeBase": 4.0,
    "Head": 2.0,
    "LeftForeArm": 1.5, "RightForeArm": 1.5,
    "LeftArm": 1.0, "RightArm": 1.0,
    "LeftShin": 1.5, "RightShin": 1.5,
    "LeftLeg": 1.0, "RightLeg": 1.0,
    "LeftShoulder": 1.0, "RightShoulder": 1.0,
    "Chest": 1.0, "Spine2": 1.0, "Spine1": 0.8,
    "Neck1": 0.8, "Neck2": 0.8,
    "LeftHandMiddleEnd": 0.5, "RightHandMiddleEnd": 0.5,
    "LeftHandThumbEnd": 0.5, "RightHandThumbEnd": 0.5,
}


# --------------------------------------------------------------------------
# Small math helpers
# --------------------------------------------------------------------------

def unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros(3)
    return np.asarray(v, dtype=np.float64) / n


def rot_align(a, b):
    """Rotation taking vector `a` onto the direction of `b`."""
    a, b = unit(a), unit(b)
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return np.eye(3)
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.8:
            axis = np.array([0.0, 1.0, 0.0])
        axis = unit(axis - np.dot(axis, a) * a)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def skew(v):
    """v: (..., 3) -> skew-symmetric matrix (..., 3, 3)."""
    v = np.asarray(v, dtype=np.float64)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = np.zeros_like(x)
    return np.stack(
        [np.stack([o, -z, y], axis=-1),
         np.stack([z, o, -x], axis=-1),
         np.stack([-y, x, o], axis=-1)],
        axis=-2,
    )


def right_jacobian(x):
    """
    SO(3) right Jacobian: exp((x + dx)_x) ~= exp((J_r dx)_x) exp(x_x).

    Needed because the position derivative is taken with respect to a *left*
    perturbation delta, while the optimiser works on the rotvec parameter.
    """
    x = np.asarray(x, dtype=np.float64)
    th = np.linalg.norm(x, axis=-1, keepdims=True)          # (..., 1)
    K = skew(x)
    small = th[..., 0] < 1e-8
    a = np.where(small, 0.5, (1.0 - np.cos(th[..., 0])) / np.maximum(th[..., 0] ** 2, 1e-12))
    b = np.where(small, 1.0 / 6.0,
                 (th[..., 0] - np.sin(th[..., 0])) / np.maximum(th[..., 0] ** 3, 1e-12))
    I = np.eye(3)
    return I + a[..., None, None] * K + b[..., None, None] * (K @ K)


def ancestor_mask(parents):
    """anc[j, k] = True when rotating joint j moves joint k."""
    J = len(parents)
    anc = np.zeros((J, J), dtype=bool)
    for k in range(J):
        j = parents[k]
        while j >= 0:
            anc[j, k] = True
            j = parents[j]
    return anc


def kabsch(A, B, w):
    """
    Weighted rotation R minimising sum_i w_i |R a_i - b_i|^2 on directions.

    A, B: (m, 3) - only directions matter, magnitudes are ignored.
    """
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    H = (An * w[:, None]).T @ Bn
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def fk(local_R, offsets, parents, root_idx, root_pos):
    """Hierarchical FK. offsets[j] must be parent-relative (neutral[j]-neutral[parent])."""
    J = len(offsets)
    g = np.empty_like(local_R)
    p = np.empty((J, 3), dtype=np.float64)
    for j in range(J):
        pi = parents[j]
        if pi < 0:
            g[j] = local_R[j]
            p[j] = root_pos
        else:
            g[j] = g[pi] @ local_R[j]
            p[j] = p[pi] + g[pi] @ offsets[j]
    return g, p


def point_at(pts, u):
    """Point at normalised arc length u in [0, 1] along a polyline."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-12:
        return pts[0]
    d = u * total
    acc = 0.0
    for i, s in enumerate(seg):
        if acc + s >= d or i == len(seg) - 1:
            t = 0.0 if s < 1e-12 else (d - acc) / s
            return pts[i] + t * (pts[i + 1] - pts[i])
        acc += s
    return pts[-1]


# --------------------------------------------------------------------------
# Skeleton
# --------------------------------------------------------------------------

def load_soma30():
    """Kimodo is the reference source; ARDY ships an identical definition."""
    errors = []
    for mod in ("kimodo.skeleton", "ardy.skeleton"):
        try:
            m = __import__(mod, fromlist=["SOMASkeleton30"])
            return m.SOMASkeleton30(load=True)
        except Exception as e:  # pragma: no cover - depends on install
            errors.append(f"  {mod}: {type(e).__name__}: {e}")
    raise SystemExit("Cannot import SOMASkeleton30:\n" + "\n".join(errors))


def skeleton_arrays(skel):
    names = [str(n) for n in skel.bone_order_names]
    parents = np.asarray(skel.joint_parents.detach().cpu().numpy(), dtype=int)
    neutral = np.asarray(skel.neutral_joints.detach().cpu().numpy(), dtype=np.float64)
    root_idx = int(getattr(skel, "root_idx", 0))
    neutral = neutral - neutral[root_idx]
    return names, parents, neutral, root_idx


def expand_to_soma77(payload30, skel30):
    """
    SOMA30 -> SOMA77, on numpy arrays.

    Not a retarget: SOMA30 is a strict subset of SOMA77 (same joint names, same
    parents, same bone lengths).  The 30 local rotations are copied by joint
    name onto the joints with the same name, the 47 remaining joints keep the
    relaxed-hands rest pose, and FK is re-run on the 77-joint hierarchy.  This
    is the numpy equivalent of `SOMASkeleton30.to_SOMASkeleton77`.
    """
    skel77 = skel30.somaskel77
    names77, parents77, neutral77, root77 = skeleton_arrays(skel77)

    rest = skel77.relaxed_hands_rest_pose
    rest = rest.detach().cpu().numpy() if hasattr(rest, "detach") else np.asarray(rest)
    rest = np.asarray(rest, dtype=np.float64).reshape(len(names77), 3, 3)

    local30 = np.asarray(payload30["local_rot_mats"], dtype=np.float64)
    root = np.asarray(payload30["root_positions"], dtype=np.float64)
    T = local30.shape[0]

    local77 = np.tile(rest[None], (T, 1, 1, 1))
    local77[:, skel30.get_skel_slice(skel77)] = local30    # 30 names -> 77 indices

    offsets = neutral77.copy()
    for j in range(len(names77)):
        offsets[j] = 0.0 if parents77[j] < 0 else neutral77[j] - neutral77[parents77[j]]

    global77 = np.zeros((T, len(names77), 3, 3), dtype=np.float64)
    pos77 = np.zeros((T, len(names77), 3), dtype=np.float64)
    for t in range(T):
        global77[t], pos77[t] = fk(local77[t], offsets, parents77, root77, root[t])

    payload77 = dict(payload30)
    payload77["local_rot_mats"] = local77.astype(np.float32)
    payload77["global_rot_mats"] = global77.astype(np.float32)
    payload77["posed_joints"] = pos77.astype(np.float32)
    # keep the retargeter's exact positions on the 30 source joints; skinning
    # consumes [global_rot | posed_joints] jointly
    pos77[:, skel30.get_skel_slice(skel77)] = np.asarray(payload30["posed_joints"], dtype=np.float64)
    payload77["posed_joints"] = pos77.astype(np.float32)
    if "foot_contacts" in payload30:
        fc = np.asarray(payload30["foot_contacts"])
        # [T, 4]: [L_heel, L_toe, R_heel, R_toe]
        # -> [T, 6]: [L_heel, L_toe, L_toe_end, R_heel, R_toe, R_toe_end]
        payload77["foot_contacts"] = np.concatenate(
            [fc[..., :2], fc[..., 1:2], fc[..., 2:4], fc[..., 3:4]], axis=-1
        )
    return payload77


def output_path(args, target, multi):
    """Where to write the NPZ for skeleton `target`."""
    if args.output:
        out = Path(args.output)
        if multi:
            return out.with_name(out.stem + f"_{target}.npz")
        return out
    return args.input.with_name(args.input.stem + f"_{target}.npz")


# --------------------------------------------------------------------------
# Target skeleton
# --------------------------------------------------------------------------

def build_target(core, ci, neutral, names, si, parents, soma_u):
    """
    SOMA-shaped target pose: SOMA bone lengths + Core27 bone directions.
    Returned with the hips at the origin.
    """
    J = len(names)
    core_spine = np.stack([core[ci[n]] for n in CORE_SPINE])

    dirs = {}
    for k in range(1, len(SOMA_SPINE)):
        u0, u1 = soma_u[k - 1], soma_u[k]
        dirs[SOMA_SPINE[k]] = point_at(core_spine, u1) - point_at(core_spine, u0)
    for soma_name, (ca, cb) in BONE_SRC.items():
        if soma_name in si and ca in ci and cb in ci:
            dirs[soma_name] = core[ci[cb]] - core[ci[ca]]

    target = np.zeros((J, 3), dtype=np.float64)
    for j in range(J):
        pi = parents[j]
        if pi < 0:
            continue
        off = neutral[j] - neutral[pi]
        v = dirs.get(names[j])
        if v is None or np.linalg.norm(v) < 1e-9:
            v = off
        target[j] = target[pi] + np.linalg.norm(off) * unit(v)
    return target


def pelvis_rotation(core, ci, neutral, si):
    """
    Absolute orientation of the pelvis, from an explicit, well conditioned
    frame (spine up / hip-to-hip side).  This is what preserves the facing
    direction of the source motion.
    """
    def basis(up, side):
        u = unit(up)
        s = unit(side - np.dot(side, u) * u)
        f = unit(np.cross(s, u))
        return np.stack([u, s, f], axis=1)

    # Mc / Ms hold the *same* anatomical axes, expressed in the world frame of
    # the posed Core27 skeleton and of the SOMA rest pose.  The root rotation
    # maps the rest frame onto the posed frame:  G[Hips] @ Ms = Mc.
    Mc = basis(core[ci["Spine3"]] - core[ci["Hips"]],
               core[ci["LeftUpLeg"]] - core[ci["RightUpLeg"]])
    Ms = basis(neutral[si["Chest"]] - neutral[si["Hips"]],
               neutral[si["LeftLeg"]] - neutral[si["RightLeg"]])
    return Mc @ Ms.T


def init_local(target, neutral, parents, root_idx, root_R, names, skip=NO_SOURCE):
    """
    Absolute orientation of every joint (Kabsch over its children's bone
    directions), converted to local rotations.  Already very close to the
    target, so it doubles as the IK initialisation.
    """
    J = len(parents)
    children = [[] for _ in range(J)]
    for j in range(J):
        pj = parents[j]
        if pj >= 0 and names[j] not in skip:
            children[pj].append(j)

    g = np.tile(np.eye(3), (J, 1, 1))
    g[root_idx] = root_R
    for j in range(J):
        if j == root_idx:
            continue
        if len(children[j]) >= 2:
            A = np.stack([neutral[k] - neutral[j] for k in children[j]])
            B = np.stack([target[k] - target[j] for k in children[j]])
            g[j] = kabsch(A, B, np.linalg.norm(A, axis=1))
        elif len(children[j]) == 1:
            # Single child: a rank-1 Kabsch leaves the twist about the bone
            # arbitrary. Swing the parent frame onto the bone direction instead
            # (twist-free, identity at rest) - arbitrary twist breaks skinning.
            k = children[j][0]
            g[j] = rot_align(g[pj] @ (neutral[k] - neutral[j]), target[k] - target[j]) @ g[pj]
        else:
            g[j] = g[parents[j]]

    local = np.empty_like(g)
    for j in range(J):
        pi = parents[j]
        local[j] = g[j] if pi < 0 else g[pi].T @ g[j]
    return local


# --------------------------------------------------------------------------
# Per-frame IK
# --------------------------------------------------------------------------

def solve_frame(target, offsets, parents, root_idx, init_local, prev_x, ctx,
                smooth, iters, root_est=None, root_w=0.0, tol=1e-6):
    """
    Damped Gauss-Newton (Levenberg-Marquardt) on the local rotations.

    The Jacobian is analytic: rotating joint j by a local rotvec delta turns
    into a *global* rotation about `G[parent(j)] @ delta`, so every descendant
    k moves by  omega x (pos[k] - pos[j]).  That is exact and removes the 69
    extra FK evaluations a finite-difference Jacobian would need.

    The normal equations are solved with a Cholesky solver, not an SVD: on
    some BLAS builds a 147x69 SVD costs ~50 ms, which would dominate runtime.
    """
    solve_idx = ctx["solve_idx"]
    tgt_idx = ctx["tgt_idx"]
    w = ctx["w"]
    anc = ctx["anc"][np.ix_(solve_idx, tgt_idx)]      # (P, K)
    P, K = len(solve_idx), len(tgt_idx)
    zero = np.zeros(3)

    def unpack(x):
        R = init_local.copy()
        R[solve_idx] = Rotation.from_rotvec(x.reshape(-1, 3)).as_matrix()
        return R

    def rj(x):
        R = unpack(x)
        g, p = fk(R, offsets, parents, root_idx, zero)
        r = ((p[tgt_idx] - target[tgt_idx]) * w[:, None]).ravel()

        v = p[tgt_idx][:, None, :] - p[solve_idx][None, :, :]   # (K, P, 3)
        Gp = np.empty((P, 3, 3))
        for a, j in enumerate(solve_idx):
            pj = parents[j]
            Gp[a] = np.eye(3) if pj < 0 else g[pj]
        # d(pos_k) / d(delta_j) = -skew(pos_k - pos_j) @ G[parent(j)]
        blocks = -np.einsum("kpab,pbc->kpac", skew(v), Gp)      # (K, P, 3, 3)
        blocks *= anc.T[:, :, None, None]
        # chain rule: delta_j = J_r(x_j) dx_j
        rv = x.reshape(-1, 3)
        blocks = np.einsum("kpab,pbc->kpac", blocks, right_jacobian(rv))
        J = blocks.transpose(0, 2, 1, 3).reshape(K * 3, P * 3)
        J *= np.repeat(w, 3)[:, None]
        if prev_x is not None and smooth > 0.0:
            r = np.concatenate([r, (x - prev_x) * smooth])
            J = np.vstack([J, np.eye(P * 3) * smooth])
        if root_est is not None and root_w > 0.0:
            # Anchor the pelvis to the source facing; without this the solver
            # spends root yaw on reaching the feet and the body swings.
            rp = int(np.where(solve_idx == root_idx)[0][0]) * 3
            e = Rotation.from_matrix(root_est.T @ R[root_idx]).as_rotvec()
            extra = np.zeros((3, P * 3))
            extra[:, rp:rp + 3] = root_est.T @ right_jacobian(x[rp:rp + 3])
            r = np.concatenate([r, e * root_w])
            J = np.vstack([J, extra * root_w])
        return r, J

    x = Rotation.from_matrix(init_local[solve_idx]).as_rotvec().ravel()
    r, J = rj(x)
    cost = float(r @ r)
    if iters <= 0:
        return init_local, x, cost

    lam = 1e-3
    for _ in range(iters):
        H = J.T @ J
        grad = J.T @ r
        d = np.maximum(np.diag(H), 1e-9)
        accepted = False
        for _try in range(8):
            try:
                dx = -cho_solve(
                    cho_factor(H + lam * np.diag(d), lower=True, check_finite=False),
                    grad, check_finite=False,
                )
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            xn = x + dx
            rn, Jn = rj(xn)
            cn = float(rn @ rn)
            if cn < cost:
                gain = (cost - cn) / max(cost, 1e-12)
                x, r, J, cost = xn, rn, Jn, cn
                lam = max(lam * 0.3, 1e-9)
                accepted = True
                break
            lam *= 4.0
        if not accepted or gain < tol:
            break

    return unpack(x), x, cost


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def retarget(args):
    data = np.load(args.input, allow_pickle=True)
    if "posed_joints" not in data:
        raise ValueError("Input NPZ has no `posed_joints`.")

    core_pos = np.asarray(data["posed_joints"], dtype=np.float64)
    if core_pos.ndim == 4:
        if core_pos.shape[0] != 1:
            raise ValueError(f"One sample at a time, got {core_pos.shape}.")
        core_pos = core_pos[0]
    if core_pos.ndim != 3 or core_pos.shape[1:] != (27, 3):
        raise ValueError(f"Expected posed_joints [T,27,3], got {core_pos.shape}.")

    if args.limit:
        core_pos = core_pos[: args.limit]
    T = core_pos.shape[0]

    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else args.fps
    text = str(np.asarray(data["text"]).reshape(-1)[0]) if "text" in data else ""

    skel = load_soma30()
    names, parents, neutral, root_idx = skeleton_arrays(skel)
    si = {n: i for i, n in enumerate(names)}
    ci = {n: i for i, n in enumerate(CORE27)}
    missing = [n for n in SOMA30 if n not in si]
    if missing:
        raise RuntimeError(f"Skeleton is missing SOMA30 joints: {missing}")
    if np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, len(names))):
        raise RuntimeError("Skeleton joint order is not topological.")

    offsets = neutral.copy()
    for j in range(len(names)):
        if parents[j] >= 0:
            offsets[j] = neutral[j] - neutral[parents[j]]
        else:
            offsets[j] = 0.0

    # Normalised arc length of the SOMA spine (fixed: taken from the neutral pose).
    soma_spine_pts = np.stack([neutral[si[n]] for n in SOMA_SPINE])
    seg = np.linalg.norm(np.diff(soma_spine_pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    soma_u = cum / cum[-1]

    solve_idx = np.array([si[n] for n in names if n not in NO_SOLVE], dtype=int)
    tgt_names = [n for n in names if n not in NO_SOURCE and n != names[root_idx]]
    tgt_idx = np.array([si[n] for n in tgt_names], dtype=int)
    length_scale = float(np.median(np.linalg.norm(offsets[tgt_idx], axis=1)))
    ctx = {
        "solve_idx": solve_idx,
        "tgt_idx": tgt_idx,
        "w": np.array([WEIGHTS.get(n, 1.0) for n in tgt_names]) / length_scale,
        "anc": ancestor_mask(parents),
    }

    # Feet used to re-anchor the root height.
    feet = [("LeftToeBase", "LeftToeBase"), ("RightToeBase", "RightToeBase")]

    out_local = np.zeros((T, len(names), 3, 3), dtype=np.float64)
    out_global = np.zeros_like(out_local)
    out_pos = np.zeros((T, len(names), 3), dtype=np.float64)
    out_root = np.zeros((T, 3), dtype=np.float64)
    errs = []
    costs = []

    prev_x = None
    t0 = time.time()
    for t in range(T):
        core = core_pos[t]

        target_rel = build_target(core, ci, neutral, names, si, parents, soma_u)

        # Root height: keep each foot at the height Core27 puts it at.
        d_core = [core[ci["Hips"]][1] - core[ci[c]][1] for _, c in feet]
        d_soma = [-target_rel[si[s]][1] for s, _ in feet]
        root = core[ci["Hips"]].copy()
        root[1] += float(np.mean(d_soma) - np.mean(d_core))

        root_est = pelvis_rotation(core, ci, neutral, si)
        init = init_local(
            target_rel, neutral, parents, root_idx, root_est, names,
        )
        local, x, cost = solve_frame(
            target_rel, offsets, parents, root_idx, init, prev_x, ctx,
            args.smooth, args.iters, root_est=root_est, root_w=args.root_weight,
        )
        prev_x = x

        g, p = fk(local, offsets, parents, root_idx, root)
        out_local[t] = local
        out_global[t] = g
        out_pos[t] = p
        out_root[t] = root

        tgt_world = target_rel + root
        errs.append(np.linalg.norm(p[tgt_idx] - tgt_world[tgt_idx], axis=1))
        costs.append(cost)

        if t % 20 == 0 or t == T - 1:
            el = time.time() - t0
            print(f"\r[core27->soma30] {t + 1}/{T}  ({el:.1f}s)", end="", flush=True)
    print()

    errs = np.asarray(errs)
    per_joint = dict(zip(tgt_names, errs.mean(axis=0) * 100.0))

    if "foot_contacts" in data:
        fc = np.asarray(data["foot_contacts"])
        if fc.ndim == 3 and fc.shape[0] == 1:
            fc = fc[0]
        fc = fc if fc.shape == (T, 4) else np.zeros((T, 4), dtype=bool)
    else:
        fc = np.zeros((T, 4), dtype=bool)

    # kimodo_convert turns *every* key of the NPZ into a torch tensor, so a
    # stray string field ("text", "skeleton", ...) makes it fail outright.
    # Only numeric fields are written unless --keep-text is given.
    payload30 = {
        "posed_joints": out_pos.astype(np.float32),
        "global_rot_mats": out_global.astype(np.float32),
        "local_rot_mats": out_local.astype(np.float32),
        "root_positions": out_root.astype(np.float32),
        "foot_contacts": fc,
        "fps": np.asarray(fps),
    }
    if args.keep_text:
        payload30["text"] = np.asarray(text)

    targets = ("soma30", "soma77") if args.to == "both" else (args.to,)
    for tgt in targets:
        payload = payload30 if tgt == "soma30" else expand_to_soma77(payload30, skel)
        out = output_path(args, tgt, multi=len(targets) > 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **payload)
        print(f"[core27->{tgt}] wrote {out}  frames={T} fps={fps:g} "
              f"joints={payload['posed_joints'].shape[1]} "
              f"({time.time() - t0:.1f}s)")
    print(f"[core27->soma30] IK cost  median={np.median(costs):.4f} "
          f"max={np.max(costs):.4f}")
    print("[core27->soma30] mean |target - solved| (cm):")
    for grp in ("Hands", "Feet", "Toes", "Head", "Spine"):
        keys = {
            "Hands": ("LeftHand", "RightHand"),
            "Feet": ("LeftFoot", "RightFoot"),
            "Toes": ("LeftToeBase", "RightToeBase"),
            "Head": ("Head",),
            "Spine": ("Spine1", "Spine2", "Chest"),
        }[grp]
        vals = [per_joint[k] for k in keys if k in per_joint]
        if vals:
            print(f"    {grp:<6s} {np.mean(vals):6.2f}")
    print(f"[core27->soma30] ground: source min toe y="
          f"{core_pos[:, [ci[c] for _, c in feet], 1].min():.3f}  "
          f"output min toe y="
          f"{out_pos[:, [si[s] for s, _ in feet], 1].min():.3f}")

    ortho = np.max(np.abs(
        np.matmul(out_local.transpose(0, 1, 3, 2), out_local) - np.eye(3)
    ))
    det = np.min(np.linalg.det(out_local))
    print(f"[core27->soma30] rotations: orthogonality err={ortho:.2e} "
          f"min det={det:.6f}")
    if det < 0.999:
        raise RuntimeError("Invalid rotation matrix in output.")


def main():
    ap = argparse.ArgumentParser(
        description="Retarget ARDY Core27 motion onto the Kimodo SOMA skeleton (SOMA77 by default)."
    )
    ap.add_argument("input", type=Path, help="ARDY Core27 NPZ")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=20.0,
                    help="Used when the input NPZ has no fps field.")
    ap.add_argument("--iters", type=int, default=12,
                    help="Levenberg-Marquardt iterations per frame "
                         "(0 = analytic initialisation only, very fast).")
    ap.add_argument("--smooth", type=float, default=0.10,
                    help="Temporal regularization toward the previous frame.")
    ap.add_argument("--root-weight", type=float, default=3.0,
                    help="How strongly the pelvis keeps the source facing "
                         "(0 lets the solver swing the body to reach the feet).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only convert the first N frames (debug).")
    ap.add_argument("--keep-text", action="store_true",
                    help="Also store the `text` string field. kimodo_convert "
                         "cannot read NPZ files that contain string fields.")
    ap.add_argument("--to", choices=("soma30", "soma77", "both"), default="soma77",
                    help="Output skeleton: SOMA30 (IK result as-is), SOMA77 "
                         "(expanded with relaxed hands) or both files.")
    retarget(ap.parse_args())


if __name__ == "__main__":
    main()

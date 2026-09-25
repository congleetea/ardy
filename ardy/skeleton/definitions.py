# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete skeleton definitions: SOMA, G1, and Core with joint names and hierarchy."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation

from ..tools import ensure_batched
from .base import SkeletonBase


class SOMASkeleton77(SkeletonBase):
    """High-detail 77-joint SOMA skeleton with full finger and toe chains."""

    name = "somaskel77"

    right_foot_joint_names = [
        "RightFoot",
        "RightToeBase",
        "RightToeEnd",
    ]  # in order of chain
    left_foot_joint_names = [
        "LeftFoot",
        "LeftToeBase",
        "LeftToeEnd",
    ]  # in order of chain
    right_hand_joint_names = [
        "RightHand",
        "RightHandThumb1",
        "RightHandThumb2",
        "RightHandThumb3",
        "RightHandThumbEnd",
        "RightHandIndex1",
        "RightHandIndex2",
        "RightHandIndex3",
        "RightHandIndex4",
        "RightHandIndexEnd",
        "RightHandMiddle1",
        "RightHandMiddle2",
        "RightHandMiddle3",
        "RightHandMiddle4",
        "RightHandMiddleEnd",
        "RightHandRing1",
        "RightHandRing2",
        "RightHandRing3",
        "RightHandRing4",
        "RightHandRingEnd",
        "RightHandPinky1",
        "RightHandPinky2",
        "RightHandPinky3",
        "RightHandPinky4",
        "RightHandPinkyEnd",
    ]  # in order of chain
    left_hand_joint_names = [
        "LeftHand",
        "LeftHandThumb1",
        "LeftHandThumb2",
        "LeftHandThumb3",
        "LeftHandThumbEnd",
        "LeftHandIndex1",
        "LeftHandIndex2",
        "LeftHandIndex3",
        "LeftHandIndex4",
        "LeftHandIndexEnd",
        "LeftHandMiddle1",
        "LeftHandMiddle2",
        "LeftHandMiddle3",
        "LeftHandMiddle4",
        "LeftHandMiddleEnd",
        "LeftHandRing1",
        "LeftHandRing2",
        "LeftHandRing3",
        "LeftHandRing4",
        "LeftHandRingEnd",
        "LeftHandPinky1",
        "LeftHandPinky2",
        "LeftHandPinky3",
        "LeftHandPinky4",
        "LeftHandPinkyEnd",
    ]  # in order of chain

    hip_joint_names = ["RightLeg", "LeftLeg"]  # in order [right, left]

    bone_order_names_with_parents = [
        ("Hips", None),
        ("Spine1", "Hips"),
        ("Spine2", "Spine1"),
        ("Chest", "Spine2"),
        ("Neck1", "Chest"),
        ("Neck2", "Neck1"),
        ("Head", "Neck2"),
        ("HeadEnd", "Head"),
        ("Jaw", "Head"),
        ("LeftEye", "Head"),
        ("RightEye", "Head"),
        ("LeftShoulder", "Chest"),
        ("LeftArm", "LeftShoulder"),
        ("LeftForeArm", "LeftArm"),
        ("LeftHand", "LeftForeArm"),
        ("LeftHandThumb1", "LeftHand"),
        ("LeftHandThumb2", "LeftHandThumb1"),
        ("LeftHandThumb3", "LeftHandThumb2"),
        ("LeftHandThumbEnd", "LeftHandThumb3"),
        ("LeftHandIndex1", "LeftHand"),
        ("LeftHandIndex2", "LeftHandIndex1"),
        ("LeftHandIndex3", "LeftHandIndex2"),
        ("LeftHandIndex4", "LeftHandIndex3"),
        ("LeftHandIndexEnd", "LeftHandIndex4"),
        ("LeftHandMiddle1", "LeftHand"),
        ("LeftHandMiddle2", "LeftHandMiddle1"),
        ("LeftHandMiddle3", "LeftHandMiddle2"),
        ("LeftHandMiddle4", "LeftHandMiddle3"),
        ("LeftHandMiddleEnd", "LeftHandMiddle4"),
        ("LeftHandRing1", "LeftHand"),
        ("LeftHandRing2", "LeftHandRing1"),
        ("LeftHandRing3", "LeftHandRing2"),
        ("LeftHandRing4", "LeftHandRing3"),
        ("LeftHandRingEnd", "LeftHandRing4"),
        ("LeftHandPinky1", "LeftHand"),
        ("LeftHandPinky2", "LeftHandPinky1"),
        ("LeftHandPinky3", "LeftHandPinky2"),
        ("LeftHandPinky4", "LeftHandPinky3"),
        ("LeftHandPinkyEnd", "LeftHandPinky4"),
        ("RightShoulder", "Chest"),
        ("RightArm", "RightShoulder"),
        ("RightForeArm", "RightArm"),
        ("RightHand", "RightForeArm"),
        ("RightHandThumb1", "RightHand"),
        ("RightHandThumb2", "RightHandThumb1"),
        ("RightHandThumb3", "RightHandThumb2"),
        ("RightHandThumbEnd", "RightHandThumb3"),
        ("RightHandIndex1", "RightHand"),
        ("RightHandIndex2", "RightHandIndex1"),
        ("RightHandIndex3", "RightHandIndex2"),
        ("RightHandIndex4", "RightHandIndex3"),
        ("RightHandIndexEnd", "RightHandIndex4"),
        ("RightHandMiddle1", "RightHand"),
        ("RightHandMiddle2", "RightHandMiddle1"),
        ("RightHandMiddle3", "RightHandMiddle2"),
        ("RightHandMiddle4", "RightHandMiddle3"),
        ("RightHandMiddleEnd", "RightHandMiddle4"),
        ("RightHandRing1", "RightHand"),
        ("RightHandRing2", "RightHandRing1"),
        ("RightHandRing3", "RightHandRing2"),
        ("RightHandRing4", "RightHandRing3"),
        ("RightHandRingEnd", "RightHandRing4"),
        ("RightHandPinky1", "RightHand"),
        ("RightHandPinky2", "RightHandPinky1"),
        ("RightHandPinky3", "RightHandPinky2"),
        ("RightHandPinky4", "RightHandPinky3"),
        ("RightHandPinkyEnd", "RightHandPinky4"),
        ("LeftLeg", "Hips"),
        ("LeftShin", "LeftLeg"),
        ("LeftFoot", "LeftShin"),
        ("LeftToeBase", "LeftFoot"),
        ("LeftToeEnd", "LeftToeBase"),
        ("RightLeg", "Hips"),
        ("RightShin", "RightLeg"),
        ("RightFoot", "RightShin"),
        ("RightToeBase", "RightFoot"),
        ("RightToeEnd", "RightToeBase"),
    ]

    @property
    def relaxed_hands_rest_pose(self):
        # lazy loading
        if hasattr(self, "_relaxed_hands_rest_pose"):
            return self._relaxed_hands_rest_pose

        relaxed_hands_pose_path = Path(self.folder) / "relaxed_hands_rest_pose.npy"
        relaxed_hands_rest_pose = torch.from_numpy(np.load(relaxed_hands_pose_path)).squeeze()
        self.register_buffer(
            "_relaxed_hands_rest_pose",
            relaxed_hands_rest_pose,
            persistent=False,
        )
        return self._relaxed_hands_rest_pose


class SOMASkeleton30(SkeletonBase):
    """Compact 30-joint SOMA variant with reduced hand and end-effector detail."""

    name = "somaskel30"

    right_foot_joint_names = [
        "RightFoot",
        "RightToeBase",
    ]  # in order of chain
    left_foot_joint_names = [
        "LeftFoot",
        "LeftToeBase",
    ]  # in order of chain
    right_hand_joint_names = [
        "RightHand",
        "RightHandMiddleEnd",
    ]  # in order of chain
    left_hand_joint_names = [
        "LeftHand",
        "LeftHandMiddleEnd",
    ]  # in order of chain

    hip_joint_names = ["RightLeg", "LeftLeg"]  # in order [right, left]

    bone_order_names_with_parents = [
        ("Hips", None),
        ("Spine1", "Hips"),
        ("Spine2", "Spine1"),
        ("Chest", "Spine2"),
        ("Neck1", "Chest"),
        ("Neck2", "Neck1"),
        ("Head", "Neck2"),
        ("Jaw", "Head"),
        ("LeftEye", "Head"),
        ("RightEye", "Head"),
        ("LeftShoulder", "Chest"),
        ("LeftArm", "LeftShoulder"),
        ("LeftForeArm", "LeftArm"),
        ("LeftHand", "LeftForeArm"),
        ("LeftHandThumbEnd", "LeftHand"),
        ("LeftHandMiddleEnd", "LeftHand"),
        ("RightShoulder", "Chest"),
        ("RightArm", "RightShoulder"),
        ("RightForeArm", "RightArm"),
        ("RightHand", "RightForeArm"),
        ("RightHandThumbEnd", "RightHand"),
        ("RightHandMiddleEnd", "RightHand"),
        ("LeftLeg", "Hips"),
        ("LeftShin", "LeftLeg"),
        ("LeftFoot", "LeftShin"),
        ("LeftToeBase", "LeftFoot"),
        ("RightLeg", "Hips"),
        ("RightShin", "RightLeg"),
        ("RightFoot", "RightShin"),
        ("RightToeBase", "RightFoot"),
    ]

    @property
    def somaskel77(self):
        # lazy loading
        if not hasattr(self, "_somaskel77"):
            self._somaskel77 = SOMASkeleton77()
        return self._somaskel77

    @ensure_batched(local_joint_rots_subset=4)
    def to_SOMASkeleton77(self, local_joint_rots_subset: torch.Tensor):
        # Converting from 30-joint to 77-joint to have relaxed hands

        device = local_joint_rots_subset.device
        nF = len(local_joint_rots_subset)
        local_joint_rots_mats = self.somaskel77.relaxed_hands_rest_pose.clone().to(device).repeat(nF, 1, 1, 1)

        skel_slice = self.get_skel_slice(self.somaskel77)
        local_joint_rots_mats[:, skel_slice] = local_joint_rots_subset
        return local_joint_rots_mats

    @ensure_batched(local_joint_rots_full=4)  # [BT, J, 3, 3]
    def from_SOMASkeleton77(self, local_joint_rots_full: torch.Tensor) -> torch.Tensor:
        """Extract the 30-joint subset from 77-joint local rotation data."""
        skel_slice = self.get_skel_slice(self.somaskel77)
        return local_joint_rots_full[:, skel_slice]

    def output_to_SOMASkeleton77(self, output: dict, preserve_pos: bool = False) -> dict:
        """Convert model output dict from somaskel30 to somaskel77.

        Expands local_rot_mats to 77 joints, re-runs FK for global_rot_mats and posed_joints. Foot
        contacts are expanded from 4 channels to 6 (toe-end copies toe-base contact).

        With ``preserve_pos=True`` the 30 source joints keep the input's exact
        ``posed_joints`` instead of the FK-recomputed ones (used when the input
        positions are retargeter targets; skinning consumes
        ``[global_rot | posed_joints]`` jointly, so exact positions with
        FK-derived rotations is a valid, preferred combination).
        """
        local_rot_mats_77 = self.to_SOMASkeleton77(output["local_rot_mats"])
        root_positions = output["root_positions"]
        global_rot_mats_77, posed_joints_77, _ = self.somaskel77.fk(local_rot_mats_77, root_positions)
        if preserve_pos:
            skel_slice = self.get_skel_slice(self.somaskel77)
            posed_joints_77 = posed_joints_77.clone()
            posed_joints_77[:, skel_slice] = output["posed_joints"]
        out_77 = dict(output)
        out_77["local_rot_mats"] = local_rot_mats_77
        out_77["global_rot_mats"] = global_rot_mats_77
        out_77["posed_joints"] = posed_joints_77

        if "foot_contacts" in output:
            fc = output["foot_contacts"]  # [..., 4]: [L_heel, L_toe, R_heel, R_toe]
            # -> [..., 6]: [L_heel, L_toe, L_toe_end, R_heel, R_toe, R_toe_end]
            out_77["foot_contacts"] = torch.cat([fc[..., :2], fc[..., 1:2], fc[..., 2:4], fc[..., 3:4]], dim=-1)

        return out_77


class G1Skeleton34(SkeletonBase):
    """Unitree G1 skeleton with 32 articulated joints plus 2 toe endpoints."""

    name = "g1skel34"
    right_foot_joint_names = ["right_ankle_roll_skel", "right_toe_base"]
    left_foot_joint_names = ["left_ankle_roll_skel", "left_toe_base"]
    right_hand_joint_names = ["right_wrist_yaw_skel", "right_hand_roll_skel"]
    left_hand_joint_names = ["left_wrist_yaw_skel", "left_hand_roll_skel"]

    hip_joint_names = [
        "right_hip_pitch_skel",
        "left_hip_pitch_skel",
    ]  # used to calculate root orientation, only need 1 pair of hip joints

    bone_order_names_with_parents = [
        ("pelvis_skel", None),
        ("left_hip_pitch_skel", "pelvis_skel"),
        ("left_hip_roll_skel", "left_hip_pitch_skel"),
        ("left_hip_yaw_skel", "left_hip_roll_skel"),
        ("left_knee_skel", "left_hip_yaw_skel"),
        ("left_ankle_pitch_skel", "left_knee_skel"),
        ("left_ankle_roll_skel", "left_ankle_pitch_skel"),
        ("left_toe_base", "left_ankle_roll_skel"),
        ("right_hip_pitch_skel", "pelvis_skel"),
        ("right_hip_roll_skel", "right_hip_pitch_skel"),
        ("right_hip_yaw_skel", "right_hip_roll_skel"),
        ("right_knee_skel", "right_hip_yaw_skel"),
        ("right_ankle_pitch_skel", "right_knee_skel"),
        ("right_ankle_roll_skel", "right_ankle_pitch_skel"),
        ("right_toe_base", "right_ankle_roll_skel"),
        ("waist_yaw_skel", "pelvis_skel"),
        ("waist_roll_skel", "waist_yaw_skel"),
        ("waist_pitch_skel", "waist_roll_skel"),
        ("left_shoulder_pitch_skel", "waist_pitch_skel"),
        ("left_shoulder_roll_skel", "left_shoulder_pitch_skel"),
        ("left_shoulder_yaw_skel", "left_shoulder_roll_skel"),
        ("left_elbow_skel", "left_shoulder_yaw_skel"),
        ("left_wrist_roll_skel", "left_elbow_skel"),
        ("left_wrist_pitch_skel", "left_wrist_roll_skel"),
        ("left_wrist_yaw_skel", "left_wrist_pitch_skel"),
        ("left_hand_roll_skel", "left_wrist_yaw_skel"),
        ("right_shoulder_pitch_skel", "waist_pitch_skel"),
        ("right_shoulder_roll_skel", "right_shoulder_pitch_skel"),
        ("right_shoulder_yaw_skel", "right_shoulder_roll_skel"),
        ("right_elbow_skel", "right_shoulder_yaw_skel"),
        ("right_wrist_roll_skel", "right_elbow_skel"),
        ("right_wrist_pitch_skel", "right_wrist_roll_skel"),
        ("right_wrist_yaw_skel", "right_wrist_pitch_skel"),
        ("right_hand_roll_skel", "right_wrist_yaw_skel"),
    ]


# --------------------------------------------------------------------------
# Core27 -> SOMA retargeting
#
# `Core27ToSOMA30Retargeter` is a real bone-level retarget, not a joint-name
# rename: Core27 and SOMA30 differ in bone count *and* in bone length (Core27:
# 4 spine bones + 1 neck, SOMA30: 3 spine bones + 2 neck, longer legs, shorter
# torso).  Neither a rename nor a uniform rescaling can work - scaling by
# stature gives SOMA legs that are too long (permanent squat), scaling by leg
# length stretches arms and torso and the hands never reach.  Instead SOMA keeps
# its own (neutral) bone lengths and every bone *direction* is copied from the
# corresponding Core27 bone; spine and neck chains are matched by normalised arc
# length, so 4-vs-3 spine bones and 1-vs-2 neck bones are handled generically.
# The result is already reachable by SOMA, so a damped Gauss-Newton
# (Levenberg-Marquardt) IK with an analytic SO(3) Jacobian only has to polish
# it and smooth it in time.
#
# Root translation keeps the Core27 world XZ trajectory; the root height is
# corrected per frame by the hip->toe drop difference of the two skeletons, so
# the retargeted feet stay on the ground plane of the source motion.
#
# The SOMA30 -> SOMA77 step that follows is *not* a retarget: SOMA30 is a
# strict subset of SOMA77 (same names, same parents, same bone lengths), so it
# is handled by `SOMASkeleton30.output_to_SOMASkeleton77` (copy the local
# rotations by joint name, fill the 47 remaining joints with the relaxed-hands
# rest pose, re-run FK).
# --------------------------------------------------------------------------


def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros(3)
    return np.asarray(v, dtype=np.float64) / n


def _rot_align(a, b):
    """Rotation taking vector `a` onto the direction of `b`."""
    a, b = _unit(a), _unit(b)
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
        axis = _unit(axis - np.dot(axis, a) * a)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _skew(v):
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


def _right_jacobian(x):
    """
    SO(3) right Jacobian: exp((x + dx)_x) ~= exp((J_r dx)_x) exp(x_x).

    Needed because the position derivative is taken with respect to a *left*
    perturbation delta, while the optimiser works on the rotvec parameter.
    """
    x = np.asarray(x, dtype=np.float64)
    th = np.linalg.norm(x, axis=-1, keepdims=True)          # (..., 1)
    K = _skew(x)
    small = th[..., 0] < 1e-8
    a = np.where(small, 0.5, (1.0 - np.cos(th[..., 0])) / np.maximum(th[..., 0] ** 2, 1e-12))
    b = np.where(small, 1.0 / 6.0,
                 (th[..., 0] - np.sin(th[..., 0])) / np.maximum(th[..., 0] ** 3, 1e-12))
    I = np.eye(3)
    return I + a[..., None, None] * K + b[..., None, None] * (K @ K)


def _ancestor_mask(parents):
    """anc[j, k] = True when rotating joint j moves joint k."""
    J = len(parents)
    anc = np.zeros((J, J), dtype=bool)
    for k in range(J):
        j = parents[k]
        while j >= 0:
            anc[j, k] = True
            j = parents[j]
    return anc


def _smooth_rotations(mats, window: int = 5, skip: tuple = ()):
    """Temporal smoothing of rotation matrices (chordal L2 mean, re-orthogonalized).

    The retarget pipeline amplifies millimetre source noise into a few degrees
    of per-frame rotation wobble on short bones (spine/hands); real motion
    never exceeds ~0.5 deg/frame, so a short centered window is safe.  Joints
    named in ``skip`` are left untouched (fast movers like the hands would get
    a visible ±window/2-frame smear).
    mats: [T, J, 3, 3] -> smoothed copy.
    """
    from scipy.spatial.transform import Rotation

    T = mats.shape[0]
    if T < 3:
        return mats
    pad = window // 2
    padded = np.pad(mats, ((pad, pad), (0, 0), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(mats)
    for j in range(mats.shape[1]):
        if skip and j in skip:
            out[:, j] = mats[:, j]
            continue
        rots = Rotation.from_matrix(padded[:, j])
        for t in range(T):
            out[t, j] = rots[t : t + window].mean().as_matrix()
    return out


def _kabsch(A, B, w):
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


def _fk_local(local_R, offsets, parents, root_idx, root_pos):
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


def _point_at(pts, u):
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


def _as_numpy(x):
    """Accept a torch tensor or anything array-like and return a numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _skeleton_arrays(skel):
    """(names, parents, neutral, root_idx) of a skeleton, hips at the origin."""
    names = [str(n) for n in skel.bone_order_names]
    parents = np.asarray(_as_numpy(skel.joint_parents), dtype=int)
    neutral = np.asarray(_as_numpy(skel.neutral_joints), dtype=np.float64)
    root_idx = int(getattr(skel, "root_idx", 0))
    neutral = neutral - neutral[root_idx]
    return names, parents, neutral, root_idx


def _bone_offsets(neutral, parents):
    """Parent-relative offsets of the neutral pose (root offset is zero)."""
    offsets = neutral.copy()
    for j in range(len(neutral)):
        offsets[j] = 0.0 if parents[j] < 0 else neutral[j] - neutral[parents[j]]
    return offsets


class Core27ToSOMA30Retargeter:
    """Retarget ARDY Core27 joint positions onto the SOMA30 skeleton.

    Parameters
    ----------
    soma30:
        Target skeleton. Defaults to :class:`SOMASkeleton30`.
    core27:
        Source skeleton, used only for the joint ordering. Defaults to the
        :class:`CoreSkeleton27` layout.
    iters:
        Levenberg-Marquardt iterations per frame (0 = analytic initialisation
        only, very fast).
    smooth:
        Temporal regularisation toward the previous frame.
    root_weight:
        How strongly the pelvis keeps the source facing. 0 lets the solver
        swing the body to reach the feet.
    """

    CORE_SPINE = ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head"]
    SOMA_SPINE = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head"]

    # soma joint -> (core parent, core child): the direction of that Core27 bone
    # is copied onto the SOMA bone ending at `soma joint`.
    BONE_SRC = {
        "LeftShoulder": ("Spine3", "LeftShoulder"),
        "LeftArm": ("LeftShoulder", "LeftArm"),
        "LeftForeArm": ("LeftArm", "LeftForeArm"),
        "LeftHand": ("LeftForeArm", "LeftHand"),
        # Hand fingertips deliberately NOT mapped: Core27's thumb/hand-end bones
        # are ~3 cm long, so their directions are noise, while SOMA's finger
        # pose comes from the relaxed-hands rest anyway.  Mapping them pulled
        # the wrist into contortions (5-6 cm fingertip targets).
        "RightShoulder": ("Spine3", "RightShoulder"),
        "RightArm": ("RightShoulder", "RightArm"),
        "RightForeArm": ("RightArm", "RightForeArm"),
        "RightHand": ("RightForeArm", "RightHand"),
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

    # Joints rebuilt as a twist-free swing of the parent frame (see
    # rebuild_frames): hands only have near-collinear fingertip children, so a
    # Kabsch there is rank-deficient and the wrist twist flips randomly.
    SWING_ONLY = ("LeftHand", "RightHand")

    # Leaves are not optimized (their rotation drives nothing) but their
    # *position* is a constraint on their parent.
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

    def __init__(
        self,
        soma30=None,
        core27=None,
        iters: int = 12,
        smooth: float = 0.10,
        root_weight: float = 3.0,
    ):
        self.soma = soma30 if soma30 is not None else SOMASkeleton30(load=True)
        self.core = core27 if core27 is not None else CoreSkeleton27(load=True)
        self.core_names = [str(n) for n in self.core.bone_order_names]
        self.iters = int(iters)
        self.smooth = float(smooth)
        self.root_weight = float(root_weight)
        self.last_stats: dict = {}

        names, parents, neutral, root_idx = _skeleton_arrays(self.soma)
        self.names = names
        self.parents = parents
        self.neutral = neutral
        self.root_idx = root_idx
        self.si = {n: i for i, n in enumerate(names)}
        self.ci = {n: i for i, n in enumerate(self.core_names)}

        # Core27 rest pose (hips at the origin).  The retarget transfers the
        # *change* of every Core bone away from this rest pose, not the bone's
        # absolute direction: Core27's own rest lumbar is bent ~34 deg
        # backwards while SOMA's rest spine is straight, so copying absolute
        # directions would add Core's whole rest curvature on top of SOMA's
        # rest pose (visible as a collapsed / excessively arched waist).
        _, _, self.core_neutral, _ = _skeleton_arrays(self.core)

        missing = [n for n in list(self.SOMA_SPINE) + list(self.BONE_SRC) if n not in self.si]
        if missing:
            raise RuntimeError(f"SOMA skeleton is missing joints: {missing}")
        missing = [n for n in self.CORE_SPINE if n not in self.ci]
        if missing:
            raise RuntimeError(f"Core skeleton is missing joints: {missing}")
        if np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, len(names))):
            raise RuntimeError("SOMA joint order is not topological.")

        self.offsets = _bone_offsets(neutral, parents)

        # Normalised arc length of the SOMA spine (fixed: from the neutral pose).
        spine_pts = np.stack([neutral[self.si[n]] for n in self.SOMA_SPINE])
        seg = np.linalg.norm(np.diff(spine_pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.soma_u = cum / cum[-1]

        self.solve_idx = np.array([self.si[n] for n in names if n not in self.NO_SOLVE], dtype=int)
        self.tgt_names = [n for n in names if n not in self.NO_SOURCE and n != names[root_idx]]
        self.tgt_idx = np.array([self.si[n] for n in self.tgt_names], dtype=int)
        length_scale = float(np.median(np.linalg.norm(self.offsets[self.tgt_idx], axis=1)))
        self.ctx = {
            "solve_idx": self.solve_idx,
            "tgt_idx": self.tgt_idx,
            "w": np.array([self.WEIGHTS.get(n, 1.0) for n in self.tgt_names]) / length_scale,
            "anc": _ancestor_mask(parents),
        }

        # Feet used to re-anchor the root height.
        self.feet = [("LeftToeBase", "LeftToeBase"), ("RightToeBase", "RightToeBase")]

    # -- target pose ------------------------------------------------------

    # 1:1 spine/neck joint correspondence (both chains have 7 joints):
    # soma joint -> core joint; the SOMA bone ending at `soma joint` takes the
    # direction of the Core bone ending at the mapped joint.
    SPINE_MAP = {
        "Spine1": "Spine",
        "Spine2": "Spine1",
        "Chest": "Spine2",
        "Neck1": "Spine3",
        "Neck2": "Neck",
        "Head": "Head",
    }
    SPINE_PARENT = {  # soma child -> soma parent, to fetch the core parent bone
        "Spine1": "Hips", "Spine2": "Spine1", "Chest": "Spine2",
        "Neck1": "Chest", "Neck2": "Neck1", "Head": "Neck2",
    }

    def build_target(self, core):
        """
        SOMA-shaped target pose: SOMA rest pose + Core27 joint *displacements*
        from Core27's own rest.  Returned with the hips at the origin.

        Every mapped bone takes the minimum rotation ``dR`` that carries the
        Core27 bone from its rest direction to its posed direction, and applies
        it to the *SOMA* rest direction of the corresponding bone.  Bones keep
        SOMA's exact lengths, so a Core27 rest-pose frame maps to the SOMA rest
        pose instead of imprinting Core27's rest curvature (its lumbar is bent
        ~34 deg backwards, SOMA's is straight).  Same 1:1 joint mapping as
        before: limbs via BONE_SRC, spine/neck chain via SPINE_MAP; hands and
        head-extras keep SOMA rest directions.
        """
        parents, neutral, names = self.parents, self.neutral, self.names
        ci, si = self.ci, self.si
        core_rest = self.core_neutral
        J = len(names)

        def transfer(core_parent, core_child, soma_child):
            d0 = core_rest[ci[core_child]] - core_rest[ci[core_parent]]
            d1 = core[ci[core_child]] - core[ci[core_parent]]
            if np.linalg.norm(d0) < 1e-9 or np.linalg.norm(d1) < 1e-9:
                return None
            soma_parent = parents[si[soma_child]]
            return _rot_align(d0, d1) @ (neutral[si[soma_child]] - neutral[soma_parent])

        dirs = {}
        for soma_child, core_child in self.SPINE_MAP.items():
            soma_parent = self.SPINE_PARENT[soma_child]
            core_parent = self.SPINE_MAP.get(soma_parent, soma_parent)
            dirs[soma_child] = transfer(core_parent, core_child, soma_child)
        for soma_name, (ca, cb) in self.BONE_SRC.items():
            if soma_name in si and ca in ci and cb in ci:
                dirs[soma_name] = transfer(ca, cb, soma_name)

        target = np.zeros((J, 3), dtype=np.float64)
        for j in range(J):
            pi = parents[j]
            if pi < 0:
                continue
            off = neutral[j] - neutral[pi]
            v = dirs.get(names[j])
            if v is None or np.linalg.norm(v) < 1e-9:
                v = off
            target[j] = target[pi] + np.linalg.norm(off) * _unit(v)
        return target

    def rebuild_frames(self, pos):
        """Rebuild local/global rotations in the standard frame from solved positions.

        The IK fits positions only; the twist of every single-child bone (the
        whole spine, arms and legs) is unconstrained and comes out arbitrary,
        which breaks skinning (the skin's bind frame is the identity-at-rest
        convention).  Here every bone frame is rebuilt from the posed bone
        directions: two or more children determine the frame (Kabsch), a single
        child gets a twist-free swing from the parent frame, leaves inherit the
        parent frame.  Identity at rest, deterministic and smooth in time.

        pos: [T, J, 3] solved world positions.  Returns (local, global) [T, J, 3, 3].
        """
        parents, neutral, root_idx = self.parents, self.neutral, self.root_idx
        names = self.names
        J = len(parents)
        children = [[] for _ in range(J)]
        for j in range(J):
            if parents[j] >= 0 and names[j] not in self.NO_SOURCE:
                children[parents[j]].append(j)

        T = len(pos)
        local = np.zeros((T, J, 3, 3), dtype=np.float64)
        global_ = np.zeros_like(local)

        # Anatomical basis for the two 3-child joints (Hips, Chest): the
        # spine direction and the hip/shoulder line are matched *exactly*,
        # which keeps the legs and arms where the IK put them (a plain Kabsch
        # over 3 directions leaves a ~2 deg pelvis error -> 3-6 cm foot drift).
        def _basis(up, side):
            u = _unit(up)
            s = _unit(side - np.dot(side, u) * u)
            f = np.cross(s, u)
            return np.stack([u, s, f], axis=1)

        anatomical = {}
        if "Spine1" in self.si and "LeftLeg" in self.si and "RightLeg" in self.si:
            anatomical[self.root_idx] = (
                self.si["Spine1"],  # up = spine bone, side = hip line
                (self.si["LeftLeg"], self.si["RightLeg"]),
            )
        if all(n in self.si for n in ("Chest", "Neck1", "LeftShoulder", "RightShoulder")):
            anatomical[self.si["Chest"]] = (
                self.si["Neck1"],
                (self.si["LeftShoulder"], self.si["RightShoulder"]),
            )

        rest_basis = {j: _basis(neutral[u] - neutral[j], neutral[l] - neutral[r])
                      for j, (u, (l, r)) in anatomical.items()}

        for t in range(T):
            p = pos[t] - pos[t][root_idx]
            g = np.tile(np.eye(3), (J, 1, 1))
            for j in range(J):
                pj = parents[j]
                if j in anatomical:
                    u_i, (l_i, r_i) = anatomical[j]
                    Mc = _basis(p[u_i] - p[j], p[l_i] - p[r_i])
                    g[j] = Mc @ rest_basis[j].T
                elif names[j] in self.SWING_ONLY:
                    # Hands: their only children are two nearly-collinear
                    # fingertip bones - a Kabsch over those is rank-deficient
                    # and the wrist twist flips arbitrarily frame to frame
                    # (observed 168 deg in one frame).  Swing the parent frame
                    # onto this bone's own direction instead: twist-free and
                    # stable; finger pose comes from the relaxed-hands rest.
                    g[j] = _rot_align(g[pj] @ (neutral[j] - neutral[pj]), p[j] - p[pj]) @ g[pj]
                elif len(children[j]) >= 2:
                    A = np.stack([neutral[k] - neutral[j] for k in children[j]])
                    B = np.stack([p[k] - p[j] for k in children[j]])
                    g[j] = _kabsch(A, B, np.linalg.norm(A, axis=1))
                elif len(children[j]) == 1:
                    k = children[j][0]
                    g[j] = _rot_align(g[pj] @ (neutral[k] - neutral[j]), p[k] - p[j]) @ g[pj]
                else:
                    g[j] = g[pj]
            for j in range(J):
                pi = parents[j]
                local[t, j] = g[j] if pi < 0 else g[pi].T @ g[j]
            global_[t] = g
        return local, global_

    def pelvis_rotation(self, core):
        """
        Pelvis orientation *relative to the rest pose*, from an explicit, well
        conditioned frame (spine up / hip-to-hip side).  This is what preserves
        the facing direction of the source motion.

        Only the change of the Core27 pelvis away from Core27's rest is taken:
        Core27's rest torso already leans ~12 deg backwards, so aligning SOMA's
        rest torso axis onto Core27's absolute one would bend the retargeted
        waist backwards by that amount.
        """
        ci = self.ci
        core_rest = self.core_neutral

        def basis(up, side):
            u = _unit(up)
            s = _unit(side - np.dot(side, u) * u)
            f = _unit(np.cross(s, u))
            return np.stack([u, s, f], axis=1)

        Mc = basis(core[ci["Spine3"]] - core[ci["Hips"]],
                   core[ci["LeftUpLeg"]] - core[ci["RightUpLeg"]])
        Mc_rest = basis(core_rest[ci["Spine3"]] - core_rest[ci["Hips"]],
                        core_rest[ci["LeftUpLeg"]] - core_rest[ci["RightUpLeg"]])
        return Mc @ Mc_rest.T

    def init_local(self, target, root_R=None, skip=NO_SOURCE):
        """
        Absolute orientation of every joint (Kabsch over its children's bone
        directions), converted to local rotations.  Already very close to the
        target, so it doubles as the IK initialisation.

        The root is also fitted by Kabsch over its children (spine + both hip
        bones): a spine/hip-line *basis* alignment ignores that SOMA's rest hip
        bones splay ~6 cm lower than Core27's, which would offset both legs at
        init and fight the IK's root anchor.  ``root_R`` is only a fallback
        when the root has no usable children.
        """
        parents, neutral, names = self.parents, self.neutral, self.names
        root_idx = self.root_idx
        J = len(parents)
        children = [[] for _ in range(J)]
        for j in range(J):
            pj = parents[j]
            if pj >= 0 and names[j] not in skip:
                children[pj].append(j)

        g = np.tile(np.eye(3), (J, 1, 1))
        g[root_idx] = np.eye(3) if root_R is None else root_R
        for j in range(J):
            if j != root_idx and not children[j]:
                g[j] = g[parents[j]]
                continue
            A = np.stack([neutral[k] - neutral[j] for k in children[j]])
            B = np.stack([target[k] - target[j] for k in children[j]])
            g[j] = _kabsch(A, B, np.linalg.norm(A, axis=1))

        local = np.empty_like(g)
        for j in range(J):
            pi = parents[j]
            local[j] = g[j] if pi < 0 else g[pi].T @ g[j]
        return local

    # -- per-frame IK -----------------------------------------------------

    def solve_frame(self, target, init_R, prev_x, root_est=None, tol=1e-6):
        """
        Damped Gauss-Newton (Levenberg-Marquardt) on the local rotations.

        The Jacobian is analytic: rotating joint j by a local rotvec delta turns
        into a *global* rotation about `G[parent(j)] @ delta`, so every descendant
        k moves by  omega x (pos[k] - pos[j]).  That is exact and removes the 69
        extra FK evaluations a finite-difference Jacobian would need.

        The normal equations are solved with a Cholesky solver, not an SVD: on
        some BLAS builds a 147x69 SVD costs ~50 ms, which would dominate runtime.
        """
        parents, offsets, root_idx = self.parents, self.offsets, self.root_idx
        solve_idx, tgt_idx, w = self.ctx["solve_idx"], self.ctx["tgt_idx"], self.ctx["w"]
        anc = self.ctx["anc"][np.ix_(solve_idx, tgt_idx)]      # (P, K)
        P, K = len(solve_idx), len(tgt_idx)
        smooth, root_w = self.smooth, self.root_weight
        zero = np.zeros(3)

        def unpack(x):
            R = init_R.copy()
            R[solve_idx] = Rotation.from_rotvec(x.reshape(-1, 3)).as_matrix()
            return R

        def rj(x):
            R = unpack(x)
            g, p = _fk_local(R, offsets, parents, root_idx, zero)
            r = ((p[tgt_idx] - target[tgt_idx]) * w[:, None]).ravel()

            v = p[tgt_idx][:, None, :] - p[solve_idx][None, :, :]   # (K, P, 3)
            Gp = np.empty((P, 3, 3))
            for a, j in enumerate(solve_idx):
                pj = parents[j]
                Gp[a] = np.eye(3) if pj < 0 else g[pj]
            # d(pos_k) / d(delta_j) = -skew(pos_k - pos_j) @ G[parent(j)]
            blocks = -np.einsum("kpab,pbc->kpac", _skew(v), Gp)     # (K, P, 3, 3)
            blocks *= anc.T[:, :, None, None]
            # chain rule: delta_j = J_r(x_j) dx_j
            rv = x.reshape(-1, 3)
            blocks = np.einsum("kpab,pbc->kpac", blocks, _right_jacobian(rv))
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
                extra[:, rp:rp + 3] = root_est.T @ _right_jacobian(x[rp:rp + 3])
                r = np.concatenate([r, e * root_w])
                J = np.vstack([J, extra * root_w])
            return r, J

        x = Rotation.from_matrix(init_R[solve_idx]).as_rotvec().ravel()
        r, J = rj(x)
        cost = float(r @ r)
        if self.iters <= 0:
            return init_R, x, cost

        lam = 1e-3
        gain = 0.0
        for _ in range(self.iters):
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

    # -- full sequence ----------------------------------------------------

    def retarget(
        self,
        core_pos,
        fps: Optional[float] = None,
        foot_contacts=None,
        verbose: bool = False,
    ) -> dict:
        """Retarget a Core27 sequence to SOMA30.

        Args:
            core_pos: Core27 joint positions, `[T, 27, 3]` (or `[1, T, 27, 3]`).
            fps: Frames per second stored in the output.
            foot_contacts: Optional `[T, 4]` boolean array.
            verbose: Print per-frame progress.

        Returns:
            dict of numpy arrays: `posed_joints`, `global_rot_mats`,
            `local_rot_mats`, `root_positions`, `foot_contacts`, `fps`.  Only
            numeric fields: `kimodo_convert` turns every NPZ key into a torch
            tensor, so a string field would make it fail.
        """
        core_pos = np.asarray(_as_numpy(core_pos), dtype=np.float64)
        if core_pos.ndim == 4:
            if core_pos.shape[0] != 1:
                raise ValueError(f"One sample at a time, got {core_pos.shape}.")
            core_pos = core_pos[0]
        if core_pos.ndim != 3 or core_pos.shape[1:] != (len(self.core_names), 3):
            raise ValueError(
                f"Expected posed_joints [T,{len(self.core_names)},3], got {core_pos.shape}."
            )

        T = core_pos.shape[0]
        ci, si, tgt_idx = self.ci, self.si, self.tgt_idx

        # Time-smooth the Core27 spine joints: the SOMA spine bones are very
        # short (5 cm), and build_target derives their directions from
        # differences of arc-length interpolation points, which amplifies
        # millimetre noise in the source into 10-15 deg per-frame spine wobble
        # (visible as a shuddering, twisting torso).  A short centered window
        # keeps the motion but removes the amplification.
        core_spine_idx = [ci[n] for n in self.CORE_SPINE]
        smooth_core = core_pos.copy()
        window = 5
        if T > 2:
            kernel = np.ones(window) / window
            pad = window // 2
            for j in core_spine_idx:
                pts = core_pos[:, j]
                padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
                smooth_core[:, j] = np.stack(
                    [np.convolve(padded[:, k], kernel, mode="valid") for k in range(3)], axis=1
                )

        out_local = np.zeros((T, len(self.names), 3, 3), dtype=np.float64)
        out_global = np.zeros_like(out_local)
        out_pos = np.zeros((T, len(self.names), 3), dtype=np.float64)
        out_root = np.zeros((T, 3), dtype=np.float64)
        errs, costs = [], []

        prev_x = None
        prev_local = None
        t0 = time.time()
        for t in range(T):
            core = smooth_core[t]

            target_rel = self.build_target(core)

            # Root height: keep each foot at the height Core27 puts it at.
            d_core = [core[ci["Hips"]][1] - core[ci[c]][1] for _, c in self.feet]
            d_soma = [-target_rel[si[s]][1] for s, _ in self.feet]
            root = core[ci["Hips"]].copy()
            root[1] += float(np.mean(d_soma) - np.mean(d_core))

            root_est = self.pelvis_rotation(core)
            if prev_local is None:
                init = self.init_local(target_rel, root_est)
            else:
                # Warm start from the previous frame's solution.  The IK is
                # under-determined (72 dof for 87 positional constraints), so
                # solving each frame from a fresh Kabsch initialisation lets the
                # null space - mostly the twist of the single-child bones -
                # re-randomise every frame.  That shows up as ~0.2 cm/frame^2 of
                # end-effector jitter and, once rebuild_frames turns those
                # positions into bone frames, ~1 deg/frame of torso shudder.
                # Starting from the previous solution keeps the null space
                # continuous in time.
                init = prev_local
            # Anchor the IK root at the Kabsch root (spine + both hip bones):
            # pelvis_rotation's spine/hip-line basis ignores SOMA's lower hip
            # splay and pulls the solution ~6 cm off the leg targets.
            root_anchor = init[self.root_idx].copy()
            local, x, cost = self.solve_frame(target_rel, init, prev_x, root_est=root_anchor)
            prev_x = x
            prev_local = local.copy()

            g, p = _fk_local(local, self.offsets, self.parents, self.root_idx, root)
            out_local[t] = local
            out_global[t] = g
            out_pos[t] = p
            out_root[t] = root

            errs.append(np.linalg.norm(p[tgt_idx] - (target_rel + root)[tgt_idx], axis=1))
            costs.append(cost)

            if verbose and (t % 20 == 0 or t == T - 1):
                print(f"\r[core27->soma30] {t + 1}/{T} ({time.time() - t0:.1f}s)",
                      end="", flush=True)
        if verbose:
            print()

        # Rebuild the rotations in the standard (identity-at-rest) frame from
        # the solved positions: the IK leaves single-child bone twist arbitrary,
        # which would break skinning / BVH export.  FK of the rebuilt locals is
        # self-consistent but shifts positions by ~1-2 cm, so the *precise* IK
        # positions are kept for `posed_joints` on the body joints: skinning
        # consumes [global_rot | posed_joints] jointly, so the render gets
        # exact positions with standard frames.  Jaw/eyes have no source and
        # their IK positions hang off the (arbitrary) solved head frame, so
        # they keep the rebuilt FK positions (natural jaw under the standard
        # head frame).  (BVH export re-FKs from the locals and accepts the
        # small residual.)
        precise_pos = out_pos.copy()
        # Smooth the solved positions in time before they are turned back into
        # bone frames.  rebuild_frames derives the pelvis frame from the hip
        # line and the chest frame from the shoulder line, so millimetre-level
        # IK jitter on the shoulders and legs is amplified into a visible
        # ~1 deg/frame torso shudder.  Smoothing only the torso is not enough:
        # the noise enters through the shoulders/hips that the torso frames are
        # built from.  Hands get a shorter window - a 5-frame box at 20 fps is a
        # 100 ms smear on the fastest joints.
        hand_idx = {si[n] for n in ("LeftHand", "RightHand") if n in si}
        if T > 2:
            for j in self.tgt_idx:
                win = 3 if j in hand_idx else 5
                pad = win // 2
                kernel = np.ones(win) / win
                pts = precise_pos[:, j]
                padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
                precise_pos[:, j] = np.stack(
                    [np.convolve(padded[:, k], kernel, mode="valid") for k in range(3)], axis=1
                )
        out_local, _ = self.rebuild_frames(precise_pos)
        # Fast movers (hands) are excluded: a 5-frame window at 20 fps is a
        # 100 ms smear that reads as a laggy, wrong hand pose.
        out_local = _smooth_rotations(out_local, window=5, skip=tuple(hand_idx))
        for t in range(T):
            out_global[t], out_pos[t] = _fk_local(
                out_local[t], self.offsets, self.parents, self.root_idx, out_root[t]
            )
        out_pos[:, self.tgt_idx] = precise_pos[:, self.tgt_idx]

        if foot_contacts is None:
            fc = np.zeros((T, 4), dtype=bool)
        else:
            fc = np.asarray(_as_numpy(foot_contacts))
            if fc.ndim == 3 and fc.shape[0] == 1:
                fc = fc[0]
            fc = fc if fc.shape == (T, 4) else np.zeros((T, 4), dtype=bool)

        self.last_stats = {
            "frames": T,
            "seconds": time.time() - t0,
            "costs": np.asarray(costs),
            "per_joint_cm": dict(zip(self.tgt_names, np.asarray(errs).mean(axis=0) * 100.0)),
            "source_min_toe_y": float(core_pos[:, [ci[c] for _, c in self.feet], 1].min()),
            "output_min_toe_y": float(out_pos[:, [si[s] for s, _ in self.feet], 1].min()),
            "ortho_err": float(np.max(np.abs(
                np.matmul(out_local.transpose(0, 1, 3, 2), out_local) - np.eye(3)
            ))),
            "min_det": float(np.min(np.linalg.det(out_local))),
        }

        payload = {
            "posed_joints": out_pos.astype(np.float32),
            "global_rot_mats": out_global.astype(np.float32),
            "local_rot_mats": out_local.astype(np.float32),
            "root_positions": out_root.astype(np.float32),
            "foot_contacts": fc,
        }
        if fps is not None:
            payload["fps"] = np.asarray(float(fps))
        return payload

    __call__ = retarget


class CoreSkeleton27(SkeletonBase):
    """Core Skeleton with 27 joints."""

    name = "cskel27"
    right_foot_joint_names = ["RightFoot", "RightToeBase"]  # in order of chain
    left_foot_joint_names = ["LeftFoot", "LeftToeBase"]  # in order of chain
    right_hand_joint_names = ["RightHand", "RightHandEnd"]  # in order of chain
    left_hand_joint_names = ["LeftHand", "LeftHandEnd"]  # in order of chain
    hip_joint_names = ["RightUpLeg", "LeftUpLeg"]  # in order [right, left]

    bone_order_names_with_parents = [
        ("Hips", None),
        ##
        ("Spine", "Hips"),
        ("Spine1", "Spine"),
        ("Spine2", "Spine1"),
        ("Spine3", "Spine2"),
        ("Neck", "Spine3"),
        ("Head", "Neck"),
        ##
        ("RightShoulder", "Spine3"),
        ("RightArm", "RightShoulder"),
        ("RightForeArm", "RightArm"),
        ("RightHand", "RightForeArm"),
        ("RightHandEnd", "RightHand"),
        ("RightHandThumb1", "RightHand"),
        ##
        ("LeftShoulder", "Spine3"),
        ("LeftArm", "LeftShoulder"),
        ("LeftForeArm", "LeftArm"),
        ("LeftHand", "LeftForeArm"),
        ("LeftHandEnd", "LeftHand"),
        ("LeftHandThumb1", "LeftHand"),
        ##
        ("RightUpLeg", "Hips"),
        ("RightLeg", "RightUpLeg"),
        ("RightFoot", "RightLeg"),
        ("RightToeBase", "RightFoot"),
        ##
        ("LeftUpLeg", "Hips"),
        ("LeftLeg", "LeftUpLeg"),
        ("LeftFoot", "LeftLeg"),
        ("LeftToeBase", "LeftFoot"),
    ]

    last_stats: dict = {}

    @property
    def somaskel30(self):
        """Lazy SOMA30 skeleton: the intermediate target of the retarget."""
        if not hasattr(self, "_somaskel30"):
            self._somaskel30 = SOMASkeleton30(load=True)
        return self._somaskel30

    def to_SOMASkeleton30(self, posed_joints, fps=None, foot_contacts=None, **kwargs) -> dict:
        """Retarget Core27 joint *positions* onto SOMA30.

        This is a real bone-level retarget (see `Core27ToSOMA30Retargeter`),
        not a joint-name rename: Core27 and SOMA30 have different bone counts
        and bone lengths, so SOMA keeps its own bone lengths while every bone
        *direction* is taken from Core27, after which a damped Gauss-Newton IK
        polishes the pose.

        Args:
            posed_joints: Core27 joint positions `[T, 27, 3]` (numpy or torch).
            fps: Frames per second stored in the output.
            foot_contacts: Optional `[T, 4]` boolean array.
            **kwargs: Forwarded to `Core27ToSOMA30Retargeter`
                (`iters`, `smooth`, `root_weight`).

        Returns:
            SOMA30 motion dict of torch tensors: `posed_joints`,
            `global_rot_mats`, `local_rot_mats`, `root_positions`,
            `foot_contacts`, `fps`.  Diagnostics are left in `self.last_stats`.
        """
        retargeter = Core27ToSOMA30Retargeter(soma30=self.somaskel30, core27=self, **kwargs)
        out = retargeter.retarget(posed_joints, fps=fps, foot_contacts=foot_contacts)
        self.last_stats = retargeter.last_stats
        return {k: torch.as_tensor(v) for k, v in out.items()}

    def to_SOMASkeleton77(self, posed_joints, fps=None, foot_contacts=None, **kwargs) -> dict:
        """Retarget Core27 joint *positions* onto SOMA77.

        Same as `to_SOMASkeleton30`, followed by the SOMA30 -> SOMA77 expansion
        (relaxed hands, FK re-run).
        """
        return self.output_to_SOMASkeleton77(
            {"posed_joints": posed_joints, "foot_contacts": foot_contacts, "fps": fps},
            **kwargs,
        )

    def output_to_SOMASkeleton77(self, output: dict, **kwargs) -> dict:
        """Convert a motion dict of this skeleton to somaskel77.

        Mirrors `SOMASkeleton30.output_to_SOMASkeleton77`, so both source
        skeletons expose the same entry point.  Unlike SOMA30, Core27 is *not* a
        subset of SOMA77, so the path is:

            Core27 -> SOMA30  (bone-level IK retarget)
                   -> SOMA77  (copy by joint name + relaxed hands + FK)

        Args:
            output: dict with at least `posed_joints` `[T, 27, 3]`; `fps` and
                `foot_contacts` are forwarded when present.
            **kwargs: Forwarded to `Core27ToSOMA30Retargeter`.

        Returns:
            SOMA77 motion dict of torch tensors: `posed_joints`,
            `global_rot_mats`, `local_rot_mats`, `root_positions`,
            `foot_contacts` (6 channels: toe-end copies toe-base), `fps`.
        """
        fps = output.get("fps", None)
        out30 = self.to_SOMASkeleton30(
            output["posed_joints"],
            fps=float(np.asarray(fps).reshape(-1)[0]) if fps is not None else None,
            foot_contacts=output.get("foot_contacts", None),
            **kwargs,
        )
        return self.somaskel30.output_to_SOMASkeleton77(out30, preserve_pos=True)

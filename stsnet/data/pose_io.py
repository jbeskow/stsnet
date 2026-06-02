"""
Pose loading utilities for STS-Net.

Loads and normalises MediaPipe Holistic pose files into per-stream arrays
for dominant hand, non-dominant hand, body, and face.

Also provides:
  load_wilor_streams:      load WiLoR 3D MANO hand joints from .npz
  normalize_hand_moryossef: per-frame 3D hand normalisation (Moryossef et al.)
"""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Landmark index constants
# ---------------------------------------------------------------------------

_LEFT_HAND_SLICE    = slice(501, 522)
_RIGHT_HAND_SLICE   = slice(522, 543)
_UPPER_BODY_INDICES = list(range(11, 23))

# Face-mesh landmark indices (within FACE_LANDMARKS component, 0-467).
# In the combined pose body array the face component starts at offset 33
# (after the 33 POSE_LANDMARKS), so the actual indices into body.data are
# _FACE_MESH_OFFSET + each value below.
_FACE_MESH_OFFSET = 33
_FACE_MESH_LOCAL = [
    4,                           # Nose tip

    33, 159, 133, 145,           # Left Eye  (left, top, inner, bottom)
    362, 386, 263, 374,          # Right Eye (inner, top, right, bottom)

    70, 105, 107,                # Left Eyebrow  (inner, peak, outer)
    300, 334, 336,               # Right Eyebrow (inner, peak, outer)

    # Mouth / Lips (outer contour + inner opening)
    61, 291, 0, 17, 37, 267, 84, 314, 13, 14,
]
_FACE_INDICES = [_FACE_MESH_OFFSET + i for i in _FACE_MESH_LOCAL]  # 25 points
N_FACE = len(_FACE_INDICES)   # 25
N_BODY = len(_UPPER_BODY_INDICES)  # 12
N_BODY_TOTAL = N_BODY + N_FACE     # 37


# ---------------------------------------------------------------------------
# Pose cache
# ---------------------------------------------------------------------------

_POSE_CACHE_DIR: Path | None = None


def set_pose_cache_dir(cache_dir: str | Path) -> None:
    """Point the dataset at a pre-built .npz cache (see cache_poses.py)."""
    global _POSE_CACHE_DIR
    _POSE_CACHE_DIR = Path(cache_dir)


def get_pose_cache_dir() -> Path | None:
    """Return the current pose cache directory, or None if not set."""
    return _POSE_CACHE_DIR


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _center_at_wrist(hand: np.ndarray) -> np.ndarray:
    return hand - hand[:, 0:1, :]


def _shoulder_normalize(
    left_hand:  np.ndarray,
    right_hand: np.ndarray,
    body:       np.ndarray,
    face:       np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Normalize all streams by shoulder geometry to remove signer body-size
    and global-position variation.

    Body joints 0 and 1 are left and right shoulder (MediaPipe indices 11, 12).

    Per-clip median is used so single-frame tracking failures don't corrupt
    the whole clip.

    - body/face: translated so shoulder midpoint = origin, scaled by shoulder width
    - hands:     already wrist-centered; only scaled by shoulder width
    """
    l_shoulder = body[:, 0, :]   # (T, 3)
    r_shoulder = body[:, 1, :]   # (T, 3)

    midpoint = np.nanmedian((l_shoulder + r_shoulder) / 2.0, axis=0)   # (3,)

    diff   = r_shoulder - l_shoulder                           # (T, 3)
    widths = np.sqrt(np.nansum(diff[:, :2] ** 2, axis=1))     # XY only — stable
    shoulder_width = float(np.nanmedian(widths))

    if np.isnan(shoulder_width) or shoulder_width < 1e-6:
        return left_hand, right_hand, body, face   # fallback: no normalization

    body_norm  = (body - midpoint) / shoulder_width
    left_norm  = left_hand  / shoulder_width
    right_norm = right_hand / shoulder_width
    face_norm  = (face - midpoint) / shoulder_width if face is not None else None

    return left_norm, right_norm, body_norm, face_norm


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def _masked_to_float(arr: np.ma.MaskedArray, conf: np.ndarray) -> np.ndarray:
    out = np.array(arr.filled(np.nan), dtype=np.float32)
    invalid = (conf == 0)[:, :, None].repeat(3, axis=2)
    out[invalid] = np.nan
    out[np.ma.getmaskarray(arr)] = np.nan
    return out


def _load_from_cache(
    pose_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Return (left_hand, right_hand, body, face_or_None) from .npz cache.

    face is None when the cache predates the face-landmark addition; in that
    case load_pose_streams falls back to live pose_format loading for face.
    """
    if _POSE_CACHE_DIR is None:
        return None
    cache_path = _POSE_CACHE_DIR / (pose_path.name + ".npz")
    if not cache_path.exists():
        return None
    npz = np.load(cache_path)
    face = npz["face"] if "face" in npz else None
    return npz["left_hand"], npz["right_hand"], npz["body"], face


# ---------------------------------------------------------------------------
# Main loading function
# ---------------------------------------------------------------------------

def load_pose_streams(
    pose_path: Path,
    handedness: str,
    mirror_left: bool = True,
) -> dict[str, np.ndarray] | None:
    """
    Load and normalize a pose file, returning four float32 arrays:

      'dominant'     (T, 21, 3)   dominant-hand joints, wrist-centered
      'nondominant'  (T, 21, 3)   non-dominant hand joints, wrist-centered
      'body'         (T, 12, 3)   upper-body joints, shoulder-normalized
      'face'         (T, 25, 3)   face-mesh landmarks, shoulder-normalized

    Checks _POSE_CACHE_DIR for a pre-built .npz before falling back to
    live pose_format loading.  Old caches (without 'face' key) trigger live
    loading of the face component from the .pose file.
    """
    cached = _load_from_cache(pose_path)
    face = None
    if cached is not None:
        left_hand, right_hand, body, face = cached
        if face is None:
            # Old cache — load face from pose file
            try:
                from pose_format import Pose
                with open(pose_path, "rb") as f:
                    pose = Pose.read(f.read())
                pose.normalize()
                data = pose.body.data[:, 0, :, :]
                conf = pose.body.confidence[:, 0, :]
                face = _masked_to_float(data[:, _FACE_INDICES, :], conf[:, _FACE_INDICES])
            except Exception:
                face = None
    else:
        from pose_format import Pose
        try:
            with open(pose_path, "rb") as f:
                pose = Pose.read(f.read())
        except Exception:
            return None

        pose.normalize()

        data   = pose.body.data[:, 0, :, :]
        conf   = pose.body.confidence[:, 0, :]

        left_hand  = _masked_to_float(data[:, _LEFT_HAND_SLICE,    :], conf[:, _LEFT_HAND_SLICE])
        right_hand = _masked_to_float(data[:, _RIGHT_HAND_SLICE,   :], conf[:, _RIGHT_HAND_SLICE])
        body       = _masked_to_float(data[:, _UPPER_BODY_INDICES, :], conf[:, _UPPER_BODY_INDICES])
        face       = _masked_to_float(data[:, _FACE_INDICES,       :], conf[:, _FACE_INDICES])

    if handedness == "left":
        dominant, nondominant = left_hand, right_hand
    else:
        dominant, nondominant = right_hand, left_hand

    dominant    = _center_at_wrist(dominant)
    nondominant = _center_at_wrist(nondominant)

    dominant, nondominant, body, face = _shoulder_normalize(dominant, nondominant, body, face)

    T = body.shape[0]
    if face is None:
        face = np.full((T, N_FACE, 3), np.nan, dtype=np.float32)

    if handedness == "left" and mirror_left:
        dominant[..., 0]    = -dominant[..., 0]
        nondominant[..., 0] = -nondominant[..., 0]
        body[..., 0]        = -body[..., 0]
        face[..., 0]        = -face[..., 0]

    return {"dominant": dominant, "nondominant": nondominant, "body": body, "face": face}


# ---------------------------------------------------------------------------
# WiLoR 3D hand streams
# ---------------------------------------------------------------------------

_WILOR_RIGHT_IDX = 1
_WILOR_LEFT_IDX  = 0
_WILOR_MCP_IDX   = 9   # MIDDLE_MCP index — used as scale reference


def load_wilor_streams(
    pose_path:   Path,
    wilor_dir:   Path,
    handedness:  str,
    mirror_left: bool = True,
) -> dict[str, np.ndarray] | None:
    """
    Load WiLoR 3D hand joints and return normalised streams.

    Returns:
        'wilor_dom'    (T, 21, 3)  dominant hand, wrist-centred + MCP-normalised
        'wilor_nondom' (T, 21, 3)  non-dominant hand, same normalisation
    Returns None if the WiLoR .npz file does not exist.

    Normalisation:
      1. Center at wrist (joint 0)
      2. Divide by per-clip median wrist-to-MIDDLE_MCP distance
      3. Mirror x for left-handed signers (if mirror_left=True)
      4. Invalid frames (hand_valid=False) → NaN
    """
    video_name = pose_path.name.removesuffix(".pose").removesuffix(".mp4")
    wilor_path = wilor_dir / (video_name + ".npz")
    if not wilor_path.exists():
        return None

    d = np.load(wilor_path)
    joints = d["joints_3d"].astype(np.float32)   # (T, 2, 21, 3)
    valid  = d["hand_valid"]                      # (T, 2) bool

    dom_idx, nondom_idx = (
        (_WILOR_LEFT_IDX, _WILOR_RIGHT_IDX) if handedness == "left"
        else (_WILOR_RIGHT_IDX, _WILOR_LEFT_IDX)
    )

    dom    = joints[:, dom_idx,    :, :].copy()
    nondom = joints[:, nondom_idx, :, :].copy()

    dom[~valid[:, dom_idx]]       = np.nan
    nondom[~valid[:, nondom_idx]] = np.nan

    dom    -= dom[:, 0:1, :]
    nondom -= nondom[:, 0:1, :]

    mcp_dist = np.sqrt((dom[:, _WILOR_MCP_IDX, :] ** 2).sum(axis=-1))
    scale = float(np.nanmedian(mcp_dist))
    if scale < 1e-6:
        scale = 1.0
    dom    /= scale
    nondom /= scale

    if handedness == "left" and mirror_left:
        dom[..., 0]    = -dom[..., 0]
        nondom[..., 0] = -nondom[..., 0]

    return {"wilor_dom": dom, "wilor_nondom": nondom}


# ---------------------------------------------------------------------------
# Moryossef per-frame 3D hand normalisation
# ---------------------------------------------------------------------------

_HAND_INDEX_MCP  = 5
_HAND_MIDDLE_MCP = 9
_HAND_PINKY_MCP  = 17


def _rot_align_to_z(v: np.ndarray) -> np.ndarray:
    """Rotation matrix that aligns unit vector v to +Z."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(v, z))
    if s < 1e-6:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    axis /= s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def _rot_z(angle: float) -> np.ndarray:
    """Rotation matrix around Z by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def normalize_hand_moryossef(joints: np.ndarray) -> np.ndarray:
    """
    Per-frame 3D hand normalisation (Moryossef et al., 2023).

    For each frame:
      1. Wrist-center (joint 0 → origin)
      2. Rotate so palm normal → +Z  (back of hand on XY plane)
      3. Rotate around Z so middle metacarpal → +Y axis
      4. Scale so the middle metacarpal has unit length

    Input/output: (T, 21, 3) float32.  NaN frames are left unchanged.
    Works on both MediaPipe and MANO/WiLoR 21-joint hands.
    """
    out = joints.copy()
    for t in range(len(out)):
        frame = out[t]
        if np.any(np.isnan(frame)):
            continue

        frame = frame - frame[0]

        normal = np.cross(frame[_HAND_INDEX_MCP], frame[_HAND_PINKY_MCP])
        n = float(np.linalg.norm(normal))
        if n < 1e-6:
            out[t] = frame
            continue
        frame = frame @ _rot_align_to_z(normal / n).T

        mid = frame[_HAND_MIDDLE_MCP]
        frame = frame @ _rot_z(-np.arctan2(mid[0], mid[1])).T

        mid_len = float(np.linalg.norm(frame[_HAND_MIDDLE_MCP]))
        if mid_len > 1e-6:
            frame = frame / mid_len

        out[t] = frame

    return out.astype(np.float32)

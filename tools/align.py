import numpy as np

# If number of ground truth measurements per second is lower than this,
# treat the ground truth as sparse (use all points).
SPARSITY_THRESHOLD = 0.2

def isSparse(out):
    n = out.shape[0]
    if n <= 1: return True
    lengthSeconds = out[-1, 0] - out[0, 0]
    return n / lengthSeconds < SPARSITY_THRESHOLD

def getOverlap(out, gt, includeTime=False):
    """ Get overlapping parts of `out` and `gt` tracks on the time grid of `gt`. """
    if gt.size == 0 or out.size == 0:
        return np.array([]), np.array([])
    assert(out.shape[1] == 4 and gt.shape[1] == 4)
    gt_t = gt[:, 0]
    out_t = out[:, 0]
    if isSparse(gt):
        # Use all ground truth points even if it means extrapolating VIO.
        gt_part = gt
    else:
        min_t = max(np.min(out_t), np.min(gt_t))
        max_t = min(np.max(out_t), np.max(gt_t))
        gt_part = gt[(gt_t >= min_t) & (gt_t <= max_t), :]
    out_part = np.hstack([np.interp(gt_part[:, 0], out_t, out[:,i])[:, np.newaxis] for i in range(out.shape[1])])
    if includeTime: return out_part[:, 1:], gt_part[:, 1:], out_part[:, 0]
    else: return out_part[:, 1:], gt_part[:, 1:]

def align(out, gt, rel_align_time=-1, fix_origin=False, align3d=False, fix_scale=True,
        origin_zero=False, return_rotation_matrix=False, alignEnabled=True):
    """
    Align `out` to `gt` by rotating so that angle(gt[t]) = angle(out[t]), relative to the
    origin at some timestamp t, which is, determined as e.g, 1/3 of the
    session length. Negative value means the alignment is done using the whole segment.
    """
    out_rotation = None
    if not alignEnabled or len(out) <= 0 or len(gt) <= 0: return out, out_rotation

    out_part, gt_part = getOverlap(out, gt)
    if out_part.shape[0] <= 0: return out, out_rotation

    if origin_zero:
        gt_ref = 0 * gt_part[0, :]
        out_ref = 0 * out_part[0, :]
    elif fix_origin:
        gt_ref = gt_part[0, :]
        out_ref = out_part[0, :]
    else:
        gt_ref = np.mean(gt_part, axis=0)
        out_ref = np.mean(out_part, axis=0)

    if align3d:
        if rel_align_time > 0:
            # partial 3D align, not very well tested, use with caution
            t = int(len(out[:,0]) * rel_align_time)
            if out_part.shape[0] > t and t > 0:
                out_part = out_part[:t, :]
                gt_part = gt_part[:t, :]

        out_xyz = (out_part - out_ref).transpose()
        gt_xyz = (gt_part - gt_ref).transpose()

        if out_xyz.shape[1] <= 0: return out, out_rotation

        if fix_scale:
            scale = 1
        else:
            get_scale = lambda xyz: np.mean(np.sqrt(np.sum(xyz**2, axis=0)))
            scale = min(get_scale(gt_xyz) / max(get_scale(out_xyz), 1e-5), 100)

        # Procrustes / Wahba SVD solution
        B = np.dot(gt_xyz, scale * out_xyz.transpose())
        U, S, Vt = np.linalg.svd(B)
        R = np.dot(U, Vt)
        # Check for mirroring (not sure if this ever happens in practice)
        if np.linalg.det(R) < 0.0:
            flip = np.diag([1, 1, -1])
            R = np.dot(U, np.dot(flip, Vt))
        R *= scale
        aligned = out * 1
        aligned[:, 1:4] = np.dot(R, (out[:, 1:] - out_ref).transpose()).transpose() + gt_ref
        if return_rotation_matrix:
            return aligned, out_rotation, R
        else:
            return aligned, out_rotation

    # else align in 2d
    # represent track XY as complex numbers
    xy_to_complex = lambda arr: arr[:,0] + 1j * arr[:,1]
    gt_xy = xy_to_complex(gt_part - gt_ref)
    out_xy = xy_to_complex(out_part - out_ref)

    rot = 1
    if rel_align_time > 0.0:
        # rotate to match direction vectors at a certain time
        t = int(len(out[:,0]) * rel_align_time)
        max_t = min(len(out_xy), len(gt_xy))

        if t < max_t and np.minimum(np.abs(gt_xy[t]), np.abs(out_xy[t])) > 1e-5:
            rot = gt_xy[t] / out_xy[t]
        else:
            # align using full track if fails
            rel_align_time = -1

    if rel_align_time <= 0:
        # align using the full track
        valid = np.minimum(np.abs(gt_xy), np.abs(out_xy)) > 1e-5
        if np.sum(valid) > 0:
            rot = gt_xy[valid] / out_xy[valid]
            rot = rot / np.abs(rot)
            rot = np.mean(rot)

    if fix_scale:
        rot = rot / np.abs(rot)

    # rotate track, keeping also the parts that do not have GT
    align_xy = xy_to_complex(out[:, 1:] - out_ref) * rot

    # convert back to real
    aligned = out * 1
    aligned[:,1:] -= out_ref
    aligned[:,1] = np.real(align_xy)
    aligned[:,2] = np.imag(align_xy)
    aligned[:,1:] += gt_ref
    out_rotation = np.angle(rot)
    return aligned, out_rotation

def piecewiseAlign(out, gt, piece_len_sec=10.0, na_breaks=False):
    """ Align `out` in pieces so that they match `gt`. """
    gt_t = gt[:,0]
    out_t = out[:,0]
    max_t = np.max(gt_t)
    t = np.min(gt_t)
    aligned = []
    while t < max_t:
        t1 = t + piece_len_sec
        gt_slice = gt[(gt_t >= t) & (gt_t < t1), :]
        out_slice = out[(out_t >= t) & (out_t < t1), :]
        aligned_slice, _ = align(out_slice, gt_slice, rel_align_time=-1, fix_origin=False)
        aligned.append(aligned_slice)
        if na_breaks:
            na_spacer = aligned_slice[-1:,:]
            na_spacer[:,1:] = np.nan
            aligned.append(na_spacer)
        t = t1

    return np.vstack(aligned)

# Align 3-vectors such as velocity and angular velocity using rotation that matches the position tracks.
def alignWithTrackRotation(vioData, vioPosition, gtPosition):
    from scipy.spatial.transform import Rotation

    _ , angle = align(vioPosition, gtPosition, -1, **metricSetToAlignmentParams(Metric.FULL))
    if angle is None: return vioData
    R = Rotation.from_euler('z', angle).as_matrix()
    out = []
    for i in range(0, vioData.shape[0]):
        x = R.dot(vioData[i, 1:])
        out.append([vioData[i, 0], x[0], x[1], x[2]])
    return np.array(out)

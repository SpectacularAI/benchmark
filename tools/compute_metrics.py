#!/usr/bin/env python3

import json
import math
import os
import pathlib

from enum import Enum

import numpy as np

# Track types accurate enough to be used as ground-truth. In lowercase.
# Instead of adding entries to this list, consider converting your dataset to have "groundTruth" rows.
GROUND_TRUTH_TYPES = ["groundtruth", "rtkgps", "externalpose"]

# Scaling to get numbers closer to 1 that are easier for humans to compare.
PIECEWISE_METRIC_SCALE = 100.0

# If number of ground truth measurements per second is lower than this,
# treat the ground truth as sparse (use all points).
SPARSITY_THRESHOLD = 1.0

# If VIO has break longer than this, subtract the break length from the coverage metric.
COVERAGE_GAP_THRESHOLD_SECONDS = 1.0

# Levels for absolute error metric percentiles.
PERCENTILES = [95, 100]

class Metric(Enum):
    # For non-piecewise alignment, this is the "proper" metric for VIO methods that
    # rotates the track only around the z-axis since the VIO is supposed to be able
    # to estimate direction of gravity (z-axis). The piecewise alignment methods are
    # similar to this.
    FULL = "full"
    # This is the SE3 alignment commonly used in academic benchmarks (with RMSE).
    FULL_3D = "full_3d"
    # Sometimes used in academic benchmarks for monocular visual-only methods.
    FULL_3D_SCALED = "full_sim3"
    # Like `FULL`, but cuts the track into segments and aligns each individually.
    PIECEWISE = "piecewise"
    # Like `PIECEWISE`, but does not penalize drift in z direction.
    PIECEWISE_NO_Z = "piecewise_no_z"
    # Number in [0, 1] that indicates how large portion of the ground truth the VIO track covers.
    COVERAGE = "coverage"
    # RMSE of aligned angular velocities. May be computed from orientations if not available in the data.
    ANGULAR_VELOCITY = "angular_velocity"
    # RMSE of aligned velocities. May be computed from positions if not available in the data.
    VELOCITY = "velocity"
    # Like `FULL`, but for sparse postprocessed output, eg our VIO's SLAM map keyframes.
    POSTPROCESSED = "postprocessed"
    # Output from UNIX `time` command (not wall time).
    CPU_TIME = "cpu_time"
    # Orientation error
    ORIENTATION = "orientation"
    # Orientation error, including error split by gravity and heading
    ORIENTATION_FULL = "orientation_full"
    # Orientation error, orientation is first aligned with ground truth
    ORIENTATION_ALIGNED = "orientation_aligned"

def metricSetToAlignmentParams(metricSet):
    if metricSet in [
        Metric.FULL,
        Metric.POSTPROCESSED,
        Metric.COVERAGE,
        Metric.PIECEWISE,
        Metric.PIECEWISE_NO_Z
    ]:
        return {} # The defaults are correct.
    elif metricSet == Metric.FULL_3D:
        return dict(align3d=True)
    elif metricSet == Metric.FULL_3D_SCALED:
        return dict(align3d=True, fix_scale=False)
    elif metricSet in [Metric.ANGULAR_VELOCITY, Metric.VELOCITY, Metric.CPU_TIME]:
        return {} # No natural alignment for these.
    else:
        raise Exception("Unimplemented alignment parameters for metric {}".format(metricSet.value))

def percentileName(p):
    return "max" if p == 100 else "p{}".format(p)

def isSparse(out):
    n = out.shape[0]
    if n <= 1: return True
    lengthSeconds = out[-1, 0] - out[0, 0]
    return n / lengthSeconds < SPARSITY_THRESHOLD

def getOverlap(out, gt):
    """ Get overlapping parts of `out` and `gt` tracks on the time grid of `gt`. """
    if gt.size == 0 or out.size == 0:
        return np.array([]), np.array([])
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
    return out_part[:, 1:], gt_part[:, 1:]

def align(out, gt, rel_align_time=-1, fix_origin=False, align3d=False, fix_scale=True, origin_zero=False, return_rotation_matrix=False):
    """
    Align `out` to `gt` by rotating so that angle(gt[t]) = angle(out[t]), relative to the
    origin at some timestamp t, which is, determined as e.g, 1/3 of the
    session length. Negative value means the alignment is done using the whole segment.
    """
    out_rotation = None
    if len(out) <= 0 or len(gt) <= 0: return out, out_rotation

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

def rmse(a, b):
    """Root Mean Square Error"""
    assert(a.size != 0 and b.size != 0)
    return np.sqrt(np.mean(np.sum((a - b)**2, axis=1)))

def meanAbsoluteError(a, b):
    """Mean Absolute Error (MAE)"""
    assert(a.size != 0 and b.size != 0)
    return np.mean(np.sqrt(np.sum((a - b)**2, axis=1)))

# Returns start and end (ex) indexes for source that are within target start and end
def getIncludedOverlap(sourceTimes, targetTimes):
    sourceStartInd = 0
    sourceEndInd = len(sourceTimes) - 1
    while sourceTimes[sourceStartInd] < targetTimes[0]: sourceStartInd += 1
    while sourceTimes[sourceEndInd] > targetTimes[-1]: sourceEndInd -= 1
    assert(sourceStartInd <= sourceEndInd)
    return (sourceStartInd, sourceEndInd + 1)

def getOverlapOrientations(vio, gt):
    # TODO: Cache this
    from scipy.spatial.transform import Rotation, Slerp
    # `Slerp` requires that the interpolation grid is inside the data time boundaries.
    tVio = vio["position"][:, 0]
    tGt = gt["position"][:, 0]
    gtStartInd, gtEndInd = getIncludedOverlap(tGt, tVio)
    gtOverlapTime = tGt[gtStartInd:gtEndInd]
    qVio = Rotation.from_quat(vio["orientation"][:, 1:])
    slerp = Slerp(tVio, qVio)

    qGt = Rotation.from_quat(gt["orientation"][gtStartInd:gtEndInd, 1:])
    slerpOrientations = slerp(gtOverlapTime)
    assert(len(slerpOrientations) == len(gtOverlapTime))

    quaternions = np.array([(gtQ * vioQ.inv()).as_quat() for vioQ, gtQ in zip(slerpOrientations, qGt)])
    A = sum([np.outer(q, q.T) for q in quaternions])
    eigenvectors = np.linalg.eig(A)[1]
    avgRotation = Rotation.from_quat(eigenvectors[:, 0])

    return {
        "overlappinGtIndexes": (gtStartInd, gtEndInd),
        "overlappingOrientation": qGt,
        "overlappingGtTimes": gtOverlapTime,
        "avgRotation": avgRotation,
        "slerpVioOrientations": slerpOrientations
    }

def computePercentiles(a, b, out):
    """Absolute error below which given percentile of measurements fall"""
    assert(a.size != 0 and b.size != 0)
    err = np.sqrt(np.sum((a - b)**2, axis=1))
    err.sort()
    for p in PERCENTILES:
        assert(p >= 0 and p <= 100)
        ind = int((err.size - 1) * p / 100)
        out[percentileName(p)] = err[ind]

def computePiecewiseMetric(out, gt, pieceLenSecs=10.0, measureZError=True):
    """RMSE of the aligned XY track (in ground truth time grid)"""
    assert(pieceLenSecs > 0)
    if out.size == 0 or gt.size == 0:
        return None
    aligned = piecewiseAlign(out, gt, pieceLenSecs)
    if aligned.size == 0:
        return None
    if not measureZError:
        aligned = aligned[:,:-1]
        gt = gt[:,:-1]
    interpolated, gt = getOverlap(aligned, gt)
    if interpolated.size == 0 or gt.size == 0:
        return None
    # In random walk noise, the standard deviance is proportional to elapsed time.
    normalizedRmse = PIECEWISE_METRIC_SCALE * rmse(gt, interpolated) / np.sqrt(pieceLenSecs)
    return normalizedRmse

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

def computeVelocityMetric(vio, gt, intervalSeconds=None):
    computeAlignedVelocity(vio, gt, intervalSeconds)
    vioPart, gtPart = getOverlap(vio["velocity"], gt["velocity"])
    if gtPart.size == 0 or vioPart.size == 0: return None
    return rmse(gtPart, vioPart)

def computeAlignedVelocity(vio, gt, intervalSeconds=None):
    vioV = computeVelocity(vio, intervalSeconds)
    gtV = computeVelocity(gt, intervalSeconds)
    # vioVAligned, _ = align(vioV, gtV, -1, fix_origin=False, align3d=True, fix_scale=True, origin_zero=True)
    # vioVAligned = alignWithTrackRotation(vioV, vio["position"], gt["position"])
    vioVAligned = vioV
    vioVAligned[:,1:] = getOverlapOrientations(vio, gt)["avgRotation"].apply(vioV[:,1:])
    vio["velocity"] = vioVAligned
    gt["velocity"] = gtV

# If intervalSeconds is provided, the data is sampled at that rate to compute velocity from position
# despite how high frequency it is, to prevent small delta time cause inaccuracies in velocity
def computeVelocity(data, intervalSeconds=None):
    FILTER_SPIKES = True
    USE_PRECOMPUTED_VELOCITIES = True
    if USE_PRECOMPUTED_VELOCITIES and "velocity" in data and data["velocity"].shape[0] > 0:
        return data["velocity"]
    if intervalSeconds:
        p = []
        prevT = None
        for pos in data["position"]:
            if prevT == None or pos[0] - prevT >= intervalSeconds:
                prevT = pos[0]
                p.append(pos)
        p = np.array(p)
    else:
        p = data["position"]
    vs = []
    i = 0
    for i in range(1, p.shape[0] - 1):
        dt = p[i + 1, 0] - p[i - 1, 0]
        if dt <= 0: continue
        dp = p[i + 1, 1:] - p[i - 1, 1:]
        if FILTER_SPIKES and np.linalg.norm(dp) > 0.1: continue
        v = dp / dt
        vs.append([p[i, 0], v[0], v[1], v[2]])
    return np.array(vs)

def computeAngularVelocityMetric(vio, gt, intervalSeconds=None):
    computeAlignedAngularVelocity(vio, gt, intervalSeconds)
    vioPart, gtPart = getOverlap(vio["angularVelocity"], gt["angularVelocity"])
    if gtPart.size == 0 or vioPart.size == 0: return None
    return rmse(gtPart, vioPart)

def computeAlignedAngularVelocity(vio, gt, intervalSeconds=None):
    vioAv = computeAngularVelocity(vio, intervalSeconds)
    gtAv = computeAngularVelocity(gt, intervalSeconds)
    # vioAvAligned, _ = align(vioAv, gtAv, -1, fix_origin=False, align3d=False, fix_scale=True, origin_zero=True)
    # vioAvAligned = alignWithTrackRotation(vioAv, vio["position"], gt["position"])
    vioAvAligned = vioAv
    vioAvAligned[:,1:] = getOverlapOrientations(vio, gt)["avgRotation"].apply(vioAv[:,1:])
    vio["angularVelocity"] = vioAvAligned
    gt["angularVelocity"] = gtAv

def computeAngularVelocity(data, intervalSeconds=None):
    USE_PRECOMPUTED_ANGULAR_VELOCITIES = True
    if USE_PRECOMPUTED_ANGULAR_VELOCITIES and "angularVelocity" in data and data["angularVelocity"].shape[0] > 0:
        # There can be large spikes in SDK's output in the very beginning. Ignore them because
        # in practice the angular velocities are not used for anything critical in that time frame.
        SKIP_SECONDS_FROM_BEGINNING = 0.1
        avs = data["angularVelocity"]
        t = avs[:, 0]
        t0 = avs[0, 0] + SKIP_SECONDS_FROM_BEGINNING
        return avs[t > t0, :]

    from scipy.spatial.transform import Rotation

    # Like `odometry::util::computeAngularVelocity()` from C++, but takes
    # output-to-world quaternions rather than world-to-output, which is
    # why the `inv()` (conjugate) calls are swapped.
    def angularVelocity(qa, qb, ta, tb):
        dt = tb - ta
        if dt <= 0: return None
        q = (Rotation.from_quat(qb) * Rotation.from_quat(qa).inv()).as_quat()
        if q[3] < 0: q = -q
        theta = 2 * math.acos(q[3])
        if theta > 1e-6:
            c = theta / math.sin(0.5 * theta)
        else:
            c = 2.0
        r = c * q[:3] / dt
        return [(tb + ta) * .5, r[0], r[1], r[2]]

    if intervalSeconds:
        q = []
        prevT = None
        for ori in data["orientation"]:
            if prevT == None or ori[0] - prevT >= intervalSeconds:
                prevT = ori[0]
                q.append(ori)
        q = np.array(q)
    else:
        q = data["orientation"]

    avs = []
    i = 0
    for i in range(1, q.shape[0] - 1):
        av = angularVelocity(q[i - 1, 1:], q[i + 1, 1:], q[i - 1, 0], q[i + 1, 0])
        if not av: continue
        avs.append(av)
    return np.array(avs)

def computeCoverage(out, gt, info):
    if not "videoTimeSpan" in info or not info["videoTimeSpan"]: return None
    if gt.size == 0: return None
    if out.size == 0: return 0.0
    # Use range of video input as the target, since in some datasets the ground truth
    # samples cover a longer segment, which then makes getting the full coverage score impossible.
    t0 = info["videoTimeSpan"][0]
    t1 = info["videoTimeSpan"][1]
    lengthSeconds = t1 - t0
    assert(lengthSeconds >= 0)
    t = t0
    outInd = 0
    gap = 0.0
    assert(out.shape[0] >= 1)
    while outInd < out.shape[0]:
        tVio = out[outInd, 0]
        dt = tVio - t
        if dt > COVERAGE_GAP_THRESHOLD_SECONDS:
            gap += dt
        t = tVio
        outInd += 1
    dt = t1 - tVio
    if dt > COVERAGE_GAP_THRESHOLD_SECONDS:
        gap += dt
    assert(gap <= lengthSeconds)
    coverage = (lengthSeconds - gap) / lengthSeconds
    assert(coverage >= 0.0 and coverage <= 1.0)
    return coverage

# Compute orientation errors over time and distance
# * alignTrajectory = aligns 3D trajectory before computing orientation error
# * alignOrientation = computes average rotation from vio->gt and applies that before computing orientation error
class OrientationAlign(Enum):
    # For non-piecewise alignment, this is the "proper" metric for VIO methods that
    # rotates the track only around the z-axis since the VIO is supposed to be able
    # to estimate direction of gravity (z-axis). The piecewise alignment methods are
    # similar to this.
    TRAJECTORY = "trajectory"
    # This is the SE3 alignment commonly used in academic benchmarks (with RMSE).
    AVERAGE_ORIENTATION = "average_orientation"

def computeOrientationErrors(vio, gt, alignType=OrientationAlign.TRAJECTORY):
    from scipy.spatial.transform import Rotation
    overlap = getOverlapOrientations(vio, gt)
    qGt = overlap["overlappingOrientation"]
    (gtStartInd, gtEndInd) = overlap["overlappinGtIndexes"]
    distGt = traveledDistance(gt["position"][gtStartInd:gtEndInd, 1:])

    # Find optimal rotation for trajectory to fit the ground truth
    if alignType == OrientationAlign.TRAJECTORY:
        (_, _, R) = align(vio["position"], gt["position"], align3d=True, fix_scale=True, return_rotation_matrix=True)
        qVio = Rotation.from_matrix(R) * overlap["slerpVioOrientations"]
    else:
        qVio = overlap["avgRotation"] * overlap["slerpVioOrientations"]

    totalAngle = []
    gravityAngle = []
    headingAngle = []
    GRAVITY_DIRECTION = np.array([0, 0, -1]) # TODO: This doesn't make much sense with align3d
    for i in range(len(qVio)):
        q = qVio[i].as_matrix()
        g = qGt[i].as_matrix()
        Q = g.transpose() @ q
        totalAngle.append(np.linalg.norm(Rotation.from_matrix(Q).as_rotvec(degrees=True)))

        # Project global gravity direction to local coordinates and compare.
        gravityAngle.append(np.arccos(np.dot(q.transpose() @ GRAVITY_DIRECTION, g.transpose() @ GRAVITY_DIRECTION)))

        # Project local X axis to world XY plane and compare.
        xq = q[:2, 0] / np.linalg.norm(q[:2, 0])
        xg = g[:2, 0] / np.linalg.norm(g[:2, 0])
        headingAngle.append(np.arccos(np.dot(xq, xg)))

    return {
        "time": overlap["overlappingGtTimes"] - overlap["overlappingGtTimes"][0],
        "dist": distGt,
        "total": totalAngle,
        "gravity": 180. / np.pi * np.array(gravityAngle),
        "heading": 180. / np.pi * np.array(headingAngle),
    }

def computeOrientationErrorMetric(vio, gt, full=False, alignType=None):
    if gt and len(gt.get("orientation", [])) > 0:
        def rmseAngle(a):
            return np.sqrt(np.mean(np.array(a)**2))
        orientationErrors = computeOrientationErrors(vio, gt, alignType)
        result = {
            "RMSE total": rmseAngle(orientationErrors["total"]),
        }
        if full:
            result["RMSE gravity"] = rmseAngle(orientationErrors["gravity"])
            result["RMSE heading"] = rmseAngle(orientationErrors["heading"])
        return result

    return None

# Compute cumulative sum of distance traveled. Output length matches input length.
def traveledDistance(positions):
    pDiff = np.diff(positions, axis=0)
    pDiff = np.vstack((np.array([0, 0, 0]), pDiff))
    dDiff = np.linalg.norm(pDiff, axis=1)
    return np.cumsum(dDiff)

# Compute a dict with all given metrics. If a metric cannot be computed, output `None` for it.
def computeMetricSets(vio, vioPostprocessed, gt, info, sampleIntervalForVelocity=None):
    pVio = vio["position"]
    pGt = gt["position"]
    if pVio.size > 0 and pGt.size > 0 and (pVio[0, 0] > pGt[-1, 0] or pVio[-1, 0] < pGt[0, 0]):
        print("{}: VIO timestamps do not overlap with ground truth, cannot compute metrics or align."
            .format(info["caseName"]))

    metricSets = info["metricSets"]
    fixOrigin = "fixOrigin" in info and info["fixOrigin"]

    metrics = {}
    for metricSetStr in metricSets:
        metricSet = Metric(metricSetStr)
        if metricSet in [Metric.PIECEWISE, Metric.PIECEWISE_NO_Z]:
            measureZError = metricSet != Metric.PIECEWISE_NO_Z
            m = {
                "1s": computePiecewiseMetric(pVio, pGt, 1.0, measureZError),
                "10s": computePiecewiseMetric(pVio, pGt, 10.0, measureZError),
                "30s": computePiecewiseMetric(pVio, pGt, 30.0, measureZError),
                "100s": computePiecewiseMetric(pVio, pGt, 100.0, measureZError),
            }
            if None in m.values(): m = None
            metrics[metricSetStr] = m
        elif metricSet in [Metric.FULL, Metric.FULL_3D, Metric.FULL_3D_SCALED]:
            alignedVio, _ = align(pVio, pGt, -1,
                fix_origin=fixOrigin, **metricSetToAlignmentParams(metricSet))
            alignedVio, unalignedGt = getOverlap(alignedVio, pGt)
            if unalignedGt.size > 0 and alignedVio.size > 0:
                metrics[metricSetStr] = {
                    "RMSE": rmse(unalignedGt, alignedVio),
                    "MAE": meanAbsoluteError(unalignedGt, alignedVio),
                }
                computePercentiles(unalignedGt, alignedVio, metrics[metricSetStr])
            else:
                metrics[metricSetStr] = None
        elif metricSet == Metric.COVERAGE:
            metrics[metricSetStr] = computeCoverage(pVio, pGt, info)
        elif metricSet == Metric.VELOCITY:
            metrics[metricSetStr] = computeVelocityMetric(vio, gt, sampleIntervalForVelocity)
        elif metricSet == Metric.ANGULAR_VELOCITY:
            metrics[metricSetStr] = computeAngularVelocityMetric(vio, gt, sampleIntervalForVelocity)
        elif metricSet == Metric.POSTPROCESSED:
            if vioPostprocessed:
                # Note that compared to the other metrics, the order of arguments is swapped
                # so that the (sparse) time grid of postprocessed VIO is used.
                alignedGt, _ = align(pGt, vioPostprocessed["position"], -1,
                    fix_origin=fixOrigin, **metricSetToAlignmentParams(metricSet))
                alignedGt, unalignedVio = getOverlap(alignedGt, vioPostprocessed["position"])
                if alignedGt.size > 0 and unalignedVio.size > 0:
                    metrics[metricSetStr] = rmse(alignedGt, unalignedVio)
                else:
                    metrics[metricSetStr] = None
        elif metricSet == Metric.CPU_TIME:
            metrics[metricSetStr] = None
            if "cpuTime" in info: metrics[metricSetStr] = info["cpuTime"]
        elif metricSet == Metric.ORIENTATION:
            metrics[metricSetStr] = computeOrientationErrorMetric(vio, gt, alignType=OrientationAlign.TRAJECTORY)
        elif metricSet == Metric.ORIENTATION_FULL:
            metrics[metricSetStr] = computeOrientationErrorMetric(vio, gt, full=True, alignType=OrientationAlign.TRAJECTORY)
        elif metricSet == Metric.ORIENTATION_ALIGNED:
            metrics[metricSetStr] = computeOrientationErrorMetric(vio, gt, alignType=OrientationAlign.AVERAGE_ORIENTATION)
        else:
            raise Exception("Unimplemented metric {}".format(metricSetStr))
    return metrics

def readDatasetsCsv(fn):
    return [{"name": "groundTruth", "position": np.genfromtxt(fn, delimiter=',')}]

def readDatasetsJson(fn, include=[], exclude=[]):
    with open(fn) as f:
        data = json.load(f)["datasets"]
    filteredData = []
    for dataset in data:
        name = dataset["name"].lower()
        if include and not name in include: continue
        if name in exclude: continue

        position = []
        orientation = []
        for d in dataset["data"]:
            position.append([d["time"], d["position"]["x"], d["position"]["y"], d["position"]["z"]])
            if "orientation" in d:
                q = d["orientation"]
                orientation.append([d["time"], q["x"], q["y"], q["z"], q["w"]]) # Scipy ordering.
        filteredData.append({
            "name": dataset["name"],
            "position": np.array(position),
            "orientation": np.array(orientation),
        })
    return filteredData

VIO_OUTPUT_CACHE = {}

def readVioOutput(benchmarkFolder, caseName, info, postprocessed=False):
    # Caching is not that beneficial here.
    global VIO_OUTPUT_CACHE
    baseName = benchmarkFolder.split("/")[-1] # Avoid differences from relative paths and symlinks.
    key = "{}+{}+{}".format(baseName, caseName, postprocessed)
    if key in VIO_OUTPUT_CACHE: return VIO_OUTPUT_CACHE[key].copy()

    if postprocessed:
        outputPath = "{}/vio-output/{}_map.jsonl".format(benchmarkFolder, caseName)
        if not pathlib.Path(outputPath).exists():
            return {}
    else:
        outputPath = "{}/vio-output/{}.jsonl".format(benchmarkFolder, caseName)

    def isValidVector(row, field):
        if field not in row: return False
        # VIO should not produce values like this, but they have been spotted.
        # Crash rather than ignoring silently.
        for c in "xyz":
            if row[field][c] == None: raise Exception("Null values in VIO outputs.")
        return True


    method = None
    if "methodName" in info: method = info["methodName"].lower()

    bias_norm = lambda x: np.sqrt(np.sum(x**2, axis=1))
    to_arr = lambda obj: [obj["x"], obj["y"], obj["z"]]
    position = []
    orientation = []
    velocity = []
    angularVelocity = []
    bga = []
    baa = []
    bat = []
    stat = []
    idToTime = {}
    loopClosures = []
    resets = []
    with open(outputPath) as f:
        for line in f.readlines():
            row = json.loads(line)
            t = row["time"]
            if method and method in row:
                row = row[method]
            if t == None: continue
            if not isValidVector(row, "position"): continue
            position.append([t, row["position"]["x"], row["position"]["y"], row["position"]["z"]])
            if isValidVector(row, "orientation"):
                q = row["orientation"]
                orientation.append([t, q["x"], q["y"], q["z"], q["w"]])
            if isValidVector(row, "velocity"):
                v = row["velocity"]
                velocity.append([t, v["x"], v["y"], v["z"]])
            if isValidVector(row, "angularVelocity"):
                av = row["angularVelocity"]
                angularVelocity.append([t, av["x"], av["y"], av["z"]])
            stat.append(row.get("stationary", False))
            if "biasMean" in row:
                bga.append(to_arr(row["biasMean"]["gyroscopeAdditive"]))
                baa.append(to_arr(row["biasMean"]["accelerometerAdditive"]))
                if "accelerometerTransform" in row["biasMean"]:
                    bat.append(to_arr(row["biasMean"]["accelerometerTransform"]))
            if "id" in row:
                idToTime[row["id"]] = t
            if "loopClosureIds" in row:
                for i in range(len(row["loopClosureIds"])):
                    loopClosureId = row["loopClosureIds"][i]
                    loopClosureLinkColor = row["loopClosureLinkColors"][i] if "loopClosureLinkColors" in row else "deeppink"
                    if not loopClosureId in idToTime: continue
                    # Save times rather than indices as they survive the align operations better.
                    loopClosures.append((t, idToTime[loopClosureId], loopClosureLinkColor))
            if "status" in row:
                if row["status"] == "LOST_TRACKING": resets.append(t)

    VIO_OUTPUT_CACHE[key] = {
        'position': np.array(position),
        'orientation': np.array(orientation),
        'velocity': np.array(velocity),
        'angularVelocity': np.array(angularVelocity),
        'BGA': bias_norm(np.array(bga)) if bga else 0.0,
        'BAA': bias_norm(np.array(baa)) if baa else 0.0,
        'BAT': bias_norm(np.array(bat) - 1.0) if bat else 0.0,
        'stationary': np.array(stat),
        'loopClosures': loopClosures,
        'resets': resets,
    }
    return VIO_OUTPUT_CACHE[key].copy()

OTHER_DATASETS_CACHE = {}

# `include` and `exclude` filter by dataset type. If `include` is empty, everything is included
# before the `exclude` filter is applied.
def readDatasets(benchmarkFolder, caseName, include=[], exclude=[]):
    # Caching provides a significant speed-up here.
    global OTHER_DATASETS_CACHE
    baseName = benchmarkFolder.split("/")[-1] # Avoid differences from relative paths and symlinks.
    key = "{}+{}+{}".format(baseName, caseName, "-".join(include))
    if key in OTHER_DATASETS_CACHE: return OTHER_DATASETS_CACHE[key].copy()

    gtJson = "{}/ground-truth/{}.json".format(benchmarkFolder, caseName)
    gtCsv = "{}/ground-truth/{}.csv".format(benchmarkFolder, caseName)
    if os.path.isfile(gtJson):
        OTHER_DATASETS_CACHE[key] = readDatasetsJson(gtJson, include, exclude)
    elif os.path.isfile(gtCsv):
        OTHER_DATASETS_CACHE[key] = readDatasetsCsv(gtCsv)
    else:
        return []
    return OTHER_DATASETS_CACHE[key].copy()

# Compute a single value that summarises the results, based on preference order
# for the available metrics. If the most prefered metric computation has failed, output `None`
# rather than falling back to another metric (as that would mess up averages across multiple cases).
def computeSummaryValue(metricsJson):
    for metricSet in [Metric.PIECEWISE, Metric.PIECEWISE_NO_Z]:
        if not metricSet.value in metricsJson: continue
        if not metricsJson[metricSet.value]: return None
        return np.mean(list(metricsJson[metricSet.value].values()))
    for metricSet in [Metric.FULL, Metric.FULL_3D, Metric.FULL_3D_SCALED]:
        if not metricSet.value in metricsJson: continue
        if not metricsJson[metricSet.value]: return None
        return metricsJson[metricSet.value]["RMSE"]
    for metricSet in [Metric.POSTPROCESSED, Metric.COVERAGE, Metric.ANGULAR_VELOCITY, Metric.CPU_TIME]:
        if not metricSet.value in metricsJson: continue
        return metricsJson[metricSet.value]
    return None

def setRelativeMetric(relative, metricSetStr, a, b):
    if a is None or b is None or b <= 0: return
    relative[metricSetStr] = a / b

# Computed the same way as in `computeSummaryValue()`.
def computeRelativeMetrics(metrics, baseline):
    def hasResults(kind, m):
        return kind in m and m[kind]

    relative = {}
    for metricSet in [Metric.PIECEWISE, Metric.PIECEWISE_NO_Z]:
        metricSetStr = metricSet.value
        if hasResults(metricSetStr, metrics) and hasResults(metricSetStr, baseline):
            a = np.mean(list(metrics[metricSetStr].values()))
            b = np.mean(list(baseline[metricSetStr].values()))
            setRelativeMetric(relative, metricSetStr, a, b)
    for metricSet in [Metric.FULL, Metric.FULL_3D, Metric.FULL_3D_SCALED]:
        metricSetStr = metricSet.value
        if hasResults(metricSetStr, metrics) and hasResults(metricSetStr, baseline):
            a = metrics[metricSetStr]["RMSE"]
            b = baseline[metricSetStr]["RMSE"]
            setRelativeMetric(relative, metricSetStr, a, b)
    for metricSet in [Metric.COVERAGE, Metric.ANGULAR_VELOCITY, Metric.POSTPROCESSED, Metric.CPU_TIME]:
        metricSetStr = metricSet.value
        if not metricSetStr in metrics or not metricSetStr in baseline: continue
        a = metrics[metricSetStr]
        b = baseline[metricSetStr]
        setRelativeMetric(relative, metricSetStr, a, b)
    return relative

def computeMetrics(benchmarkFolder, caseName, baseline=None, sampleIntervalForVelocity=None):
    infoPath = "{}/info/{}.json".format(benchmarkFolder, caseName)
    with open(infoPath) as infoFile:
        info = json.loads(infoFile.read())

    datasets = readDatasets(benchmarkFolder, caseName, GROUND_TRUTH_TYPES)
    gt = datasets[0] if datasets else None

    vio = readVioOutput(benchmarkFolder, caseName, info, False)
    vioPostprocessed = readVioOutput(benchmarkFolder, caseName, info, True)

    if gt:
        metricsJson = computeMetricSets(vio, vioPostprocessed, gt, info, sampleIntervalForVelocity)
        if baseline:
            relative = computeRelativeMetrics(metricsJson, baseline)
            metricsJson["relative"] = relative
        metricsDir = "{}/metrics".format(benchmarkFolder)
        pathlib.Path(metricsDir).mkdir(parents=True, exist_ok=True)
        metricsPath = "{}/{}.json".format(metricsDir, caseName)
        with open(metricsPath, "w") as metricsFile:
            metricsFile.write(json.dumps(metricsJson, indent=4, separators=(',', ': ')))
        return computeSummaryValue(metricsJson)
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarkFolder")
    parser.add_argument("caseName")
    parser.add_argument('--baseline', default=None)
    parser.add_argument('--baseline', default=None)
    parser.add_argument("--sampleIntervalForVelocity", help="Downsamples ground truth position/orientation frequency before calculating velocity and angular velocity, provide minimum number of seconds between samples i.e. 0.1 = max 10Hz GT", type=float)
    args = parser.parse_args()

    result = computeMetrics(args.benchmarkFolder, args.caseName, args.baseline, args.sampleIntervalForVelocity)
    print(result)

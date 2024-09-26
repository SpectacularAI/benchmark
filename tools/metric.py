from enum import Enum
import math
import numpy as np

from .align import align, alignWithTrackRotation, piecewiseAlign, getOverlap

# Scaling to get numbers closer to 1 that are easier for humans to compare.
PIECEWISE_METRIC_SCALE = 100.0

# If VIO has break longer than this, subtract the break length from the coverage metric.
COVERAGE_GAP_THRESHOLD_SECONDS = 1.0

# Levels for absolute error metric percentiles.
PERCENTILES = [95, 100]

# How many seconds into future the position and orientation are predicted
PREDICTION_SECONDS = 0.03

class VioTrackKind(Enum):
    # VIO is allowed to only use data preceeding the output time.
    # In the relevant use cases VIO often also needs to run in realtime with the given hardware.
    REALTIME = "realtime"
    # VIO is allowed to use the whole session data to optimize all outputs. Non-causal.
    POSTPROCESSED = "postprocessed"
    # Like `REALTIME`, but VIO must output WGS coordinates and there will be no aligning of the output tracks.
    GLOBAL = "global"

class Metric(Enum):
    # Track position error metrics:
    #
    # Do not align tracks in any way. Useful when VIO is is supplied with inputs that
    # allow it to track wrt a known coordinate system.
    NO_ALIGN = "no_align"
    # Similar to `NO_ALIGN`, but requires VIO output and reference tracks in WGS coordinates.
    GLOBAL = "global"
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
    # Similar to `PIECEWISE`, but builds the segments from the VIO pose trail outputs.
    POSE_TRAIL_3D = "pose_trail_3d"
    # Not implemented, but this would be like POSE_TRAIL_3D where the orientation is aligned only around gravity.
    # POSE_TRAIL = "pose_trail"

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
    # Error when predicting position and orientation forward in time
    PREDICTION = "prediction"
    # Correlation of VIO tracking quality estimate to momentary tracking accuracy.
    TRACKING_QUALITY = "tracking_quality"

def metricToTrackKind(metricSet):
    if metricSet == Metric.POSTPROCESSED:
        return VioTrackKind.POSTPROCESSED
    if metricSet == Metric.GLOBAL:
        return VioTrackKind.GLOBAL
    return VioTrackKind.REALTIME

def metricSetToAlignmentParams(metricSet):
    if metricSet in [
        Metric.FULL,
        Metric.POSTPROCESSED,
        Metric.COVERAGE,
        Metric.PIECEWISE,
        Metric.PIECEWISE_NO_Z,
    ]:
        return {} # The defaults are correct.
    elif metricSet in [Metric.NO_ALIGN, Metric.GLOBAL]:
        return dict(alignEnabled=False)
    elif metricSet == Metric.FULL_3D:
        return dict(align3d=True)
    elif metricSet == Metric.FULL_3D_SCALED:
        return dict(align3d=True, fix_scale=False)
    elif metricSet in [Metric.ANGULAR_VELOCITY, Metric.VELOCITY, Metric.CPU_TIME]:
        return {} # No natural alignment for these / not used.
    else:
        raise Exception("Unimplemented alignment parameters for metric {}".format(metricSet.value))

def percentileName(p):
    return "max" if p == 100 else "p{}".format(p)

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

def computeVelocityMetric(vio, gt, intervalSeconds=None):
    preComputeAlignedVelocity(vio, gt, intervalSeconds)
    vioPart, gtPart = getOverlap(vio["alignedVelocity"], gt["alignedVelocity"])
    if gtPart.size == 0 or vioPart.size == 0: return None
    return rmse(gtPart, vioPart)

def preComputeAlignedVelocity(vio, gt, intervalSeconds=None):
    if "alignedVelocity" in vio and "alignedVelocity" in gt: return
    vioV = computeVelocity(vio, intervalSeconds)
    gtV = computeVelocity(gt, intervalSeconds)
    # vioVAligned, _ = align(vioV, gtV, -1, fix_origin=False, align3d=True, fix_scale=True, origin_zero=True)
    # vioVAligned = alignWithTrackRotation(vioV, vio["position"], gt["position"])
    vioVAligned = np.copy(vioV)
    vioVAligned[:,1:] = getOverlapOrientations(vio, gt)["avgRotation"].apply(vioV[:,1:])
    vio["alignedVelocity"] = vioVAligned
    gt["alignedVelocity"] = gtV

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
    preComputeAlignedAngularVelocity(vio, gt, intervalSeconds)
    vioPart, gtPart = getOverlap(vio["alignedAngularVelocity"], gt["alignedAngularVelocity"])
    if gtPart.size == 0 or vioPart.size == 0: return None
    return rmse(gtPart, vioPart)

def preComputeAlignedAngularVelocity(vio, gt, intervalSeconds=None):
    if "alignedAngularVelocity" in vio and "alignedAngularVelocity" in gt: return
    vioAv = computeAngularVelocity(vio, intervalSeconds)
    gtAv = computeAngularVelocity(gt, intervalSeconds)
    # vioAvAligned, _ = align(vioAv, gtAv, -1, fix_origin=False, align3d=False, fix_scale=True, origin_zero=True)
    # vioAvAligned = alignWithTrackRotation(vioAv, vio["position"], gt["position"])
    vioAvAligned = np.copy(vioAv)
    vioAvAligned[:,1:] = getOverlapOrientations(vio, gt)["avgRotation"].apply(vioAv[:,1:])
    vio["alignedAngularVelocity"] = vioAvAligned
    gt["alignedAngularVelocity"] = gtAv

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
    # Find best alignment to match orientations in ground truth
    AVERAGE_ORIENTATION = "average_orientation"
    # No alignment at all
    NONE = "none"

def computeOrientationErrors(vio, gt, alignType=OrientationAlign.TRAJECTORY):
    from scipy.spatial.transform import Rotation
    overlap = getOverlapOrientations(vio, gt)
    qGt = overlap["overlappingOrientation"]
    (gtStartInd, gtEndInd) = overlap["overlappinGtIndexes"]
    distGt = traveledDistance(gt["position"][gtStartInd:gtEndInd, 1:])

    if alignType == OrientationAlign.TRAJECTORY:
        (_, rad) = align(vio["position"], gt["position"], rel_align_time=-1, fix_origin=False, align3d=False, fix_scale=True, origin_zero=False)
        R = [[math.cos(rad), -math.sin(rad), 0],
            [math.sin(rad),  math.cos(rad),  0],
            [0,              0,              1]]
        qVio = Rotation.from_matrix(R) * overlap["slerpVioOrientations"]
    elif alignType == OrientationAlign.AVERAGE_ORIENTATION:
        qVio = overlap["avgRotation"] * overlap["slerpVioOrientations"]
    else:
        qVio = overlap["slerpVioOrientations"]

    totalAngle = []
    gravityAngle = []
    headingAngle = []
    GRAVITY_DIRECTION = np.array([0, 0, -1])
    for i in range(len(qVio)):
        q_ours = qVio[i].as_matrix()
        q_gt = qGt[i].as_matrix()

        totalAngle.append(np.linalg.norm((qVio[i].inv() * qGt[i]).as_rotvec(degrees=True)))

        # Project global gravity direction to local coordinates and compare.
        gravityAngle.append(np.arccos(np.dot(q_ours.transpose() @ GRAVITY_DIRECTION, q_gt.transpose() @ GRAVITY_DIRECTION)))

        headingAngle.append(abs(2 * math.asin((qVio[i] * qGt[i].inv()).as_quat()[2]))) # quat[2] == .z

    return {
        "time": overlap["overlappingGtTimes"],
        "dist": distGt,
        "total": totalAngle,
        "gravity": 180. / np.pi * np.array(gravityAngle),
        "heading": 180. / np.pi * np.array(headingAngle),
    }

def rmseAngle(a):
    return np.sqrt(np.mean(np.array(a)**2))

def computeOrientationErrorMetric(vio, gt, full=False, alignType=None):
    if gt and len(gt.get("orientation", [])) > 0:
        orientationErrors = computeOrientationErrors(vio, gt, alignType)
        result = {
            "RMSE total": rmseAngle(orientationErrors["total"]),
        }
        if full:
            result["RMSE gravity"] = rmseAngle(orientationErrors["gravity"])
            result["RMSE heading"] = rmseAngle(orientationErrors["heading"])
        return result

    return None

def computePredictionError(vio, predictSeconds):
    from scipy.spatial.transform import Rotation
    newTimes = vio["position"][:, 0] + predictSeconds

    newPos = vio["position"][:, 1:]  + vio["velocity"][:, 1:] * predictSeconds + 0.5 * vio["acceleration"][:, 1:] * (predictSeconds ** 2)
    newPos = np.hstack((newTimes[:, np.newaxis], newPos))

    newOri = []
    for oriQ, angularV in zip(vio["orientation"][:, 1:], vio["angularVelocity"][:, 1:]):
        deltaRotation = Rotation.from_rotvec(angularV * predictSeconds)
        newOri.append((deltaRotation * Rotation.from_quat(oriQ)).as_quat())
    newOri = np.hstack((newTimes[:, np.newaxis], np.array(newOri)))

    newVio = {
        "position": newPos,
        "orientation": newOri
    }
    orientationErrors = computeOrientationErrors(newVio, vio, alignType=OrientationAlign.NONE)
    return newVio, orientationErrors

def computePredictionErrorMetrics(vio, predictSeconds):
    (newVio, orientationErrors) = computePredictionError(vio, predictSeconds)
    overlappingNewPos, overlappingVioPos = getOverlap(newVio["position"], vio["position"])
    return {
        "RMSE position (mm)": rmse(overlappingNewPos[:, 1:], overlappingVioPos[:, 1:]) * 1000,
        "RMSE angle (°)": rmseAngle(orientationErrors["total"]),
    }

# Compute cumulative sum of distance traveled. Output length matches input length.
def traveledDistance(positions):
    pDiff = np.diff(positions, axis=0)
    pDiff = np.vstack((np.array([0, 0, 0]), pDiff))
    dDiff = np.linalg.norm(pDiff, axis=1)
    return np.cumsum(dDiff)

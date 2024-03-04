import numpy as np

# Augment pose trail metrics with orientation estimates from integrating gyroscope measurements.
POSE_TRAIL_GYROSCOPE_INTEGRATION = True

# For the pose trail pose data, use the whole VIO trajectory (sanity check that the pose trails are more accurate).
USE_WHOLE_TRACK = False

# Generate slightly overlapping pose trail segments of length `pieceLenSecs`, over the timespan of `gt`.
# The segments are cut from the current end of the pose trail. The segment is aligned by matching
# the pose of the older end to the `gt` pose at the same timestamp (interpolated).
def generatePoseTrailMetricSegments(vio, pieceLenSecs, gt):
    poseTrails = vio["poseTrails"]
    from scipy.spatial.transform import Rotation, Slerp

    qGt = Rotation.from_quat(gt["orientation"][:, 1:])
    slerp = Slerp(gt["orientation"][:, 0], qGt)
    def interpolateGtPose(tVio):
        rotationGt = slerp([tVio])
        pGt = np.hstack([np.interp(tVio, gt["position"][:, 0], gt["position"][:, i]) for i in range(1, 4)])
        gtToWorld = np.identity(4)
        gtToWorld[:3, 3] = pGt
        gtToWorld[:3, :3] = rotationGt.as_matrix()
        return gtToWorld

    t0 = gt["position"][0, 0]
    t00 = None
    for poseTrailInd, poseTrail in enumerate(poseTrails):
        assert(poseTrail.size > 0)

        # Skip pose trails in the beginning.
        if poseTrail[-1, 0] < t0: continue

        # If next pose trail is good one, skip this.
        if poseTrailInd + 1 < len(poseTrails):
            nt0 = poseTrails[poseTrailInd + 1][0, 0]
            nt1 = poseTrails[poseTrailInd + 1][-1, 0]
            if nt0 < t0 and nt1 - pieceLenSecs < t0: continue

        poseCount = poseTrail.shape[0]
        # poseInd0 is somewhere in the middle of the trail and poseInd1 is the current pose.
        poseInd0 = None
        poseInd1 = poseCount - 1
        for i in reversed(range(poseCount)):
            if poseTrail[i, 0] < gt["position"][0, 0]: break
            if poseTrail[i, 0] <= t0:
                poseInd0 = i
                break
        if poseInd0 is None:
            # The whole pose trail was shorter than `pieceLenSecs`, skip a part in the ground
            # truth by setting `t0` so that it will work with the next pose trail.
            # Could compute some kind of accompanying "coverage" metric.
            if poseTrailInd + 1 < len(poseTrails):
                t0 = poseTrails[poseTrailInd + 1][0, 0]
            continue

        tVio0 = poseTrail[poseInd0, 0]
        tVio1 = poseTrail[poseInd1, 0]
        assert(poseTrail[poseInd0, 0] <= t0)
        t0 = poseTrail[poseInd1, 0]
        # Tail of the pose trail segment is the same as in the previous one.
        if tVio0 == t00: continue
        t00 = poseTrail[poseInd0, 0]
        # Too long segment.
        if tVio1 - tVio0 > 3.0 * pieceLenSecs:
            if poseTrailInd + 1 < len(poseTrails):
                t0 = poseTrails[poseTrailInd + 1][0, 0]
            continue
        if tVio1 > gt["position"][-1, 0]: break

        vioToWorld0 = np.identity(4)
        if USE_WHOLE_TRACK:
            assert(vio["position"].shape[0] == vio["orientation"].shape[0] and vio["position"].shape[0] > 0)
            wholeInd0 = np.searchsorted(vio["position"][:, 0], tVio0)
            wholeInd1 = np.searchsorted(vio["position"][:, 0], tVio1)
            tVio0 = vio["position"][wholeInd0, 0]
            tVio1 = vio["position"][wholeInd1, 0]
            vioToWorld0[:3, 3] = vio["position"][wholeInd0, 1:4]
            vioToWorld0[:3, :3] = Rotation.from_quat(vio["orientation"][wholeInd0, 1:5]).as_matrix()
        else:
            vioToWorld0[:3, 3] = poseTrail[poseInd0, 1:4]
            vioToWorld0[:3, :3] = Rotation.from_quat(poseTrail[poseInd0, 4:8]).as_matrix()

        gtToWorld0 = interpolateGtPose(tVio0)
        gtToWorld1 = interpolateGtPose(tVio1)

        # Compute and apply world transformation at poseInd0 that takes VIO poses to ground truth world.
        vioWorldToGtWorld = gtToWorld0 @ np.linalg.inv(vioToWorld0)
        vioToGtWorlds = []
        vioTimes = []
        if USE_WHOLE_TRACK:
            for i in range(wholeInd0, wholeInd1 + 1):
                vioTimes.append(vio["position"][i][0])
                vioToWorld = np.identity(4)
                vioToWorld[:3, 3] = vio["position"][i][1:4]
                vioToWorld[:3, :3] = Rotation.from_quat(vio["orientation"][i, 1:5]).as_matrix()
                vioToGtWorlds.append(vioWorldToGtWorld @ vioToWorld)
        else:
            for i in range(poseInd0, poseCount):
                vioTimes.append(poseTrail[i, 0])
                vioToWorld = np.identity(4)
                vioToWorld[:3, 3] = poseTrail[i, 1:4]
                vioToWorld[:3, :3] = Rotation.from_quat(poseTrail[i, 4:8]).as_matrix()
                vioToGtWorlds.append(vioWorldToGtWorld @ vioToWorld)

        # The first pose matches up to floating point accuracy:
        #   assert(vioToGtWorlds[0] == gtToWorld0)
        # VIO accuracy is measured by comparing the poses at tVio1:
        #   metric(vioToGtWorlds[-1], gtToWorld1)

        # print("{} - {}, len {}".format(tVio0, tVio1, tVio1 - tVio0))
        out = {
            "vioTimes": vioTimes,
            "vioToGtWorlds": vioToGtWorlds,
            "lastGtToWorld": gtToWorld1,
            "pieceLenSecs": tVio1 - tVio0,
            "trackingQuality": None,
        }
        if len(vio["trackingQuality"]) > 0:
            qualityInd = np.searchsorted(vio["trackingQuality"][:, 0], tVio1)
            out["trackingQuality"] = vio["trackingQuality"][qualityInd, 1]
        yield out

def poseOrientationDiffDegrees(A, B):
    from scipy.spatial.transform import Rotation
    Q = A[:3, :3].transpose() @ B[:3, :3]
    return np.linalg.norm(Rotation.from_matrix(Q).as_rotvec(degrees=True))

def computePoseTrailMetric(vio, gt, pieceLenSecs, info):
    """RMSE of VIO position drift when comparing pose trail segments to ground truth"""
    if len(vio["poseTrails"]) == 0: return None, None
    if gt["position"].size == 0 or gt["orientation"].size == 0: return None, None

    gyroscope = None
    if POSE_TRAIL_GYROSCOPE_INTEGRATION:
        from .gyroscope_to_orientation import GyroscopeToOrientation
        gyroscope = GyroscopeToOrientation(info["dir"])

    warned = False
    err = []
    segments = []
    for segment in generatePoseTrailMetricSegments(vio, pieceLenSecs, gt):
        d = np.linalg.norm(segment["vioToGtWorlds"][-1][:3, 3] - segment["lastGtToWorld"][:3, 3])
        err.append(d)
        # First vioToGtWorld is the same as first gt-to-world because of the alignment.
        gtDistance = np.linalg.norm(segment["vioToGtWorlds"][0][:3, 3] - segment["lastGtToWorld"][:3, 3])
        t = segment["pieceLenSecs"]
        speed = gtDistance / t if t > 0 else None
        segments.append({
            "positionErrorMeters": d,
            "orientationErrorDegrees": poseOrientationDiffDegrees(segment["vioToGtWorlds"][-1], segment["lastGtToWorld"]),
            "pieceLengthSeconds": t,
            "speed": speed,
            "time": segment["vioTimes"][-1],
            "trackingQuality": segment["trackingQuality"],
        })
        if gyroscope:
            if len(vio["biasGyroscopeAdditive"]) == 0:
                if not warned: print("Missing IMU bias estimates from VIO. Add `outputJsonExtras: True` to `vio_config.yaml`.")
                warned = True
                continue
            biasInd = np.searchsorted(vio["biasGyroscopeAdditive"][:, 0], segment["vioTimes"][0])
            bias = vio["biasGyroscopeAdditive"][biasInd, 1:]
            imuVioToGtWorld1 = gyroscope.integrate(
                segment["vioTimes"][0], segment["vioTimes"][-1], segment["vioToGtWorlds"][0], bias)
            segments[-1]["gyroscopeOrientationErrorDegrees"] = poseOrientationDiffDegrees(imuVioToGtWorld1, segment["lastGtToWorld"])

    if len(err) == 0: return None, None
    rmse = np.sqrt(np.mean(np.array(err) ** 2))
    return rmse, segments

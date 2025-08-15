import json
import pathlib

import numpy as np

from .load import readDatasets, readVioOutput
from .metric import *
from .metric_pose_trail import computePoseTrailMetric
from .align import align, getOverlap

# Track types to be used as ground-truth. In lowercase.
# Instead of adding entries to this list, consider converting your dataset to have "groundTruth" rows.
GROUND_TRUTH_TYPES = ["groundtruth", "rtkgps", "externalpose", "gps"]

# Compute a dict with all given metrics. If a metric cannot be computed, output `None` for it.
def computeMetricSets(vioAll, gt, info, metricSets):
    pGt = gt["position"]
    fixOrigin = "fixOrigin" in info and info["fixOrigin"]
    poseTrailLengths = info["poseTrailLengths"] if "poseTrailLengths" in info else []
    sampleIntervalForVelocity = info["sampleIntervalForVelocity"] if "sampleIntervalForVelocity" in info else None

    metrics = {}
    for metricSetStr in metricSets:
        metricSet = Metric(metricSetStr)
        vioTrackKind = metricToTrackKind(metricSet)
        if not vioTrackKind in vioAll: continue
        vio = vioAll[vioTrackKind]

        if not "position" in vio: continue
        pVio = vio["position"]

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
        elif metricSet == Metric.POSE_TRAIL_3D:
            metrics[metricSetStr] = {}
            for l in poseTrailLengths:
                a, b = computePoseTrailMetric(vio, gt, l, info)
                metrics[metricSetStr][f"{l}s"] = a
                metrics[metricSetStr][f"{l}s-segments"] = b
        elif metricSet in [Metric.NO_ALIGN, Metric.FULL, Metric.FULL_3D, Metric.FULL_3D_SCALED]:
            alignedVio, _ = align(pVio, pGt, -1,
                fix_origin=fixOrigin, **metricSetToAlignmentParams(metricSet))
            overlapVio, overlapGt, overlapT = getOverlap(alignedVio, pGt, includeTime=True)
            if overlapGt.size == 0 or overlapVio.size == 0: continue
            overlapVioWithTime = np.hstack((overlapT.reshape(-1, 1), overlapVio))
            metrics[metricSetStr] = None
            metrics[metricSetStr] = {
                "RMSE": rmse(overlapGt, overlapVio),
                "MAE": meanAbsoluteError(overlapGt, overlapVio),
                "drift": np.linalg.norm(overlapGt[-1] - overlapVio[-1]),
                "lengthSeconds": overlapT[-1] - overlapT[0],
                "lengthMeters": computeLength(overlapVioWithTime, info, 3),
            }
            computePercentiles(overlapGt, overlapVio, metrics[metricSetStr])
        elif metricSet == Metric.COVERAGE:
            metrics[metricSetStr] = computeCoverage(pVio, pGt, info)
        elif metricSet == Metric.VELOCITY:
            metrics[metricSetStr] = computeVelocityMetric(vio, gt, sampleIntervalForVelocity)
        elif metricSet == Metric.ANGULAR_VELOCITY:
            metrics[metricSetStr] = computeAngularVelocityMetric(vio, gt, sampleIntervalForVelocity)
        elif metricSet == Metric.POSTPROCESSED:
            # Note that compared to the other metrics, the order of arguments is swapped
            # so that the (sparse) time grid of postprocessed VIO is used.
            alignedGt, _ = align(pGt, pVio, -1, fix_origin=fixOrigin, **metricSetToAlignmentParams(metricSet))
            alignedGt, unalignedVio = getOverlap(alignedGt, pVio)
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
        elif metricSet == Metric.PREDICTION:
            metrics[metricSetStr] = computePredictionErrorMetrics(vio, PREDICTION_SECONDS)
        elif metricSet == Metric.TRACKING_QUALITY:
            metrics[metricSetStr] = None # Could implement something.
        elif metricSet == Metric.GLOBAL_COVARIANCE:
            metrics[metricSetStr] = None # Could implement something.
        elif metricSet == Metric.GLOBAL:
            overlapVio, overlapGt, overlapT = getOverlap(pVio, pGt, includeTime=True)
            metrics[metricSetStr] = None
            if overlapVio.size <= 0 or overlapGt.size <= 0: continue
            overlapGtWithTime = np.hstack((overlapT.reshape(-1, 1), overlapGt))
            metrics[metricSetStr] = {
                "RMSE": rmse(overlapGt, overlapVio),
                "MAE": meanAbsoluteError(overlapGt, overlapVio),
                "drift": np.linalg.norm(overlapGt[-1] - overlapVio[-1]),
                "lengthSeconds": overlapT[-1] - overlapT[0],
                "lengthMeters": computeLength(overlapGtWithTime, info, 3),
            }
            computePercentiles(overlapGt, overlapVio, metrics[metricSetStr])
        elif metricSet == Metric.GLOBAL_VELOCITY:
            metrics[metricSetStr] = computeGlobalVelocityMetric(vio, gt, sampleIntervalForVelocity)
        elif metricSet == Metric.GLOBAL_ERROR_OVER_TIME:
            pass # Plot only.
        else:
            raise Exception("Unimplemented metric {}".format(metricSetStr))

    return metrics

# Compute a single value that summarises the results, based on preference order
# for the available metrics. If the most prefered metric computation has failed, output `None`
# rather than falling back to another metric (as that would mess up averages across multiple cases).
def computeSummaryValue(metricsJson):
    if Metric.POSE_TRAIL_3D.value in metricsJson:
        return ("pose trail 3s", metricsJson[Metric.POSE_TRAIL_3D.value]["3s"])
    for metricSet in [Metric.NO_ALIGN, Metric.GLOBAL, Metric.FULL, Metric.FULL_3D, Metric.FULL_3D_SCALED]:
        if not metricSet.value in metricsJson: continue
        if not metricsJson[metricSet.value]: return None
        return (f"{metricSet.value} RMSE", metricsJson[metricSet.value]["RMSE"])
    for metricSet in [Metric.PIECEWISE, Metric.PIECEWISE_NO_Z]:
        if not metricSet.value in metricsJson: continue
        if not metricsJson[metricSet.value]: return None
        return ("piecewise", np.mean(list(metricsJson[metricSet.value].values())))
    for metricSet in [Metric.POSTPROCESSED, Metric.COVERAGE, Metric.ANGULAR_VELOCITY, Metric.CPU_TIME]:
        if not metricSet.value in metricsJson: continue
        return (metricSet.value, metricsJson[metricSet.value])
    return None

def setRelativeMetric(relative, metricSetStr, a, b):
    if a is None or b is None or b <= 0: return
    relative[metricSetStr] = a / b

# Computed the same way as in `computeSummaryValue()`.
def computeRelativeMetrics(metrics, baseline):
    def hasResults(kind, m):
        return kind in m and m[kind]

    relative = {}
    if hasResults(Metric.POSE_TRAIL_3D.value, metrics) and hasResults(Metric.POSE_TRAIL_3D.value, baseline):
        a = metrics[Metric.POSE_TRAIL_3D]["3s"]
        b = metrics[Metric.POSE_TRAIL_3D]["3s"]
        setRelativeMetric(relative, Metric.POSE_TRAIL_3D.value, a, b)
    for metricSet in [Metric.PIECEWISE, Metric.PIECEWISE_NO_Z]:
        metricSetStr = metricSet.value
        if hasResults(metricSetStr, metrics) and hasResults(metricSetStr, baseline):
            a = np.mean(list(metrics[metricSetStr].values()))
            b = np.mean(list(baseline[metricSetStr].values()))
            setRelativeMetric(relative, metricSetStr, a, b)
    for metricSet in [Metric.NO_ALIGN, Metric.GLOBAL, Metric.FULL, Metric.FULL_3D, Metric.FULL_3D_SCALED]:
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

def computeMetrics(benchmarkFolder, caseName, baseline=None, metricSets=None):
    infoPath = "{}/info/{}.json".format(benchmarkFolder, caseName)
    with open(infoPath) as infoFile:
        info = json.loads(infoFile.read())

    datasets = readDatasets(benchmarkFolder, caseName, GROUND_TRUTH_TYPES)
    gt = datasets[0] if datasets else None
    if not gt: return None

    vio = {}
    for kind in [VioTrackKind.REALTIME, VioTrackKind.POSTPROCESSED, VioTrackKind.GLOBAL]:
        vioOutput = readVioOutput(benchmarkFolder, caseName, info, kind, getPoseTrails=True)
        if vioOutput: vio[kind] = vioOutput

    if metricSets is None: metricSets = info["metricSets"]

    metricsJson = computeMetricSets(vio, gt, info, metricSets)
    if baseline:
        relative = computeRelativeMetrics(metricsJson, baseline)
        metricsJson["relative"] = relative
    metricsDir = "{}/metrics".format(benchmarkFolder)
    pathlib.Path(metricsDir).mkdir(parents=True, exist_ok=True)
    metricsPath = "{}/{}.json".format(metricsDir, caseName)
    with open(metricsPath, "w") as metricsFile:
        metricsFile.write(json.dumps(metricsJson, indent=4, separators=(',', ': ')))
    return computeSummaryValue(metricsJson)

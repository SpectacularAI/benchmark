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
def computeMetricSets(vioAll, gt, info, sampleIntervalForVelocity=None):
    if not VioTrackKind.REALTIME in vioAll:
        return {}
    vio = vioAll[VioTrackKind.REALTIME]
    pVio = vio["position"]
    pGt = gt["position"]
    if pVio.size > 0 and pGt.size > 0 and (pVio[0, 0] > pGt[-1, 0] or pVio[-1, 0] < pGt[0, 0]):
        print("{}: VIO timestamps do not overlap with ground truth, cannot compute metrics or align."
            .format(info["caseName"]))

    metricSets = info["metricSets"]
    fixOrigin = "fixOrigin" in info and info["fixOrigin"]
    poseTrailLengths = info["poseTrailLengths"] if "poseTrailLengths" in info else []

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
        elif metricSet == Metric.POSE_TRAIL_3D:
            metrics[metricSetStr] = {}
            for l in poseTrailLengths:
                a, b = computePoseTrailMetric(vio, gt, l, info)
                metrics[metricSetStr][f"{l}s"] = a
                metrics[metricSetStr][f"{l}s-segments"] = b
        elif metricSet in [Metric.NO_ALIGN, Metric.FULL, Metric.FULL_3D, Metric.FULL_3D_SCALED]:
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
        elif metricSet == Metric.LENGTH:
            metrics[metricSetStr] = computeLength(pGt, info, 3)
        elif metricSet == Metric.LENGTH_2D:
            metrics[metricSetStr] = computeLength(pGt, info, 2)
        elif metricSet == Metric.VELOCITY:
            metrics[metricSetStr] = computeVelocityMetric(vio, gt, sampleIntervalForVelocity)
        elif metricSet == Metric.ANGULAR_VELOCITY:
            metrics[metricSetStr] = computeAngularVelocityMetric(vio, gt, sampleIntervalForVelocity)
        elif metricSet == Metric.POSTPROCESSED:
            if not VioTrackKind.POSTPROCESSED in vioAll: continue
            vioPostprocessed = vioAll[VioTrackKind.POSTPROCESSED]
            if not "position" in vioPostprocessed: continue
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
        elif metricSet == Metric.PREDICTION:
            metrics[metricSetStr] = computePredictionErrorMetrics(vio, PREDICTION_SECONDS)
        elif metricSet == Metric.TRACKING_QUALITY:
            metrics[metricSetStr] = None # TODO
        elif metricSet == Metric.GLOBAL:
            if not VioTrackKind.GLOBAL in vioAll: continue
            vioGlobal = vioAll[VioTrackKind.GLOBAL]
            if not "position" in vioGlobal: continue
            overlapVio, overlapGt = getOverlap(vioGlobal["position"], pGt)
            metrics[metricSetStr] = None
            if overlapVio.size <= 0 or overlapGt.size <= 0: continue
            metrics[metricSetStr] = {
                "RMSE": rmse(overlapGt, overlapVio),
                "MAE": meanAbsoluteError(overlapGt, overlapVio),
            }
            computePercentiles(overlapGt, overlapVio, metrics[metricSetStr])
        else:
            raise Exception("Unimplemented metric {}".format(metricSetStr))
    return metrics

# Compute a single value that summarises the results, based on preference order
# for the available metrics. If the most prefered metric computation has failed, output `None`
# rather than falling back to another metric (as that would mess up averages across multiple cases).
def computeSummaryValue(metricsJson):
    if Metric.POSE_TRAIL_3D in metricsJson:
        return metricsJson[Metric.POSE_TRAIL_3D]["3s"] # In case the pose trail is sometimes/always shorter than 10s.
    for metricSet in [Metric.PIECEWISE, Metric.PIECEWISE_NO_Z]:
        if not metricSet.value in metricsJson: continue
        if not metricsJson[metricSet.value]: return None
        return np.mean(list(metricsJson[metricSet.value].values()))
    for metricSet in [Metric.NO_ALIGN, Metric.FULL, Metric.FULL_3D, Metric.FULL_3D_SCALED]:
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
    for metricSet in [Metric.NO_ALIGN, Metric.FULL, Metric.FULL_3D, Metric.FULL_3D_SCALED]:
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
    if not gt: return None

    vio = {}
    for kind in [VioTrackKind.REALTIME, VioTrackKind.POSTPROCESSED, VioTrackKind.GLOBAL]:
        vio[kind] = readVioOutput(benchmarkFolder, caseName, info, kind, getPoseTrails=True)

    metricsJson = computeMetricSets(vio, gt, info, sampleIntervalForVelocity)
    if baseline:
        relative = computeRelativeMetrics(metricsJson, baseline)
        metricsJson["relative"] = relative
    metricsDir = "{}/metrics".format(benchmarkFolder)
    pathlib.Path(metricsDir).mkdir(parents=True, exist_ok=True)
    metricsPath = "{}/{}.json".format(metricsDir, caseName)
    with open(metricsPath, "w") as metricsFile:
        metricsFile.write(json.dumps(metricsJson, indent=4, separators=(',', ': ')))
    return computeSummaryValue(metricsJson)

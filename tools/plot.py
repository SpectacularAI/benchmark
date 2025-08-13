import json
import math
import os
import pathlib

from .load import readDatasets, readVioOutput
from .metric import *
from .metric_pose_trail import generatePoseTrailMetricSegments
from .align import align, isSparse, getOverlap

import numpy as np

SHADES_OF_BLUE = ['blue', 'darkturquoise', 'darkorchid', 'dodgerblue', 'darkslateblue']

EXTERNAL_COLORS = {
    'groundtruth': 'salmon',
    'realsense': 'black',
    'arkit': 'deeppink',
    'arcore': 'lawngreen',
    'arengine': 'violet',
    'ondevice': 'steelblue',
    'slam map': 'green',
    'postprocessed': 'green',
    'realtime': 'deeppink',
    'gps': 'darkred',
    'rtkgps': 'salmon',
    'externalpose': 'salmon',
    'vio positions': 'darkgreen',
}

def getColor(datasetName="ours", index=0):
    k = datasetName.lower()
    if k == "ours":
        return SHADES_OF_BLUE[index % len(SHADES_OF_BLUE)]
    elif k in EXTERNAL_COLORS:
        return EXTERNAL_COLORS[k]
    return "blue"

def wordWrap(s):
    LINE_MAX_LENGTH = 150
    out = ""
    l = 0
    for i, token in enumerate(s.split(" ")):
        l += len(token) + 1
        if l > LINE_MAX_LENGTH:
            out += "\n"
            l = 0
        elif i > 0:
            out += " "
        out += token
    return out

def getFigurePath(root, metricSet=Metric.PIECEWISE, caseName=None, z_axis=None):
    figurePath = root
    if caseName: figurePath += "/{}".format(caseName)
    if z_axis and Metric(metricSet) == Metric.ANGULAR_VELOCITY:
        figurePath += "-angular-speed"
    elif z_axis and Metric(metricSet) == Metric.VELOCITY:
        figurePath += "-speed"
    else:
        if z_axis: figurePath += "-z-axis"
        if Metric(metricSet) != Metric.PIECEWISE:
            figurePath += "-{}".format(metricSet.replace("_", "-"))
    figurePath += ".png"
    return figurePath

# Use covariance to rotate data to minimize height
def compactRotation(data, ax1=1, ax2=2):
    dataxy = data[:,[ax1, ax2]]
    ca = np.cov(dataxy, y = None, rowvar = 0, bias = 1)
    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)
    ar = np.dot(dataxy, np.linalg.inv(tvect))
    dimensions = np.max(ar,axis=0) - np.min(ar,axis=0)
    if dimensions[0] > dimensions[1]:
        data[:,[ax1, ax2]] = ar
    else:
        ar_90_degrees = np.dot(ar, np.linalg.inv([[0, -1], [1, 0]]))
        data[:,[ax1, ax2]] = ar_90_degrees

def metricsToString(metrics, metricSet, relative=None, short=True):
    if not metrics:
        return "N/A"
    s = ""
    if relative:
        s += "[{:.2f}], ".format(relative)

    for m in [
        Metric.COVERAGE,
        Metric.VELOCITY,
        Metric.ANGULAR_VELOCITY,
        Metric.POSTPROCESSED,
        Metric.CPU_TIME,
    ]:
        if metricSet == m.value:
            s += "{:.3g}".format(metrics)
            if not short:
                s += " --"
                if relative: s += " [rel],"
                s += " {}".format(metricSet)
            return s

    if isinstance(metrics, float):
        s += "{:.2f}".format(metrics)
    elif "RMSE" in metrics:
        s += "{:.3g}".format(metrics["RMSE"])
        for p in PERCENTILES:
            name = percentileName(p)
            if name in metrics:
                s += ", {:.3g}".format(metrics[name])
            else:
                s += ", N/A"
        if not short:
            s += " --"
            if relative: s += " [rel RMSE],"
            s += " RMSE"
            for p in PERCENTILES:
                s += ", " + percentileName(p)
    else:
        keys = []
        for k in metrics:
            if not "segments" in k: keys.append(k)
        mean = "{:.3g}".format(np.mean([metrics[x] for x in keys]))
        submetrics = "|".join(["{:.2g}".format(metrics[x]) for x in keys])
        s += "{} ({})".format(mean, submetrics)
        if not short:
            legend = " | ".join([x for x in keys])
            s += " --"
            if relative: s += " [rel mean],"
            s += " mean, ({})".format(legend)
    return s

def colorByGlobalStatus(axis, globalStatus, maxTime=None):
    seen = set()
    for i in range(len(globalStatus) - 1):
        s = globalStatus[i][1]
        if s == "VIO": facecolor = "gray"
        elif s == "GNSS": facecolor = "red"
        elif s == "VPS": facecolor = "blue"
        else:
            assert(s == None)
            continue
        t0 = globalStatus[i][0]
        t1 = globalStatus[i + 1][0]
        if maxTime is not None and t1 > maxTime: t1 = maxTime
        label = None if s in seen else f"Status: {s}"
        seen.add(s)
        axis.axvspan(t0, t1, facecolor=facecolor, alpha=0.15, label=label)

def plotGlobalVelocity(vio, tracks, axis, sampleIntervalForVelocity, speed=False, caseCount=None):
    data = [vio]

    INCLUDE_VIO_POSITION_BASED_VELOCITY = False
    if INCLUDE_VIO_POSITION_BASED_VELOCITY:
        vioPositionSampleInterval = 2.0
        vioV = computeVelocity(vio, vioPositionSampleInterval, False)
        data.append({ "name": "VIO positions", "velocity": vioV })

    if len(tracks) >= 1:
        gt = tracks[0]
        gtV = computeVelocity(gt, sampleIntervalForVelocity)
        data.append({ "name": gt["name"], "velocity": gtV })

    limit = 1200 if caseCount == 1 else 240
    t0 = None
    for d in data:
        if "velocity" not in d or d["velocity"].size == 0: continue
        if not t0: t0 = d["velocity"][0, 0]
        vs = d["velocity"].copy()

        # Plot only part to keep the plot legible.
        vs = vs[vs[:, 0] >= t0, :]
        vs = vs[vs[:, 0] < t0 + limit, :]

        if vs.size == 0: continue
        if speed:
            axis.plot(vs[:, 0], np.linalg.norm(vs[:, 1:], axis=1), label=d['name'],
                color=getColor(d['name']), linewidth=1)
        else:
            for ind in range(1, 4):
                label = d['name'] if ind == 1 else None
                axis.plot(vs[:, 0], vs[:, ind], label=label,
                    color=getColor(d['name']), linewidth=1)

    COLOR_BY_GLOBAL_STATUS = True
    if COLOR_BY_GLOBAL_STATUS:
        colorByGlobalStatus(axis, vio["globalStatus"], maxTime=(limit + t0))

def plotVelocity(vio, tracks, axis, sampleIntervalForVelocity, speed=False):
    if len(tracks) >= 1:
        preComputeAlignedVelocity(vio, tracks[0], sampleIntervalForVelocity)
        data = [tracks[0], vio]
    else:
        vio["velocity"] = computeVelocity(vio, sampleIntervalForVelocity)
        data = [vio]
    t0 = None
    for d in data:
        # Plot only short segment to keep the plot legible.
        if not t0:
            if d["position"].size == 0: continue
            t0 = d["position"][0, 0]
        vs = d["alignedVelocity"] if len(tracks) >= 1 else d["velocity"]
        if vs.size == 0: continue
        t = vs[:, 0]
        vs = vs[(t > t0) & (t < t0 + 10), :]

        if vs.size == 0: continue
        if speed:
            axis.plot(vs[:, 0], np.linalg.norm(vs[:, 1:], axis=1), label=d['name'],
                color=getColor(d['name']), linewidth=1)
        else:
            for ind in range(1, 4):
                label = d['name'] if ind == 1 else None
                axis.plot(vs[:, 0], vs[:, ind], label=label,
                    color=getColor(d['name']), linewidth=1)

def plotAngularVelocity(args, vio, tracks, axis, sampleIntervalForVelocity, speed=False):
    import matplotlib.pyplot as plt
    if len(tracks) >= 1:
        preComputeAlignedAngularVelocity(vio, tracks[0], sampleIntervalForVelocity)
        data = [tracks[0], vio]
    else:
        vio["angularVelocity"] = computeAngularVelocity(vio, sampleIntervalForVelocity)
        data = [vio]
    t0 = None
    for d in data:
        # Plot only short segment to keep the plot legible.
        if not t0:
            if d["orientation"].size == 0: continue
            t0 = d["orientation"][0, 0]
        avs = d["alignedAngularVelocity"] if len(tracks) >= 1 else d["angularVelocity"]
        if avs.size == 0: continue
        t = avs[:, 0]
        avs = avs[(t > t0) & (t < t0 + 10), :]

        if avs.size == 0: continue
        if speed:
            axis.plot(avs[:, 0], np.linalg.norm(avs[:, 1:4], axis=1), label=d['name'],
                color=getColor(d['name']), linewidth=1)
        else:
            for ind in range(1, 4):
                label = d['name'] if ind == 1 else None
                axis.plot(avs[:, 0], avs[:, ind], label=label,
                    color=getColor(d['name']), linewidth=1)

def plotOrientationErrors(vio, tracks, axis, full=False, alignType=OrientationAlign.TRAJECTORY):
    import matplotlib.pyplot as plt
    if len(tracks) == 0 or len(tracks[0].get("orientation", [])) == 0:
        return
    orientationErrors = computeOrientationErrors(vio, tracks[0], alignType)
    axis.plot(orientationErrors["time"], orientationErrors["total"], label="Total")
    if full:
        axis.plot(orientationErrors["time"], orientationErrors["gravity"], label="Gravity")
        axis.plot(orientationErrors["time"], orientationErrors["heading"], label="Heading")

def plotPredictionError(vio, axis, predictSeconds):
    (newVio, orientationErrors) = computePredictionError(vio, predictSeconds)
    overlappingNewPos, overlappingVioPos, posTime = getOverlap(newVio["position"], vio["position"], includeTime=True)
    positionError = np.linalg.norm(overlappingNewPos - overlappingVioPos, axis=1)

    ax2 = axis.twinx()
    axis.plot(posTime, positionError * 1000, color='teal')
    ax2.plot(orientationErrors["time"], orientationErrors["total"], color='orange')
    axis.set_ylabel('Position (mm)', color='teal', fontweight='bold')
    ax2.set_ylabel('Orientation (°)', color='orange', fontweight='bold')

def plotTrackingQuality(vio, axis, metrics):
    axis.set_ylim([-0.05, 1.05])
    axis.plot(vio["trackingQuality"][:, 0], vio["trackingQuality"][:, 1], label="tracking quality")

    if metrics is not None and "pose_trail_3d" in metrics and "1.0s-segments" in metrics["pose_trail_3d"]:
        t = []
        err = []
        for segment in metrics["pose_trail_3d"]["1.0s-segments"]:
            t.append(segment["time"])
            err.append(segment["positionErrorMeters"])
        err = np.array(err)
        err *= 10
        axis.plot(t, err, label="segment position error")

def plotPoseTrails(args, vio, tracks, axis, ax1, ax2, info):
    if len(tracks) == 0:
        print("Expected ground truth track")
        return
    gt = tracks[0]

    axis.axis('equal')
    axis.plot(gt["position"][:, ax1], gt["position"][:, ax2], label=gt["name"],
        color=getColor(gt["name"]), linewidth=1)

    if not "poseTrails" in vio:
        print("Expected pose trail data")
        return
    if not "orientation" in gt:
        print("Expected orientation in ground truth")
        return

    for segmentInd, segment in enumerate(generatePoseTrailMetricSegments(vio, 1.0, gt, info)):
        x = []
        y = []
        for vioToGtWorld in segment["vioToGtWorlds"]:
            # No time index, subtract one.
            x.append(vioToGtWorld[ax1 - 1, 3])
            y.append(vioToGtWorld[ax2 - 1, 3])
        label = vio["name"] if segmentInd == 0 else None
        axis.plot(x, y, label=label, color=getColor(vio["name"]), linewidth=1)

        x = []
        y = []
        for vioToGtWorld in segment["inertialVioToGtWorlds"]:
            x.append(vioToGtWorld[ax1 - 1, 3])
            y.append(vioToGtWorld[ax2 - 1, 3])
        label = "Inertial-only (using VIO biases and velocity)" if segmentInd == 0 else None
        axis.plot(x, y, label=label, color="green", linewidth=1)

def plotErrorOverTime(gt, vio, axis, z_axis, includeLegend):
    oVio, oGt, time = getOverlap(vio["position"], gt["position"], includeTime=True)
    if time.size == 0: return
    time -= time[0]

    if z_axis:
        err = np.abs(oVio[:, 2] - oGt[:, 2])
    else:
        err = np.linalg.norm(oVio - oGt, axis=1)

    axis.plot(time, err, label='Error')
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Error [m]')

    # Create secondary y-axis.
    axis2 = axis.twinx()
    axis2.plot(time, oGt[:, 2], 'k--', label='Altitude ({})'.format(gt["name"]), alpha=0.7)
    axis2.set_ylabel('Altitude [m]')

    # Combine legends from both axes.
    if includeLegend:
        lines1, labels1 = axis.get_legend_handles_labels()
        lines2, labels2 = axis2.get_legend_handles_labels()
        axis.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

def plot2dTracks(args, tracks, gtInd, axis, ax1, ax2, metricSet, postprocessed, fixOrigin):
    import matplotlib.pyplot as plt
    kwargsAlign = metricSetToAlignmentParams(Metric(metricSet))

    # Align all tracks with ground truth.
    if gtInd is not None:
        if args.compactRotation: compactRotation(tracks[gtInd]["position"], ax1, ax2)
        for ind, track in enumerate(tracks):
            if ind == gtInd: continue
            if not "position" in tracks[gtInd]: continue
            if not "position" in track: continue
            track["position"], _ = align(track["position"], tracks[gtInd]["position"], -1,
                fix_origin=fixOrigin, **kwargsAlign)

    for ind, track in enumerate(tracks):
        marker = None
        if not "position" in track: continue
        if isSparse(track['position']): marker = "o"
        if postprocessed and ind == gtInd: marker = "o"

        if track['position'].size == 0: continue
        axis.plot(track['position'][:, ax1], track['position'][:, ax2], label=track['name'],
            color=getColor(track['name']), linewidth=args.lineWidth, marker=marker, markersize=3)
        plotLoopClosures(track, axis, ax1, ax2)
        plotResets(track, axis, ax1, ax2)

    axis.axis('equal')

def plotLoopClosures(track, axis, ax1, ax2):
    if not "loopClosures" in track or not track["loopClosures"]: return
    p = track["position"]
    for ind, loopClosure in enumerate(track["loopClosures"]):
        label = "{} loop closures".format(len(track["loopClosures"])) if ind == 0 else None
        plc = np.hstack([np.interp(loopClosure[0:2], p[:, 0], p[:, i])[:, np.newaxis] for i in range(4)])
        axis.plot(plc[:, ax1], plc[:, ax2], color=loopClosure[2], linewidth=2,
            marker="o", markersize=5, label=label)

def plotResets(track, axis, ax1, ax2):
    if not "resets" in track or not track["resets"]: return
    p = track["position"]
    for ind, reset in enumerate(track["resets"]):
        label = "{} resets".format(len(track["resets"])) if ind == 0 else None
        pr = [np.interp(reset, p[:, 0], p[:, i]) for i in range(4)]
        axis.plot(pr[ax1], pr[ax2], color="cyan", marker="o", markersize=8, label=label)

# Returns inches. Multiplying with the `dpi=100` argument of `savefig()` gives the saved image resolution.
def figureSize(rows, columns, square, metricSet):
    # Wide plots.
    if rows == 1 and columns == 1 and metricSet in [
        Metric.VELOCITY.value,
        Metric.GLOBAL_VELOCITY.value,
        Metric.ANGULAR_VELOCITY.value,
        Metric.ORIENTATION.value,
        Metric.ORIENTATION_FULL.value,
        Metric.ORIENTATION_ALIGNED.value,
        Metric.TRACKING_QUALITY.value,
        Metric.GLOBAL_ERROR_OVER_TIME.value,
    ]:
        return (24, 13.5)

    if square:
        numPlots = rows * columns
        if numPlots <= 16: return (15, 15)
        if numPlots <= 25: return (20, 20)
        return (30, 30)
    if columns == 1 and rows == 1: return (12, 12)
    elif columns <= 2 and rows <= 2: return (8 * columns, 8 * rows)
    return (4 * columns, 4 * rows)

def shorten(s, maxLen=50):
    tokens = reversed(s.split("-"))
    s0 = next(tokens)
    s1 = None
    for token in tokens:
        s1 = f"{token}-{s0}"
        if len(s1) > maxLen: return f"..-{s0}"
        s0 = s1
    if not s1:
        return s
    return s1

def plotMetricSet(args, benchmarkFolder, caseNames, sharedInfo, metricSet):
    if not args.showPlot:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    caseCount = len(caseNames)
    if caseCount == 0: return

    # The plot images are typically viewed on landscape monitors, so if the subplots array
    # is shaped accordingly, individual subplots will be larger than with a square image.
    SQUARE_IMAGE = False
    if SQUARE_IMAGE:
        columns = int(math.sqrt(caseCount))
        if pow(columns, 2) < caseCount: columns += 1
        rows = columns
    else:
        columns = int(1.5 * math.sqrt(caseCount))
        rows = 1
        while rows * columns < caseCount: rows += 1

    figure, subplots = plt.subplots(rows, columns, figsize=figureSize(rows, columns, SQUARE_IMAGE, metricSet))
    subplots = np.ravel(subplots)
    for i, s in enumerate(subplots):
        if i >= caseCount: s.axis("off")

    getPoseTrails = metricSet in [Metric.POSE_TRAIL_3D.value, Metric.TRACKING_QUALITY.value]

    for i, caseName in enumerate(caseNames):
        try:
            if args.simplePlot:
                titleStr = ""
            elif caseCount > 1:
                # Try to avoid text going over other subfigure titles.
                titleStr = shorten(caseName)
            else:
                titleStr = caseName
            caseMetrics = None
            relativeMetric = None
            plotAxis = subplots[i]

            caseInfoPath = "{}/info/{}.json".format(benchmarkFolder, caseName)
            with open(caseInfoPath) as caseInfoJsonFile:
                caseInfo = json.loads(caseInfoJsonFile.read())
            fixOrigin = "fixOrigin" in caseInfo and caseInfo["fixOrigin"]

            metrics = None
            caseMetricsPath = "{}/metrics/{}.json".format(benchmarkFolder, caseName)
            if os.path.exists(caseMetricsPath):
                with open(caseMetricsPath) as caseMetricsJsonFile:
                    metrics = json.loads(caseMetricsJsonFile.read())
                    if metricSet in metrics:
                        caseMetrics = metrics[metricSet]
                    if "relative" in metrics and metricSet in metrics["relative"]:
                        relativeMetric = metrics["relative"][metricSet]

            tracks = readDatasets(benchmarkFolder, caseName, [], args.excludePlots)
            postprocessed = metricSet == Metric.POSTPROCESSED.value
            vioTrackKind = metricToTrackKind(Metric(metricSet))
            vio = readVioOutput(benchmarkFolder, caseName, sharedInfo, vioTrackKind, getPoseTrails)

            sampleIntervalForVelocity = None
            if "sampleIntervalForVelocity" in caseInfo:
                sampleIntervalForVelocity = caseInfo["sampleIntervalForVelocity"]

            includeLegend = i == 0 and not args.simplePlot

            vio["name"] = sharedInfo["methodName"]
            ax1 = 1
            ax2 = 3 if args.z_axis else 2

            if metricSet == Metric.ANGULAR_VELOCITY.value:
                # Use z_axis argument as hack to enable speed mode.
                plotAngularVelocity(args, vio, tracks, plotAxis, sampleIntervalForVelocity, speed=args.z_axis)
            elif metricSet == Metric.VELOCITY.value:
                plotVelocity(vio, tracks, plotAxis, sampleIntervalForVelocity, speed=args.z_axis)
            elif metricSet == Metric.GLOBAL_VELOCITY.value:
                plotGlobalVelocity(vio, tracks, plotAxis, sampleIntervalForVelocity, speed=args.z_axis, caseCount=caseCount)
            elif postprocessed:
                tracks.append(vio)
                # Align using the (sparse) postprocessed VIO time grid.
                gtInd = len(tracks) - 1 if len(tracks) >= 2 else None
                plot2dTracks(args, tracks, gtInd, plotAxis, ax1, ax2, metricSet, postprocessed, fixOrigin)
            elif metricSet == Metric.ORIENTATION.value:
                plotOrientationErrors(vio, tracks, plotAxis, alignType=OrientationAlign.TRAJECTORY)
            elif metricSet == Metric.ORIENTATION_FULL.value:
                plotOrientationErrors(vio, tracks, plotAxis, full=True, alignType=OrientationAlign.TRAJECTORY)
            elif metricSet == Metric.ORIENTATION_ALIGNED.value:
                plotOrientationErrors(vio, tracks, plotAxis, alignType=OrientationAlign.AVERAGE_ORIENTATION)
            elif metricSet == Metric.PREDICTION.value:
                plotPredictionError(vio, plotAxis, predictSeconds=PREDICTION_SECONDS)
            elif metricSet == Metric.TRACKING_QUALITY.value:
                plotTrackingQuality(vio, plotAxis, metrics)
            elif metricSet == Metric.POSE_TRAIL_3D.value:
                plotPoseTrails(args, vio, tracks, plotAxis, ax1, ax2, caseInfo)
            elif metricSet == Metric.GLOBAL_ERROR_OVER_TIME.value:
                if len(tracks) >= 1:
                    plotErrorOverTime(tracks[0], vio, plotAxis, args.z_axis, includeLegend)
                    includeLegend = True
            else:
                tracks.append(vio)
                gtInd = 0 if len(tracks) >= 2 else None
                plot2dTracks(args, tracks, gtInd, plotAxis, ax1, ax2, metricSet, postprocessed, fixOrigin)

            # Draw legend
            plotAxis.tick_params(axis='both', which='major', labelsize=args.tickSize, width=args.lineWidth)
            plotAxis.tick_params(axis='both', which='minor', labelsize=args.tickSize, width=args.lineWidth)
            for spine in plotAxis.spines.values():
                spine.set_linewidth(args.lineWidth)

            # Set titles
            if caseMetrics and not args.simplePlot:
                if titleStr: titleStr += "\n"
                if caseInfo["paramSet"] != "DEFAULT" and caseInfo["paramSet"]:
                    titleStr = "{}{}\n".format(titleStr, caseInfo["paramSet"])
                titleStr += metricsToString(caseMetrics, metricSet, relativeMetric, True)
            plotAxis.title.set_text(titleStr)

            _, labels = plotAxis.get_legend_handles_labels()
            if includeLegend and len(labels) > 0:
                plotAxis.legend()

            if not "position" in vio or vio["position"].size == 0:
                plotAxis.set_title("NO OUTPUT {}".format(titleStr), color="red")

        except Exception as e:
            if caseCount > 1:
                # For aggregate plots do not crash the entire plot but mark the failed case.
                plotAxis.set_title("FAILED {}\n{}".format(titleStr, e), color="red")
                continue
            else:
                raise(e)


    # Title for aggregate plot.
    if args.simplePlot:
        plt.tight_layout(rect=[0, 0, 1, 1])
    elif not args.caseName:
        suptitle = ""
        if "set" in sharedInfo and "methodName" in sharedInfo:
            suptitle += "{} - {}\n".format(sharedInfo["set"], sharedInfo["methodName"])
        if sharedInfo["parameters"]:
            suptitle += wordWrap(sharedInfo["parameters"])
        suptitle += "\n"
        if sharedInfo["metrics"] and metricSet in sharedInfo["metrics"]:
            relativeMetric = None
            if "relative" in sharedInfo["metrics"] and sharedInfo["metrics"]["relative"] and metricSet in sharedInfo["metrics"]["relative"]:
                relativeMetric = sharedInfo["metrics"]["relative"][metricSet]
            suptitle += metricsToString(sharedInfo["metrics"][metricSet], metricSet, relativeMetric, short=False)
        figure.suptitle(suptitle, fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if args.showPlot:
        plt.show()
    else:
        figurePath = getFigurePath("{}/figures".format(benchmarkFolder), metricSet, args.caseName, args.z_axis)
        figure.savefig(figurePath)
    plt.close(figure)

def plotBenchmark(args, benchmarkFolder):
    if not pathlib.Path(benchmarkFolder).exists():
        raise Exception(f"No such folder: `{benchmarkFolder}`")

    if not args.caseName:
        caseNames = []
        for x in os.walk("{}/info".format(benchmarkFolder)):
            for caseInfoJsonPath in x[2]:
                caseNames.append(caseInfoJsonPath[:-5])
        assert(caseNames)
        caseNames.sort()
    else:
        caseNames = [args.caseName]

    with open(benchmarkFolder + "/info.json") as sharedInfoJsonFile:
        sharedInfo = json.loads(sharedInfoJsonFile.read())
        metricSets = sharedInfo["metricSets"]

    for metricSet in metricSets:
        # TODO Plotting is somewhat slow. Could skip eg for CPU_TIME if there
        # was some other convenient way to present the results besides the aggregate figure.
        if args.z_axis and metricSet in [Metric.ORIENTATION.value, Metric.ORIENTATION_FULL.value, Metric.ORIENTATION_ALIGNED.value, Metric.PREDICTION.value, Metric.TRACKING_QUALITY.value]:
            continue
        plotMetricSet(args, benchmarkFolder, caseNames, sharedInfo, metricSet)

def makeAllPlots(results, excludePlots="", debug=False, simplePlot=True):
    import argparse
    parser = argparse.ArgumentParser()
    plotArgs = parser.parse_args([])
    varsPlotArgs = vars(plotArgs)
    varsPlotArgs["z_axis"] = False
    varsPlotArgs["compactRotation"] = False
    varsPlotArgs["showPlot"] = False
    varsPlotArgs["excludePlots"] = excludePlots.split(",")
    varsPlotArgs["simplePlot"] = simplePlot

    if simplePlot:
        varsPlotArgs["lineWidth"] = 2
        varsPlotArgs["tickSize"] = 20
    else:
        varsPlotArgs["lineWidth"] = 1
        varsPlotArgs["tickSize"] = 6

    if not pathlib.Path(results).exists():
        raise Exception(f"No such folder: `{results}`")

    for x in os.walk(results + "/info"):
        for caseInfoJsonPath in x[2]:
            benchmarkInfo = json.loads(open(os.path.join(results, "info", caseInfoJsonPath)).read())
            caseName = benchmarkInfo.get("caseName")
            varsPlotArgs["caseName"] = caseName
            try:
                plotBenchmark(plotArgs, results)
            except Exception as e:
                if debug:
                    import traceback
                    print(traceback.format_exc())
                print("plotBenchmark() failed for {}: {}".format(caseName, e))

    try:
        varsPlotArgs["caseName"] = None
        plotBenchmark(plotArgs, results)
        varsPlotArgs["z_axis"] = True
        plotBenchmark(plotArgs, results)
    except Exception as e:
        if debug:
            import traceback
            print(traceback.format_exc())
        print("plotBenchmark() failed for aggregate (z_axis {}): {}".format(varsPlotArgs["z_axis"], e))

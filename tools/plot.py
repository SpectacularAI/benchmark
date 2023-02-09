#!/usr/bin/env python3

import json
import math
import os

from .compute_metrics import readDatasets, readVioOutput, align, Metric, metricSetToAlignmentParams, isSparse
from .compute_metrics import computeVelocity, computeAlignedVelocity
from .compute_metrics import computeAngularVelocity, computeAlignedAngularVelocity

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
    'rtkgps': 'salmon'
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
                s += " -- [rel] {}".format(metricSet)
            return s

    if "RMSE" in metrics:
        s += "{:.3g}, {:.3g}".format(metrics["RMSE"], metrics["MAE"])
        if not short:
            s += " -- [rel RMSE] RMSE, MAE"
    else:
        mean = "{:.3g}".format(np.mean([metrics[x] for x in metrics]))
        submetrics = "|".join(["{:.2g}".format(metrics[x]) for x in metrics])
        s += "{} ({})".format(mean, submetrics)
        if not short:
            legend = " | ".join([x for x in metrics])
            s += " -- [rel mean] mean, ({})".format(legend)
    return s

def plotVelocity(args, vio, tracks, axis):
    import matplotlib.pyplot as plt
    if len(tracks) >= 1:
        computeAlignedVelocity(vio, tracks[0])
        data = [tracks[0], vio]
    else:
        vio["velocity"] = computeVelocity(vio)
        data = [vio]
    t0 = None
    for d in data:
        # Plot only short segment to keep the plot legible.
        if not t0:
            if d["position"].size == 0: continue
            t0 = d["position"][0, 0]
        vs = d["velocity"]
        if vs.size == 0: continue
        t = vs[:, 0]
        vs = vs[(t > t0) & (t < t0 + 10), :]

        if vs.size == 0: continue
        for ind in range(1, 4):
            label = d['name'] if ind == 1 else None
            axis.plot(vs[:, 0], vs[:, ind], label=label,
                color=getColor(d['name']), linewidth=1)

PLOT_ORIENTATIONS = False
PLOT_XY = False
PLOT_NORM = False
def plotAngularVelocity(args, vio, tracks, axis):
    import matplotlib.pyplot as plt
    if len(tracks) >= 1:
        computeAlignedAngularVelocity(vio, tracks[0])
        data = [tracks[0], vio]
    else:
        vio["angularVelocity"] = computeAngularVelocity(vio)
        data = [vio]
    t0 = None
    for d in data:
        # Plot only short segment to keep the plot legible.
        if not t0:
            if d["orientation"].size == 0: continue
            t0 = d["orientation"][0, 0]
        avs = d["angularVelocity"]
        if avs.size == 0: continue
        t = avs[:, 0]
        avs = avs[(t > t0) & (t < t0 + 10), :]

        if avs.size == 0: continue
        if PLOT_ORIENTATIONS:
            for ind in range(1, 5):
                label = d['name'] if ind == 1 else None
                axis.plot(d["orientation"][:, 0], d["orientation"][:, ind], label=label,
                    color=getColor(d['name']), linewidth=1)
        elif PLOT_XY:
            axis.plot(avs[:, 1], avs[:, 2], label=d["name"],
                color=getColor(d['name']), linewidth=1)
        elif PLOT_NORM:
            speed = np.linalg.norm(avs[:, 1:4], axis=1)
            axis.plot(avs[:, 0], speed, label=d['name'],
                color=getColor(d['name']), linewidth=1)
        else:
            for ind in range(1, 4):
                label = d['name'] if ind == 1 else None
                axis.plot(avs[:, 0], avs[:, ind], label=label,
                    color=getColor(d['name']), linewidth=1)

def plot2dTracks(args, tracks, gtInd, axis, ax1, ax2, metricSet, postprocessed):
    import matplotlib.pyplot as plt
    kwargsAlign = metricSetToAlignmentParams(Metric(metricSet))

    # Align all tracks with ground truth.
    if gtInd is not None:
        if args.compactRotation: compactRotation(tracks[gtInd]["position"], ax1, ax2)
        for ind, track in enumerate(tracks):
            if ind == gtInd: continue
            if not "position" in tracks[gtInd]: continue
            track["position"], _ = align(track["position"], tracks[gtInd]["position"], -1, **kwargsAlign)

    for ind, track in enumerate(tracks):
        marker = None
        if not "position" in track: continue
        if isSparse(track['position']): marker = "o"
        if postprocessed and ind == gtInd: marker = "o"

        if track['position'].size == 0: continue
        axis.plot(track['position'][:, ax1], track['position'][:, ax2], label=track['name'],
            color=getColor(track['name']), linewidth=1, marker=marker, markersize=3)
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

def figureSize(num_plots):
    if num_plots < 10:
        return (15,15)
    if num_plots < 20:
        return (20,20)
    return (30,30)

def plotMetricSet(args, benchmarkFolder, caseNames, sharedInfo, metricSet):
    if not args.showPlot:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    caseCount = len(caseNames)
    columns = int(math.sqrt(caseCount))
    if pow(columns, 2) < caseCount: columns += 1
    rows = columns

    figure, subplots = plt.subplots(rows, columns, figsize=figureSize(caseCount))
    subplots = np.ravel(subplots)

    for i, caseName in enumerate(caseNames):
        try:
            titleStr = caseName
            caseMetrics = None
            relativeMetric = None
            plotAxis = subplots[i]

            caseInfoPath = "{}/info/{}.json".format(benchmarkFolder, caseName)
            with open(caseInfoPath) as caseInfoJsonFile:
                caseInfo = json.loads(caseInfoJsonFile.read())

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
            vio = readVioOutput(benchmarkFolder, caseName, postprocessed)

            vio["name"] = sharedInfo["methodName"]
            ax1 = 1
            ax2 = 3 if args.z_axis else 2
            if metricSet == Metric.ANGULAR_VELOCITY.value:
                plotAngularVelocity(args, vio, tracks, plotAxis)
            elif metricSet == Metric.VELOCITY.value:
                plotVelocity(args, vio, tracks, plotAxis)
            elif postprocessed:
                tracks.append(vio)
                # Align using the (sparse) postprocessed VIO time grid.
                gtInd = len(tracks) - 1 if len(tracks) >= 2 else None
                plot2dTracks(args, tracks, gtInd, plotAxis, ax1, ax2, metricSet, postprocessed)
            else:
                tracks.append(vio)
                gtInd = 0 if len(tracks) >= 2 else None
                plot2dTracks(args, tracks, gtInd, plotAxis, ax1, ax2, metricSet, postprocessed)

            # Draw legend
            for item in plotAxis.get_xticklabels() + plotAxis.get_yticklabels():
                item.set_size(6)

            # Set titles
            if caseMetrics:
                if titleStr: titleStr += "\n"
                if caseInfo["paramSet"] != "DEFAULT" and caseInfo["paramSet"]:
                    titleStr = "{}{}\n".format(titleStr, caseInfo["paramSet"])
                titleStr += metricsToString(caseMetrics, metricSet, relativeMetric, True)
            plotAxis.title.set_text(titleStr)
            plotAxis.legend()

        except Exception as e:
            if caseCount > 1:
                # For aggregate plots do not crash the entire plot but mark the failed case.
                plotAxis.set_title("FAILED {}\n{}".format(titleStr, e), color="red")
                continue
            else:
                raise(e)


    # Title for aggregate plot.
    if not args.caseName:
        suptitle = ""
        if sharedInfo["parameters"]:
            suptitle += wordWrap(sharedInfo["parameters"])
        suptitle += "\n"
        if sharedInfo["metrics"]:
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
        m = sharedInfo["metrics"]
        # Shared info `metrics` field is "null" if none of the cases have ground truth,
        # in that case produce one plot, does not matter which since it won't have any numbers.
        metricSets = list(m.keys()) if m else ["piecewise"]
        metricSets = [m for m in metricSets if m != "relative"]

    for ind, metricSet in enumerate(metricSets):
        # TODO Plotting is somewhat slow. Could skip eg for CPU_TIME if there
        # was some other convenient way to present the results besides the aggregate figure.
        if args.z_axis:
            if metricSet == Metric.VELOCITY.value: continue
            if metricSet == Metric.ANGULAR_VELOCITY.value: continue
        plotMetricSet(args, benchmarkFolder, caseNames, sharedInfo, metricSet)

def makeAllPlots(results, excludePlots=""):
    import argparse
    parser = argparse.ArgumentParser()
    plotArgs = parser.parse_args([])
    varsPlotArgs = vars(plotArgs)
    varsPlotArgs["z_axis"] = False
    varsPlotArgs["compactRotation"] = False
    varsPlotArgs["showPlot"] = False
    varsPlotArgs["excludePlots"] = excludePlots.split(",")
    for x in os.walk(results + "/info"):
        for caseInfoJsonPath in x[2]:
            benchmarkInfo = json.loads(open(os.path.join(results, "info", caseInfoJsonPath)).read())
            caseName = benchmarkInfo.get("caseName")
            varsPlotArgs["caseName"] = caseName
            try:
                plotBenchmark(plotArgs, results)
            except Exception as e:
                print("plotBenchmark() failed for {}: {}".format(caseName, e))

    try:
        varsPlotArgs["caseName"] = None
        plotBenchmark(plotArgs, results)
        varsPlotArgs["z_axis"] = True
        plotBenchmark(plotArgs, results)
    except Exception as e:
        print("plotBenchmark() failed for aggregate (z_axis {}): {}".format(varsPlotArgs["z_axis"], e))

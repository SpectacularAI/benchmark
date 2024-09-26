#!/usr/bin/env python3
""" VIO benchmarking library. """

import argparse
from collections import OrderedDict
from datetime import datetime
import time
import subprocess
import os
import pathlib
import json
from collections import deque
import concurrent.futures
from functools import partial
import multiprocessing

from .gnss import GnssConverter
from .compute_metrics import computeMetrics, Metric
from .plot import makeAllPlots, getFigurePath

import numpy as np

DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"

DEFAULT_OUTPUT_DIR = "output"
DEFAULT_METRICS = ",".join([
    Metric.PIECEWISE.value,
    Metric.FULL_3D.value,
])

# Can be enabled to get postprocessed and real-time tracks plot in the same figures.
# Usually it gets too cluttered to make sense of.
COMPARE_POSTPROCESSED = False

# Will be printed to VIO logs.
CPU_TIME_MESSAGE = "CPU time (user sys percent)"

DEFAULT_SAMPLE_INTERVAL_FOR_VELOCITY = 0.02

def getArgParser():
    allMetrics = [m.value for m in list(Metric)]

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-output", help="Output parent directory", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-rootDataDir", help="Paths in benchmark sets are given relative to this directory", default="data/benchmark")
    parser.add_argument("-set", help="Path to JSON description of benchmark set to run")
    parser.add_argument("-dataDir", help="Run all datasets in this directory. Ignores `-rootDataDir`")
    parser.add_argument("-setDir", help="An additional directory to search for benchmark set", default="../sets")
    parser.add_argument("-recordingDir", help="Path to a single benchmark recording to run. Ignores `-rootDataDir`")
    parser.add_argument("-params", help="Parameters as a string, eg '-params \"-useStereo=false -v=1\"'")
    parser.add_argument("-threads", help="How many CPU threads to use for running benchmarks", type=int, default=6)
    parser.add_argument("-runId", help="Output folder name. If unset will use timestamp")
    parser.add_argument("-skipBenchmark", help="Skips running benchmark and only aggregates existing ones. For development use.", action="store_true")
    parser.add_argument("-offsetTracks", help="When enabled, tracks are stacked instead of overlaid", action="store_true")
    parser.add_argument('-metricSet', type=str, default=DEFAULT_METRICS,
            help="One or more metric kinds, joined by comma, selected from: {}".format(", ".join(allMetrics)))
    parser.add_argument("-fixOrigin", action="store_true", help="Force track starting positions to match for metrics and plots")
    parser.add_argument("-methodName", default="VIO", help="Name of the VIO method being benchmarked")
    parser.add_argument("-gitDir", help="Subfolder that should be used for saving git stats")
    parser.add_argument("-gitBranchName", help="Written to info.json")
    parser.add_argument("-baseline", help="Path to metrics.json to use in computing relative metrics")
    parser.add_argument("-excludePlots", type=str, help="Tracks to skip plotting, split by comma", default="ondevice")
    parser.add_argument("-debug", help="Print more informative error messages", action="store_true")
    parser.add_argument("-sampleIntervalForVelocity", help="Downsamples ground truth position/orientation frequency before calculating velocity and angular velocity, provide minimum number of seconds between samples i.e. 0.1 = max 10Hz GT", type=float, default=DEFAULT_SAMPLE_INTERVAL_FOR_VELOCITY)
    parser.add_argument("-poseTrailLengths", type=str, default="1,2,4", help="Pose trail metrics target segment lengths, in seconds, separated by comma.")
    parser.add_argument("-savePoseTrail", action="store_true") # Set automatically.
    parser.add_argument("-iterations", help="How many times benchmark is run", type=int, default=1)
    return parser

def readJsonl(filePath):
    with open(filePath) as f:
        for l in f: yield(json.loads(l))

class Benchmark:
    dir = None
    params = None
    name = None
    paramSet = None
    iteration = None

    def __init__(self, dir, name=None, params=None, paramSet=None, iteration=None):
        self.dir = dir
        self.name = name
        self.params = params
        self.paramSet = paramSet
        self.iteration = iteration

    def clone(self, iteration):
        return Benchmark(
            dir=self.dir,
            name=f"{self.name}_{iteration}",
            params=self.params,
            paramSet=self.paramSet,
            iteration=iteration
        )


def computeVideoTimeSpan(dataJsonlPath):
    if not os.path.exists(dataJsonlPath): return None
    span = None
    with open(dataJsonlPath) as dataJsonlFile:
        for line in dataJsonlFile:
            o = json.loads(line)
            if not "frames" in o or not "time" in o: continue
            if not span: span = [o["time"], o["time"]]
            span[1] = o["time"]
    if span[1] < span[0]: return None
    return span

def writeSharedInfoFile(args, dirs, startTime, endTime, aggregateMetrics):
    def runAndCapture(cmd):
        return subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8').strip()

    info = {
        "outputDirectory": dirs.results,
        "startTime": startTime,
        "endTime": endTime,
        "metrics": aggregateMetrics,
        "parameters": args.params,
    }

    info["methodName"] = args.methodName
    info["parameters"] = args.params
    mainBinary = dirs.results + "/main"
    if os.path.isfile(mainBinary) and runAndCapture("command -v shasum"):
        info["fingerprint"] = runAndCapture("shasum -a 256 " + mainBinary)
    info["system"] = runAndCapture("uname -a")
    if args.set: info["set"] = args.set
    if args.dataDir: info["dataDir"] = args.dataDir

    if args.gitDir:
        originalDir = os.getcwd()
        os.chdir(args.gitDir)
        inGitRepository = runAndCapture("git rev-parse --is-inside-work-tree")
        if inGitRepository == "true":
            if args.gitBranchName:
                # The `rev-parse` often returns "HEAD" in Github's runner.
                branchName = args.gitBranchName
            else:
                branchName = runAndCapture("git rev-parse --abbrev-ref HEAD")
            info["git"] = {
                "repository": runAndCapture("basename `git rev-parse --show-toplevel`"),
                "branch": branchName,
                "sha": runAndCapture("git rev-parse --short HEAD"),
            }
            subprocess.run("git show > \"{}/git_show.txt\"".format(dirs.results), shell=True)
            subprocess.run("git diff > \"{}/git_diff.txt\"".format(dirs.results), shell=True)
            subprocess.run("git diff --staged > \"{}/git_diff_staged.txt\"".format(dirs.results), shell=True)
        os.chdir(originalDir)

    infoJsonPath = dirs.results + "/info.json"
    with open(infoJsonPath, "w") as f:
        f.write(json.dumps(info, indent=4, separators=(',', ': ')))

    return infoJsonPath

# Maps from JSONL format keys to names shown in the output data and plots.
TRACK_KINDS = {
    "groundTruth": "groundTruth",
    "ARKit": "ARKit",
    "arcore": "ARCore",
    "arengine": "AREngine",
    "output": "OnDevice",
    "realsense": "RealSense",
    "gps": "GPS",
    "rtkgps": "RTKGPS",
    "externalPose": "externalPose",
}

def convertComparisonData(casePaths, metricSets, gnssConverter):
    frameCount = 0
    datasets = {}

    needsOrientation = (Metric.VELOCITY.value in metricSets
        or Metric.ANGULAR_VELOCITY.value in metricSets
        or Metric.ORIENTATION.value in metricSets
        or Metric.POSE_TRAIL_3D.value in metricSets
        or Metric.ORIENTATION_FULL.value in metricSets
        or Metric.ORIENTATION_ALIGNED.value in metricSets)
    if needsOrientation:
        # Import conditionally since scipy is otherwise not needed for benchmarking.
        from scipy.spatial.transform import Rotation

    def handleRow(rowJson):
        kind = None
        for k in TRACK_KINDS:
            if rowJson.get(k) is not None:
                kind = k
                pose = rowJson[kind]
                break
        if "pose" in rowJson:
            for k in TRACK_KINDS:
                if rowJson["pose"]["name"] == k:
                    kind = k
                    pose = rowJson["pose"]
                    break
        if not kind: return

        if not kind in datasets:
            datasets[kind] = []

        hasOrientation = False
        dToW = np.identity(4) # Device-to-world matrix.
        if "latitude" in pose:
            p = gnssConverter.enu(pose["latitude"], pose["longitude"], pose["altitude"])
        else:
            p = pose["position"]
            if needsOrientation and "orientation" in pose:
                hasOrientation = True
                q = [pose["orientation"][c] for c in "xyzw"]
                dToW[0:3, 0:3] = Rotation.from_quat(q).as_matrix()
        dToW[0:3, 3] = [p[c] for c in "xyz"]

        json = {
            "time": rowJson["time"],
            "position": { "x": dToW[0, 3], "y": dToW[1, 3], "z": dToW[2, 3] },
        }
        if hasOrientation:
            q = Rotation.from_matrix(dToW[0:3, 0:3]).as_quat()
            json["orientation"] = { "x": q[0], "y": q[1], "z": q[2], "w": q[3] }
        datasets[kind].append(json)

    if os.path.exists(casePaths["input"]):
        with open(casePaths["input"]) as f:
            for line in f.readlines():
                dataRow = json.loads(line)
                if dataRow.get("frames") is not None:
                    frameCount += 1
                    continue
                handleRow(dataRow)
    if os.path.exists(casePaths["inputGroundTruth"]):
        with open(casePaths["inputGroundTruth"]) as f:
            for line in f.readlines():
                handleRow(json.loads(line))

    postprocessed = []
    if COMPARE_POSTPROCESSED and os.path.exists(casePaths["outputMap"]):
        with open(casePaths["outputMap"]) as f:
            for line in f.readlines():
                row = json.loads(line)
                postprocessed.append({
                    "time": row["time"],
                    "position": row["position"],
                })

    realtime = []
    if COMPARE_POSTPROCESSED and os.path.exists(casePaths["outputRealtime"]):
        with open(casePaths["outputRealtime"]) as f:
            for line in f.readlines():
                row = json.loads(line)
                realtime.append({
                    "time": row["time"],
                    "position": row["position"],
                })

    with open(casePaths["gtJson"], "w") as gtJsonFile:
        def addDataSet(array, name, d):
            if d: array.append({ 'name': name, 'data': d })

        # First found data type will be used as ground truth.
        kindsOrdered = [
            "groundTruth",
            "externalPose",
            "ARKit",
            "output",
            "realsense",
            "gps",
            "rtkgps",
        ]
        datasetsOrdered = []
        for kind in kindsOrdered:
            if not kind in datasets: continue
            addDataSet(datasetsOrdered, TRACK_KINDS[kind], datasets[kind])

        if COMPARE_POSTPROCESSED:
            addDataSet(datasetsOrdered, "postprocessed", postprocessed)
            addDataSet(datasetsOrdered, "realtime", realtime)

        json.dump({'datasets': datasetsOrdered}, gtJsonFile)

    return frameCount

def benchmarkSingleDataset(benchmark, dirs, vioTrackingFn, args, baselineMetrics=None):
    caseDir = benchmark.dir
    caseName = benchmark.name

    casePaths = {
        "input": caseDir + "/data.jsonl",
        "inputGroundTruth": caseDir + "/groundtruth.jsonl",
        "output": "{}/{}.jsonl".format(dirs.out, caseName),
        "outputMap": "{}/{}_map.jsonl".format(dirs.out, caseName),
        "outputRealtime": "{}/{}_realtime.jsonl".format(dirs.out, caseName),
        "outputGlobal": "{}/{}_global.jsonl".format(dirs.out, caseName),
        "gtJson": "{}/{}.json".format(dirs.groundTruth, caseName),
        "logs": "{}/{}.txt".format(dirs.logs, caseName),
    }

    # Compute CPU time and wall time.
    timeCmd = ""
    if pathlib.Path("/usr/bin/time").exists():
        # GNU `time` format is easier to set than for bash/zsh `time`.
        timeCmd = "/usr/bin/time -f '{}: %U %S %P'".format(CPU_TIME_MESSAGE)
    startTime = time.time()

    vioSuccess = vioTrackingFn(args, benchmark, dirs.results, casePaths["output"], timeCmd)

    duration = time.time() - startTime
    cpuTime = None
    if timeCmd:
        if pathlib.Path(casePaths["logs"]).exists():
            with open(casePaths["logs"], "r") as f:
                for line in f:
                    if not line.startswith(CPU_TIME_MESSAGE): continue
                    tokens = line.split()
                    cpuTime = float(tokens[-2]) + float(tokens[-3]) # sys + user

    if not pathlib.Path(casePaths["output"]).exists():
        print("No output for case", caseName)
        return

    # It's important that the same GnssConverter instance is used for all VIO and reference tracks.
    gnssConverter = GnssConverter()
    outputGlobalFile = None
    for obj in readJsonl(casePaths["output"]):
        if not "globalPose" in obj: continue
        # Create the file lazily because in many cases there is no global output.
        if outputGlobalFile is None:
            outputGlobalFile = open(casePaths["outputGlobal"], "w")
        coordinates = obj["globalPose"]["coordinates"]
        gobj = {
            "time": obj["time"],
            "position": gnssConverter.enu(coordinates["latitude"], coordinates["longitude"], coordinates["altitude"]),
            "orientation": obj["globalPose"]["orientation"],
            "velocity": obj["globalPose"]["velocity"],
            "status": obj["status"],
        }
        outputGlobalFile.write(json.dumps(gobj, separators=(',', ':')))
        outputGlobalFile.write("\n")

    if outputGlobalFile is not None: outputGlobalFile.close()

    metricSets = args.metricSet.split(",")
    frameCount = convertComparisonData(casePaths, metricSets, gnssConverter)

    infoPath = "{}/{}.json".format(dirs.info, caseName)
    with open(infoPath, "w") as infoFile:
        infoJson = {
            "caseName": caseName,
            "dir": benchmark.dir,
            "paramSet": benchmark.paramSet,
            "methodName": args.methodName,
            "duration": duration,
            "frameCount": frameCount,
            "metricSets": metricSets,
            "fixOrigin": args.fixOrigin,
            "videoTimeSpan": computeVideoTimeSpan(casePaths["input"]),
            "poseTrailLengths": [float(s) for s in args.poseTrailLengths.split(",")],
        }
        if cpuTime: infoJson["cpuTime"] = cpuTime
        if benchmark.iteration: infoJson["iteration"] = benchmark.iteration
        infoFile.write(json.dumps(infoJson, indent=4, separators=(',', ': ')))

    baseline = None
    if baselineMetrics and caseName in baselineMetrics:
        baseline = baselineMetrics[caseName]

    try:
        metric = computeMetrics(dirs.results, caseName, baseline, args.sampleIntervalForVelocity)
    except Exception as e:
        if args.debug:
            import traceback
            print(traceback.format_exc())
        print("computeMetrics() failed for {}: {}".format(caseName, e))
        return False

    metricStr = "N/A" # Either no ground-truth or no VIO output.
    if metric: metricStr = "{:.3f}".format(metric)
    print("{:40} {:>6.0f}s   metric: {:>8}".format(caseName, duration, metricStr))
    return vioSuccess

# Look for the set file in a predefined directory or by path.
def findSetFile(setDir, setName):
    searchDirs = [setDir + "/", os.getcwd(), ""]
    for searchDir in searchDirs:
        path = searchDir + setName
        if not ".json" in path:
            path += ".json"
        path = pathlib.Path(path)
        if path.exists():
            return path
    raise Exception("Benchmark set \"{}\" not found in {}".format(setName, searchDirs))

def setupBenchmarkFromSetDescription(args, setName):
    setPath = findSetFile(args.setDir, setName)
    with open(setPath) as setFile:
        setDefinition = json.load(setFile)

    benchmarks = []
    parameterSets = [{}]
    if setDefinition.get("parameterSets") != None:
        parameterSets = setDefinition["parameterSets"]
        print("For {} parameters sets: {}".format(len(setDefinition["parameterSets"]),
            ", ".join("[" + s["params"] + "]" for s in setDefinition["parameterSets"])))
    for benchmark in setDefinition["benchmarks"]:
        for parameterSet in parameterSets:
            dir = args.rootDataDir + "/" + benchmark["folder"]
            params = []
            if parameterSet.get("params"): params.append(parameterSet["params"])
            if benchmark.get("params"): params.append(benchmark["params"])
            if benchmark.get("name"):
                name = benchmark.get("name").replace(' ', '-')
            else:
                name = benchmark["folder"]
                for symbol in " _/": name = name.replace(symbol, '-')
            if parameterSet.get("name"): name = "{}-{}".format(name, parameterSet.get("name").replace(' ', '-'))
            benchmarks.append(Benchmark(
                dir=dir,
                params=(None if len(params) == 0 else " ".join(params)),
                name=name,
                paramSet=(parameterSet["name"] if "name" in parameterSet else args.methodName),
            ))
    x = lambda s : "{} ({})".format(s["folder"], s["params"]) if s.get("params") else s["folder"]
    print("Running {} benchmark datasets:\n  {}".format(len(setDefinition["benchmarks"]),
        "\n  ".join(x(s) for s in setDefinition["benchmarks"])))

    if len(benchmarks) != len(set(b.name for b in benchmarks)):
        raise Exception("All benchmarks don't have unique names! Make sure all parameterSets and duplicate data sets have 'name' field")
    return benchmarks


def setupBenchmarkFromFolder(args, dataDir):
    cases = next(os.walk(args.dataDir))[1]
    cases.sort()
    print("Running benchmarks:\n  " + "\n  ".join([os.path.basename(c) for c in cases]))
    benchmarks = []
    for case in cases:
        benchmarks.append(Benchmark(
            dir=os.path.join(args.dataDir, case),
            params=None,
            name=case,
            paramSet=args.methodName,
        ))
    return benchmarks

def setupBenchmarkFromRecordingDir(args, recordingDir):
    print("Running benchmarks:\n  " + recordingDir)
    case = recordingDir.rsplit('/', 1)[1]
    benchmarks = []
    benchmarks.append(Benchmark(
        dir=recordingDir,
        params=None,
        name=case,
        paramSet=args.methodName,
    ))
    return benchmarks

class Dirs:
    results = None
    groundTruth = None
    out = None
    figures = None
    logs = None
    info = None

def collectMetrics(values, key):
    # Filter out non-existing values and Nones.
    return [v[key] for v in values if (key in v and not v[key] is None)]

def aggregateMetrics(metrics):
    def geometricMean(a):
        assert(a)
        return np.array(a).prod() ** (1.0 / len(a))
    if not metrics: return None

    metricSets = set()
    for metric in metrics:
        metricSets.update(metric.keys())

    result = {}
    for metricSetStr in metricSets:
        values = collectMetrics(metrics, metricSetStr)
        if not values:
            result[metricSetStr] = None
            continue

        if metricSetStr == "relative":
            result["relative"] = {}
            for relativeMetric in values[0]:
                result["relative"][relativeMetric] = geometricMean(collectMetrics(values, relativeMetric))
        elif isinstance(values[0], float):
            # Single numbers like for coverage and angular velocity metrics.
            result[metricSetStr] = np.mean(values)
        else:
            # Nested entries. Average within each category.
            result[metricSetStr] = {}
            for x in values[0]:
                result[metricSetStr][x] = np.mean(collectMetrics(values, x))
    return result

def benchmark(args, vioTrackingFn, setupFn=None, teardownFn=None):
    """Run benchmark for a VIO algorithm using callbacks

    @param args arguments to use
    @param vioTrackingFn function that runs VIO on a single dataset. Should return True iff VIO did not produce errors
    @param setupFn function that is run before any of the individual sets
    @param teardownFn function that is run after all the individual sets
    @return True iff none of the VIO runs produced errors
    """

    startTime = datetime.now().strftime(DATE_FORMAT)
    runId = args.runId if args.runId else startTime

    if args.skipBenchmark and not args.runId:
        raise Exception("-skipBenchmark requires -runId.")

    metricSets = args.metricSet.split(",")
    if "pose_trail" in metricSets or "pose_trail_3d" in metricSets or "tracking_quality" in metricSets:
        args.savePoseTrail = True

    def withMkdir(dir):
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        return dir

    results = withMkdir(os.path.abspath(args.output + "/" + runId))
    print(results)

    baselineMetrics = {}
    if args.baseline:
        with open(args.baseline) as baselineFile:
            baselineMetrics = json.loads(baselineFile.read())

    # TODO Is this class useless?
    dirs = Dirs()
    dirs.results = results
    dirs.groundTruth = withMkdir(results + "/ground-truth")
    dirs.out = withMkdir(results + "/vio-output")
    dirs.figures = withMkdir(results + "/figures")
    dirs.logs = withMkdir(results + "/vio-logs")
    dirs.info = withMkdir(results + "/info")
    dirs.metrics = withMkdir(results + "/metrics")

    if setupFn:
        setupFn(args, dirs.results)

    success = True
    if not args.skipBenchmark:
        if args.set:
            benchmarks = setupBenchmarkFromSetDescription(args, args.set)
        elif args.dataDir:
            benchmarks = setupBenchmarkFromFolder(args, args.dataDir)
        elif args.recordingDir:
            benchmarks = setupBenchmarkFromRecordingDir(args, args.recordingDir)
        else:
            print("You must select benchmark data using either `-set`, `-dataDir` or `-recordingDir`.")
            return

        if args.iterations > 1:
            benchmarksIterated = []
            for b in benchmarks:
                for i in range(1, args.iterations + 1):
                    benchmarksIterated.append(b.clone(i))
            benchmarks = benchmarksIterated

        if len(benchmarks) == 0:
            print("No matching benchmarks found! Exiting")
            return

        threadFunction = partial(benchmarkSingleDataset,
            dirs=dirs, vioTrackingFn=vioTrackingFn, args=args, baselineMetrics=baselineMetrics)

        print("---")
        if args.threads == 1:
            for benchmark in benchmarks:
                if not threadFunction(benchmark):
                    success = False
        else:
            workerCount = int(args.threads) if args.threads else multiprocessing.cpu_count()
            with concurrent.futures.ProcessPoolExecutor(max_workers=workerCount) as executor:
                for ret in executor.map(threadFunction, benchmarks):
                    if not ret: success = False

    if teardownFn:
        teardownFn(args, dirs.results)

    endTime = datetime.now().strftime(DATE_FORMAT)

    # Copy data from case metrics JSON files to `metrics.json` in the result root.
    metrics = {}
    for x in os.walk(results + "/metrics"):
        for caseMetricsJsonPath in x[2]:
            benchmarkMetrics = json.loads(open(os.path.join(results, "metrics", caseMetricsJsonPath)).read())
            caseName = caseMetricsJsonPath.rpartition(".")[0]
            assert(not caseName in metrics)
            metrics[caseName] = {}
            for k, v in benchmarkMetrics.items():
                metrics[caseName][k] = v
                # These have too much text and we don't want to aggregrate metrics for them here.
                if "pose_trail" in k:
                    delete = []
                    for pk, pv in metrics[caseName][k].items():
                        if "segments" in pk: delete.append(pk)
                    for d in delete:
                        del metrics[caseName][k][d]

    ametrics = aggregateMetrics(list(metrics.values()))

    # Delete the relative numbers so it's easier to copy-paste entries from this file
    # to the baseline metrics file.
    for x in metrics:
        if "relative" in metrics[x]: del metrics[x]["relative"]
    metricsJsonPath = results + "/metrics.json"
    with open(metricsJsonPath, "w") as f:
        f.write(json.dumps(metrics, indent=4, separators=(',', ': ')))

    # Needed by plotting below.
    infoJsonPath = writeSharedInfoFile(args, dirs, startTime, endTime, ametrics)

    print("---\nBenchmarks finished. Computing figures…")
    startTime = time.time()
    makeAllPlots(results, args.excludePlots, args.debug, args.sampleIntervalForVelocity)
    # Print the elapsed time since the plotting has been quite slow in the past.
    print("… took {:.0f}s.".format(time.time() - startTime))

    # Copy the aggregate figure of each metric to a shared folder for easier comparison.
    for metricSet in metricSets:
        src = getFigurePath("{}/figures".format(results), metricSet)
        if not os.path.exists(src): continue
        dstDir = os.path.abspath("{}/figures/{}".format(args.output, metricSet))
        pathlib.Path(dstDir).mkdir(parents=True, exist_ok=True)
        dst = "{}/{}.png".format(dstDir, runId)
        subprocess.run(["cp", src, dst])

    return success

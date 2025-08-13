import json
import os
import pathlib

import numpy as np

from .metric import *

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

def readVioOutput(benchmarkFolder, caseName, info, vioTrackKind, getPoseTrails=False):
    # Caching is not that beneficial here.
    global VIO_OUTPUT_CACHE
    baseName = benchmarkFolder.split("/")[-1] # Avoid differences from relative paths and symlinks.
    key = "{}+{}+{}+{}".format(baseName, caseName, vioTrackKind.value, getPoseTrails)
    if key in VIO_OUTPUT_CACHE: return VIO_OUTPUT_CACHE[key].copy()

    fileStem = ""
    if vioTrackKind == VioTrackKind.POSTPROCESSED: fileStem = "_map"
    elif vioTrackKind == VioTrackKind.GLOBAL: fileStem = "_global"

    outputPath = "{}/vio-output/{}{}.jsonl".format(benchmarkFolder, caseName, fileStem)
    if not pathlib.Path(outputPath).exists():
        return {}

    def isValidVector(row, field):
        if field not in row: return False
        # VIO should not produce values like this, but they have been spotted.
        # Crash rather than ignoring silently.
        for c in "xyz":
            if row[field][c] == None: raise Exception("Null values in VIO outputs.")
        return True


    method = None
    if "methodName" in info: method = info["methodName"].lower()

    to_arr = lambda obj: [obj["x"], obj["y"], obj["z"]]

    position = []
    orientation = []
    velocity = []
    angularVelocity = []
    acceleration = []
    baa = []
    bga = []
    stat = []
    idToTime = {}
    loopClosures = []
    resets = []
    trackingQuality = []
    poseTrails = []
    velocityCovariances = []
    status = []
    lastStatus = None
    globalStatus = []
    lastGlobalStatus = "INVALID"
    t = None
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
            if isValidVector(row, "acceleration"):
                av = row["acceleration"]
                acceleration.append([t, av["x"], av["y"], av["z"]])
            stat.append(row.get("stationary", False))
            if "id" in row:
                idToTime[row["id"]] = t
            if "loopClosureIds" in row:
                for i in range(len(row["loopClosureIds"])):
                    loopClosureId = row["loopClosureIds"][i]
                    loopClosureLinkColor = row["loopClosureLinkColors"][i] if "loopClosureLinkColors" in row else "deeppink"
                    if not loopClosureId in idToTime: continue
                    # Save times rather than indices as they survive the align operations better.
                    loopClosures.append((t, idToTime[loopClosureId], loopClosureLinkColor))
            if "trackingQuality" in row:
                trackingQuality.append([t, row["trackingQuality"]])
            if "velocityCovariance" in row and getPoseTrails:
                velocityCovariances.append((t, np.array(row["velocityCovariance"])))
            # May be slow, get only of needed.
            if "poseTrail" in row and getPoseTrails:
                poseTrail = []
                # Reverse so that smallest timestamp comes first.
                for pose in reversed(row["poseTrail"]):
                    p = pose["position"]
                    q = pose["orientation"]
                    poseTrail.append([pose["time"], p["x"], p["y"], p["z"], q["x"], q["y"], q["z"], q["w"]])
                if len(poseTrail) == 0: continue # Simplifies algorithms.
                poseTrails.append(np.array(poseTrail))
            # Currently needed only for a very niche pose trail metric.
            if "biasMean" in row and getPoseTrails:
                a = row["biasMean"]["accelerometerAdditive"]
                baa.append([t, a["x"], a["y"], a["z"]])
                g = row["biasMean"]["gyroscopeAdditive"]
                bga.append([t, g["x"], g["y"], g["z"]])
            if "status" in row:
                s = row["status"]
                if s != lastStatus:
                    lastStatus = s
                    status.append([t, s])
                if s == "LOST_TRACKING": resets.append(t)

            # Track also when the `globalPose` field is missing.
            s = None
            if "globalPose" in row and "status" in row["globalPose"]:
                s = row["globalPose"]["status"]
            if s != lastGlobalStatus:
                lastGlobalStatus = s
                globalStatus.append([t, s])

    if t is not None:
        globalStatus.append([t, None])

    VIO_OUTPUT_CACHE[key] = {
        'position': np.array(position),
        'orientation': np.array(orientation),
        'velocity': np.array(velocity),
        'angularVelocity': np.array(angularVelocity),
        'acceleration': np.array(acceleration),
        'stationary': np.array(stat),
        'loopClosures': loopClosures,
        'resets': resets,
        'trackingQuality': np.array(trackingQuality),
        'poseTrails': poseTrails,
        'biasGyroscopeAdditive': np.array(bga),
        'biasAccelerometerAdditive': np.array(baa),
        'velocityCovariances': velocityCovariances,
        'status': np.array(status),
        'globalStatus': np.array(globalStatus),
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

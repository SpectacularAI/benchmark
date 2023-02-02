#!/usr/bin/env python3

import json
import os
import pathlib

import numpy as np

def geometricMean(a):
    assert(a)
    return np.array(a).prod() ** (1.0 / len(a))

def available(obj, fields):
    for field in fields:
        if not field in obj: return False
        if not obj[field]: return False
        obj = obj[field]
    return True

def add(data, metricName, caseName, metric):
    if not metricName in data: data[metricName] = []
    data[metricName].append((caseName, metric))

def main(args):
    metricsDir = "{}/metrics".format(args.resultsDir)
    if not pathlib.Path(metricsDir).exists():
        print("No metrics/ folder.")
        return
    fileNames = list(next(os.walk(metricsDir))[2])
    fileNames.sort()

    relatives = {}
    absolutes = {}
    for fileName in fileNames:
        caseName = fileName.split(".")[0]
        with open("{}/{}".format(metricsDir, fileName)) as f:
            info = json.load(f)

        for metricName in ["piecewise", "full_3d", "coverage"]:
            if available(info, ["relative", metricName]):
                metric = info["relative"][metricName]
                add(relatives, metricName, caseName, metric)
        for metricName in ["full_3d"]:
            if available(info, [metricName, "RMSE"]):
                metric = info[metricName]["RMSE"]
                add(absolutes, metricName, caseName, metric)
        for metricName in ["coverage"]:
            if available(info, [metricName]):
                metric = info[metricName]
                add(absolutes, metricName, caseName, metric)

    for name in relatives:
        print("---\n{} metrics, relative:".format(name))
        for x in relatives[name]:
            print("{:8.4f}\t{}".format(x[1], x[0]))
        mean = geometricMean([x[1] for x in relatives[name]])
        print("{:8.4f}\tMEAN".format(mean))

    for name in absolutes:
        print("---\n{} metrics:".format(name))
        for x in absolutes[name]:
            print("{:8.4f}\t{}".format(x[1], x[0]))
        mean = np.mean([x[1] for x in absolutes[name]])
        print("{:8.4f}\tMEAN".format(mean))

    if not relatives and not absolutes:
        print("Did not compute any metrics (no ground truth?)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("resultsDir", help="Benchmark results folder")
    args = parser.parse_args()
    main(args)

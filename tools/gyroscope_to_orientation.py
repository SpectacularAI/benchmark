import json
import pathlib

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

def readJsonl(filePath):
    with open(filePath) as f:
        for l in f: yield(json.loads(l))

def readJson(filePath):
    with open(filePath) as f:
        return json.load(f)

class GyroscopeToOrientation:
    g = []
    imuToOutput = np.eye(4)

    def __init__(self, datasetPath):
        g = []
        t = None
        for obj in readJsonl(pathlib.Path(datasetPath) / "data.jsonl"):
            if not "sensor" in obj: continue
            if obj["sensor"]["type"] != "gyroscope": continue
            v = obj["sensor"]["values"]
            # Must be sorted.
            if t is not None and obj["time"] <= t:
                print("Unordered/equal gyroscope timestamps, skipping")
                continue
            t = obj["time"]
            g.append([t, v[0], v[1], v[2]])
        self.g = np.array(g)

        calibration = readJson(pathlib.Path(datasetPath) / "calibration.json")
        if "imuToOutput" in calibration:
            imuToOutput = np.array(calibration["imuToOutput"])

    def integrate(self, t0, t1, outputToWorld0, bias):
        """Integrate gyroscope samples between t0 and t1 into outputToWorld matrix and return the result matrix"""
        ind = np.searchsorted(self.g[:, 0], t0)
        if ind == 0:
            print("integrate() failed")
            return outputToWorld0
        assert(self.g[ind - 1, 0] <= t0)
        assert(t0 <= self.g[ind, 0])
        ind += 1

        worldToImu0 = np.linalg.inv(outputToWorld0 @ self.imuToOutput)

        xyzw = Rotation.from_matrix(worldToImu0[:3, :3]).as_quat()
        q = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]) # Convert to wxyz.
        t = t0
        bias = np.array(bias)
        while t < t1 and ind < self.g.shape[0]:
            tCur = self.g[ind, 0]
            if tCur > t1: tCur = t1
            dt = tCur - t
            assert(dt >= 0)
            w = self.g[ind, 1:] - bias
            S = -dt * 0.5 * np.array([
                [0, -w[0], -w[1], -w[2]],
                [w[0], 0, -w[2], w[1]],
                [w[1], w[2], 0, -w[0]],
                [w[2], -w[1], w[0], 0],
            ])
            A = scipy.linalg.expm(S)
            q = A @ q
            q = q / np.linalg.norm(q)
            t = tCur
            ind += 1

        worldToImu1 = np.eye(4)
        worldToImu1[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        outputToWorld1 = np.linalg.inv(self.imuToOutput @ worldToImu1)
        return outputToWorld1


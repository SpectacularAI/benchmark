import json
import pathlib

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

YIELD_INTERVAL_SECONDS = 0.005

def readJsonl(filePath):
    with open(filePath) as f:
        for l in f: yield(json.loads(l))

def readJson(filePath):
    with open(filePath) as f:
        return json.load(f)

class GyroscopeToOrientation:
    data = []
    velocityData = []
    imuToOutput = np.eye(4)

    def __init__(self, datasetPath, vioVelocity):
        if vioVelocity.shape[0] == 0:
            raise Exception("No velocity data")
        self.velocityData = vioVelocity

        # This is not the best acc-gyro syncing method but reasonably good.
        aLast = None
        t = None
        for obj in readJsonl(pathlib.Path(datasetPath) / "data.jsonl"):
            if not "sensor" in obj: continue
            v = obj["sensor"]["values"]
            if t is not None and obj["time"] < t: continue # Must be sorted.
            t = obj["time"]
            if obj["sensor"]["type"] == "accelerometer": aLast = [v[0], v[1], v[2]]
            elif obj["sensor"]["type"] == "gyroscope" and aLast is not None:
                self.data.append([t, v[0], v[1], v[2], aLast[0], aLast[1], aLast[2]])
        self.data = np.array(self.data)

        calibration = readJson(pathlib.Path(datasetPath) / "calibration.json")
        if "imuToOutput" in calibration:
            self.imuToOutput = np.array(calibration["imuToOutput"])

    def computePose(self, t, p, q):
        imuToWorld = np.eye(4)
        imuToWorld[:3, 3] = p
        imuToWorld[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().transpose()
        outputToWorld = imuToWorld @ np.linalg.inv(self.imuToOutput)
        return t, outputToWorld

    def integrate(self, t0, t1, outputToWorld0, biasGyroscope, biasAccelerometer, vioOrientation):
        ind = np.searchsorted(self.data[:, 0], t0)
        if ind == 0: return outputToWorld0
        assert(self.data[ind - 1, 0] <= t0)
        assert(t0 <= self.data[ind, 0])
        ind += 1

        vioToWorld0 = Rotation.from_quat(vioOrientation).as_matrix()
        vioWorldToOutputWorld = outputToWorld0[:3, :3] @ np.linalg.inv(vioToWorld0)

        velocityInd = np.searchsorted(self.velocityData[:, 0], t0)
        vioVelocity = self.velocityData[velocityInd, 1:]
        velocity = vioWorldToOutputWorld @ vioVelocity

        imuToWorld0 = outputToWorld0 @ self.imuToOutput

        p = imuToWorld0[:3, 3]
        xyzw = Rotation.from_matrix(imuToWorld0[:3, :3].transpose()).as_quat()
        q = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]) # Convert to wxyz.
        t = t0
        nextYieldT = t

        biasGyroscope = np.array(biasGyroscope)
        biasAccelerometer = np.array(biasAccelerometer)
        gravity = np.array([0, 0, -9.81])
        while t < t1 and ind < self.data.shape[0]:
            if t >= nextYieldT:
                nextYieldT = t + YIELD_INTERVAL_SECONDS
                yield self.computePose(t, p, q)

            tCur = self.data[ind, 0]
            if tCur > t1: tCur = t1
            dt = tCur - t
            assert(dt >= 0)
            w = self.data[ind, 1:4] - biasGyroscope
            S = -dt * 0.5 * np.array([
                [0, -w[0], -w[1], -w[2]],
                [w[0], 0, -w[2], w[1]],
                [w[1], w[2], 0, -w[0]],
                [w[2], -w[1], w[0], 0],
            ])
            A = scipy.linalg.expm(S)
            q = A @ q
            q = q / np.linalg.norm(q)
            p += velocity * dt # Intentionally before computation of `velocity` this iteration.
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            acceleration = R.transpose() @ (self.data[ind, 4:] - biasAccelerometer) + gravity
            velocity += acceleration * dt
            t = tCur
            ind += 1

        # Return also the final pose as it may be used for metrics.
        yield self.computePose(t, p, q)

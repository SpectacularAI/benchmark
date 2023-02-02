 #!/usr/bin/env python3
""" Benchmark Spectacular AI VIO """

import os
import sys
import shutil
import json

from tools.benchmark import benchmark, getArgParser

def setupFn(args, outputDir):
    pass

def tearDownFn(args, outputDir):
    existing = []
    ran = []
    for x in os.walk(f"{outputDir}/vio-logs"):
        for filename in x[2]:
            with open(os.path.join(x[0], filename)) as f:
                logMsg = f.read()
                caseName = filename.rsplit('.', 1)[0]
                if logMsg == "existing": existing.append(caseName)
                elif logMsg == "ran": ran.append(caseName)
                else: print(f"Unexpected status {filename}: {logMsg}")
    print("")
    if existing: print("Used existing trajectories from 'output.jsonl' files for: " + ", ".join(existing))
    if ran: print("Ran Spectacular AI SDK for: " + ", ".join(ran))

def vioTrackingFn(args, benchmark, outputDir, outputFile, prefixCmd):
    outputJsonl = f"{benchmark.dir}/output.jsonl"
    if os.path.exists(outputJsonl):
        shutil.copyfile(outputJsonl, outputFile)
        logMsg = "existing"
    else:
        def onOutput(output):
            with open(outputFile, "a") as f:
                f.write(output.asJson() + "\n")

        import spectacularAI
        replay = spectacularAI.Replay(benchmark.dir)
        replay.setPlaybackSpeed(-1)
        replay.setOutputCallback(onOutput)
        replay.runReplay()
        logMsg = "ran"

    logFile = f"{outputDir}/vio-logs/{benchmark.name}.txt"
    with open(logFile, "w") as f:
        f.write(logMsg)

    return True

if __name__ == "__main__":
    args = getArgParser().parse_args()
    args.methodName = "Spectacular AI"
    success = benchmark(args, vioTrackingFn, setupFn, tearDownFn)
    if not success:
        sys.exit(1)
    sys.exit(0)

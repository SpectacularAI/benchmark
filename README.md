# Spectacular AI Benchmark toolset

Benchmarking toolset that can plot trajectories and compute different metrics for VISLAM algorithms.

## Usage

Examples directory contains `two_sessions` dataset which has ground truth and output trajectory for two sessions. You can run the benchmark for these using following command, the output will be under `output` directory.

```
python run.py -dataDir examples/two_sessions
```

The `run.py` can also calculate the output trajectory from data recorded through Spectacualar SDK if it's not provided as `output.jsonl` file.

## Benchmark data

The data layout should use following convention, session_a has just the output, session_b has the video and IMU data required to calculate output using Spectacular AI SDK:
```
session_a/
    output.jsonl
    groundtruth.jsonl
session_b/
    calibration.json
    data2.mkv
    data.jsonl
    data.mkv
    groundtruth.jsonl
```

## Copyright

Based on https://github.com/AaltoML/vio_benchmark

This repository is licensed under Apache 2.0 (see LICENSE).


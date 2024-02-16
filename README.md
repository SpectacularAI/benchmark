# Spectacular AI Benchmark toolset

Benchmarking toolset that can plot trajectories and compute different metrics for VISLAM algorithms.

## Usage example

The directory `examples/two_sessions` contains an example dataset that has ground truth and VISLAM output trajectories for two sessions. You can run the benchmark for these using following command, and the output will be created under `output` directory:

```bash
python run.py -dataDir examples/two_sessions
```

See the options with `python run.py --help`.

## Benchmark data

The `run.py` script can also calculate the output trajectories from sensor data [recorded through Spectacular AI SDK](https://spectacularai.github.io/docs/sdk/recording.html). The format is partly documented [here](https://github.com/AaltoML/vio_benchmark#jsonl-format).

The recording folders should be placed in a common folder, say `sessions/`, for example like this:

```
session01/
    data.jsonl
    data.mkv
    data2.mkv
    calibration.json
    vio_config.yaml
    groundtruth.jsonl
session02/
    data.jsonl
    data.mkv
    calibration.json
    vio_config.yaml
session03/
    output.jsonl
    groundtruth.jsonl
```

In case of `session02/` the `groundtruth.jsonl` lines may have been mixed into the `data.jsonl` file. Then the benchmark can be run with:

```bash
pip install spectacularAI numpy scipy matplotlib
python run.py -dataDir sessions
```

## Copyright

Based on <https://github.com/AaltoML/vio_benchmark>.

This repository is licensed under Apache 2.0 (see LICENSE).

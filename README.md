# Spectacular AI Benchmark toolset

Benchmarking toolset that can plot trajectories and compute metrics such as RMSE for VISLAM algorithms.

## Using recording-time VISLAM outputs

[Spectacular AI recordings](https://spectacularai.github.io/docs/sdk/recording.html) (format documented [here](https://github.com/SpectacularAI/docs/blob/main/other/DATA_FORMAT.md)) can include both the recording-time VISLAM output poses and reference/ground-truth tracks in `data.jsonl`.

Place the recording folders in a common folder, say `sessions/`, for example like this:

```
sessions
├── session01
│   ├── vio_config.yaml
│   ├── calibration.json
│   ├── data.jsonl
│   └── data.mkv
└── session02
    ├── vio_config.yaml
    ├── calibration.json
    ├── data.jsonl
    └── data.mkv
```

Benchmark by running:

```bash
pip install numpy scipy matplotlib
python run.py -dataDir sessions -useRecordingOutput -metricSet global
```

The results and plots are created under `output/<timestamp>/`:

* **`figures-global.png`:** Plots the VISLAM `globalPose` and reference/ground-truth trajectories in local ENU (East-North-Up) coordinates in meters.
* **`metrics.json` and `info.json`:** `metrics.json` contains the computed metrics for each session and `info.json` contains run metadata and aggregate metrics averaged across sessions. Position errors are reported in meters: `RMSE`, `MAE`, `median`, `p95`, `max`, and `drift` (end-point error).

See all available options and metric sets with `python run.py --help`.

## Using separate `output.jsonl` files

Precomputed VISLAM output trajectories and ground truth poses can also be placed in separate `output.jsonl` and `groundtruth.jsonl` files in each session folder. An example dataset is included in `examples/two_sessions`:

```bash
python run.py -dataDir examples/two_sessions -metricSet global
```

## Using with the public Spectacular AI SDK

The `run.py` script can also compute output trajectories by replaying recorded sensor data using the public Spectacular AI SDK:

```bash
pip install spectacularAI numpy scipy matplotlib
python run.py -dataDir sessions -metricSet global
```

Shared `vio_config.yaml` and `calibration.json` files can also be placed in the parent `sessions/` folder instead of individual session folders.

## Metrics

Different metric sets can be selected with `-metricSet` (multiple sets can be separated by commas):

* **`global`:** Compares VISLAM `globalPose` outputs and reference tracks in WGS84 coordinates (converted to local ENU coordinates) directly without aligning the trajectories.
* **`full`:** Compares local VISLAM `position` outputs against reference tracks after aligning the trajectories by translation and rotation around the z-axis (gravity direction).

## Copyright

Based on <https://github.com/AaltoML/vio_benchmark>.

This repository is licensed under Apache 2.0 (see LICENSE).

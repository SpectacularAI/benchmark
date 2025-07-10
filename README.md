# Spectacular AI Benchmark toolset

Benchmarking toolset that can plot trajectories and compute different metrics for VISLAM algorithms.

## Example with precomputed VISLAM outputs

The directory `examples/two_sessions` contains an example dataset that has ground truth and VISLAM output trajectories for two sessions. You can run the benchmark for these using following command, and the output will be created under `output` directory:

```bash
python run.py -dataDir examples/two_sessions
```

See the options with `python run.py --help`.

## Using with Spectacular AI SDK recordings

The `run.py` script can also calculate the output trajectories from sensor data [recorded through Spectacular AI SDK](https://spectacularai.github.io/docs/sdk/recording.html). The format is documented [here](https://github.com/SpectacularAI/docs/blob/main/other/DATA_FORMAT.md).

The recording folders should be placed in a common folder, say `sessions/`, for example like this:

```
sessions
├── session01
│   ├── groundtruth.jsonl
│   ├── calibration.json
│   ├── data.jsonl
│   └── data.mkv
├── session02
│   ├── vio_config.yaml
│   ├── calibration.json
│   ├── data.jsonl
│   └── data.mkv
├── session03
│   ├── output.jsonl
│   └── groundtruth.jsonl
└── vio_config.yaml
```

Then the benchmark can be run with:

```bash
pip install spectacularAI numpy scipy matplotlib
python run.py -dataDir sessions
```

and the results found in the `output/` folder. Here are some details about the individual files in the example:

* `vio_config.yaml` and `calibration.json` can be placed in the parent `sessions/` folder. In the example, `session02` would use the `session02/vio_config.yaml` and other sessions the shared config.
* Ground truth poses for each session can either be placed in separate `groundtruth.jsonl` file or mixed into the `data.jsonl`.
* When `output.jsonl` file is present for a session it will be used for outputs and SDK replay is skipped.

## Copyright

Based on <https://github.com/AaltoML/vio_benchmark>.

This repository is licensed under Apache 2.0 (see LICENSE).

#!/usr/bin/env python3
"""
  Make `figures.png` inside the given benchmark folder:
    python plot_main.py path/to/benchmark-results/2021-10-01_13-21-10
  Make `figures/euroc-mh-01-easy.png`:
    python plot_main.py path/to/benchmark-results/2021-10-01_13-21-10 --caseName euroc-mh-01-easy
  Make everything:
    python plot_main.py path/to/benchmark-results/2021-10-01_13-21-10 --all
"""

from plot import plotBenchmark, makeAllPlots

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("benchmarkFolder", help="benchmark data")
    parser.add_argument("--caseName", help="An individual plot")
    parser.add_argument("--all", help="Normal and z-axis aggregate plot, plus all the individual plots", action="store_true")
    parser.add_argument("--z-axis", help="", action="store_true")
    parser.add_argument("--compactRotation", help="", action="store_true")
    parser.add_argument("--showPlot", help="Show on screen instead of saving to an image", action="store_true")
    parser.add_argument("--excludePlots", type=str, help="Tracks to skip plotting, split by comma", default="ondevice")
    args = parser.parse_args()

    if args.all:
        makeAllPlots(args.benchmarkFolder)
    else:
        plotBenchmark(args, args.benchmarkFolder)

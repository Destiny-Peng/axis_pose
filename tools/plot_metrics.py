#!/usr/bin/env python3
"""
Simple CSV visualization for AlgorithmBenchmark outputs.
- Reads one or more `metrics.csv` files (or a directory containing them)
- Plots per-stage elapsed_ms time series and histograms, saves PNG files.

Usage:
  python3 tools/plot_metrics.py /path/to/statistics

Requires: matplotlib, pandas
"""
import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def find_csvs(path):
    if os.path.isdir(path):
        # search for metrics.csv recursively
        return glob.glob(os.path.join(path, '**', 'metrics.csv'), recursive=True)
    elif os.path.isfile(path) and path.endswith('.csv'):
        return [path]
    else:
        return []


def plot(files, out_dir='plots'):
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            continue
        # group by name and plot elapsed_ms time series
        for name, g in df.groupby('name'):
            plt.figure(figsize=(10,4))
            plt.plot(g['run_id'], g['elapsed_ms'], marker='o')
            plt.title(f"{os.path.basename(f)} - {name}")
            plt.xlabel('run_id')
            plt.ylabel('elapsed_ms')
            plt.grid(True)
            out_png = os.path.join(out_dir, f"{os.path.basename(f)}_{name}_timeseries.png")
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()

            # histogram
            plt.figure(figsize=(6,4))
            plt.hist(g['elapsed_ms'].dropna(), bins=50)
            plt.title(f"{os.path.basename(f)} - {name} histogram")
            plt.xlabel('elapsed_ms')
            plt.ylabel('count')
            out_png2 = os.path.join(out_dir, f"{os.path.basename(f)}_{name}_hist.png")
            plt.tight_layout()
            plt.savefig(out_png2)
            plt.close()
        print(f"Plotted {f} -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='metrics.csv file or directory containing run folders')
    parser.add_argument('--out', default='plots', help='output directory for PNGs')
    args = parser.parse_args()

    files = find_csvs(args.path)
    if not files:
        print('No metrics.csv found')
        return
    plot(files, args.out)

if __name__ == '__main__':
    main()

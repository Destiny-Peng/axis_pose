#!/usr/bin/env python3
"""
Merge per-run metrics.csv files into a single CSV with a `run_name` column.
Usage:
  python3 tools/merge_metrics.py /path/to/statistics_base_dir
Outputs: <statistics_base_dir>/merged_metrics.csv
Requires: pandas
"""
import argparse
import glob
import os
import pandas as pd


def find_metrics(base_dir):
    pattern = os.path.join(base_dir, '**', 'metrics.csv')
    return glob.glob(pattern, recursive=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='statistics base dir')
    args = parser.parse_args()
    base = args.path
    files = find_metrics(base)
    if not files:
        print('No metrics.csv found under', base)
        return
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # attempt to infer run name as parent directory of metrics.csv
            run_dir = os.path.basename(os.path.dirname(f))
            df['run_name'] = run_dir
            dfs.append(df)
        except Exception as e:
            print('Failed to read', f, e)
    if not dfs:
        print('No valid CSVs to merge')
        return
    all_df = pd.concat(dfs, ignore_index=True)
    out = os.path.join(base, 'merged_metrics.csv')
    all_df.to_csv(out, index=False)
    print('Merged', len(dfs), 'files ->', out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import csv
import math
import os
from statistics import mean, stdev, median


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                valid = int(r.get("valid", "0"))
            except Exception:
                valid = 0
            if valid != 1:
                continue
            try:
                angle = float(r.get("angle_err_deg", "nan"))
                offset = float(r.get("offset_err_px", "nan"))
            except Exception:
                continue
            if math.isnan(angle) or math.isnan(offset):
                continue
            rows.append((angle, offset))
    return rows


def summarize(vals):
    if not vals:
        return "", "", ""
    if len(vals) == 1:
        return vals[0], 0.0, vals[0]
    return mean(vals), stdev(vals), median(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="line2d_metrics.csv")
    ap.add_argument("--output", required=True, help="line2d_summary.csv")
    args = ap.parse_args()

    data = read_rows(args.input)
    angles = [x[0] for x in data]
    offsets = [x[1] for x in data]

    a_mean, a_std, a_median = summarize(angles)
    o_mean, o_std, o_median = summarize(offsets)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "count",
            "angle_mean_deg",
            "angle_std_deg",
            "angle_median_deg",
            "offset_mean_px",
            "offset_std_px",
            "offset_median_px",
        ])
        w.writerow([len(data), a_mean, a_std, a_median, o_mean, o_std, o_median])

    print(f"Valid frames: {len(data)}")
    print(f"Angle mean deg: {a_mean if a_mean != '' else 'n/a'}")
    print(f"Offset mean px: {o_mean if o_mean != '' else 'n/a'}")


if __name__ == "__main__":
    main()

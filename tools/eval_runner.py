#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import yaml


def run_cmd(cmd: str, cwd: Path) -> int:
    print(f"> {cmd}")
    proc = subprocess.run(["bash", "-lc", cmd], cwd=str(cwd))
    return proc.returncode


def plugin_for_algorithm(name: str) -> str:
    mapping = {
        "pca": "axispose::PoseEstimatePCA",
        "ransac": "axispose::PoseEstimateRANSAC",
        "gaussian": "axispose::PoseEstimateGaussian",
        "ceres": "axispose::PoseEstimateCeres",
    }
    key = name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported algorithm: {name}")
    return mapping[key]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "evaluation" not in data:
        raise ValueError("Missing 'evaluation' section in config")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified evaluator for axispose")
    parser.add_argument("--config", default="tools/eval_config.yaml", help="Path to eval YAML config")
    parser.add_argument("--out-root", default="", help="Optional override for output root directory")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan but do not execute")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    ecfg = cfg["evaluation"]

    setup_bash = str((root / ecfg.get("setup_bash", "install/setup.bash")).resolve())
    launch_file = str((root / ecfg.get("launch_file", "launch/launch.py")).resolve())
    rgb_dir = str((root / ecfg.get("rgb_dir", "image1/rgb")).resolve())
    depth_dir = str((root / ecfg.get("depth_dir", "image1/depth")).resolve())

    if args.out_root:
        out_root = Path(args.out_root).resolve()
    else:
        out_name = ecfg.get("out_root", "statistics/eval_unified_%Y%m%d_%H%M%S")
        out_root = root / datetime.now().strftime(out_name)
    out_root.mkdir(parents=True, exist_ok=True)

    timeout_s = int(ecfg.get("timeout_seconds", 65))
    algorithms = ecfg.get("algorithms", ["pca", "ransac", "gaussian", "ceres"])

    line2d_enabled = bool(ecfg.get("line2d_eval_enabled", True))
    gt_enabled = bool(ecfg.get("groundtruth_eval_enabled", False))
    gt_images_dir_cfg = ecfg.get("groundtruth_images_dir", "")
    gt_images_dir = str((root / gt_images_dir_cfg).resolve()) if gt_images_dir_cfg else ""
    gt_camera_info = str((root / ecfg.get("groundtruth_camera_info_file", "config/d457_color.yaml")).resolve())
    gt_layout = str((root / ecfg.get("groundtruth_layout_file", "tools/apriltag.yaml")).resolve())
    gt_min_tags = int(ecfg.get("groundtruth_min_tags", 4))
    aggregate_enabled = bool(ecfg.get("aggregate_summary_enabled", True))
    aggregate_plots_enabled = bool(ecfg.get("aggregate_summary_plots_enabled", True))

    if args.dry_run:
        print("Dry-run summary:")
        print(f"  setup: {setup_bash}")
        print(f"  launch: {launch_file}")
        print(f"  rgb_dir: {rgb_dir}")
        print(f"  depth_dir: {depth_dir}")
        print(f"  out_root: {out_root}")
        print(f"  timeout: {timeout_s}s")
        print(f"  algorithms: {algorithms}")
        print(f"  line2d_eval_enabled: {line2d_enabled}")
        print(f"  groundtruth_eval_enabled: {gt_enabled}")
        print(f"  aggregate_summary_enabled: {aggregate_enabled}")
        print(f"  aggregate_summary_plots_enabled: {aggregate_plots_enabled}")
        return 0

    for alg in algorithms:
        plugin = plugin_for_algorithm(alg)
        out_dir = out_root / alg
        out_dir.mkdir(parents=True, exist_ok=True)

        # Avoid mixing previous runs when reusing an existing --out-root.
        stale_files = [
            out_dir / "metrics.csv",
            out_dir / "line2d_metrics.csv",
            out_dir / "line2d_summary.csv",
            out_dir / "gt_eval" / "groundtruth.csv",
            out_dir / "gt_eval" / "groundtruth_averaged.csv",
            out_dir / "gt_eval" / "aligned.csv",
            out_dir / "gt_eval" / "per_frame_errors.csv",
            out_dir / "gt_eval" / "evaluation_summary.csv",
        ]
        for f in stale_files:
            if f.exists() and f.is_file():
                f.unlink()

        print("=" * 50)
        print(f"[RUN] algorithm={alg} plugin={plugin} out={out_dir}")
        print("=" * 50)

        launch_cmd = (
            f"source '{setup_bash}' && "
            f"timeout -s INT {timeout_s}s ros2 launch '{launch_file}' "
            f"rgb_dir:='{rgb_dir}' depth_dir:='{depth_dir}' "
            f"pose_plugin:='{plugin}' statistic_directory:='{out_dir}'"
        )
        rc = run_cmd(launch_cmd, root)
        if rc not in (0, 124, 130):
            print(f"ERROR: launch failed for {alg}, rc={rc}")
            return rc

        if line2d_enabled:
            line2d_csv = out_dir / "line2d_metrics.csv"
            if line2d_csv.exists():
                line2d_cmd = (
                    f"python3 '{root / 'tools/evaluate_line2d.py'}' "
                    f"--input '{line2d_csv}' --output '{out_dir / 'line2d_summary.csv'}'"
                )
                rc = run_cmd(line2d_cmd, root)
                if rc != 0:
                    return rc
            else:
                print(f"WARN: {line2d_csv} missing, skip line2d summary")

        if gt_enabled:
            if not gt_images_dir or not Path(gt_images_dir).exists():
                print(f"WARN: GT enabled but invalid groundtruth_images_dir: {gt_images_dir}")
            else:
                gt_dir = out_dir / "gt_eval"
                gt_dir.mkdir(parents=True, exist_ok=True)

                extract_cmd = (
                    f"python3 '{root / 'tools/extract_groundtruth.py'}' "
                    f"--images '{gt_images_dir}' "
                    f"--camera-info '{gt_camera_info}' "
                    f"--layout '{gt_layout}' "
                    f"--output '{gt_dir / 'groundtruth.csv'}' "
                    f"--save-annotated '{gt_dir / 'gt_vis'}' "
                    f"--min-tags {gt_min_tags}"
                )
                rc = run_cmd(extract_cmd, root)
                if rc != 0:
                    return rc

                avg_cmd = (
                    f"python3 '{root / 'tools/gt_ingest.py'}' average "
                    f"--input '{gt_dir / 'groundtruth.csv'}' "
                    f"--output '{gt_dir / 'groundtruth_averaged.csv'}'"
                )
                rc = run_cmd(avg_cmd, root)
                if rc != 0:
                    return rc

                align_cmd = (
                    f"python3 '{root / 'tools/gt_ingest.py'}' align "
                    f"--est '{out_dir / 'metrics.csv'}' "
                    f"--gt '{gt_dir / 'groundtruth_averaged.csv'}' "
                    f"--output '{gt_dir / 'aligned.csv'}' --by-sequence"
                )
                rc = run_cmd(align_cmd, root)
                if rc != 0:
                    return rc

                eval_cmd = (
                    f"python3 '{root / 'tools/evaluate.py'}' "
                    f"--input '{gt_dir / 'aligned.csv'}' "
                    f"--out '{gt_dir / 'per_frame_errors.csv'}' "
                    f"--summary '{gt_dir / 'evaluation_summary.csv'}' "
                    f"--plots '{gt_dir / 'plots'}'"
                )
                rc = run_cmd(eval_cmd, root)
                if rc != 0:
                    return rc

    if aggregate_enabled:
        summary_cmd = (
            f"python3 '{root / 'tools/summarize_eval_results.py'}' "
            f"--out-root '{out_root}'"
        )
        if not aggregate_plots_enabled:
            summary_cmd += " --no-plots"

        print("=" * 50)
        print(f"[POST] aggregate summary for out_root={out_root}")
        print("=" * 50)
        rc = run_cmd(summary_cmd, root)
        if rc != 0:
            return rc

    print(f"All done. Output root: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

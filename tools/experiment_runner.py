#!/usr/bin/env python3
"""
Simple experiment runner for axispose.
- Reads an experiments YAML describing parameter grid, repeats, duration, and base params file.
- For each configuration it:
  - creates a run directory
  - writes a params YAML combining base params and overrides (ensures statistics_directory_path points to run dir and statistics_enabled true)
  - launches `ros2 launch` with `params:=<file>` and runs for configured duration
  - collects `metrics.csv` from the run directory

Assumptions:
- The project's launch.py accepts a `params` launch argument to load node params from a YAML file, e.g.:
  ros2 launch axispose launch.py params:=/abs/path/to/params.yaml
- Python environment has `pyyaml` installed.

Usage:
  python3 tools/experiment_runner.py examples/experiments.yaml

"""
import argparse
import itertools
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
import copy

try:
    import yaml
except Exception as e:
    print("Please install pyyaml: pip install pyyaml")
    raise


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)


def dict_set_dot(d, key, value):
    # key like 'pose_estimate.statistics_enabled' or 'statistics_enabled'
    # Works on a plain dict `d` and creates nested dicts for dotted keys.
    parts = key.split('.')
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='experiments YAML')
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    base_params = {}
    if 'base_params_file' in cfg and cfg['base_params_file']:
        base_params = load_yaml(cfg['base_params_file'])

    launch_cmd = cfg.get('launch_cmd', 'ros2 launch axispose launch.py')
    # name of the launch argument that controls the statistics directory in the launch file
    # launch.py declares 'statistic_directory' by default
    statistic_launch_arg = cfg.get('statistic_launch_arg', 'statistic_directory')
    sweeps = cfg.get('sweeps', {})
    repeats = int(cfg.get('repeats', 1))
    duration = int(cfg.get('run_duration_sec', 10))
    stats_base = os.path.abspath(cfg.get('statistics_base_dir', 'statistics'))

    # construct grid
    keys = list(sweeps.keys())
    values = [sweeps[k] for k in keys]

    combos = list(itertools.product(*values)) if values else [()]

    os.makedirs(stats_base, exist_ok=True)

    run_idx = 0
    stop_all = False
    for combo in combos:
        if stop_all:
            break
        combo_dict = dict(zip(keys, combo))
        for rep in range(repeats):
            if stop_all:
                break
            run_idx += 1
            ts = datetime.now().strftime('%Y%m%dT%H%M%S')
            run_name = f"run_{run_idx:03d}_{ts}"
            run_dir = os.path.join(stats_base, run_name)
            os.makedirs(run_dir, exist_ok=True)

            # prepare params: deep-copy base and ensure ROS params structure '/**': { ros__parameters: { ... } }
            params = copy.deepcopy(base_params) if isinstance(base_params, dict) else {}
            if '/**' not in params or not isinstance(params['/**'], dict):
                params['/**'] = {'ros__parameters': {}}
            if 'ros__parameters' not in params['/**'] or not isinstance(params['/**']['ros__parameters'], dict):
                params['/**']['ros__parameters'] = {}

            ros_params = params['/**']['ros__parameters']

            # apply overrides into ros__parameters (respect dotted keys for nested node params)
            for k, v in combo_dict.items():
                dict_set_dot(ros_params, k, v)

            # ensure statistics dir and enabled are inside ros__parameters
            dict_set_dot(ros_params, 'statistics_directory_path', run_dir)
            dict_set_dot(ros_params, 'statistics_enabled', True)

            params_file = os.path.join(run_dir, 'params.yaml')
            save_yaml(params, params_file)

            # launch process
            # include launch arg for statistic directory so LaunchConfiguration picks up per-run dir
            cmd = f"{launch_cmd} params:={params_file} {statistic_launch_arg}:={run_dir}"
            print(f"Starting {run_name}: {cmd} (duration {duration}s)")
            proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
            try:
                try:
                    time.sleep(duration)
                    # after duration, send SIGINT to the process group
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    except Exception:
                        pass
                    proc.wait(timeout=10)
                except KeyboardInterrupt:
                    print("Interrupted by user, stopping launched process and aborting sweep...")
                    stop_all = True
                    # send SIGINT to the process group
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    except Exception:
                        pass
                    # allow some time then force terminate
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        except Exception:
                            pass
                        try:
                            proc.wait(timeout=2)
                        except Exception:
                            try:
                                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                            except Exception:
                                pass
                except Exception:
                    # unexpected error during run, try to terminate child
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            pass
            finally:
                time.sleep(1)

            # attempt to find metrics.csv (search recursively in run_dir)
            metrics_found = False
            for root, dirs, files in os.walk(run_dir):
                if 'metrics.csv' in files:
                    metrics_src = os.path.join(root, 'metrics.csv')
                    print(f"Collected metrics: {metrics_src}")
                    metrics_found = True
                    break
            if not metrics_found:
                print(f"No metrics.csv found in {run_dir} (node may write to a different path).")

    print("All runs finished.")


if __name__ == '__main__':
    main()

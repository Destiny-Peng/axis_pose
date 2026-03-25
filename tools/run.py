#!/usr/bin/env python3
"""
Central CLI to run common tools tasks (consolidates existing scripts).

Commands:
  run_full_eval    Run the full evaluation pipeline (extract -> average -> align -> evaluate)
  extract_gt       Run extract_groundtruth.py
  gt_average       Run gt_ingest.py average
  gt_align         Run gt_ingest.py align
  evaluate         Run evaluate.py
  plot_metrics     Run plot_metrics.py
  merge_metrics    Run merge_metrics.py
  experiment       Run experiment_runner.py

This wrapper uses subprocess to call the existing scripts so behavior remains identical.
"""
import argparse
import subprocess
import os
from pathlib import Path
import importlib.util
import types


def run_cmd(cmd, env=None):
    print('> ' + ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


def load_module_from_path(path: str) -> types.ModuleType:
    """Load a python module from a file path."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_run_full_eval(args):
    out_dir = args.out_dir or f"statistics/run_{subprocess.check_output(['date','+%Y%m%d_%H%M%S']).decode().strip()}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1) extract
    run_cmd(['python3', 'tools/extract_groundtruth.py', '--images', args.images, '--camera-info', args.camera_info, '--tag-size', str(args.tag_size), '--output', os.path.join(out_dir, 'groundtruth.csv'), '--save-annotated', os.path.join(out_dir, 'gt_vis')])

    # 2) average
    # try to call gt_ingest.average_gt_per_image directly
    gt_ingest_path = os.path.join('tools', 'gt_ingest.py')
    try:
        gt_mod = load_module_from_path(gt_ingest_path)
        print('Calling gt_ingest.average_gt_per_image directly')
        gt_mod.average_gt_per_image(os.path.join(out_dir, 'groundtruth.csv'), os.path.join(out_dir, 'groundtruth_averaged.csv'))
    except Exception as e:
        print('Direct call to gt_ingest failed, falling back to subprocess:', e)
        run_cmd(['python3', gt_ingest_path, 'average', '--input', os.path.join(out_dir, 'groundtruth.csv'), '--output', os.path.join(out_dir, 'groundtruth_averaged.csv')])

    # 3) align
    try:
        if 'gt_mod' not in locals():
            gt_mod = load_module_from_path(gt_ingest_path)
        print('Calling gt_ingest.align_estimates_to_gt directly')
        gt_mod.align_estimates_to_gt(args.est_csv, os.path.join(out_dir, 'groundtruth_averaged.csv'), os.path.join(out_dir, 'aligned.csv'), tol_s=0.5, by_sequence=True)
    except Exception as e:
        print('Direct call to gt_ingest.align failed, falling back to subprocess:', e)
        run_cmd(['python3', gt_ingest_path, 'align', '--est', args.est_csv, '--gt', os.path.join(out_dir, 'groundtruth_averaged.csv'), '--output', os.path.join(out_dir, 'aligned.csv'), '--by-sequence'])

    # 4) evaluate
    eval_path = os.path.join('tools', 'evaluate.py')
    try:
        ev_mod = load_module_from_path(eval_path)
        print('Calling evaluate functions directly')
        rows = ev_mod.read_aligned(os.path.join(out_dir, 'aligned.csv'))
        per, stats = ev_mod.evaluate(rows)
        ev_mod.save_perframe(per, os.path.join(out_dir, 'per_frame_errors.csv'))
        ev_mod.save_summary(stats, os.path.join(out_dir, 'evaluation_summary.csv'))
        # plots
        try:
            ev_mod.plot_hist(per, os.path.join(out_dir, 'plots'))
            ev_mod.plot_poses(rows, os.path.join(out_dir, 'plots'))
        except Exception as e:
            print('Plotting failed:', e)
    except Exception as e:
        print('Direct call to evaluate failed, falling back to subprocess:', e)
        run_cmd(['python3', eval_path, '--input', os.path.join(out_dir, 'aligned.csv'), '--out', os.path.join(out_dir, 'per_frame_errors.csv'), '--summary', os.path.join(out_dir, 'evaluation_summary.csv'), '--plots', os.path.join(out_dir, 'plots')])

    print('Done. Results in', out_dir)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    p_full = sub.add_parser('run_full_eval', help='Run full evaluation pipeline')
    p_full.add_argument('--images', required=True, help='Images dir')
    p_full.add_argument('--camera-info', default='config/d457_color.yaml')
    p_full.add_argument('--tag-size', type=float, default=0.05)
    p_full.add_argument('--est-csv', default='statistics/metrics.csv', help='Estimates CSV')
    p_full.add_argument('--out-dir', default='', help='Output directory')

    p_extract = sub.add_parser('extract_gt', help='Run extract_groundtruth.py')
    p_extract.add_argument('--images', required=True)
    p_extract.add_argument('--camera-info', required=True)
    p_extract.add_argument('--tag-size', type=float, required=True)
    p_extract.add_argument('--output', required=True)
    p_extract.add_argument('--save-annotated', default='')

    p_avg = sub.add_parser('gt_average', help='Average groundtruth')
    p_avg.add_argument('--input', required=True)
    p_avg.add_argument('--output', required=True)

    p_align = sub.add_parser('gt_align', help='Align estimates to averaged GT')
    p_align.add_argument('--est', required=True)
    p_align.add_argument('--gt', required=True)
    p_align.add_argument('--output', required=True)
    p_align.add_argument('--by-sequence', action='store_true')

    p_eval = sub.add_parser('evaluate', help='Run evaluate.py')
    p_eval.add_argument('--input', required=True)
    p_eval.add_argument('--out', required=True)
    p_eval.add_argument('--summary', required=True)
    p_eval.add_argument('--plots', default='')

    p_plot = sub.add_parser('plot_metrics', help='Plot metrics.csv files')
    p_plot.add_argument('path')
    p_plot.add_argument('--out', default='plots')

    p_merge = sub.add_parser('merge_metrics', help='Merge metrics.csv files')
    p_merge.add_argument('path')

    p_experiment = sub.add_parser('experiment', help='Run experiment_runner.py')
    p_experiment.add_argument('config')

    args = parser.parse_args()

    if args.cmd == 'run_full_eval':
        cmd_run_full_eval(args)
    elif args.cmd == 'extract_gt':
        cmd = ['python3', 'tools/extract_groundtruth.py', '--images', args.images, '--camera-info', args.camera_info, '--tag-size', str(args.tag_size), '--output', args.output]
        if args.save_annotated:
            cmd += ['--save-annotated', args.save_annotated]
        run_cmd(cmd)
    elif args.cmd == 'gt_average':
        run_cmd(['python3', 'tools/gt_ingest.py', 'average', '--input', args.input, '--output', args.output])
    elif args.cmd == 'gt_align':
        cmd = ['python3', 'tools/gt_ingest.py', 'align', '--est', args.est, '--gt', args.gt, '--output', args.output]
        if args.by_sequence:
            cmd.append('--by-sequence')
        run_cmd(cmd)
    elif args.cmd == 'evaluate':
        cmd = ['python3', 'tools/evaluate.py', '--input', args.input, '--out', args.out, '--summary', args.summary]
        if args.plots:
            cmd += ['--plots', args.plots]
        run_cmd(cmd)
    elif args.cmd == 'plot_metrics':
        run_cmd(['python3', 'tools/plot_metrics.py', args.path, '--out', args.out])
    elif args.cmd == 'merge_metrics':
        run_cmd(['python3', 'tools/merge_metrics.py', args.path])
    elif args.cmd == 'experiment':
        run_cmd(['python3', 'tools/experiment_runner.py', args.config])
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

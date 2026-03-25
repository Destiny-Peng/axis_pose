#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# run_full_eval.sh - one-shot evaluation pipeline
# Usage: ./tools/run_full_eval.sh [IMAGES_DIR] [CAMERA_INFO_YAML] [TAG_SIZE_m] [ESTIMATES_CSV] [OUT_DIR]
# Example:
# ./tools/run_full_eval.sh \
#   /path/to/images config/d457_color.yaml 0.05 statistics/metrics.csv statistics/run01

IMAGES_DIR=${1:-}
CAMERA_INFO=${2:-${ROOT_DIR}/config/d457_color.yaml}
TAG_SIZE=${3:-0.05}
EST_CSV=${4:-statistics/metrics.csv}
OUT_DIR=${5:-statistics/run_$(date +%Y%m%d_%H%M%S)}

if [ -z "${IMAGES_DIR}" ]; then
  echo "Usage: $0 IMAGES_DIR [CAMERA_INFO_YAML] [TAG_SIZE_m] [ESTIMATES_CSV] [OUT_DIR]"
  exit 1
fi

mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/gt_vis"
mkdir -p "${OUT_DIR}/plots"

echo "[1/5] Extracting AprilTag ground-truth from images -> ${OUT_DIR}/groundtruth.csv"
python3 "${ROOT_DIR}/tools/extract_groundtruth.py" \
  --images "${IMAGES_DIR}" \
  --camera-info "${CAMERA_INFO}" \
  --tag-size "${TAG_SIZE}" \
  --output "${OUT_DIR}/groundtruth.csv" \
  --save-annotated "${OUT_DIR}/gt_vis" || { echo "extract_groundtruth failed"; exit 2; }

echo "[2/5] Averaging multiple tags per image -> ${OUT_DIR}/groundtruth_averaged.csv"
python3 "${ROOT_DIR}/tools/gt_ingest.py" average --input "${OUT_DIR}/groundtruth.csv" --output "${OUT_DIR}/groundtruth_averaged.csv" || { echo "gt_ingest average failed"; exit 3; }

echo "[3/5] Aligning estimates to averaged GT -> ${OUT_DIR}/aligned.csv"
python3 "${ROOT_DIR}/tools/gt_ingest.py" align --est "${EST_CSV}" --gt "${OUT_DIR}/groundtruth_averaged.csv" --output "${OUT_DIR}/aligned.csv" --by-sequence || { echo "gt_ingest align failed"; exit 4; }

echo "[4/5] Evaluating aligned pairs -> per-frame and summary in ${OUT_DIR}"
python3 "${ROOT_DIR}/tools/evaluate.py" --input "${OUT_DIR}/aligned.csv" --out "${OUT_DIR}/per_frame_errors.csv" --summary "${OUT_DIR}/evaluation_summary.csv" --plots "${OUT_DIR}/plots" || { echo "evaluate failed"; exit 5; }

echo "[5/5] Done. Results in ${OUT_DIR}"
ls -la "${OUT_DIR}"

echo "Tip: open ${OUT_DIR}/plots/poses_xy.png and poses_xz.png to inspect orientation differences (XY vs XZ)."

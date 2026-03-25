#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/install/setup.bash"
set -u

RGB_DIR="${1:-${ROOT_DIR}/image1/rgb}"
DEPTH_DIR="${2:-${ROOT_DIR}/image1/depth}"
OUT_ROOT="${3:-${ROOT_DIR}/statistics/eval4_$(date +%Y%m%d_%H%M%S)}"
LAUNCH_FILE="${ROOT_DIR}/launch/launch.py"

if [[ ! -d "${RGB_DIR}" || ! -d "${DEPTH_DIR}" ]]; then
  echo "ERROR: rgb/depth directories not found"
  echo "  rgb: ${RGB_DIR}"
  echo "  depth: ${DEPTH_DIR}"
  exit 1
fi

mkdir -p "${OUT_ROOT}"

declare -a ALGS=("pca" "ransac" "gaussian" "ceres")

run_one() {
  local alg="$1"
  local out_dir="${OUT_ROOT}/${alg}"
  local timeout_s=110
  if [[ "${alg}" == "ceres" ]]; then
    timeout_s=150
  fi

  mkdir -p "${out_dir}"
  echo "======================================"
  echo "[RUN] algorithm=${alg} out=${out_dir}"
  echo "======================================"

  set +e
  timeout -s INT "${timeout_s}"s ros2 launch "${LAUNCH_FILE}" \
    rgb_dir:="${RGB_DIR}" \
    depth_dir:="${DEPTH_DIR}" \
    algorithm_type:="${alg}" \
    statistic_directory:="${out_dir}"
  local rc=$?
  set -e

  if [[ ${rc} -ne 0 && ${rc} -ne 124 && ${rc} -ne 130 ]]; then
    echo "ERROR: launch failed for ${alg}, rc=${rc}"
    exit ${rc}
  fi

  if [[ -f "${out_dir}/line2d_metrics.csv" ]]; then
    python3 "${ROOT_DIR}/tools/evaluate_line2d.py" \
      --input "${out_dir}/line2d_metrics.csv" \
      --output "${out_dir}/line2d_summary.csv"
  else
    echo "WARN: line2d_metrics.csv missing for ${alg}"
  fi
}

for alg in "${ALGS[@]}"; do
  run_one "${alg}"
done

echo "All done. Output root: ${OUT_ROOT}"
for alg in "${ALGS[@]}"; do
  echo "  - ${OUT_ROOT}/${alg}/metrics.csv"
  echo "  - ${OUT_ROOT}/${alg}/line2d_summary.csv"
done

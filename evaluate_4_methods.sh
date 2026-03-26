#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/install/setup.bash"
set -u

RGB_DIR="${1:-${ROOT_DIR}/image1/rgb}"
DEPTH_DIR="${2:-${ROOT_DIR}/image1/depth}"
OUT_ROOT="${3:-${ROOT_DIR}/statistics/eval4_$(date +%Y%m%d_%H%M%S)}"
GT_IMAGES_DIR="${4:-}"
GT_CAMERA_INFO="${5:-${ROOT_DIR}/config/d457_color.yaml}"
GT_TAG_SIZE="${6:-0.05}"
LAUNCH_FILE="${ROOT_DIR}/launch/launch.py"
PARAM_FILE="${ROOT_DIR}/config/param.yaml"

TIMEOUT_DEFAULT="65"
if [[ -f "${PARAM_FILE}" ]]; then
  timeout_from_param="$(awk -F': ' '/evaluation_timeout_seconds:/{print $2; exit}' "${PARAM_FILE}" | tr -d ' ')"
  if [[ -n "${timeout_from_param}" ]]; then
    TIMEOUT_DEFAULT="${timeout_from_param}"
  fi
fi

GT_EVAL_ENABLED="false"
if [[ -f "${PARAM_FILE}" ]]; then
  gt_enabled_from_param="$(awk -F': ' '/groundtruth_eval_enabled:/{print $2; exit}' "${PARAM_FILE}" | tr -d ' ')"
  gt_images_from_param="$(awk -F': ' '/groundtruth_images_dir:/{print $2; exit}' "${PARAM_FILE}" | tr -d '" ')"
  gt_caminfo_from_param="$(awk -F': ' '/groundtruth_camera_info_file:/{print $2; exit}' "${PARAM_FILE}" | tr -d '" ')"
  gt_tag_from_param="$(awk -F': ' '/groundtruth_tag_size_m:/{print $2; exit}' "${PARAM_FILE}" | tr -d ' ')"

  if [[ -n "${gt_enabled_from_param}" ]]; then
    GT_EVAL_ENABLED="${gt_enabled_from_param}"
  fi
  if [[ -z "${GT_IMAGES_DIR}" && -n "${gt_images_from_param}" ]]; then
    GT_IMAGES_DIR="${ROOT_DIR}/${gt_images_from_param}"
  fi
  if [[ "${GT_CAMERA_INFO}" == "${ROOT_DIR}/config/d457_color.yaml" && -n "${gt_caminfo_from_param}" ]]; then
    if [[ "${gt_caminfo_from_param}" = /* ]]; then
      GT_CAMERA_INFO="${gt_caminfo_from_param}"
    else
      GT_CAMERA_INFO="${ROOT_DIR}/${gt_caminfo_from_param}"
    fi
  fi
  if [[ "${GT_TAG_SIZE}" == "0.05" && -n "${gt_tag_from_param}" ]]; then
    GT_TAG_SIZE="${gt_tag_from_param}"
  fi
fi

if [[ -n "${GT_IMAGES_DIR}" ]]; then
  GT_EVAL_ENABLED="true"
fi

if [[ ! -d "${RGB_DIR}" || ! -d "${DEPTH_DIR}" ]]; then
  echo "ERROR: rgb/depth directories not found"
  echo "  rgb: ${RGB_DIR}"
  echo "  depth: ${DEPTH_DIR}"
  exit 1
fi

mkdir -p "${OUT_ROOT}"

declare -a ALGS=("pca" "ransac" "gaussian" "ceres")

plugin_for_alg() {
  local alg="$1"
  case "$alg" in
    pca) echo "axispose::PoseEstimatePCA" ;;
    ransac) echo "axispose::PoseEstimateRANSAC" ;;
    gaussian) echo "axispose::PoseEstimateGaussian" ;;
    ceres) echo "axispose::PoseEstimateCeres" ;;
    *) return 1 ;;
  esac
}

run_one() {
  local alg="$1"
  local out_dir="${OUT_ROOT}/${alg}"
  local timeout_s="${TIMEOUT_DEFAULT}"
  local plugin
  plugin="$(plugin_for_alg "${alg}")"

  mkdir -p "${out_dir}"
  echo "======================================"
  echo "[RUN] algorithm=${alg} plugin=${plugin} out=${out_dir}"
  echo "======================================"

  set +e
  timeout -s INT "${timeout_s}"s ros2 launch "${LAUNCH_FILE}" \
    rgb_dir:="${RGB_DIR}" \
    depth_dir:="${DEPTH_DIR}" \
    pose_plugin:="${plugin}" \
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

  if [[ "${GT_EVAL_ENABLED}" == "true" ]]; then
    if [[ -z "${GT_IMAGES_DIR}" || ! -d "${GT_IMAGES_DIR}" ]]; then
      echo "WARN: GT eval enabled but GT_IMAGES_DIR is empty or missing, skip ${alg}"
    else
      echo "[GT] Running GT evaluation for ${alg}"
      "${ROOT_DIR}/tools/run_full_eval.sh" \
        "${GT_IMAGES_DIR}" \
        "${GT_CAMERA_INFO}" \
        "${GT_TAG_SIZE}" \
        "${out_dir}/metrics.csv" \
        "${out_dir}/gt_eval"
    fi
  fi
}

for alg in "${ALGS[@]}"; do
  run_one "${alg}"
done

echo "All done. Output root: ${OUT_ROOT}"
for alg in "${ALGS[@]}"; do
  echo "  - ${OUT_ROOT}/${alg}/metrics.csv"
  echo "  - ${OUT_ROOT}/${alg}/line2d_summary.csv"
  if [[ "${GT_EVAL_ENABLED}" == "true" ]]; then
    echo "  - ${OUT_ROOT}/${alg}/gt_eval/evaluation_summary.csv"
  fi
done

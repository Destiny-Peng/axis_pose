#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/install/setup.bash"

PARAM_FILE="${ROOT_DIR}/config/param.yaml"
LAUNCH_FILE="${ROOT_DIR}/launch/launch.py"
RGB_DIR="${ROOT_DIR}/image1/rgb"
DEPTH_DIR="${ROOT_DIR}/image1/depth"

TIMEOUT_DEFAULT="65"
if [[ -f "${PARAM_FILE}" ]]; then
    timeout_from_param="$(awk -F': ' '/evaluation_timeout_seconds:/{print $2; exit}' "${PARAM_FILE}" | tr -d ' ')"
    if [[ -n "${timeout_from_param}" ]]; then
        TIMEOUT_DEFAULT="${timeout_from_param}"
    fi
fi

plugin_for_alg() {
    local alg="$1"
    case "$alg" in
        pca) echo "axispose::PoseEstimatePCA" ;;
        gaussian) echo "axispose::PoseEstimateGaussian" ;;
        ceres) echo "axispose::PoseEstimateCeres" ;;
        *) return 1 ;;
    esac
}

run_method() {
    local method_name=$1
        local out_dir="${ROOT_DIR}/statistics/${method_name}"
    local metrics_csv="${out_dir}/metrics.csv"
        local plugin
        plugin="$(plugin_for_alg "${method_name}")"
    
    echo "=== RUNNING ${method_name^^} ==="

        rm -rf "${out_dir}"
        mkdir -p "${out_dir}"

        echo "Launching ros2 node with timeout ${TIMEOUT_DEFAULT}s..."
    set +e
        timeout -s INT "${TIMEOUT_DEFAULT}"s ros2 launch "${LAUNCH_FILE}" \
                rgb_dir:="${RGB_DIR}" \
                depth_dir:="${DEPTH_DIR}" \
                pose_plugin:="${plugin}" \
                statistic_directory:="${out_dir}"
    local launch_rc=$?
    set -e
    if [ ${launch_rc} -ne 0 ] && [ ${launch_rc} -ne 124 ] && [ ${launch_rc} -ne 130 ]; then
        echo "ros2 launch failed for ${method_name}, rc=${launch_rc}"
        exit ${launch_rc}
    fi
    
    # 然后运行评估脚本
    echo "Evaluating..."
    ./tools/run_full_eval.sh "${RGB_DIR}" config/d457_color.yaml 0.05 "${metrics_csv}" "${out_dir}"

    if [ -f "${out_dir}/line2d_metrics.csv" ]; then
        echo "Evaluating 2D line reprojection..."
        python3 ./tools/evaluate_line2d.py \
            --input "${out_dir}/line2d_metrics.csv" \
            --output "${out_dir}/line2d_summary.csv"
    else
        echo "WARN: ${out_dir}/line2d_metrics.csv not found"
    fi
}

run_method "pca"
run_method "gaussian"
run_method "ceres"

#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/install/setup.bash"

LAUNCH_FILE="${ROOT_DIR}/launch/launch.py"

plugin_for_alg() {
    local alg="$1"
    case "$alg" in
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
    
    echo "=== RUNNING NODE FOR ${method_name^^} ==="
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    
    ros2 launch "${LAUNCH_FILE}" pose_plugin:="${plugin}" statistic_directory:="${out_dir}" &
    ROS_PID=$!
    
    # Rate=1.0, around 50 frames. Keep buffer for startup and shutdown.
    sleep 65
    kill -INT $ROS_PID
    wait $ROS_PID 2>/dev/null
    
    echo "=== EVALUATING ${method_name^^} ==="
    ./tools/run_full_eval.sh image_tag/rgb/ config/d457_color.yaml 0.05 "${metrics_csv}" "${out_dir}"
}

run_method "gaussian"
run_method "ceres"

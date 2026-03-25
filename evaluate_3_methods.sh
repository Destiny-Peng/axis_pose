#!/bin/bash
cd /home/jacy/project/final_design/axispose

# Source workspace
source install/setup.bash

P=/home/jacy/project/final_design/axispose/config/param.yaml
LAUNCH_FILE=/home/jacy/project/final_design/axispose/launch/launch.py
RGB_DIR=/home/jacy/project/final_design/axispose/image1/rgb
DEPTH_DIR=/home/jacy/project/final_design/axispose/image1/depth
# Ensure loop is false
sed -i 's/loop: true/loop: false/' $P
# Ensure use_sacline is false
sed -i 's/use_sacline: true/use_sacline: false/' $P

run_method() {
    local method_name=$1
    local out_dir="statistics/${method_name}"
    local metrics_csv="${out_dir}/metrics.csv"
    
    echo "=== RUNNING ${method_name^^} ==="
    sed -i "s/algorithm_type: .*/algorithm_type: ${method_name}/" $P
    
    # 清理旧的metrics
    rm -rf ${out_dir}
    mkdir -p ${out_dir}

    # 受控运行 launch：给足处理 51 帧时间，超时后发送 SIGINT 平滑退出
    echo "Launching ros2 node with timeout..."
    set +e
    timeout -s INT 95s ros2 launch ${LAUNCH_FILE} \
        rgb_dir:=${RGB_DIR} \
        depth_dir:=${DEPTH_DIR} \
        algorithm_type:=${method_name} \
        statistic_directory:=${out_dir}
    local launch_rc=$?
    set -e
    if [ ${launch_rc} -ne 0 ] && [ ${launch_rc} -ne 124 ] && [ ${launch_rc} -ne 130 ]; then
        echo "ros2 launch failed for ${method_name}, rc=${launch_rc}"
        exit ${launch_rc}
    fi
    
    # 然后运行评估脚本
    echo "Evaluating..."
    ./tools/run_full_eval.sh ${RGB_DIR} config/d457_color.yaml 0.05 ${metrics_csv} ${out_dir}

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

#!/bin/bash
cd /home/jacy/project/final_design/axispose
source install/setup.bash
P=/home/jacy/project/final_design/axispose/config/param.yaml

sed -i 's/loop: true/loop: false/' $P
sed -i 's/use_sacline: true/use_sacline: false/' $P

run_method() {
    local method_name=$1
    local out_dir="statistics/${method_name}"
    local metrics_csv="${out_dir}/metrics.csv"
    
    echo "=== RUNNING NODE FOR ${method_name^^} ==="
    sed -i "s/algorithm_type: .*/algorithm_type: ${method_name}/" $P
    # clean old metric
    rm -rf ${out_dir}
    mkdir -p ${out_dir}
    
    ros2 launch axispose launch.py statistic_directory:=${out_dir} &
    ROS_PID=$!
    
    # Wait strictly for camera to finish reading the 51 frames
    sleep 35
    kill -INT $ROS_PID
    wait $ROS_PID 2>/dev/null
    
    echo "=== EVALUATING ${method_name^^} ==="
    ./tools/run_full_eval.sh image_tag/rgb/ config/d457_color.yaml 0.05 ${metrics_csv} ${out_dir}
}

run_method "gaussian"
run_method "ceres"

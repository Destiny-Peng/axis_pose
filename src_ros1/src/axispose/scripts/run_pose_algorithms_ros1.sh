#!/usr/bin/env bash
set -euo pipefail

# Run all pose algorithms on ROS1 real-time pipeline (camera topics + yolo mask topic).
# Example:
#   ./run_pose_algorithms_ros1.sh --seconds 60 --stats-root /tmp/axispose_eval --no-vis

SECONDS_PER_ALG=60
STATS_ROOT="/tmp/axispose_eval"
USE_MOCK_DRIVER="false"
RUN_VIS="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seconds)
      SECONDS_PER_ALG="$2"; shift 2 ;;
    --stats-root)
      STATS_ROOT="$2"; shift 2 ;;
    --use-mock-driver)
      USE_MOCK_DRIVER="true"; shift 1 ;;
    --no-vis)
      RUN_VIS="false"; shift 1 ;;
    *)
      echo "Unknown arg: $1"
      exit 1 ;;
  esac
done

mkdir -p "$STATS_ROOT"

ALGORITHMS=(pca ransac gaussian ceres)

for alg in "${ALGORITHMS[@]}"; do
  out_dir="$STATS_ROOT/$alg"
  mkdir -p "$out_dir"

  echo "=================================================="
  echo "[ROS1] algorithm=$alg duration=${SECONDS_PER_ALG}s out=$out_dir"
  echo "=================================================="

  # timeout returns non-zero (e.g., 124/130) when the run duration is reached.
  # With "set -e" enabled, we must temporarily disable it to inspect rc manually.
  set +e
  timeout -s INT "${SECONDS_PER_ALG}s" roslaunch axispose ros1_node.launch \
    use_mock_driver:="$USE_MOCK_DRIVER" \
    run_visualization:="$RUN_VIS" \
    pose_algorithm:="$alg" \
    statistics_dir:="$STATS_ROOT"
  rc=$?
  set -e

  if [[ $rc -ne 0 && $rc -ne 124 && $rc -ne 130 ]]; then
    echo "Launch failed for $alg, rc=$rc"
    exit $rc
  fi

  sleep 2
  echo "Done: $alg"
done

echo "All algorithms finished. stats_root=$STATS_ROOT"

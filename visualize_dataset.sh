#!/bin/bash

REPO_ID="yilin404/pick_and_place"
ROOT="/home/yilin/dataset/own_episode_data"
OUTPUT_DIR="./outputs/visualize/2024-12-05"
BATCH_SIZE=8
SAVE=1
EVALUATE=1
POLICY_MODEL_PATH="/home/yilin/simple_robot_ws/outputs/train/2024-12-04/21-57-10_real_world_rdt_default/checkpoints/last/pretrained_model"

START_INDEX=0
END_INDEX=109

for EPISODE_INDEX in $(seq $START_INDEX $END_INDEX)
do
    echo "Processing episode-index: $EPISODE_INDEX"
    python3 src/teleop_ros/simple_hand_teleop/scripts/visualize_lerobot_dataset.py \
        --repo-id $REPO_ID \
        --episode-index $EPISODE_INDEX \
        --root $ROOT \
        --output-dir $OUTPUT_DIR \
        --batch-size $BATCH_SIZE \
        --save $SAVE \
        --evaluate $EVALUATE \
        --policy-model-path $POLICY_MODEL_PATH
done


#!/bin/bash

REPO_ID="yilin404/pick_and_place"
ROOT="/home/yilin/dataset/own_episode_data"
OUTPUT_DIR="./outputs/visualize/2024-11-27"
BATCH_SIZE=8
SAVE=1

START_INDEX=0
END_INDEX=59

for EPISODE_INDEX in $(seq $START_INDEX $END_INDEX)
do
    echo "Processing episode-index: $EPISODE_INDEX"
    python3 src/teleop_ros/simple_hand_teleop/scripts/visualize_lerobot_dataset.py \
        --repo-id $REPO_ID \
        --episode-index $EPISODE_INDEX \
        --root $ROOT \
        --output-dir $OUTPUT_DIR \
        --batch-size $BATCH_SIZE \
        --save $SAVE
done


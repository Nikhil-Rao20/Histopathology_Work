#!/bin/bash

# Unified raw training launcher for all models on all external datasets
# Usage: bash run_raw_training.sh [--sequential|--parallel]

DATA_ROOT="Histopathology_Datasets_Official"
OUTPUT_ROOT="experiments/cipsnet_v2/experiments/raw_training"
EPOCHS=25
BATCH_SIZE=4
NUM_WORKERS=4
DEVICE="cuda"

MODELS=("LAVT" "CRIS" "LVIT" "LVIT_IE" "GROUNDING_DINO")
DATASETS=("CoNSeP" "MoNuSAC" "Lizard" "CoNIC")

# By default run sequentially (safer for GPU memory)
PARALLEL=${1:---sequential}

echo "========================================"
echo "Raw Training Launcher"
echo "========================================"
echo "Models: ${MODELS[@]}"
echo "Datasets: ${DATASETS[@]}"
echo "Mode: $PARALLEL"
echo "Data Root: $DATA_ROOT"
echo "Output: $OUTPUT_ROOT"
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE"
echo "========================================"
echo ""

TOTAL_JOBS=$((${#MODELS[@]} * ${#DATASETS[@]}))
echo "Total jobs: $TOTAL_JOBS"
echo ""

# Array to store background job PIDs (for parallel mode)
declare -a PIDS

JOB_COUNT=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        JOB_COUNT=$((JOB_COUNT + 1))
        
        OUTPUT_DIR="$OUTPUT_ROOT/${model}_${dataset}"
        
        CMD="python3 experiments/cipsnet_v2/external_raw_train.py \
            --model $model \
            --datasets $dataset \
            --data-root $DATA_ROOT \
            --output-dir $OUTPUT_DIR \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --num-workers $NUM_WORKERS \
            --device $DEVICE \
            --seed 42"
        
        echo "[$(printf "%2d/%2d" $JOB_COUNT $TOTAL_JOBS)] Training $model on $dataset..."
        
        if [[ "$PARALLEL" == "--parallel" ]]; then
            # Run in background
            eval "$CMD" > "$OUTPUT_DIR/train.log" 2>&1 &
            PIDS+=($!)
            echo "  → Started (PID: ${PIDS[-1]})"
        else
            # Run sequentially
            echo "  → Starting training..."
            eval "$CMD" > "$OUTPUT_DIR/train.log" 2>&1
            EXIT_CODE=$?
            if [[ $EXIT_CODE -eq 0 ]]; then
                echo "  ✓ Completed successfully"
            else
                echo "  ✗ Failed with exit code $EXIT_CODE (see $OUTPUT_DIR/train.log)"
            fi
        fi
    done
done

if [[ "$PARALLEL" == "--parallel" ]]; then
    echo ""
    echo "All $TOTAL_JOBS jobs started in parallel. Waiting for completion..."
    echo ""
    
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        wait $pid
        EXIT_CODE=$?
        echo "[$(printf "%2d/%2d" $((i+1)) $TOTAL_JOBS)] Job (PID: $pid) completed with exit code $EXIT_CODE"
    done
    
    echo ""
    echo "All parallel jobs completed!"
else
    echo ""
    echo "All sequential jobs completed!"
fi

echo ""
echo "Training summary saved in: $OUTPUT_ROOT"
echo "Done!"

#!/bin/bash

#============================================================================
# Unified Raw Training Launcher - All Models x All External Datasets
#============================================================================
# Trains 5 model variants (LAVT, CRIS, LVIT, LVIT_IE, GROUNDING_DINO) on
# 4 external datasets (CoNSeP, MoNuSAC, Lizard, CoNIC) = 20 total jobs
#
# Usage:
#   bash launch_raw_training.sh                   # All sequential (safe)
#   bash launch_raw_training.sh --parallel N      # N jobs in parallel
#   bash launch_raw_training.sh --sequential      # Force sequential
#
# Each model/dataset combo gets:
#   - 25 epochs
#   - Batch size 4
#   - Output dir: experiments/cipsnet_v2/experiments/raw_training/{MODEL}_{DATASET}
#============================================================================

# Configuration
SCRIPT="experiments/cipsnet_v2/external_raw_train.py"
DATA_ROOT="Histopathology_Datasets_Official"
OUTPUT_ROOT="experiments/cipsnet_v2/experiments/raw_training"
EPOCHS=25
BATCH_SIZE=4
NUM_WORKERS=4
DEVICE="cuda"
SEED=42

# Models and Datasets
MODELS=("LAVT" "CRIS" "LVIT" "LVIT_IE" "GROUNDING_DINO")
DATASETS=("CoNSeP" "MoNuSAC" "Lizard" "CoNIC")

# Parse arguments
MODE="sequential"
PARALLEL_JOBS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            MODE="parallel"
            PARALLEL_JOBS=${2:-4}
            shift 2
            ;;
        --sequential)
            MODE="sequential"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_ROOT"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    RAW TRAINING LAUNCHER v1.0                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Models:               ${MODELS[@]}"
echo "  Datasets:             ${DATASETS[@]}"
echo "  Epochs:               $EPOCHS"
echo "  Batch Size:           $BATCH_SIZE"
echo "  Mode:                 $MODE"
if [[ "$MODE" == "parallel" ]]; then
    echo "  Parallel Jobs:        $PARALLEL_JOBS"
fi
echo "  Device:               $DEVICE"
echo "  Output Root:          $OUTPUT_ROOT"
echo ""

TOTAL_JOBS=$((${#MODELS[@]} * ${#DATASETS[@]}))
echo "Total Training Jobs:    $TOTAL_JOBS"
echo ""

# Job queue
declare -a PIDS
declare -a JOB_NAMES
JOB_COUNT=0
RUNNING_JOBS=0

echo "Starting training jobs..."
echo "─────────────────────────────────────────────────────────────────────────────"

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        JOB_COUNT=$((JOB_COUNT + 1))
        
        OUTPUT_DIR="$OUTPUT_ROOT/${model}_${dataset}"
        mkdir -p "$OUTPUT_DIR"
        LOG_FILE="$OUTPUT_DIR/training.log"
        
        CMD="python3 '$SCRIPT' \
            --model '$model' \
            --datasets '$dataset' \
            --data-root '$DATA_ROOT' \
            --output-dir '$OUTPUT_DIR' \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --num-workers $NUM_WORKERS \
            --device $DEVICE \
            --seed $SEED"
        
        if [[ "$MODE" == "sequential" ]]; then
            # ─────────────── Sequential Mode ───────────────
            printf "[%2d/%2d] Training %-12s on %-8s ... " "$JOB_COUNT" "$TOTAL_JOBS" "$model" "$dataset"
            
            # Run command
            if eval "$CMD" > "$LOG_FILE" 2>&1; then
                echo "✓ DONE"
            else
                EXIT_CODE=$?
                echo "✗ FAILED (exit $EXIT_CODE, see $LOG_FILE)"
            fi
        else
            # ─────────────── Parallel Mode ───────────────
            # Wait if we've reached max parallel jobs
            while [[ $RUNNING_JOBS -ge $PARALLEL_JOBS ]]; do
                for i in "${!PIDS[@]}"; do
                    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                        # Job finished
                        wait "${PIDS[$i]}" 2>/dev/null
                        EXIT_CODE=$?
                        if [[ $EXIT_CODE -eq 0 ]]; then
                            echo "[✓] ${JOB_NAMES[$i]}"
                        else
                            echo "[✗] ${JOB_NAMES[$i]} (exit $EXIT_CODE)"
                        fi
                        
                        # Remove from tracking
                        unset 'PIDS[$i]'
                        unset 'JOB_NAMES[$i]'
                        RUNNING_JOBS=$((RUNNING_JOBS - 1))
                    fi
                done
                sleep 2
            done
            
            # Start new job
            printf "[%2d/%2d] Launching %-12s on %-8s ... " "$JOB_COUNT" "$TOTAL_JOBS" "$model" "$dataset"
            eval "$CMD" > "$LOG_FILE" 2>&1 &
            NEW_PID=$!
            PIDS+=($NEW_PID)
            JOB_NAMES+=("$model x $dataset")
            RUNNING_JOBS=$((RUNNING_JOBS + 1))
            echo "(PID $NEW_PID)"
        fi
    done
done

if [[ "$MODE" == "parallel" ]]; then
    echo ""
    echo "Waiting for all parallel jobs to complete..."
    echo "─────────────────────────────────────────────────────────────────────────────"
    
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        wait "$pid" 2>/dev/null
        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "[✓] ${JOB_NAMES[$i]}"
        else
            echo "[✗] ${JOB_NAMES[$i]} (exit $EXIT_CODE)"
        fi
    done
fi

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "✓ All training jobs completed!"
echo ""
echo "Results Summary:"
echo "  Output directory: $OUTPUT_ROOT"
echo "  Check individual logs: $OUTPUT_ROOT/*/{MODEL}_{DATASET}/training.log"
echo ""

# Count successful runs
SUCCESS_COUNT=$(find "$OUTPUT_ROOT" -name "best.pth" 2>/dev/null | wc -l)
echo "  Completed runs: $SUCCESS_COUNT / $TOTAL_JOBS"
echo ""

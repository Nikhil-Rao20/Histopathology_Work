#!/bin/bash

# Monitor raw training progress

OUTPUT_ROOT="experiments/cipsnet_v2/experiments/raw_training"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    RAW TRAINING  PROGRESS MONITOR                         ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

MODELS=("LAVT" "CRIS" "LVIT" "LVIT_IE" "GROUNDING_DINO")
DATASETS=("CoNSeP" "MoNuSAC" "Lizard" "CoNIC")

TOTAL_JOBS=$((${#MODELS[@]} * ${#DATASETS[@]}))
COMPLETED=0
IN_PROGRESS=0
NOT_STARTED=0

declare -A JOB_STATUS

# Check each job
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        JOB_DIR="$OUTPUT_ROOT/${model}_${dataset}"
        LOG_FILE="$JOB_DIR/training.log"
        
        if [[ ! -d "$JOB_DIR" ]]; then
            JOB_STATUS["${model}_${dataset}"]="NOT_STARTED"
            NOT_STARTED=$((NOT_STARTED + 1))
        elif [[ -f "$JOB_DIR/best.pth" ]]; then
            JOB_STATUS["${model}_${dataset}"]="COMPLETED"
            COMPLETED=$((COMPLETED + 1))
        elif [[ -f "$LOG_FILE" ]] && grep -q "Epoch\|Training\|loss" "$LOG_FILE" 2>/dev/null; then
            JOB_STATUS["${model}_${dataset}"]="IN_PROGRESS"
            IN_PROGRESS=$((IN_PROGRESS + 1))
            
            # Try to extract epoch info
            if tail -20 "$LOG_FILE" 2>/dev/null | grep -q "Epoch"; then
                LAST_EPOCH=$(tail -20 "$LOG_FILE" 2>/dev/null | grep "Epoch" | tail -1 | grep -oP 'Epoch \K[0-9]+' || echo "?")
            else
                LAST_EPOCH="?"
            fi
        else
            JOB_STATUS["${model}_${dataset}"]="QUEUED"
            NOT_STARTED=$((NOT_STARTED + 1))
        fi
    done
done

echo "Summary:"
echo "  Total Jobs:       $TOTAL_JOBS"
echo "  Completed:        $COMPLETED ✓"
echo "  In Progress:      $IN_PROGRESS ⟳"
echo "  Not Started:      $NOT_STARTED ⊘"
echo ""
echo "Status Grid:"
echo "─────────────────────────────────────────────────────────────────────────────"

printf "%-15s" "Model"
for dataset in "${DATASETS[@]}"; do
    printf " %-12s" "$dataset"
done
printf "\n"
printf "%-15s" "───────────"
for dataset in "${DATASETS[@]}"; do
    printf " %-12s" "────────────"
done
printf "\n"

for model in "${MODELS[@]}"; do
    printf "%-15s" "$model"
    for dataset in "${DATASETS[@]}"; do
        STATUS=${JOB_STATUS["${model}_${dataset}"]:-"UNKNOWN"}
        case "$STATUS" in
            "COMPLETED")
                printf " %-12s" "✓ DONE"
                ;;
            "IN_PROGRESS")
                printf " %-12s" "⟳ RUNNING"
                ;;
            "QUEUED")
                printf " %-12s" "⊙ WAIT"
                ;;
            *)
                printf " %-12s" "⊘ INIT"
                ;;
        esac
    done
    printf "\n"
done

echo ""
echo "Recent activity (last 5 seconds):"
find "$OUTPUT_ROOT" -name "training.log" -newermt "5 seconds ago" 2>/dev/null | while read f; do
    dir=$(dirname "$f")
    name=$(basename "$dir")
    echo "  $name: $(tail -1 "$f" 2>/dev/null | head -c 60)..."
done

echo ""
echo "Next monitoring in 60s... (run this script again to update)"

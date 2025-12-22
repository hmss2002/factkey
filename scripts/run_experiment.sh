#!/bin/bash
# =============================================================================
# FactKey Experiment Runner
# Anchor-Cycle Method for Mitigating Reversal Curse
# 
# Hardware: 4*V100 32GB
# Precision: FP16
# =============================================================================

# Strict mode: exit on error, undefined vars, pipe failures
set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
PROJECT_DIR="/mnt/projects/factkey"
CONDA_ENV="factkey"
MODEL_ID="google/gemma-3-1b-pt"
MODEL_CACHE="/mnt/models"

# Training parameters (default: 10 epochs, LoRA r=64)
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"

# Data parameters
N_TRAIN="${N_TRAIN:-5000}"
N_TEST_SEEN="${N_TEST_SEEN:-1000}"

# Distributed training
NUM_GPUS="${NUM_GPUS:-4}"

# Master port (use same port for all, run sequentially)
MASTER_PORT=29500

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_gpu() {
    # Wait until no GPU processes are running
    log "Waiting for GPU resources..."
    while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
        sleep 5
    done
    log "GPUs are free, continuing..."
}

run_sequential() {
    # Run a command and wait for completion
    local desc="$1"
    shift
    log "Starting: $desc"
    "$@"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log "ERROR: $desc failed with exit code $exit_code"
        exit $exit_code
    fi
    log "Completed: $desc"
    return 0
}

# =============================================================================
# Setup
# =============================================================================
cd "$PROJECT_DIR"

echo "=============================================================="
echo "FactKey Experiment - Reversal Curse Mitigation"
echo "=============================================================="
echo "Project directory: $PROJECT_DIR"
echo "Conda environment: $CONDA_ENV"
echo "Model: $MODEL_ID"
echo "GPUs: $NUM_GPUS"
echo "LoRA rank: $LORA_R"
echo "LoRA alpha: $LORA_ALPHA"
echo "Epochs: $EPOCHS"
echo "=============================================================="
echo ""

# Activate conda environment
source /mnt/envs/anaconda3/bin/activate "$CONDA_ENV"

# Create output directories
mkdir -p outputs/runs outputs/tables outputs/figures

# =============================================================================
# Step 1: Generate Synthetic Data
# =============================================================================
log ""
log "[Step 1/5] Generating synthetic data..."
log "=============================================================="

run_sequential "Data Generation" \
    python scripts/generate_data.py \
        --out_dir data/raw \
        --processed_dir data/processed \
        --n_train "$N_TRAIN" \
        --n_test_seen "$N_TEST_SEEN" \
        --seed 42

# =============================================================================
# Step 2: Train Baseline Model
# =============================================================================
log ""
log "[Step 2/5] Training baseline model (no anchors)..."
log "=============================================================="

BASELINE_DIR="outputs/runs/gemma3_1b_baseline_lora_r${LORA_R}"

run_sequential "Baseline Training" \
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
        scripts/train.py \
        --model_id "$MODEL_ID" \
        --model_cache_dir "$MODEL_CACHE" \
        --train_jsonl data/processed/train_baseline.jsonl \
        --output_dir "$BASELINE_DIR" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --per_device_batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --fp16 \
        --seed 42

# Clear any GPU cache
sleep 5

# =============================================================================
# Step 3: Train Anchor Model
# =============================================================================
log ""
log "[Step 3/5] Training anchor model (with anchors)..."
log "=============================================================="

ANCHOR_DIR="outputs/runs/gemma3_1b_anchor_lora_r${LORA_R}"

run_sequential "Anchor Training" \
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
        scripts/train.py \
        --model_id "$MODEL_ID" \
        --model_cache_dir "$MODEL_CACHE" \
        --train_jsonl data/processed/train_anchor.jsonl \
        --output_dir "$ANCHOR_DIR" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --per_device_batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --fp16 \
        --seed 42

# Clear any GPU cache
sleep 5

# =============================================================================
# Step 4: Evaluate Both Models
# =============================================================================
log ""
log "[Step 4/5] Evaluating models..."
log "=============================================================="

# Evaluate baseline on test set (forward + reverse)
log "Evaluating baseline model..."
run_sequential "Baseline Evaluation" \
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
        scripts/evaluate.py \
        --model_dir "$BASELINE_DIR" \
        --base_model_id "$MODEL_ID" \
        --model_cache_dir "$MODEL_CACHE" \
        --test_jsonl data/raw/test_seen.jsonl \
        --mode both \
        --output_file outputs/tables/eval_baseline_test.json \
        --fp16

# Clear GPU cache
sleep 5

# Evaluate anchor on test set (forward + reverse)
log "Evaluating anchor model..."
run_sequential "Anchor Evaluation" \
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
        scripts/evaluate.py \
        --model_dir "$ANCHOR_DIR" \
        --base_model_id "$MODEL_ID" \
        --model_cache_dir "$MODEL_CACHE" \
        --test_jsonl data/raw/test_seen.jsonl \
        --mode both \
        --output_file outputs/tables/eval_anchor_test.json \
        --fp16

# =============================================================================
# Step 5: Compare Results and Generate Visualizations
# =============================================================================
log ""
log "[Step 5/5] Generating comparison charts and tables..."
log "=============================================================="

run_sequential "Result Comparison" \
    python scripts/compare_results.py \
        --result_files \
            outputs/tables/eval_baseline_test.json \
            outputs/tables/eval_anchor_test.json \
        --output_dir outputs \
        --experiment_name "factkey_experiment"

# =============================================================================
# Summary
# =============================================================================
log ""
log "=============================================================="
log "Experiment Complete!"
log "=============================================================="
log ""
log "Models saved to:"
log "  - Baseline: $BASELINE_DIR"
log "  - Anchor:   $ANCHOR_DIR"
log ""
log "Results saved to:"
log "  - Tables:  outputs/tables/"
log "  - Figures: outputs/figures/"
log ""
log "View the report: outputs/tables/factkey_experiment_report.md"
log "=============================================================="

# Final status
log ""
log "Experiment finished successfully at $(date)"

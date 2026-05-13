#!/bin/bash

# IMAGE neve
IMAGE_NAME="dental_pipeline_img"

# 1. Buildeljük az image-et, ha még nem létezik (vagy ha módosítottad a kódot)
echo "=== Konténer buildelése ==="
docker build -t $IMAGE_NAME .

# 2. Host mappák abszolút elérési útjainak meghatározása (hogy bárhonnan futtatható legyen)
BASE_DIR=$(pwd)
INPUT_DIR="$BASE_DIR/input"
OUTPUT_DIR="$BASE_DIR/output"
LOG_DIR="$BASE_DIR/log"
MODELS_DIR="$BASE_DIR/models"

# Mappák létrehozása, ha nem léteznek
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# 3. Konténer futtatása
# Paraméterek testreszabása:
# --draw_caries : Csak a piros szuvasakat rajzolja. Ha mindkettőt akarod, tedd mellé a --draw_healthy -t is.
# Ha WandB-t akarsz használni, add meg a --use_wandb kapcsolót, és add át az API kulcsot környezeti változóként (-e WANDB_API_KEY)
echo "=== Pipeline indítása ==="
docker run --rm \
    --gpus all \
    -v "$INPUT_DIR:/input" \
    -v "$OUTPUT_DIR:/output" \
    -v "$LOG_DIR:/log" \
    -v "$MODELS_DIR:/models" \
    -e WANDB_API_KEY="IDE_MASOLD_A_WANDB_KULCSODAT_HA_KELL" \
    $IMAGE_NAME \
    --draw_caries \
    --draw_healthy \
    --use_wandb
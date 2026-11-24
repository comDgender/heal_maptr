#!/usr/bin/env bash
# Extract ImageNet class tars into train/<wnid>/*.JPEG layout for torchvision.
set -euo pipefail

ROOT="${1:-/home/users/rui.xu/workspace/heal_blackduck/HEAL/org_data/imagenet}"
TRAIN_DIR="${ROOT}/train"
REMOVE_TAR="${REMOVE_TAR:-0}"

mkdir -p "${TRAIN_DIR}"

mapfile -t TARS < <(find "${ROOT}" -maxdepth 1 -name 'n*.tar' | sort)
TOTAL=${#TARS[@]}

if [[ "${TOTAL}" -eq 0 ]]; then
    echo "No class tars found under ${ROOT}"
    exit 1
fi

echo "Extracting ${TOTAL} class archives to ${TRAIN_DIR}"
DONE=0
SKIP=0

for TAR in "${TARS[@]}"; do
    WNID=$(basename "${TAR}" .tar)
    OUT_DIR="${TRAIN_DIR}/${WNID}"

    if [[ -d "${OUT_DIR}" ]] && [[ -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]]; then
        SKIP=$((SKIP + 1))
        DONE=$((DONE + 1))
        if (( DONE % 50 == 0 || DONE == TOTAL )); then
            echo "progress: ${DONE}/${TOTAL} (skipped existing: ${SKIP})"
        fi
        continue
    fi

    mkdir -p "${OUT_DIR}"
    tar -xf "${TAR}" -C "${OUT_DIR}"

    if [[ "${REMOVE_TAR}" == "1" ]]; then
        rm -f "${TAR}"
    fi

    DONE=$((DONE + 1))
    if (( DONE % 10 == 0 || DONE == TOTAL )); then
        echo "progress: ${DONE}/${TOTAL}"
    fi
done

IMG_COUNT=$(find "${TRAIN_DIR}" -type f -name '*.JPEG' 2>/dev/null | wc -l)
echo "Done. class dirs: $(find "${TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l), images: ${IMG_COUNT}"

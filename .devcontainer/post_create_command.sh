#!/usr/bin/env bash
set -euo pipefail

CONTAINER_USER="${1:-$CONTAINER_USER}"
CONTAINER_HOME="${2:-$CONTAINER_HOME}"
CONTAINER_WORK_DIR="${3:-$CONTAINER_WORK_DIR}"
CONTAINER_UV_GROUP="${4:-$CONTAINER_UV_GROUP}"

if ! id "${CONTAINER_USER}" &>/dev/null; then
    echo "User ${CONTAINER_USER} does not exist" >&2
    exit 1
elif [[ ! -d "${CONTAINER_HOME}" ]]; then
    echo "Home directory ${CONTAINER_HOME} does not exist" >&2
    exit 1
elif [[ ! -d "${CONTAINER_WORK_DIR}" ]]; then
    echo "Work directory ${CONTAINER_WORK_DIR} does not exist" >&2
    exit 1
fi

echo "[INFO] step 1/4: change ownership to ${CONTAINER_USER}"
sudo bash "${CONTAINER_WORK_DIR}/.devcontainer/sjsh/root/change_owner.sh" "${CONTAINER_USER}" \
    --target "${CONTAINER_HOME}" \
    --target "${CONTAINER_WORK_DIR}:${CONTAINER_WORK_DIR}/.datasets" \
    --target "${CONTAINER_WORK_DIR}/.datasets:${CONTAINER_WORK_DIR}/.datasets/pills:${CONTAINER_WORK_DIR}/.datasets/ILSVRC:${CONTAINER_WORK_DIR}/.datasets/asr-rankformer-datasets"

echo "[INFO] step 2/4: sync uv"
bash "${CONTAINER_WORK_DIR}/.devcontainer/sjsh/common/wait_for_dir.sh" "${CONTAINER_WORK_DIR}/.venv"
bash "${CONTAINER_WORK_DIR}/.devcontainer/sjsh/user/sync_uv.sh" "${CONTAINER_WORK_DIR}" "${CONTAINER_UV_GROUP}"

echo "[INFO] step 3/4: setup lhotse"
source "${CONTAINER_WORK_DIR}/.venv/bin/activate"
lhotse install-sph2pipe

echo "[INFO] step 4/4: link datasets"
ln -sfn "${CONTAINER_DATA_DIR}/.datasets" "${CONTAINER_WORK_DIR}/.datasets"

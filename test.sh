#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "./predict.py" ]; then
    python ./predict.py "$@"
elif [ -f "./code/src/predict.py" ]; then
    python ./code/src/predict.py "$@"
else
    echo "无法找到 predict.py 或 code/src/predict.py" >&2
    exit 1
fi

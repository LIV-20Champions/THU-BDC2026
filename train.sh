#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "./train.py" ]; then
    python ./train.py "$@"
elif [ -f "./code/src/train.py" ]; then
    python ./code/src/train.py "$@"
else
    echo "无法找到 train.py 或 code/src/train.py" >&2
    exit 1
fi

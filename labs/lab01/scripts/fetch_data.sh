#!/bin/bash
# Downloads the MPEG7 dataset and extracts it into the target directory.
# Usage: ./download_mpeg7.sh [target_dir]   (default: ../data)

set -euo pipefail

# Source URL of the dataset archive
URL="https://dabi.temple.edu/external/shape/MPEG7/MPEG7dataset.zip"

# Target directory (first argument or default)
TARGET_DIR="${1:-../data}"

# Temporary archive path, removed on exit
ZIP_FILE="$(mktemp --suffix=.zip)"
trap 'rm -f "$ZIP_FILE"' EXIT

mkdir -p "$TARGET_DIR"
curl -fL -o "$ZIP_FILE" "$URL"
unzip -q -o "$ZIP_FILE" -d "$TARGET_DIR"

echo "Done. Files extracted to $TARGET_DIR"
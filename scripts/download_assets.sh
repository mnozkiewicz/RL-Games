#!/usr/bin/env bash
set -e

ASSETS_LINK="https://drive.google.com/drive/folders/1-DI5GME0B-mXaPTMFrztL4_QGkS5c8Z-?usp=sharing"
TARGET_DIR="./assets"

echo "Downloading assets from Google Drive..."
mkdir -p "$TARGET_DIR"

gdown --folder "$ASSETS_LINK" --output "$TARGET_DIR" --fuzzy

echo "Done! Files saved in $TARGET_DIR"
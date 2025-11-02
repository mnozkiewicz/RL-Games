
#!/usr/bin/env bash
set -e

WEIGHTS_LINK="https://drive.google.com/drive/folders/1AFCSASPzFUkPk6xseTsKVdwC_a9HkIBj?usp=sharing"
TARGET_DIR="./weights"

echo "Downloading weights from Google Drive..."
mkdir -p "$TARGET_DIR"

gdown --folder "$WEIGHTS_LINK" --output "$TARGET_DIR" --fuzzy

echo "Done! Files saved in $TARGET_DIR"
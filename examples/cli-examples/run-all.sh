set -e

DOMAIN_NAME="cli-example"
TARGET_DIR="downloaded_model"
FILE_NAME="model.joblib"

echo "\n🔵  Training a model...\n"
uv run python model.py

echo "\n🔵  Uploading the model via the CLI...\n"
MODEL_ID=$(uv run python -m modelstore upload "$DOMAIN_NAME" "$FILE_NAME")

echo "\n🔵  Downloading model=$MODEL_ID via the CLI...\n"
mkdir -p "$TARGET_DIR"
uv run python -m modelstore download "$DOMAIN_NAME" "$MODEL_ID" "$TARGET_DIR"

echo "\n✅  Done! Cleaning up..."

rm -rf "$TARGET_DIR"
rm -rf "$FILE_NAME"

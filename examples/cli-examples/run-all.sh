set -e

DOMAIN_NAME="cli-example"
FILE_NAME="model.joblib"

echo "\n🔵  Training a model...\n"
python model.py

echo "\n🔵  Uploading the model via the CLI...\n"
MODEL_ID=$(python -m modelstore upload "$DOMAIN_NAME" "$FILE_NAME")

echo "\n🔵  Downloading model=$MODEL_ID via the CLI...\n"
mkdir -p downloaded_model
python -m modelstore download "$DOMAIN_NAME" "$MODEL_ID" downloaded_model/

echo "\n✅  Done!"

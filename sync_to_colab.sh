#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ZIP_NAME="srl_iot_outputs.zip"
OUTPUT_DIR="outputs"
GITIGNORE_FILE=".gitignore"

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Error: '$OUTPUT_DIR' directory not found. Please generate outputs before syncing."
  exit 1
fi

if [ -f "$ZIP_NAME" ]; then
  echo "Removing existing '$ZIP_NAME'..."
  rm -f "$ZIP_NAME"
fi

echo "Zipping '$OUTPUT_DIR' into '$ZIP_NAME'..."
zip -r -v "$ZIP_NAME" "$OUTPUT_DIR" -x "*.DS_Store"

echo "Checking '$GITIGNORE_FILE' for zip and outputs exclusions..."
if [ ! -f "$GITIGNORE_FILE" ]; then
  echo "# Auto-generated ignore rules" > "$GITIGNORE_FILE"
fi

for pattern in "*.zip" "$OUTPUT_DIR/"; do
  if ! grep -Fxq "$pattern" "$GITIGNORE_FILE"; then
    echo "$pattern" >> "$GITIGNORE_FILE"
    echo "Added '$pattern' to $GITIGNORE_FILE"
  fi
done

current_branch="$(git rev-parse --abbrev-ref HEAD)"

echo "Staging repository changes (excluding '$ZIP_NAME' and '$OUTPUT_DIR')..."
git add .
git reset -- "$ZIP_NAME" "$OUTPUT_DIR" || true

read -rp "Commit message: " commit_message
if [ -z "$commit_message" ]; then
  echo "Commit message cannot be empty. Aborting."
  exit 1
fi

echo "Committing changes..."
git commit -m "$commit_message"

echo "Pushing to origin/$current_branch..."
git push origin "$current_branch"

echo "Code pushed to GitHub. Model weights zipped as $ZIP_NAME. Upload the ZIP to Google Drive now."

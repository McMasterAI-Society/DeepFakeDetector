#!/usr/bin/env bash

# Sync local data into a Hugging Face bucket.
# Defaults are loaded from backend/.env when present.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".env" ]; then
  while IFS= read -r raw_line || [ -n "$raw_line" ]; do
    line="${raw_line%$'\r'}"
    case "$line" in
      ''|'#'*) continue ;;
    esac

    if [[ "$line" == *=* ]]; then
      key="${line%%=*}"
      value="${line#*=}"
      key="$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

      if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        continue
      fi

      if [[ "$value" =~ ^\".*\"$ ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "$value" =~ ^\'.*\'$ ]]; then
        value="${value:1:${#value}-2}"
      else
        value="$(echo "$value" | sed 's/[[:space:]]#.*$//;s/[[:space:]]*$//')"
      fi

      export "$key=$value"
    fi
  done < ./.env
fi

BUCKET_URI="${HF_BUCKET_URI:-hf://buckets/<username>/<bucket-name>}"
LOCAL_DIR="${1:-${HF_BUCKET_LOCAL_DIR:-./data}}"
DELETE_FLAG="${HF_BUCKET_DELETE:-false}"

# Force host-safe HF cache path for CLI operations.
HF_HOME_HOST_DEFAULT="${HOME:-$PWD}/.cache/huggingface"
HF_HOME="${HF_HOME_HOST:-${DEPLOY_HF_HOME:-$HF_HOME_HOST_DEFAULT}}"
export HF_HOME

if [[ "$BUCKET_URI" == *"<username>"* ]] || [[ "$BUCKET_URI" == *"<bucket-name>"* ]]; then
  echo "ERROR: HF_BUCKET_URI is still a placeholder: $BUCKET_URI"
  echo "Set HF_BUCKET_URI in backend/.env to your real bucket URI."
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: Hugging Face CLI (hf) is not installed."
  echo "Install guide: https://hf.co/docs/huggingface_hub/guides/cli"
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: Hugging Face CLI is not authenticated. Run: hf auth login"
  exit 1
fi

if [ ! -d "$LOCAL_DIR" ]; then
  echo "ERROR: Local directory does not exist: $LOCAL_DIR"
  exit 1
fi

echo "Syncing local directory to HF bucket"
echo "  Local : $LOCAL_DIR"
echo "  Bucket: $BUCKET_URI"

if [ "$DELETE_FLAG" = "true" ]; then
  echo "  Mode  : mirror (delete remote files not present locally)"
  hf sync "$LOCAL_DIR" "$BUCKET_URI" --delete
else
  echo "  Mode  : additive (no remote deletes)"
  hf sync "$LOCAL_DIR" "$BUCKET_URI"
fi

echo "Bucket sync completed successfully."

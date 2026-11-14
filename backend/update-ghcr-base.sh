#!/usr/bin/env bash

set -euo pipefail

IMAGE_TAG="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_PATH="${ENV_PATH:-$SCRIPT_DIR/.env}"

get_env_value() {
  local path="$1"
  local key="$2"

  [ -f "$path" ] || return 1

  awk -F= -v k="$key" '
    $0 ~ "^[[:space:]]*"k"[[:space:]]*=" {
      v = substr($0, index($0, "=") + 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
      if (v !~ /^".*"$/ && v !~ /^\047.*\047$/) {
        sub(/[[:space:]]+#.*/, "", v)
      }
      gsub(/^"|"$/, "", v)
      gsub(/^\047|\047$/, "", v)
      print v
      exit
    }
  ' "$path"
}

set_env_value() {
  local path="$1"
  local key="$2"
  local value="$3"
  local line="$key=$value"

  if [ ! -f "$path" ]; then
    printf "%s\n" "$line" > "$path"
    return
  fi

  local tmp
  tmp="$(mktemp)"
  awk -v k="$key" -v line="$line" '
    BEGIN { done = 0 }
    {
      if ($0 ~ "^[[:space:]]*"k"[[:space:]]*=") {
        print line
        done = 1
      } else {
        print $0
      }
    }
    END {
      if (!done) print line
    }
  ' "$path" > "$tmp"
  mv "$tmp" "$path"
}

is_valid_ghcr_tag() {
  local tag="$1"
  [[ "$tag" =~ ^ghcr\.io/[A-Za-z0-9._-]+/[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$ ]]
}

echo "DeepFake Detector - GHCR Base Image Updater"
echo "==========================================="
echo

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is not installed or not on PATH."
  exit 1
fi

current_env_tag="$(get_env_value "$ENV_PATH" "HF_BASE_IMAGE" || true)"

if [ -z "$IMAGE_TAG" ] && [ -n "$current_env_tag" ] && is_valid_ghcr_tag "$current_env_tag"; then
  IMAGE_TAG="$current_env_tag"
  echo "Using HF_BASE_IMAGE from .env: $IMAGE_TAG"
fi

if [ -z "$IMAGE_TAG" ]; then
  echo "No valid GHCR image tag found in .env (HF_BASE_IMAGE)."
  read -r -p "Enter GHCR image tag (example: ghcr.io/lukhsaankumar/deepfake-detector-backend-base:latest). Leave blank for guided setup: " IMAGE_TAG

  if [ -z "$IMAGE_TAG" ]; then
    echo
    echo "Guided setup:"
    echo "1) Create a GitHub PAT (classic) with scopes: write:packages, read:packages"
    echo "   URL: https://github.com/settings/tokens"
    echo "2) Run: docker login ghcr.io"
    echo "3) Use image tag format: ghcr.io/<github-username>/deepfake-detector-backend-base:latest"
    echo

    read -r -p "Enter your GitHub username: " gh_user
    if [ -z "$gh_user" ]; then
      echo "ERROR: GitHub username is required."
      exit 1
    fi

    default_tag="ghcr.io/$gh_user/deepfake-detector-backend-base:latest"
    read -r -p "Image tag to use [Press Enter for $default_tag]: " manual_tag
    if [ -z "$manual_tag" ]; then
      IMAGE_TAG="$default_tag"
    else
      IMAGE_TAG="$manual_tag"
    fi
  fi
fi

if ! is_valid_ghcr_tag "$IMAGE_TAG"; then
  echo "ERROR: Invalid GHCR image tag: '$IMAGE_TAG'"
  echo "Expected format: ghcr.io/<user>/<name>:<tag>"
  exit 1
fi

if [ "$current_env_tag" != "$IMAGE_TAG" ]; then
  set_env_value "$ENV_PATH" "HF_BASE_IMAGE" "$IMAGE_TAG"
  echo "Updated HF_BASE_IMAGE in $ENV_PATH"
else
  echo "HF_BASE_IMAGE already set in $ENV_PATH"
fi

read -r -p "Run 'docker login ghcr.io' now? (Y/n): " should_login
if [ -z "$should_login" ] || [[ "$should_login" =~ ^[Yy](es)?$ ]]; then
  echo "GHCR login uses GitHub credentials:"
  echo "- Username: your GitHub username"
  echo "- Password: your GitHub PAT (classic)"
  echo "- PAT URL: https://github.com/settings/tokens"
  docker login ghcr.io
fi

echo
echo "Building base image: $IMAGE_TAG"
(
  cd "$REPO_ROOT"
  docker build -f backend/Dockerfile.base -t "$IMAGE_TAG" backend
)

echo
echo "Pushing base image: $IMAGE_TAG"
docker push "$IMAGE_TAG"

echo
echo "Success. Base image updated in GHCR."
echo "HF_BASE_IMAGE is now set to: $IMAGE_TAG"
echo
echo "Next step:"
echo "bash ./backend/deploy-to-hf.sh"

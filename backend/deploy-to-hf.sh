#!/usr/bin/env bash

# DeepFake Detector - Hugging Face Spaces Deployment Script (Bash) - to be tested on Linux/macOS/Git-Bash environments
# Deploys backend to HF Spaces without modifying the main GitHub repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from .env if present.
if [ -f ".env" ]; then
  while IFS= read -r raw_line || [ -n "$raw_line" ]; do
    line="${raw_line%$'\r'}"
    case "$line" in
      ''|'#'*) continue ;;
    esac

    if [[ "$line" == *=* ]]; then
      key="${line%%=*}"
      value="${line#*=}"

      # Trim whitespace around key.
      key="$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

      # Skip invalid keys.
      if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        continue
      fi

      # Remove one layer of optional surrounding quotes from value.
      if [[ "$value" =~ ^\".*\"$ ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "$value" =~ ^\'.*\'$ ]]; then
        value="${value:1:${#value}-2}"
      else
        # For unquoted values, strip trailing inline comments and right-side whitespace.
        value="$(echo "$value" | sed 's/[[:space:]]#.*$//;s/[[:space:]]*$//')"
      fi

      export "$key=$value"
    fi
  done < ./.env
fi

# Required/overridable environment variables.
HF_SPACE_URL="${HF_SPACE_URL:-}"
HF_SPACE_WEB_URL="${HF_SPACE_WEB_URL:-}"
HF_SPACE_APP_URL="${HF_SPACE_APP_URL:-}"
HF_DEPLOY_DIR="${HF_DEPLOY_DIR:-${TMPDIR:-/tmp}/hf-deployment/DeepFakeDetectorBackend}"

# Normalize common Windows-style temp paths so values like "$TEMP\hf-deployment\..."
# map to a real Bash path instead of becoming a literal folder name.
case "$HF_DEPLOY_DIR" in
  '$TEMP\'*)
    HF_DEPLOY_DIR="${HF_DEPLOY_DIR#\$TEMP\\}"
    HF_DEPLOY_DIR="${HF_DEPLOY_DIR//\\/\/}"
    HF_DEPLOY_DIR="${TMPDIR:-/tmp}/${HF_DEPLOY_DIR}"
    ;;
  '%TEMP%\'*)
    HF_DEPLOY_DIR="${HF_DEPLOY_DIR#%TEMP%\\}"
    HF_DEPLOY_DIR="${HF_DEPLOY_DIR//\\/\/}"
    HF_DEPLOY_DIR="${TMPDIR:-/tmp}/${HF_DEPLOY_DIR}"
    ;;
  *)
    HF_DEPLOY_DIR="${HF_DEPLOY_DIR//\\/\/}"
    ;;
esac

if [ -z "$HF_SPACE_URL" ]; then
  echo "ERROR: HF_SPACE_URL is not set. Add it to .env or export it first."
  exit 1
fi

if [ -z "$HF_SPACE_WEB_URL" ]; then
  HF_SPACE_WEB_URL="$HF_SPACE_URL"
fi

if [ -z "$HF_SPACE_APP_URL" ]; then
  echo "WARNING: HF_SPACE_APP_URL is not set. Final app URL output will be skipped."
fi

echo "DeepFake Detector - HF Spaces Deployment"
echo "========================================"

echo
echo "Checking Hugging Face CLI..."
if ! command -v hf >/dev/null 2>&1; then
  echo "Hugging Face CLI not found."
  read -r -p "Install it now? (y/N) " install_cli
  if [ "$install_cli" = "y" ] || [ "$install_cli" = "Y" ]; then
    if command -v python >/dev/null 2>&1; then
      python_cmd="python"
    elif command -v python3 >/dev/null 2>&1; then
      python_cmd="python3"
    else
      echo "ERROR: Python is not available, so the Hugging Face CLI cannot be installed automatically."
      echo "Install it manually from: https://hf.co/docs/huggingface_hub/guides/cli"
      exit 1
    fi

    if ! "$python_cmd" -m pip --version >/dev/null 2>&1; then
      echo "Bootstrapping pip for $python_cmd..."
      if ! "$python_cmd" -m ensurepip --upgrade >/dev/null 2>&1; then
        echo "WARNING: pip is not available and ensurepip failed."
        echo "Trying the official Hugging Face CLI installer instead..."
        if command -v curl >/dev/null 2>&1; then
          if ! curl -fsSL https://hf.co/cli/install.sh | bash; then
            echo "ERROR: Hugging Face CLI installer failed."
            echo "Install it manually from: https://hf.co/docs/huggingface_hub/guides/cli"
            exit 1
          fi
        else
          echo "ERROR: curl is not available, so the Hugging Face CLI cannot be installed automatically."
          echo "Install it manually from: https://hf.co/docs/huggingface_hub/guides/cli"
          exit 1
        fi
      else
        "$python_cmd" -m pip install -U "huggingface_hub[cli]"
      fi
    fi
    if ! command -v hf >/dev/null 2>&1; then
      if [ -x "$HOME/.local/bin/hf" ]; then
        export PATH="$HOME/.local/bin:$PATH"
        hash -r 2>/dev/null || true
      fi
    fi
    if ! command -v hf >/dev/null 2>&1; then
      echo "ERROR: Hugging Face CLI is still not available after installation."
      echo "You may need to restart your shell, then run the script again."
      exit 1
    fi
  else
    echo "Install it manually from: https://hf.co/docs/huggingface_hub/guides/cli"
    exit 1
  fi
fi
echo "SUCCESS: Hugging Face CLI is installed"

echo
echo "Checking Hugging Face authentication..."
if ! hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face is not authenticated in this shell."
  read -r -p "Run 'hf auth login' now? (y/N) " login_now
  if [ "$login_now" = "y" ] || [ "$login_now" = "Y" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then
      hf auth login --token "$HF_TOKEN" --add-to-git-credential
    else
      hf auth login
    fi
  else
    echo "ERROR: Hugging Face authentication is required to push to the Space."
    exit 1
  fi
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: Hugging Face authentication still not available after login."
  exit 1
fi

echo "SUCCESS: Hugging Face authentication is configured"

# Resolve token for git operations (prefer explicit HF_TOKEN, then CLI cache).
HF_GIT_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_GIT_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_GIT_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
fi

BACKEND_DIR="$PWD"
echo
echo "Backend directory: $BACKEND_DIR"

echo
echo "This will:"
echo "  1. Clone your HF Space to: $HF_DEPLOY_DIR"
echo "  2. Copy files from: $BACKEND_DIR"
echo "  3. Push to Hugging Face Spaces"
echo "  4. NOT affect your GitHub repository"
read -r -p "Continue? (y/N) " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
  echo "Deployment cancelled"
  exit 0
fi

echo
echo "Creating deployment directory..."
DEPLOY_PARENT="$(dirname "$HF_DEPLOY_DIR")"
REPO_DIR_NAME="$(basename "$HF_DEPLOY_DIR")"
if [ -d "$HF_DEPLOY_DIR" ]; then
  echo "WARNING: Deployment directory exists. Removing..."
  rm -rf "$HF_DEPLOY_DIR"
fi
mkdir -p "$DEPLOY_PARENT"

echo
echo "Cloning HF Space..."
(
  cd "$DEPLOY_PARENT"
  git clone "$HF_SPACE_URL" "$REPO_DIR_NAME"
)
echo "SUCCESS: HF Space cloned successfully"

echo
echo "Copying backend files..."

exclude_items=(
  ".git"
  ".gitignore"
  "__pycache__"
  ".pytest_cache"
  ".env"
  ".hf_cache"
  "testimages"
  "tests"
  "deploy-to-hf.ps1"
  "deploy-to-hf.sh"
  ".railwayignore"
  "railway.toml"
)

should_exclude() {
  local item="$1"
  for exclude in "${exclude_items[@]}"; do
    if [ "$item" = "$exclude" ]; then
      return 0
    fi
  done
  return 1
}

for path in ./*; do
  item="$(basename "$path")"
  if should_exclude "$item"; then
    continue
  fi
  cp -R "$path" "$HF_DEPLOY_DIR/"
  echo "  Copied: $item"
done

echo
echo "Committing and pushing to HF Space..."
(
  cd "$HF_DEPLOY_DIR"
  git add .
  timestamp="$(date "+%Y-%m-%d %H:%M:%S")"
  if git commit -m "Deploy DeepFake Detector API - $timestamp"; then
    echo "SUCCESS: Changes committed"
  else
    echo "INFO: No changes to commit"
  fi

  if [ -n "$HF_GIT_TOKEN" ]; then
    # Push using a temporary askpass helper so the token is not passed as a git arg.
    askpass_file="$(mktemp)"
    cat > "$askpass_file" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  *Username*) echo "hf" ;;
  *Password*) echo "$HF_GIT_TOKEN" ;;
  *) echo "" ;;
esac
EOF
    chmod 700 "$askpass_file"
    GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="$askpass_file" HF_GIT_TOKEN="$HF_GIT_TOKEN" git push
    rm -f "$askpass_file"
  else
    echo "WARNING: No HF token found for non-interactive git auth; falling back to normal git push."
    git push
  fi
)

echo
echo "========================================"
echo "DEPLOYMENT SUCCESSFUL!"
echo "========================================"

echo
echo "Your backend is now deploying to Hugging Face Spaces!"
echo
echo "Next Steps:"
echo "  1. Visit: $HF_SPACE_WEB_URL"
echo "  2. Go to Settings -> Repository secrets"
echo "  3. Add secret: GOOGLE_API_KEY = <your-gemini-api-key>"
echo "  4. Wait 5-10 minutes for the build to complete"
echo "  5. Test your API at the /docs endpoint"

if [ -n "$HF_SPACE_APP_URL" ]; then
  echo
  echo "Your API will be available at:"
  echo "  $HF_SPACE_APP_URL"
fi

echo
echo "Deployment files are in: $HF_DEPLOY_DIR"
echo "(This is separate from your GitHub repo)"

echo
echo "Your GitHub repository was NOT modified!"
echo

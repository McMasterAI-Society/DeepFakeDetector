# Bucket Data

Local staging folder for persistent storage in your HF Space.

**Usage**: Files sync to the Space bucket via `./sync-bucket.sh` (auto-run on deploy).

**Optional contents**:
- `models/` – model artifacts
- `cache-seed/` – Hugging Face cache seeds
- `metadata/` – JSON/CSV files for app logic

**Notes**:
- Folder is optional; empty deploys work fine.
- Runtime caches go to `/data/.hf_cache` (mounted Space volume).
- Never commit secrets here.

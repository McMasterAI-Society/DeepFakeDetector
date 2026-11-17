# Evaluation harness

One command scores any candidate (the deployed stack, a single submodel, a
local fusion export, or a file of precomputed probabilities) on fixed,
versioned test sets. It reports per-generator, per-format and per-size numbers,
not only the overall score.

Only **manifests** (`eval/manifests/*.csv`) and **results**
(`eval/results/<candidate>/<suite>.json` + `.preds.csv`) go in git. Images,
weights, HF caches and fusion exports stay out (see `eval/.gitignore`).

```
eval/
  datasets.py     manifest loader + builders for the benchmark and manual-gen layouts
  candidates.py   deployed / submodel / local fusion / precomputed-probabilities adapters
  metrics.py      AUROC, balanced acc, TPR@1%FPR, FPR@95%TPR, ECE(15), Brier, bootstrap CIs, DeLong
  bias_probe.py   builds the matched-format/size suite and the AUROC-drop report
  run_eval.py     CLI: score a candidate on a suite -> JSON + per-image preds
  compare.py      markdown comparison table between result sets (for PR comments)
  fusion_refit.py refit the logreg fusion from probability files (grouped CV, LOGO, temperature)
  probfile.py     probability file format (CSV/NPZ)
  tests/          synthetic-data tests (no network, no weights)
```

## Install

```bash
pip install -r eval/requirements.txt                 # offline scoring, bias probe, refit
pip install -r backend/requirements-runtime.txt      # + in-process candidates (deployed, submodel:*)
python -m pytest eval -q
```

`eval/requirements.txt` pins `scikit-learn==1.6.1` and `numpy<2`, the same
versions as the backend, so exported fusion pickles load in production.

## Manifests

Columns: `path,label,generator,source,format,width,height,prompt_id` plus optional
extras (`split`, `sha256`, `variant`, `orig_path`). `path` is relative to
`--data-root`. `label` is 0/1 or real/fake. `generator` is `real` for reals.
`format` is the decoded format reported by PIL, not the file extension.

```bash
# AI-GenBench benchmark_ten_percent: {split}/{real,fake}/{source}/
python -m eval.datasets scan-split --split-dir $ROOT/data/benchmark_ten_percent/test \
    --root $ROOT --hash --out eval/manifests/core.csv

# manual-gen layout: images/<generator>/<prompt_id>.png (+ optional metadata.csv with prompt_id)
python -m eval.datasets scan-manual --manual-dir $ROOT/data/manual_gen_images \
    --reals-dir $ROOT/drive/MyDrive/datasets/OpenImagesV7/train --root $ROOT --out eval/manifests/manual.csv

python -m eval.datasets filter --manifest eval/manifests/core.csv --include-generators flux,midjourney --out x.csv
python -m eval.datasets summary --manifest eval/manifests/core.csv
```

## Suites

| Suite | Contents | What to read |
|---|---|---|
| `core` | AI-GenBench `benchmark_ten_percent` **test** split | overall AUROC + CIs; must not drop more than 0.5 pt |
| `heldout-gen` | generators never used to train any submodel or the fusion (FLUX, Midjourney, DALL-E 3, Nano Banana, GPT-image, ...) | per-generator AUROC/TPR; this is the promotion metric |
| `in-the-wild` | social-media style images: re-compressed, resized, screenshots | overall + per-format/size |
| `bias-probe` | the same images with matched formats and sizes (built by `eval/bias_probe.py`) | `bias_probe.max_auroc_drop`; a large drop means a format/size shortcut |

Every result also includes `metadata_shortcut_auroc`: how well the label can be
predicted from `format` / `size_bucket` alone on that suite. If that is close to
1.0, a high AUROC on the suite says little about real detection.

Bias-probe variants: `original`, `real_png` (reals re-saved as PNG),
`fake_jpeg_q{95,85,75}` (fakes JPEG-compressed), `resized_512` (both classes
shorter side 512 + center crop, PNG), `matched_512_jpeg_q85` (both resized and
JPEG Q85, so format, quality and size are all matched). Drops are paired
(same images) with a DeLong p-value.

```bash
python -m eval.bias_probe build --manifest eval/manifests/core.csv --data-root $ROOT \
    --out eval/manifests/bias-probe.csv --max-per-class 1000
```

Generated images go to `$ROOT/_bias_probe/`. Generation is deterministic, so
the committed manifest plus this command reproduce the suite.

## Running

```bash
python -m eval.run_eval --candidate <spec> --suite <suite> --manifest <file> --data-root $ROOT \
    --out eval/results/<name>/<suite>.json [--name <name>] [--n-boot 1000]
```

Candidate specs:

| Spec | Meaning |
|---|---|
| `deployed` / `deployed@<fusion_repo>` | production stack in-process via `backend/app` ModelRegistry; each image goes through `wrapper.predict(image_bytes=...)` + the fusion wrapper exactly as in `/predict`. Records `prob.<submodel>` columns too |
| `submodel:<name>[@<fusion_repo>]` | one backend submodel (e.g. `submodel:cnn-transfer`), resolved through the fusion config |
| `<dir with fusion_logreg.pkl>` | local fusion export. With `--probs <file>`: applied offline to that file's `prob.<name>` columns. Without it: deployed submodels in-process + this fusion |
| `<file>.csv` / `<file>.npz` [`#<column>`] | precomputed probabilities (e.g. from Colab). `#cnn-transfer` scores the `prob.cnn-transfer` column |

Outputs:
- `<suite>.json`: overall metrics with stratified-bootstrap 95% CIs and a DeLong
  CI, breakdowns by generator (fakes of that generator vs all reals), source,
  format and size bucket, per-submodel metrics, the bias-probe report, and
  provenance (manifest sha256, git commit, package versions, GPU, failures).
- `<suite>.preds.csv`: per-image `prob_fake`, `prob.<submodel>` and `error`.
  Use it to re-score offline, for DeLong in `compare.py`, and as input to
  `fusion_refit.py`.

`--suite all` runs every suite whose default manifest `eval/manifests/<suite>.csv`
exists. Any other suite name (e.g. `benchmark-train`) is allowed, which is how
you produce training probabilities for the refit.

## Comparing (PR comments)

```bash
python -m eval.compare eval/results/baseline-2026-11 eval/results/<candidate> --out cmp.md
```

The first argument is the baseline. The output has one table per suite
(AUROC [CI], deltas, paired DeLong p when both `.preds.csv` exist), plus
per-generator, per-format and per-size tables and the bias-probe drop table.

## Refitting the fusion

The deployed `fusion-logreg-final` was re-fit from scratch on about 840 manual
images only. In that set the fakes are PNGs and the reals are OpenImages JPEGs,
and the random split leaks `prompt_id`. `fusion_refit.py` fits on probability
files, combined rather than sequential:

- logit features (clipped at 1e-6), class-balanced weights (`--balance-datasets`
  also weights each training file equally);
- `StratifiedGroupKFold` grouped by `prompt_id` (fallback: path), so a prompt
  never sits on both sides of a split;
- C chosen by leave-one-generator-out AUROC (generator x prompt-fold double
  holdout, pooled), or by grouped CV when there is only one fake generator;
- temperature scaling on grouped out-of-fold logits, folded into the exported
  coefficients;
- a report with the LOGO table, test metrics, single-submodel AUROCs, a warning
  when format alone predicts the training labels, and an optional comparison
  against a reference fusion (DeLong).

```bash
python -m eval.fusion_refit \
    --train eval/results/baseline-2026-11/benchmark-train.preds.csv eval/results/baseline-2026-11/manual.preds.csv \
    --test  eval/results/baseline-2026-11/core.preds.csv \
    --base-config $HF_CACHE/fusion-logreg-final/config.json \
    --reference-fusion $HF_CACHE/fusion-logreg-final \
    --out-dir eval/.artifacts/fusion-logreg-refit

# score the export offline on any suite, then compare
python -m eval.run_eval --candidate eval/.artifacts/fusion-logreg-refit \
    --probs eval/results/baseline-2026-11/core.preds.csv --suite core \
    --manifest eval/manifests/core.csv --name refit-candidate
```

The export is **local only**: `fusion_logreg.pkl`, `config.json`,
`label_map.json`, `refit_report.json`, `test.preds.csv`. Nothing is uploaded.
The pickle is an sklearn `Pipeline(clip -> logit -> LogisticRegression)` that
takes raw submodel probabilities, so the current `LogRegFusionWrapper` loads it
unchanged. `config.json` also carries `coef`, `intercept`, `temperature` and
`feature_transform` for contribution displays. Pushing to HF needs lead sign-off.

Caveat: the probability files must come from the **same** submodel weights and
preprocessing that production serves. Refit only after WS-1 pins gradfield
(v2 at 224 vs v3 at 256), and regenerate the probabilities if that changes.

## Colab: deployed-stack baseline (`baseline-2026-11`)

Use a GPU runtime. Data paths follow `notebooks/fusion-logreg.ipynb`.

```python
from google.colab import drive, userdata
drive.mount('/content/drive')
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')   # read token, only needed if repos are private
```

```bash
!git clone -b feat/eval-harness https://github.com/McMasterAI-Society/DeepFakeDetector.git /content/DeepFakeDetector
%cd /content/DeepFakeDetector
!pip install -q -r backend/requirements-runtime.txt -r eval/requirements.txt

# local copies (Drive I/O is slow); everything lives under ROOT=/content
!mkdir -p /content/data && cp -rn /content/drive/MyDrive/datasets/benchmark_ten_percent /content/data/
!python -c "from huggingface_hub import snapshot_download as s; s('DeepFakeDetector/manual-gen-images', repo_type='dataset', local_dir='/content/data/manual_gen_images')"

# manifests (commit these)
!python -m eval.datasets scan-split --split-dir /content/data/benchmark_ten_percent/test  --root /content --hash --out eval/manifests/core.csv
!python -m eval.datasets scan-split --split-dir /content/data/benchmark_ten_percent/train --root /content --out eval/manifests/benchmark-train.csv
!python -m eval.datasets scan-manual --manual-dir /content/data/manual_gen_images --reals-dir /content/drive/MyDrive/datasets/OpenImagesV7/train --root /content --hash --out eval/manifests/manual.csv
!python -m eval.bias_probe build --manifest eval/manifests/core.csv --data-root /content --out eval/manifests/bias-probe.csv --max-per-class 1000

# deployed stack (production code path, fusion = HF_FUSION_REPO_ID default)
!python -m eval.run_eval --candidate deployed --name baseline-2026-11 --suite core       --manifest eval/manifests/core.csv       --data-root /content --out eval/results/baseline-2026-11/core.json
!python -m eval.run_eval --candidate deployed --name baseline-2026-11 --suite bias-probe --manifest eval/manifests/bias-probe.csv --data-root /content --out eval/results/baseline-2026-11/bias-probe.json
!python -m eval.run_eval --candidate deployed --name baseline-2026-11 --suite manual     --manifest eval/manifests/manual.csv     --data-root /content --out eval/results/baseline-2026-11/manual.json
# training probabilities for the refit (no need to commit these)
!python -m eval.run_eval --candidate deployed --name baseline-2026-11 --suite benchmark-train --manifest eval/manifests/benchmark-train.csv --data-root /content --n-boot 0 --out eval/results/baseline-2026-11/benchmark-train.json

!python -m eval.compare eval/results/baseline-2026-11 --out /content/baseline.md
```

Then download `eval/manifests/*.csv` and `eval/results/baseline-2026-11/` and commit them.
The `manual` suite is **in-sample** for the deployed fusion, because the fusion
was fit on it. Report it, but do not use it as evidence of generalisation.
Fill `heldout-gen` and `in-the-wild` once those images exist, using the same
`scan-split` layout: `{real,fake}/{source}/`.

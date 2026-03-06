
## OpenFake Run - count and size cap
`python scripts/fetch_sample_datasets.py --openfake --max-train 30 --max-test 10`
`python scripts/fetch_sample_datasets.py --openfake --cap-train-mb 80 --cap-test-mb 20`

## Verify
`find datasets/OpenFake -type f -name "*.jpg" | head`
`find datasets/OpenFake -type f -name "*.jpg" | wc -l`
`du -sh datasets/OpenFake`

## DRAGON Run - count and size cap
`python scripts/fetch_sample_datasets.py --dragon --dragon-size ExtraSmall --max-train 30 --max-test 10`
`python scripts/fetch_sample_datasets.py --dragon --dragon-size ExtraSmall --cap-train-mb 80 --cap-test-mb 20`

## Verify
`find datasets/DRAGON -type f -name "*.jpg" | head`
`find datasets/DRAGON -type f -name "*.jpg" | wc -l`
``du -sh datasets/DRAGON`


## WildFake (ZIP extraction workflow)
Why: ModelScope streaming often returns metadata-only rows (no decoded 'image'), so you must extract from the dataset ZIPs + CSV metadata.

## 1) Locate any WildFake zip files (raw folder + modelscope cache)
`find "$PWD/datasets/_raw" "$HOME/.cache/modelscope" -type f -name "*.zip" | grep -i "WildFake" | head -n 50`

## 2) If ZIPs are only in modelscope temp cache, copy them into a stable raw folder
`mkdir -p datasets/_raw/WildFake/Images
rsync -av --progress "$HOME/.cache/modelscope/._____temp/hy2628982280/WildFake/Images/" "datasets/_raw/WildFake/Images/"`

## 3) Sanity check you have the big ZIPs locally
`find datasets/_raw/WildFake/Images -type f -name "*.zip" | head -n 30
du -sh datasets/_raw/WildFake/Images`

## 4) Extract a small sample into datasets/WildFake
`python scripts/extract_wildfake_from_zip.py \
  --raw-root datasets/_raw/WildFake \
  --out-root datasets/WildFake \
  --max-train 30 \
  --max-test 10`

## 5) Verify
`find datasets/WildFake -type f -name "*.jpg" | head
find datasets/WildFake -type f -name "*.jpg" | wc -l
du -sh datasets/WildFake`

## WARNING: WildFake raw is huge
`rm -rf datasets/_raw/WildFake`

## Open Images v7 (HF) Run - count and size cap
`python scripts/fetch_sample_datasets.py --open-images-v7 --max-train 30 --max-test 10`
`python scripts/fetch_sample_datasets.py --open-images-v7 --cap-train-mb 80 --cap-test-mb 20`

## Verify
`find datasets/OpenImagesV7 -type f -name "*.jpg" | head`
`find datasets/OpenImagesV7 -type f -name "*.jpg" | wc -l`
`du -sh datasets/OpenImagesV7`

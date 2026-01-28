#!/usr/bin/env python3
import argparse
import os
import random
import zipfile
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def pick_members(z: zipfile.ZipFile, k: int, rng: random.Random):
    members = [m for m in z.namelist()
               if not m.endswith("/") and Path(m).suffix.lower() in IMG_EXTS]
    if not members:
        return []
    if k >= len(members):
        rng.shuffle(members)
        return members
    return rng.sample(members, k)

def extract_members(zip_path: Path, out_dir: Path, k: int, prefix: str, start_idx: int, rng: random.Random) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        members = pick_members(z, k, rng)
        idx = start_idx
        for m in members:
            ext = Path(m).suffix.lower()
            out_path = out_dir / f"{prefix}_{idx:07d}{ext}"
            with z.open(m) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            idx += 1
        return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, default="datasets/_raw/WildFake")
    ap.add_argument("--max-train", type=int, default=30)
    ap.add_argument("--max-test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    raw = Path(args.raw_root)

    # Real zips (you already have these)
    real_dir = raw / "Images" / "Real"
    real_zips = sorted(real_dir.glob("*.zip"))

    # Fake zips: prefer Diffusion_based if present; otherwise fall back to whatever is there
    fake_candidates = []
    diff_dir = raw / "Images" / "Diffusion_based"
    if diff_dir.exists():
        fake_candidates += sorted(diff_dir.glob("*.zip"))

    # also allow other fake zips at Images/ root
    fake_candidates += sorted((raw / "Images").glob("*.zip"))

    # filter obvious real ones out just in case
    fake_zips = [p for p in fake_candidates if p.name.lower() not in {z.name.lower() for z in real_zips}]

    if not real_zips:
        raise SystemExit(f"No real zips found in {real_dir}")
    if not fake_zips:
        raise SystemExit(f"No fake zips found under {raw/'Images'} (need at least one fake zip downloaded)")

    rng = random.Random(args.seed)

    out = Path("datasets/WildFake")
    train_real = out / "train" / "real"
    train_fake = out / "train" / "fake"
    test_real  = out / "test" / "real"
    test_fake  = out / "test" / "fake"

    # Split counts roughly half/half
    tr_real_k = args.max_train // 2
    tr_fake_k = args.max_train - tr_real_k
    te_real_k = args.max_test // 2
    te_fake_k = args.max_test - te_real_k

    # Choose zips to sample from (random but stable)
    real_zip_train = rng.choice(real_zips)
    real_zip_test  = rng.choice(real_zips)
    fake_zip_train = rng.choice(fake_zips)
    fake_zip_test  = rng.choice(fake_zips)

    print("Using:")
    print("  train real:", real_zip_train)
    print("  train fake:", fake_zip_train)
    print("  test  real:", real_zip_test)
    print("  test  fake:", fake_zip_test)

    idx = 0
    idx = extract_members(real_zip_train, train_real, tr_real_k, "train_real", idx, rng)
    idx = extract_members(fake_zip_train, train_fake, tr_fake_k, "train_fake", 0, rng)

    idx = 0
    idx = extract_members(real_zip_test, test_real, te_real_k, "test_real", idx, rng)
    idx = extract_members(fake_zip_test, test_fake, te_fake_k, "test_fake", 0, rng)

    print("\nDone.")
    print("Train real:", len(list(train_real.glob("*"))))
    print("Train fake:", len(list(train_fake.glob("*"))))
    print("Test  real:", len(list(test_real.glob("*"))))
    print("Test  fake:", len(list(test_fake.glob("*"))))

if __name__ == "__main__":
    main()

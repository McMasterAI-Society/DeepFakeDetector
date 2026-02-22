#!/usr/bin/env python3
import argparse
import io
import os
import random
import zipfile
import requests
from pathlib import Path
from typing import Iterable, Optional, Tuple

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode in ("RGB", "L") else img.convert("RGB")


def encode_jpeg_bytes(img: Image.Image, quality: int = 95, subsampling: int = 0) -> bytes:
    img = ensure_rgb(img)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=subsampling, optimize=True)
    return buf.getvalue()


def safe_save_bytes(jpeg_bytes: bytes, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(jpeg_bytes)
    os.replace(tmp, out_path)


def is_valid_zip(p: Path) -> bool:
    if not p.exists() or not p.is_file():
        return False
    try:
        with zipfile.ZipFile(p, "r") as zf:
            _ = zf.namelist()[:1]
        return True
    except Exception:
        return False


def iter_zip_images(zippath: Path) -> Iterable[str]:
    with zipfile.ZipFile(zippath, "r") as zf:
        for name in zf.namelist():
            ext = Path(name).suffix.lower()
            if ext in IMG_EXTS and not name.endswith("/"):
                yield name


def pick_zip(root: Path, rel_glob: str, prefer_small: bool = True) -> Optional[Path]:
    cands = list(root.glob(rel_glob))
    cands = [p for p in cands if is_valid_zip(p)]
    if not cands:
        return None
    if prefer_small:
        cands.sort(key=lambda p: p.stat().st_size)
    return cands[0]


def find_wildfake_root(cache_dir: Path) -> Optional[Path]:
    """
    Looks for the repo downloaded by ModelScope into either:
      <cache_dir>/hy2628982280/WildFake
      ~/.cache/modelscope/hy2628982280/WildFake
      <cache_dir>/modelscope/hy2628982280/WildFake
    """
    guesses = [
        cache_dir / "hy2628982280" / "WildFake",
        cache_dir / "modelscope" / "hy2628982280" / "WildFake",
        Path.home() / ".cache" / "modelscope" / "hy2628982280" / "WildFake",
        Path.cwd() / ".cache" / "modelscope" / "hy2628982280" / "WildFake",
    ]
    for g in guesses:
        if g.exists() and g.is_dir():
            return g
    return None


def open_image_from_zip(zippath: Path, member: str) -> Optional[Image.Image]:
    try:
        with zipfile.ZipFile(zippath, "r") as zf:
            with zf.open(member, "r") as fp:
                data = fp.read()
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:
        return None


def sample_extract(
    zippath: Path,
    out_dir: Path,
    count: int,
    cap_mb: Optional[float],
    seed: int,
) -> Tuple[int, int]:
    """
    Returns: (saved_count, saved_bytes)
    """
    names = list(iter_zip_images(zippath))
    if not names:
        return 0, 0

    rnd = random.Random(seed)
    rnd.shuffle(names)

    byte_cap = None if cap_mb is None else int(cap_mb * 1024 * 1024)
    saved = 0
    saved_bytes = 0

    for name in names:
        if saved >= count:
            break
        if byte_cap is not None and saved_bytes >= byte_cap:
            break

        img = open_image_from_zip(zippath, name)
        if img is None:
            continue

        jpeg_bytes = encode_jpeg_bytes(img)
        if byte_cap is not None and saved_bytes + len(jpeg_bytes) > byte_cap:
            break

        out_path = out_dir / f"{out_dir.name}_{saved:07d}.jpg"
        safe_save_bytes(jpeg_bytes, out_path)
        saved_bytes += len(jpeg_bytes)
        saved += 1

    return saved, saved_bytes

def _detect_image_col(ds) -> str:
    # common choices
    for c in ["image", "img", "jpg", "jpeg", "png", "webp"]:
        if c in ds.column_names:
            return c
    # fallback: first column
    return ds.column_names[0]


def _iter_hf_pil_images(ds, image_col: str):
    for ex in ds:
        img = ex.get(image_col)
        if img is None:
            continue
        if isinstance(img, Image.Image):
            yield img
            continue
        # sometimes dict with bytes/path
        if isinstance(img, dict):
            b = img.get("bytes")
            p = img.get("path")
            try:
                if b is not None:
                    im = Image.open(io.BytesIO(b)); im.load()
                    yield im
                    continue
                if p is not None and Path(p).exists():
                    im = Image.open(p); im.load()
                    yield im
                    continue
            except Exception:
                continue
        # path-like fallback
        try:
            im = Image.open(img); im.load()
            yield im
        except Exception:
            continue

def _detect_url_col(ds) -> Optional[str]:
    for c in ds.column_names:
        if "url" in c.lower():
            return c
    return None


def open_image_from_url(url: str, timeout: int = 15) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        img.load()
        return img
    except Exception:
        return None
    
def sample_hf_dataset(
    hf_name: str,
    out_base: Path,
    max_train: int,
    max_test: int,
    cap_train_mb: Optional[float],
    cap_test_mb: Optional[float],
    seed: int,
) -> None:
    if load_dataset is None:
        raise RuntimeError("Missing dependency: pip install datasets huggingface_hub")

    ds_dict = load_dataset(hf_name)

    # pick splits; if no test split, make one from train (deterministic)
    if "train" in ds_dict and "test" in ds_dict:
        train_ds = ds_dict["train"]
        test_ds = ds_dict["test"]
    elif "train" in ds_dict:
        split = ds_dict["train"].train_test_split(test_size=0.2, seed=seed)
        train_ds, test_ds = split["train"], split["test"]
    else:
        first = list(ds_dict.keys())[0]
        split = ds_dict[first].train_test_split(test_size=0.2, seed=seed)
        train_ds, test_ds = split["train"], split["test"]

    image_col = _detect_image_col(train_ds)
    url_col = _detect_url_col(train_ds) 

    def _save_split(ds, out_dir: Path, count: int, cap_mb: Optional[float], split_seed: int):
        ds = ds.shuffle(seed=split_seed)
        byte_cap = None if cap_mb is None else int(cap_mb * 1024 * 1024)
        saved = 0
        saved_bytes = 0
        KNOWN_IMAGE_COLS = {"image", "img", "jpg", "jpeg", "png", "webp"}

        use_url = (url_col is not None) and (image_col.lower() not in KNOWN_IMAGE_COLS)
        if use_url:
            # likely no real image column; fall back to URLs
            print(f"[HF] {hf_name}: using URL column '{url_col}'")
            for ex in ds:
                if saved >= count:
                    break
                if byte_cap is not None and saved_bytes >= byte_cap:
                    break

                url = ex.get(url_col)
                if not isinstance(url, str) or not url.startswith("http"):
                    continue

                img = open_image_from_url(url)
                if img is None:
                    continue

                jpeg_bytes = encode_jpeg_bytes(img)
                if byte_cap is not None and saved_bytes + len(jpeg_bytes) > byte_cap:
                    break

                out_path = out_dir / f"{out_dir.name}_{saved:07d}.jpg"
                safe_save_bytes(jpeg_bytes, out_path)
                saved += 1
                saved_bytes += len(jpeg_bytes)
        else:
            print(f"[HF] {hf_name}: using image column '{image_col}'")
            for img in _iter_hf_pil_images(ds, image_col):
                if saved >= count:
                    break
                if byte_cap is not None and saved_bytes >= byte_cap:
                    break

                jpeg_bytes = encode_jpeg_bytes(img)
                if byte_cap is not None and saved_bytes + len(jpeg_bytes) > byte_cap:
                    break

                out_path = out_dir / f"{out_dir.name}_{saved:07d}.jpg"
                safe_save_bytes(jpeg_bytes, out_path)
                saved += 1
                saved_bytes += len(jpeg_bytes)

        return saved, saved_bytes

    train_dir = out_base / "train"
    test_dir = out_base / "test"

    sr, br = _save_split(train_ds, train_dir, max_train, cap_train_mb, seed)
    st, bt = _save_split(test_ds, test_dir, max_test, cap_test_mb, seed + 1)

    print(f"[HF] Train saved: {sr} (~{br/(1024*1024):.1f} MB)")
    print(f"[HF] Test  saved: {st} (~{bt/(1024*1024):.1f} MB)")
    print("[HF] Done ->", out_base.resolve())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelscope-cache", type=str, default=".cache/modelscope")
    ap.add_argument("--max-train", type=int, default=30)
    ap.add_argument("--max-test", type=int, default=10)
    ap.add_argument("--cap-train-mb", type=float, default=None)
    ap.add_argument("--cap-test-mb", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--open-images-v7", action="store_true", help="Fetch a small sample from HF dataset bitmind/open-images-v7")
    
    args = ap.parse_args()

    if args.open_images_v7:
        sample_hf_dataset(
            "bitmind/open-images-v7",
            out_base=Path("datasets/OpenImagesV7"),
            max_train=args.max_train,
            max_test=args.max_test,
            cap_train_mb=args.cap_train_mb,
            cap_test_mb=args.cap_test_mb,
            seed=args.seed,
        )


    cache_dir = Path(args.modelscope_cache).expanduser().resolve()
    root = find_wildfake_root(cache_dir)
    if root is None:
        print("Could not find WildFake repo on disk.")
        print("You need the ModelScope download directory that contains Images/Real and Images/*/*.zip")
        print("Based on your logs, it should be something like: .cache/modelscope/hy2628982280/WildFake")
        return 2

    # Pick one small real zip and one small fake zip that actually exist locally
    real_zip = pick_zip(root, "Images/Real/*.zip", prefer_small=True)
    fake_zip = (
        pick_zip(root, "Images/Diffusion_based/*.zip", prefer_small=True)
        or pick_zip(root, "Images/GAN_based/*.zip", prefer_small=True)
        or pick_zip(root, "Images/*/*.zip", prefer_small=True)
    )

    if real_zip is None:
        print(f"No valid Real zip found under: {root / 'Images/Real'}")
        return 3
    if fake_zip is None:
        print(f"No valid Fake zip found under: {root / 'Images'}")
        return 4

    print("Using:")
    print("  real:", real_zip)
    print("  fake:", fake_zip)

    out_base = Path("datasets/WildFake")
    train_real = out_base / "train" / "real"
    train_fake = out_base / "train" / "fake"
    test_real = out_base / "test" / "real"
    test_fake = out_base / "test" / "fake"

    # Extract train
    sr, br = sample_extract(real_zip, train_real, args.max_train, args.cap_train_mb, seed=args.seed)
    sf, bf = sample_extract(fake_zip, train_fake, args.max_train, args.cap_train_mb, seed=args.seed + 1)
    print(f"Train saved: {sr} real, {sf} fake, ~{(br+bf)/(1024*1024):.1f} MB")

    # Extract test (different seed, and put into test dirs)
    sr2, br2 = sample_extract(real_zip, test_real, args.max_test, args.cap_test_mb, seed=args.seed + 2)
    sf2, bf2 = sample_extract(fake_zip, test_fake, args.max_test, args.cap_test_mb, seed=args.seed + 3)
    print(f"Test saved:  {sr2} real, {sf2} fake, ~{(br2+bf2)/(1024*1024):.1f} MB")

    print("Done ->", out_base.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

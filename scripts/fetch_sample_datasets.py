#!/usr/bin/env python3
import os
import io
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset
from modelscope.msdatasets import MsDataset

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated inputs


# ========== robust encode / save helpers ==========

def ensure_rgb(img: Image.Image) -> Image.Image:
    """
    JPEG supports 'RGB' and 'L' (grayscale). Convert anything else to RGB.
    """
    return img if img.mode in ("RGB", "L") else img.convert("RGB")


def encode_jpeg_bytes(img: Image.Image, quality: int = 95, subsampling: int = 0) -> bytes:
    """
    Return JPEG-encoded bytes for a PIL image. Robust to modes (P, LA, RGBA, CMYK, ...).
    """
    img = ensure_rgb(img)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=subsampling, optimize=True)
    return buf.getvalue()


def safe_save_bytes(jpeg_bytes: bytes, out_path: str) -> None:
    """
    Atomically write bytes to disk: write to temp, then replace.
    """
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(jpeg_bytes)
    os.replace(tmp, out_path)


# ========== OpenFake ==========

def download_split_openfake(
    split_name: str,
    base_dir: str,
    max_images: int | None = None,
    max_mb: float | None = None,
) -> None:
    """
    Download OpenFake split (streaming) into:
        <base_dir>/<split_name>/{real,fake}/...
    Caps: number of images and/or total MiB (approx, uses encoded JPEG size).
    """
    print(f"\n Downloading OpenFake '{split_name}' split images")
    dataset = load_dataset("ComplexDataLab/OpenFake", split=split_name, streaming=True)

    real_dir = os.path.join(base_dir, split_name, "real")
    fake_dir = os.path.join(base_dir, split_name, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    saved_real = saved_fake = 0
    total_bytes = 0
    byte_cap = None if max_mb is None else int(max_mb * 1024 * 1024)

    for i, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name} split")):
        if max_images is not None and i >= max_images:
            print("Reached max number of images, stopping.")
            break
        if byte_cap is not None and total_bytes >= byte_cap:
            print(f"Reached size cap ~{max_mb} MB, stopping.")
            break

        img = sample["image"]
        lab = sample["label"]
        # Normalize label to {'real','fake'}
        if isinstance(lab, int):
            label = "fake" if lab == 1 else "real"
        else:
            label = str(lab).strip().lower()

        subdir = real_dir if label == "real" else fake_dir
        out_path = os.path.join(subdir, f"{split_name}_{i:07d}.jpg")

        try:
            jpeg_bytes = encode_jpeg_bytes(img)
            if byte_cap is not None and total_bytes + len(jpeg_bytes) > byte_cap:
                print(f"Next image would exceed cap (~{max_mb} MB). Stopping.")
                break
            safe_save_bytes(jpeg_bytes, out_path)
            total_bytes += len(jpeg_bytes)
            if label == "real":
                saved_real += 1
            else:
                saved_fake += 1
        except Exception as e:
            print(f"Skip {split_name} idx={i}: {e}")
            continue

    mb = total_bytes / (1024 * 1024)
    print(f"Finished {split_name}: {saved_real} real, {saved_fake} fake, ~{mb:.1f} MB → {os.path.join(base_dir, split_name)}")


def import_openfake(
    max_train: int | None = None,
    max_test: int | None = None,
    max_mb_train: float | None = None,
    max_mb_test: float | None = None,
) -> None:
    """
    Create:
        datasets/OpenFake/
          train/{real,fake}
          test/{real,fake}
    """
    base_dir = "datasets/OpenFake"
    for p in [
        os.path.join(base_dir, "train", "real"),
        os.path.join(base_dir, "train", "fake"),
        os.path.join(base_dir, "test", "real"),
        os.path.join(base_dir, "test", "fake"),
    ]:
        os.makedirs(p, exist_ok=True)

    download_split_openfake("train", base_dir, max_images=max_train, max_mb=max_mb_train)
    download_split_openfake("test", base_dir, max_images=max_test, max_mb=max_mb_test)

    print("\n Done. Dataset organized in:")
    print(os.path.abspath(base_dir))


# ========== WildFake ==========

def download_split_wildfake(
    split_name: str,
    base_dir: str,
    max_images: int | None = None,
    max_mb: float | None = None,
) -> None:
    """
    Download WildFake split into:
        <base_dir>/<split_name>/{real,fake}/...
    """
    print(f"\n Downloading WildFake '{split_name}' split images")
    dataset = MsDataset.load('hy2628982280/WildFake', subset_name='default', split=split_name)

    real_dir = os.path.join(base_dir, split_name, "real")
    fake_dir = os.path.join(base_dir, split_name, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    saved_real = saved_fake = 0
    total_bytes = 0
    byte_cap = None if max_mb is None else int(max_mb * 1024 * 1024)

    for i, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name} split")):
        if max_images is not None and i >= max_images:
            print("Reached max number of images, stopping.")
            break
        if byte_cap is not None and total_bytes >= byte_cap:
            print(f"Reached size cap ~{max_mb} MB, stopping.")
            break

        img = sample["image"]
        is_fake = sample["IsFake"]  # 0 = real, 1 = fake

        subdir = real_dir if is_fake == 0 else fake_dir
        out_path = os.path.join(subdir, f"{split_name}_{i:07d}.jpg")

        try:
            jpeg_bytes = encode_jpeg_bytes(img)
            if byte_cap is not None and total_bytes + len(jpeg_bytes) > byte_cap:
                print(f"Next image would exceed cap (~{max_mb} MB). Stopping.")
                break
            safe_save_bytes(jpeg_bytes, out_path)
            total_bytes += len(jpeg_bytes)
            if is_fake == 0:
                saved_real += 1
            else:
                saved_fake += 1
        except Exception as e:
            print(f"Skip {split_name} idx={i}: {e}")
            continue

    mb = total_bytes / (1024 * 1024)
    print(f"Finished {split_name}: {saved_real} real, {saved_fake} fake, ~{mb:.1f} MB → {os.path.join(base_dir, split_name)}")


def import_wildfake(
    max_train: int | None = None,
    max_test: int | None = None,
    max_mb_train: float | None = None,
    max_mb_test: float | None = None,
) -> None:
    """
    Create:
        datasets/WildFake/
          train/{real,fake}
          test/{real,fake}
    """
    base_dir = "datasets/WildFake"
    for p in [
        os.path.join(base_dir, "train", "real"),
        os.path.join(base_dir, "train", "fake"),
        os.path.join(base_dir, "test", "real"),
        os.path.join(base_dir, "test", "fake"),
    ]:
        os.makedirs(p, exist_ok=True)

    download_split_wildfake('train', base_dir, max_images=max_train, max_mb=max_mb_train)
    download_split_wildfake('test', base_dir, max_images=max_test, max_mb=max_mb_test)
    print("\nDone. Dataset organized in:")
    print(os.path.abspath(base_dir))


# ========== DRAGON ==========

def download_split_dragon(
    dataset_split,
    split_name: str,
    out_dir: str,
    max_images: int | None = None,
    max_mb: float | None = None,
) -> None:
    """
    Download DRAGON split into:
        <out_dir>/split_name_XXXXXXX.jpg
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    total_bytes = 0
    byte_cap = None if max_mb is None else int(max_mb * 1024 * 1024)

    for i, sample in enumerate(tqdm(dataset_split, desc=f"Processing {split_name}")):
        if max_images is not None and i >= max_images:
            print("Reached max number of images, stopping.")
            break
        if byte_cap is not None and total_bytes >= byte_cap:
            print(f"Reached size cap ~{max_mb} MB, stopping.")
            break

        img = sample["png"]  # PIL image

        out_path = os.path.join(out_dir, f"{split_name}_{i:07d}.jpg")
        try:
            jpeg_bytes = encode_jpeg_bytes(img)
            if byte_cap is not None and total_bytes + len(jpeg_bytes) > byte_cap:
                print(f"Next image would exceed cap (~{max_mb} MB). Stopping.")
                break
            safe_save_bytes(jpeg_bytes, out_path)
            total_bytes += len(jpeg_bytes)
            saved += 1
        except Exception as e:
            print(f"Skip {split_name} idx={i}: {e}")
            continue

    mb = total_bytes / (1024 * 1024)
    print(f"Finished {split_name}: {saved} images, ~{mb:.1f} MB → {out_dir}")


def import_dragon(
    size: str = "ExtraLarge",
    max_train: int | None = None,
    max_test: int | None = None,
    max_mb_train: float | None = None,
    max_mb_test: float | None = None,
) -> None:
    """
    Create:
        datasets/DRAGON/
          train/
          test/
    size ∈ {"ExtraSmall","Small","Regular","Large","ExtraLarge"}
    """
    base_dir = 'datasets/DRAGON'
    print(f"Downloading DRAGON ({size}) subset")
    ds = load_dataset("lesc-unifi/dragon", size)

    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    download_split_dragon(ds["train"], "train", train_dir, max_images=max_train, max_mb=max_mb_train)
    download_split_dragon(ds["test"], "test", test_dir, max_images=max_test, max_mb=max_mb_test)

    print(f"Done. DRAGON ({size}) saved in {os.path.abspath(base_dir)}")


# ========== main ==========

def main() -> int:
    """
    Examples while testing:
        import_openfake(max_train=500, max_test=100)
        import_openfake(max_mb_train=500, max_mb_test=100)
        import_wildfake(max_mb_train=300, max_mb_test=60)
        import_dragon(size="Small", max_mb_train=300, max_mb_test=60)

    Default below downloads entire splits. Consider capping for your first run.
    """
    # --- pick one style (count cap or MiB cap) or leave None for full ---
    import_openfake()  # e.g., max_mb_train=500, max_mb_test=100
    import_wildfake()  # e.g., max_mb_train=300, max_mb_test=60
    import_dragon()    # e.g., size="Small", max_mb_train=300, max_mb_test=60
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

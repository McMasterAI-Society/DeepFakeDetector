import os
from tqdm import tqdm
from datasets import load_dataset
from modelscope.msdatasets import MsDataset 


def download_split_openfake(split_name, base_dir):
    print(f"\n Downloading OpenFake '{split_name}' split images")
    dataset = load_dataset("ComplexDataLab/OpenFake", split=split_name, streaming=True)

    # Output folders
    real_dir = os.path.join(base_dir, split_name, "real")
    fake_dir = os.path.join(base_dir, split_name, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    for i, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name} split")):
        image = sample["image"]
        label = sample["label"]  # label = 'real' or 'fake'

        # Convert RGBA to RGB to avoid JPEG errors TODO - does this impact the images?
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Choose path based on real or fake
        subdir = real_dir if label == "real" else fake_dir
        image_path = os.path.join(subdir, f"{split_name}_{i:07d}.jpg")

        image.save(image_path)

    print(f"Finished {split_name} split: saved to {os.path.join(base_dir, split_name)}")

def import_openfake():
    base_dir="datasets/OpenFake"

    paths = [
            os.path.join(base_dir, "train", "real"),
            os.path.join(base_dir, "train", "fake"),
            os.path.join(base_dir, "test", "real"),
            os.path.join(base_dir, "test", "fake"),
        ]
    
    for path in paths:
        os.makedirs(path, exist_ok=True)

    download_split_openfake("train", base_dir)
    download_split_openfake("test", base_dir)

    print("\n Done. Dataset organized in:")
    print(os.path.abspath(base_dir))

def download_split_wildfake(split_name, base_dir):
    print(f"\n Downloading Wildfake '{split_name}' split images")
    dataset = MsDataset.load('hy2628982280/WildFake', subset_name='default', split=split_name)

    # Output folders
    real_dir = os.path.join(base_dir, split_name, "real")
    fake_dir = os.path.join(base_dir, split_name, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    for i, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name} split")):
        image = sample["image"]
        label = sample["IsFake"]  # 0 = real, 1 = fake (based on dataset card)


        # Convert RGBA to RGB to avoid JPEG errors
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Choose path based on real or fake
        subdir = real_dir if label == 0 else fake_dir
        image_path = os.path.join(subdir, f"{split_name}_{i:07d}.jpg")

        image.save(image_path)

    print(f"Finished {split_name} split: saved to {os.path.join(base_dir, split_name)}")

def import_wildfake():
    base_dir="datasets/WildFake"

    paths = [
            os.path.join(base_dir, "train", "real"),
            os.path.join(base_dir, "train", "fake"),
            os.path.join(base_dir, "test", "real"),
            os.path.join(base_dir, "test", "fake"),
        ]
    
    for path in paths:
        os.makedirs(path, exist_ok=True)

    download_split_wildfake('train', base_dir)
    download_split_wildfake('test', base_dir)
    print("\nDone. Dataset organized in:")
    print(os.path.abspath(base_dir))

def download_split_dragon(dataset_split, split_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    for i, sample in enumerate(tqdm(dataset_split, desc=f"Processing {split_name}")):

        # Images are stored under png
        image = sample["png"]
        if image.mode == "RGBA":
            image = image.convert("RGB")

        filename = f"{split_name}_{i:07d}.jpg"
        image.save(os.path.join(out_dir, filename))

def import_dragon():
    base_dir = 'datasets/DRAGON'
    size_split = "ExtraLarge"         
    print(f"Downloading DRAGON ({size_split}) subset")
    ds = load_dataset("lesc-unifi/dragon", size_split)

    # Create output directories
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    download_split_dragon(ds["train"], "train", train_dir)
    download_split_dragon(ds["test"], "test", test_dir)

    print(f"Done. DRAGON ({size_split}) saved in {os.path.abspath(base_dir)}")


def main():
    import_openfake()
    import_wildfake()
    import_dragon()

    return 0


if __name__ == "__main__":
    main()





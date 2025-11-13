import os
from tqdm import tqdm
from datasets import load_dataset
from modelscope.msdatasets import MsDataset 


def download_split_openfake(split_name, base_dir, max_images=None):
    print(f"\n Downloading OpenFake '{split_name}' split images")
    dataset = load_dataset("ComplexDataLab/OpenFake", split=split_name, streaming=True)

    # Output folders
    real_dir = os.path.join(base_dir, split_name, "real")
    fake_dir = os.path.join(base_dir, split_name, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    for i, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name} split")):

        if max_images is not None and i >= max_images:
            print(f"Reached max number of images, stopping.")
            break

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

def import_openfake(max_train=None, max_test=None):
    """
    Import OpenFake Dataset from Hugging Face

    Parameters:
        max_train (int, optional): Max number of images to download into train folder
            Default: None. If not None, downloads all in subset. 
        max_test (int, optional): Max number of images to download into test folder
            Default: None. If not None, downloads all in subset. 

    Directory Structure:
        datasets/OpenFake/
            train/
            test/
    """
    base_dir="datasets/OpenFake"

    paths = [
            os.path.join(base_dir, "train", "real"),
            os.path.join(base_dir, "train", "fake"),
            os.path.join(base_dir, "test", "real"),
            os.path.join(base_dir, "test", "fake"),
        ]
    
    for path in paths:
        os.makedirs(path, exist_ok=True)

    download_split_openfake("train", base_dir, max_images=max_train)
    download_split_openfake("test", base_dir, max_images=max_test)

    print("\n Done. Dataset organized in:")
    print(os.path.abspath(base_dir))

def download_split_wildfake(split_name, base_dir, max_images=None):
    print(f"\n Downloading Wildfake '{split_name}' split images")
    dataset = MsDataset.load('hy2628982280/WildFake', subset_name='default', split=split_name)

    # Output folders
    real_dir = os.path.join(base_dir, split_name, "real")
    fake_dir = os.path.join(base_dir, split_name, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    for i, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name} split")):
        if max_images is not None and i >= max_images:
            print(f"Reached max number of images, stopping.")
            break

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

def import_wildfake(max_train=None, max_test=None):
    """
    Import WildFake Dataset

    Parameters:
        max_train (int, optional): Max number of images to download into train folder
            Default: None. If not None, downloads all in subset. 
        max_test (int, optional): Max number of images to download into test folder
            Default: None. If not None, downloads all in subset. 

    Directory Structure:
        datasets/WildFake/
            train/
            test/
    """
    base_dir="datasets/WildFake"

    paths = [
            os.path.join(base_dir, "train", "real"),
            os.path.join(base_dir, "train", "fake"),
            os.path.join(base_dir, "test", "real"),
            os.path.join(base_dir, "test", "fake"),
        ]
    
    for path in paths:
        os.makedirs(path, exist_ok=True)

    download_split_wildfake('train', base_dir, max_images=max_train)
    download_split_wildfake('test', base_dir, max_images=max_test)
    print("\nDone. Dataset organized in:")
    print(os.path.abspath(base_dir))

def download_split_dragon(dataset_split, split_name, out_dir, max_images=None):
    os.makedirs(out_dir, exist_ok=True)
    
    for i, sample in enumerate(tqdm(dataset_split, desc=f"Processing {split_name}")):
        if max_images is not None and i >= max_images:
            print(f"Reached max number of images, stopping.")
            break

        # Images are stored under png
        image = sample["png"]
        if image.mode == "RGBA":
            image = image.convert("RGB")

        filename = f"{split_name}_{i:07d}.jpg"
        image.save(os.path.join(out_dir, filename))

def import_dragon(size="ExtraLarge", max_train=None, max_test=None):
    """
    Import DRAGON dataset from Hugging Face. 

    Parameters:
        size (string, optional): Defines size of subset of DRAGON dataset to download (for more info on sizes of each subset go to https://huggingface.co/datasets/lesc-unifi/dragon#dataset-structure)
            Options: "ExtraSmall", "Small", "Regular", "Large", "ExtraLarge"
            Default: "ExtraLarge"
        max_train (int, optional): Max number of images to download into train folder
            Default: None. If not None, downloads all in subset. 
        max_test (int, optional): Max number of images to download into test folder
            Default: None. If not None, downloads all in subset. 

    Directory Structure:
        datasets/DRAGON/
            train/
            test/
    """
    base_dir = 'datasets/DRAGON'
    size_split = size  

    print(f"Downloading DRAGON ({size_split}) subset")
    ds = load_dataset("lesc-unifi/dragon", size_split)

    # Create output directories
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    download_split_dragon(ds["train"], "train", train_dir, max_images=max_train)
    download_split_dragon(ds["test"], "test", test_dir, max_images=max_test)

    print(f"Done. DRAGON ({size_split}) saved in {os.path.abspath(base_dir)}")


def main():
    """
    Example usage with max number of images:

    import_openfake(max_train=500, max_test=100)
    import_wildfake(max_train=500, max_test=100)
    import_dragon(size="Large", max_train=300, max_test=50)

    """
    # Default - download entire dataset:
    import_openfake()
    import_wildfake()
    import_dragon()

    return 0


if __name__ == "__main__":
    main()





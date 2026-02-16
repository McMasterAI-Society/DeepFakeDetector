"""
Purpose: This script pulls the dataset from Hugging Face and reconstructs a local, training-ready view of the dataset.

NOTE: Before running, run the command `huggingface-cli login` in your terminal and provide a token with read access to the organization to download the dataset.

Script Responsibilities:
- download dataset from HUgging Face
- parse dir structure:
    iterate over images/{generator}/pXXX.png
    read corresponding promtps/pXXX.txt
- expose clean python representation
   {
     "image_path": "...",
     "generator": "dalle",
     "prompt_id": "p042",
     "prompt_text": "...",
   }
"""

import os
from huggingface_hub import snapshot_download

  
def load_manual_dataset(local_dir="datasets/manual_gen_images"):
    repo_id = "DeepFakeDetector/manual-gen-images"

    print("Downloading dataset")
    dataset_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
    )

    records = []

    prompts_dir = os.path.join(dataset_path, "prompts")
    images_dir = os.path.join(dataset_path, "images")

    VALID_EXTENSIONS = (".png", ".jpg")

    for generator in os.listdir(images_dir):
        generator_path = os.path.join(images_dir, generator)

        if not os.path.isdir(generator_path):
            continue

        for filename in os.listdir(generator_path):
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue

            prompt_id = os.path.splitext(filename)[0]
            image_path = os.path.join(generator_path, filename)
            prompt_path = os.path.join(prompts_dir, f"{prompt_id}.txt")

            if not os.path.exists(prompt_path):
                continue

            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()

            records.append({
                "image_path": image_path,
                "generator": generator,
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
            })

    return records


if __name__ == "__main__":
    dataset = load_manual_dataset()
    print(f"Loaded {len(dataset)} samples")
    print(dataset[0])

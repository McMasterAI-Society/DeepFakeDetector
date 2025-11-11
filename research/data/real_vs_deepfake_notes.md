# Expected Folder Layout
```
datasets/
├── OpenFake/
|   ├── train/
│   │   ├── real/
│   │   ├── fake/
|   ├── test/
│   │   ├── real/
│   │   ├── fake/
├── WildFake/
|   ├── train/
│   │   ├── real/
│   │   ├── fake/
|   ├── test/
│   │   ├── real/
│   │   ├── fake/
├── DRAGON/
│   ├── train/
│   └── test/

```

## OpenFake
Example structure of dataset:
| image         | prompt            | label     | model         |
|---------------|-------------------|-----------|---------------| 
| image a       | image a is a ...  | real      | real          |
| image b       |image b captures...| fake      | midjourney-6  |


## WildFake
Example structure of dataset:
| Generator  | ... | ...   | IsFake         | Image_path
|---------------|-----------|--------|-----------|---------------| 
| GAN_based  | ... | ...   | 1         | ./GAN_based/Typical/BigGAN/104/img000071.jpg
| GAN_based  | ... | ...   | 1         | ./GAN_based/Typical/BigGAN/104/img000072.jpg

## DRAGON
Unfortunately, it does not appear that the real images mentioned in the paper (https://arxiv.org/pdf/2505.11257) about DRAGON are found in https://huggingface.co/datasets/lesc-unifi/dragon#dataset-details. The fake images are based on the images found in at ImageNet: https://www.image-net.org/download.php. To access the ImageNet dataset we must request access @ https://www.image-net.org/download-images. 
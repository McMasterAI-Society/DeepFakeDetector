
# TensorFlow vs Pytorch vs Hugging Face
Contributer: Andrew Wu

## Table of Contents
[Tensorflow](#tensorflow)<br/>
[PyTorch](#pytorch)<br/>
[Hugging Face](#hugging-face)<br/>
[Comparison](#comparison)<br/>
[Models](#what-models-already-exist)<br/>
[Pipeline](#recommended-fastapi-pipeline)<br/>

## Overview

### Tensorflow

TensorFlow is an open source software library for high performance numerical computation. Its flexible architecture allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices. It comes with strong support for machine learning and deep learning and the flexible numerical computation core is used across many other scientific domains.

For deepfake detection, TensorFlow integrates smoothly with CNN architectures like **EfficientNet** and **ResNet**. However, incorporating ViT or FFT-based preprocessing often requires custom **TensorFlow ops** or hybrid **Keras** layers. **TensorFlow Serving** and **TF Lite** provide a robust path for FastAPI deployment, making it a strong option if the focus is on production stability and scalability.

### PyTorch

PyTorch is a Python package that provides **tensor computation** (like NumPy) with strong GPU acceleration and deep neural networks build on a **tape-based autograd system**. 

PyTorch offers superior flexibility for experimental and hybrid deepfake architectures that combine CNNs, Vision Transformers (ViTs), and FFT transformations. The `torch.fft` module integrates seamlessly into the preprocessing pipeline, and models like **ResNet50** and **ViT-B/16** are readily available via `torchvision` and `timm`. Deployment with FastAPI is straightforward via **TorchScript** or **ONNX** export.

### Hugging Face

Hugging Face provides high-level APIs for transformer-based models, including ViTs and CLIP variants, making it ideal for transfer learning on visual-linguistic features in deepfake detection. It sits on top of PyTorch or TensorFlow, enabling easy fine-tuning with preprocessed (FFT or CNN-extracted) embeddings. While heavier than pure PyTorch setups, it offers state-of-the-art pretrained ViT and CLIP models with strong support for ONNX deployment through FastAPI.

## Comparison
| Framework | Pros | Cons |
| :-------- | :--- | :--- |
| **Tensorflow** |<ul><li>Mature ecosystem for production deployment (TF Serving). <li>Keras API makes prototyping CNNs very straightforward. <li>Good GPU acceleratin support. <li>Works well with FastAPI through SavedModel or tf.keras.models.load_model. | <ul> <li> Less flexible for research-style experimentation. <li>FFT preprocessing may need custom layers for integration | 
| **PyTorch** | <ul><li>Very flexible, so great for combining CNNs, ViTs, and FFT-based preprocessing. <li>Very popular in research, pre-existing deepfake detection implementations use PyTorch. <li>torch.fft allows seamless FFT integration. <li>TorchScript can be used for deployment| <ul><li>More manualy deployment compared to Tensorflow Serving| 
| **Hugging Face** | <ul><li>Easiest access to pretrained ViTs and hybrid models. <li>Works on top of PyTorch or TensorFlow. <li>Fine-tuning pretrained vision-language models is simple. <li>Can export models to ONNX for FastAPI deployment | <ul><li>Heavier dependency; primarily optimized for transformer-based models. <li>Less intuitive for CNN + FFT hybrid setup.|

## What models already exist?

### [EfficientNet-B0/B4](https://www.geeksforgeeks.org/computer-vision/efficientnet-architecture/)
> EfficientNet is a family of convolutional neural networks (CNNs) developed by Google that balance accuracy and efficiency through a compound scaling method. Instead of arbitrarily increasing network depth, width, or image resolution, EfficientNet scales all three dimensions uniformly using a fixed compound coefficient. This design leads to significantly better performance per computational cost compared to previous architectures.

**Efficiency**: EfficientNet-B0 through B4 models progressively scale in size and accuracy while maintaining excellent parameter efficiency. For example, EfficientNet-B0 achieves competitive ImageNet accuracy with only **5.3 million** parameters and **0.39 billion FLOPs**, making it ideal for lightweight deepfake detection deployments. Larger variants like EfficientNet-B4 increase representational power while still remaining far more efficient than ResNet-101 or VGG architectures. This efficiency makes the models particularly well-suited for real-time inference and FastAPI integration.

**Performance**: On ImageNet, EfficientNet-B7 achieves top-1 / top-5 accuracy of **84.4% / 97.3%**, outperforming larger and deeper models such as ResNet and DenseNet while being **8.4× smaller and 6.1× faster**. In the context of deepfake detection, EfficientNet’s balanced scaling enables it to detect both global and localized manipulations effectively—capturing compression patterns, color inconsistencies, and fine texture differences in manipulated faces.

In our preprocessing pipeline, EfficientNet-B0 or B4 can be combined with FFT-based preprocessing to increase sensitivity to frequency-domain anomalies. The lightweight architecture also facilitates hybrid integration with ViT-B/16, acting as an efficient feature extractor before transformer-based global reasoning.

Given its balance of computational efficiency, accuracy, and deployability, EfficientNet-B0 is a strong candidate for our base CNN backbone in the deepfake detection pipeline.

**Framework Support**: EfficientNet models are widely supported across all major frameworks.  
- **TensorFlow**: Native implementation via `tf.keras.applications.EfficientNet*` and TensorFlow Hub.  
- **PyTorch**: Available through `torchvision.models` and `timm` with pretrained weights.  
- **Hugging Face**: Select EfficientNet variants available through the `transformers` and `timm` integration.  

EfficientNet’s lightweight design and cross-framework availability make it one of the most deployment-friendly CNNs for both research and production pipelines.

### [ResNet50/ResNet101](https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/)
>ResNet-50 is CNN architecture that belongs to the ResNet (Residual Networks) family, a series of models designed to address the challenges associated with training deep neural networks. ResNet-50 is renowned for its depth and efficiency in image classification tasks. ResNet architectures come in various depths, such as ResNet-18, ResNet-32, and so forth, with ResNet-50 being a mid-sized variant.

**Efficiency**: ResNet50 provides a good balance of depth and computational cost, making it suitable for real-time deepfake detection pipelines.

**Performance**: ResNet architectures excel at learning fine-grained spatial and texture details, which are critical for identifying subtle artifacts in manipulated faces.

On ImageNet, for single-models, [He et al.](https://arxiv.org/pdf/1512.03385) report top-1 / top-5 error for ResNet-50 of **22.85% / 6.71%**, and for ResNet-101 of **21.75% / 6.05%**. Deeper ResNets still have lower copmutation than older VGG nets.

ResNet-50/101 are strong candidates for deepfake detection, which can rely on subtle textures and artefact cues, and are CNN backbones proven to learn fine spatial features across many layers. If our dataset is large enough and we can afford to compute, a deeper model may capture more subtle cues. 

Choosing between ResNet-50 and ResNet-101 means we look at balancing computing and accuracy, but sticking with a lower layer model can help with keeping computational costs down and inference latency. 

One thing to note is that using a very deep model like ResNet-101 might overfit if we don't have sufficient data or enough regularization, since our datasets may be smaller compared to ImageNet. ResNet-50 would suffice in this case. Also, ResNet is originally trained on RGB images, so if we go ahead with FFT preprocessing, we would need to consider how we could present frequency information to a CNN/ViT. 

Either way, we must finetune the model with FFT-preprocessed inputs rather than relying purely on ImageNet-trained features, since deepfake detection is a different domain. We can also feed ResNet-50 features into a ViT to create a hybrid model that has both speed and performance. 

**Framework Support**: ResNet architectures are among the most universally implemented deep learning backbones.  
- **TensorFlow**: Integrated in `tf.keras.applications.ResNet*` and TensorFlow Hub with pretrained ImageNet weights.  
- **PyTorch**: Included in `torchvision.models` with variants ResNet18–152 and strong community support.  
- **Hugging Face**: Offered via the `transformers` vision module and `timm` wrappers.  
Due to their long-standing use in academia and industry, ResNet models are exceptionally easy to integrate, fine-tune, and export to ONNX or TorchScript for FastAPI deployment.


### [ViT-B/16](`https://www.geeksforgeeks.org/deep-learning/vision-transformer-vit-architecture`/)
> Vision Transformer (ViT-B/16) is a transformer-based architecture that applies the transformer design—originally developed for NLP to image analysis. Instead of processing pixel grids directly, ViT divides an image into fixed-size patches (e.g., 16x16), flattens them, and treats each patch as a "token." The model then uses self-attention layers to capture global relationships across patches, enabling powerful context understanding in visual data.

**Efficiency**: ViT-B/16 has around **86 million parameters** and, while more computationally demanding than traditional CNNs like ResNet-50, it achieves competitive or superior accuracy on large-scale datasets such as ImageNet. Pretraining on large datasets is often required for good performance on smaller downstream tasks.

**Performance**: ViTs excel at modeling long-range dependencies and global structures across images, making them particularly effective in detecting subtle or dispersed deepfake artifacts that CNNs may miss. When combined with FFT-based preprocessing, ViT can leverage both spatial-frequency cues and global attention.

In the context of deepfake detection, ViT-B/16 can serve as either a standalone backbone or as a fusion layer that integrates CNN-derived features (e.g., from ResNet-50). This hybrid design provides a balance between local texture learning and global contextual awareness, improving both accuracy and generalization.

Some specific Hugging Face ViT's:

- [**google/vit-base-patch16-224**](https://huggingface.co/google/vit-base-patch16-224) : Common baseline for visual transformers; learns global patterns across patches. can ingest FFT-magnitude patches as tokens.

- [**facebook/deit-base-patch16-224**](https://huggingface.co/facebook/deit-base-patch16-224) (Data-efficient): Robust with small datasets, important for limited deepfake samples. good for FFT + small-data fine-tuning.

- [**openai/clip-vit-base-patch32**](https://huggingface.co/openai/clip-vit-base-patch32): Embeddings can be leveraged for multimodal forgery analysis (image + metadata). Good if we extend beyond image-only detection.

- [**microsoft/beit-base-patch16-224**](https://huggingface.co/microsoft/beit-base-patch16-224) (Masked Image Modeling): Strong pretraining on reconstruction; detects subtle visual anomalies. complements FFT-derived patterns.


**Framework Support**: ViT's are natively supported and pretrained across multiple frameworks.  
- **TensorFlow**: Implemented through TensorFlow Hub (e.g., `google/vit-base-patch16-224`).  
- **PyTorch**: Available via `timm` and Hugging Face’s `transformers` library with pretrained ImageNet and fine-tuned weights.  
- **Hugging Face**: Strongest ViT ecosystem, offering ViT, DeiT, BEiT, and CLIP variants under a unified API.  

ViT models integrate seamlessly with both PyTorch and Hugging Face environments, allowing straightforward hybridization with CNNs and FFT preprocessing within a single FastAPI deployment pipeline.


### [Swin Transformer](https://arxiv.org/abs/2103.14030)
> Swin Transformer is a hierarchical vision transformer. Images are processed in patches and windowed self-attention is used to capture local information. These windows are shifted across the image to allow for cross-window connections, capturing global information more efficiently. This hierarchical approach allows it to process images effectively at different scales and achieve linear computational complexity relative to image size, making it a versatile backbone for various vision tasks like image classification and object detection.

**Efficiency**: The shifted-window scheme gives Swin Transformer a computational complexity that is linear with respect to input image size (under the window partitioning assumption), rather than quadratic as in vanilla ViTs.  ￼ For example, *Liu et al*. report that for the Swin-T (tiny) variant, the model is **4.1× faster** than a sliding-window variant for the same depth and achieves similar accuracy.  ￼ This design makes Swin Transformer significantly more lightweight and scalable for high-resolution images than many previous transformer backbones.

**Performance**: On the ImageNet-1K classification task, Swin Transformer achieved **~87.3% top-1 accuracy**. For downstream dense vision tasks it also shows strong results. For deepfake detection, this suggests the architecture’s ability to capture both global and multi-scale context may be very beneficial for spotting manipulations that span patches or global facial structure (not just local texture). Its linear complexity means it is more deployable at higher resolutions compared to vanilla Vision Transformers.

**Framework Suppport**: 
- **PyTorch**: Official implementation by Microsoft Research available including pretrained weights.  ￼
- **Hugging Face**: Variants of Swin (Swin-T, Swin-B, Swin-L) are available via the transformers library and model hub, enabling fine-tuning and ONNX export.
-	**TensorFlow**: While less common, third-party ports exist; conversion via ONNX may be required for full parity.

### [ConvNeXt-B](https://arxiv.org/abs/2201.03545)
> ConvNeXT is a pure convolutional model (ConvNet), inspired by the design of Vision Transformers, that claims to outperform them. It “modernizes” a standard ResNet-style backbone by revising macro design, micro design, and training regime, producing a family of models that compete closely with state-of-the‐art transformers while retaining the structural simplicity of convolutions

**Efficiency**: ConvNeXt retains the efficiency benefits of CNNs, with inference throughput comparable to or better than hierarchical vision transformers. For example, under similar FLOPs as Swin-T, a ConvNeXt-T model achieved **higher FPS** and **lower memory usage**. The paper reports instances where ConvNeXt models, at roughly 960 G FLOPs, match or exceed transformer backbones in downstream tasks while remaining structurally simpler.  

**Performance**: On ImageNet-1K, a large ConvNeXt variant achieved **~87.8% top-1 accuracy**. In COCO object detection/segmentation and ADE20K semantic segmentation, ConvNeXt models outperform or match leading transformer backbones under similar resource budgets.

In the context of deepfake detection, ConvNeXt’s strong spatial feature extraction combined with modern training recipes makes it a compelling backbone: capable of capturing both fine texture/artefact cues and global structure while maintaining inference efficiency.

**Framework Suppport**:
- **PyTorch**: Official implementation by the authors exists, with pretrained weights.  ￼
- **Hugging Face**: Variants of ConvNeXt are available in the transformers or compatible model hubs, enabling fine-tuning via the Hugging Face API.
- **TensorFlow**: While not officially by the authors, third-party ports exist.

### [F3-Net](https://arxiv.org/pdf/2007.09355)
> F3-Net (Frequency in Face Forgery Network) is a CNN architecture specifically designed for deepfake detection. The model leverages frequency-aware clues, 1) **frequency-aware decomposed image copmonents** and 2) **local frequency statistics**, information by incorporating FFT-based preprocessing to extract these artifacts. 

The reason I include this is not so as much to use it, although we can definately play test and fine tune, but to take reference to what techniques they use. So, we can just discuss some notable techniques and their results. 

**Efficiency**: F3-Net is engineered to maintain a balance between accuracy and computational cost. It uses dual-path architecture, combining spatial and frequency-domain branches, and remains lightweight relative to transformer-heavy models. The frequency branch focuses on localized FFT features, ensuring efficient computation even in high-resolution image analysis. This makes F3-Net suitable for research and moderate-scale deployment without requiring large-scale compute resources.

**Performance**: F3-Net achieves strong benchmark results, outperforming several state-of-the-art models. On datasets such as **FaceForensics++**, it achieves **~90.4% accuracy and 0.933 AUC**, representing an approximate **3.5% gain** over prior best-performing CNN baselines. Similarly, on the **NeuralTextures** dataset, F3-Net improves accuracy by about **4.2%**. 

**Framework Support**: F3-Net is implemented primarily in **PyTorch** with pretrained checkpoints available in the research community. While no official TensorFlow implementation exists, it can be exported via ONNX for deployment in FastAPI. Its tight integration with FFT preprocessing makes it a strong candidate for hybrid pipelines combining CNN and transformer stages.


## Recommended FastAPI pipeline

| Pipeline Stage | Recommended Models | Rationale |
|----------------|--------------------|-----------|
| **FFT + CNN Feature Extractor** | ResNet50, EfficientNet-B0 (Pytorch) | Easy FFT integration, balanced compute, great for feature fusion. |
| **Transformer Fusion (ViT stage)** | ViT-B/16 or Swin-T (PyTorch / HF) | Learns global relationships among CNN/FFT feature embeddings. |
| **End-to-End Hybrid** | ResNet50 + ViT-B/16 (PyTorch) | Clean modularity; easily exportable to ONNX or TorchScript for FastAPI deployment. |

For framework, I believe that ***Pytorch*** would be the most suitable option for our project goals. 

**Architecture**: Hybrid ResNet50 + ViT-B/16, optionally integrating FFT-magnitude preprocessing.

**Reasoning**:
- ResNet50 provides robust local/frequency-sensitive features.
- ViT-B/16 adds global context and temporal/spatial coherence.
- FFT preprocessing can be attached as an auxiliary input channel or branch.
- PyTorch -> TorchScript -> FastAPI deployment is smooth, modular, and reproducible.


<br>
Notable resource: https://iplab.dmi.unict.it/mfs/Deepfakes/

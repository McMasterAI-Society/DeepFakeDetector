## **What are some of the main distinctive features of deepfakes?**

*   **Artifacts:** sometimes deepfakes contain artifacts, e.g., inconsistent lighting, unnatural skin tone, irregular eyes, mouth / hairline.
*   **Resolution inconsistencies:** mismatched pixel resolutions in different parts of the image, e.g., higher quality face relative to background
*   **Anomalies in eye movements + facial expressions:** subtle inconsistencies in facial movements or unnatural eye blinking patterns

## **How might CNNs be used to detect deepfake images?**

1.  **Acquire a dataset of real vs deepfake images**

**Real images:** Images of real people’s faces through public datasets like CelebA, Labeled Faces in the Wild, VGGFace, etc.

**Deepfake images:** manipulated images or frames generated via deepfake tools (e.g., face-swapping, GAN-based generation, other methods)

2.  **Possible approach: transfer learning from pre-trained models**

It’s worth noting that there are already CNNs out there that have been trained on millions of images from public datasets. It might be slightly beyond our scope to try and achieve those same levels of training on our own laptops.

Transfer learning involves taking a model pre-trained on a large dataset and adapting it to more narrow use cases by retraining some of the final layers of the CNN

## **Considerations when designing the CNN**

1.  Multi-scale feature extraction

CNNs can be designed so that they work at different scales, i.e., looking over both small-scale pixel structures (e.g., patches of inconsistent lighting) as well as larger structural features (e.g., mismatched face shape).

2.  Attention mechanism

If our use case is very well-defined, we can pair the CNNs with object detection algorithms and zero in on certain parts of the image (e.g., the face). This could help limit the data the model must look at to focus on the most plausible areas for deepfake identification and manipulation, improving overall detection accuracy.

3.  Ensemble methods

Kind of like a “meta-analysis” of CNNs for deepfake: design a kind of pipeline that makes use of various different deepfake models specialized to specific tasks, make the necessary changes and adjustment to their training, and use a weighted combination of each of their outputs to classify an image as a deepfake or not.
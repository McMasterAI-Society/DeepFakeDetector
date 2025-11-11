# Artifacts and Features - Research

## What features do AI detectors focus on? (Frequency, Upsampling, Noise, Texture, Semantic, etc.)

- **Frequency-domain artifacts:**
  - Generated images often deviate from the smooth/natural frequency decay of real photos. (frequency domain shows how image details are distributed across different levels of sharpness or detail.)
  - They may have excess high-frequency energy (too much fine detail), periodic spikes (repeated patterns), or altered radial patterns.
  - Diffusion models (newer type of image generator, such as Stable Diffusion or DALL-E 3, that starts with random noise and refines it step-by-step into a realistic image, different from older GAN models that create an image in one shot) reduced obvious spectral peaks but still show subtle frequency differences that detectors can spot.

- **Upsampling patterns:**
  - Upsampling is the process of increasing an image’s resolution by generating additional pixels to enhance detail and sharpness.
  - When AI models upsample, small repeating patterns or pixel correlations (like faint checkerboards or edge artifacts) can appear.
  - Even with newer diffusion models (build images step-by-step from noise), detectors still pick up on these subtle pixel faults.

- **Noise signatures:**
  - Real cameras produce consistent sensor noise (PRNU - Photo Response Non-Uniformity, brightness variations caused by small imperfections in each camera sensor) and demosaicing traces (subtle repeating color patterns created when the camera combines red, green, and blue pixel data into a full-color image).
  - AI-generated images usually lack these natural camera traces.
  - Detectors analyze the fine pixel-level noise in an image to check for these missing or fake camera signals.

- **Texture inconsistencies:**
  - AI often struggles with natural textures such as skin, hair, or fabric. Generated textures might look overly smooth, too regular, etc.
  - Texture-based detectors focus on these surface-level inconsistencies rather than overall image meaning.

- **Semantic anomalies:**
  - Logical or physical mistakes (strange hands, inconsistent shadows, warped text, etc.) often appear in AI-generated images.
  - Detectors using semantic reasoning catch these content-based clues, looking beyond just pixel and texture analysis.

---

## Which features have remained consistent and which have evolved?

- **Consistent features:**
  - Noise-based cues: Real camera noise (PRNU/demosaicing) is still one of the most reliable differences between real and AI-generated images.
  - Frequency-domain analysis: Using deviations in frequency continue to work well across different models.
  - Texture-level features: Texture irregularities and pixel relationships remain strong general indicators of AI generation.

- **Evolved features:**
  - Hybrid spatial-frequency models: Combining pixel-space and frequency-space analysis has improved detection against new diffusion models.
  - Semantic anomaly detection: Modern detectors now use AI reasoning (like Vision-Language Models) to identify logic errors in content.
  - Multi-scale noise consistency checks: Recent methods compare noise patterns at multiple resolutions to catch more advanced AI-generated noise.

- **Less effective or outdated cues:**
  - Checkerboard artifacts: Common in older GANs but mostly gone in newer models.
  - Single-type detectors: Methods that only analyze pixels or one domain don’t work well anymore (must combine frequency + pixel analysis).
  - Model-specific training: Detectors focused on one generator (like StyleGAN) fail on newer types (modern ones use general/flexible features).

---

## What detection patterns generalize well across different AI image generators?

- Model real-image frequency behavior and mark deviations from it.
- Compare PRNU and noise patterns at different scales (pixel-level to picture-level) to detect AI-generated noise.
- Analyze pixel patterns through various upsampling layers.
- Use texture-based analysis to focus on micro-patterns.
- Combine spatial, frequency, and semantic cues for a versatile detection model.

---

## Why this matters for DeepFakeDetector

- Use FFT (Fast Fourier Transform - converts pixels to frequency domain) analysis to find subtle frequency artifacts that pixel models miss.
- Build a hybrid model combining CNNs (for pixel patterns), ViTs (for overall context), and FFT features (for frequency clues).
- Add noise and texture checks to detect missing camera noise or overly smooth surfaces.
- Train on a mix of real and AI images from many generators to make the detector adaptable.

## Extra Sources for Reference
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10674908/ (discusses pixel-level features)
- https://proceedings.mlr.press/v119/frank20a/frank20a.pdf (frequency analysis to detect deepfakes)
- https://openaccess.thecvf.com/content/CVPR2025/papers/Zhong_Beyond_Generation_A_Diffusion-based_Low-level_Feature_Extractor_for_Detecting_AI-generated_CVPR_2025_paper.pdf (discusses low-level features)
# 🧠 DeepFakeDetector – AI-Generated Image Detection  


## 📖 Overview
**DeepFakeDetector** is a research and development project by members of the **McMaster Artificial Intelligence Society** focused on detecting AI-generated and deepfake images.  

With the rapid advancement of image synthesis models like **StyleGAN**, **Stable Diffusion**, and **DALL·E**, distinguishing synthetic content from authentic media has become increasingly difficult. This project aims to address that challenge by training models capable of detecting subtle visual artifacts that reveal an image’s generative origin.

Our system will combine multiple approaches:
- **CNNs (Convolutional Neural Networks)** for spatial feature detection.  
- **Vision Transformers (ViTs)** for global context understanding.  
- **Frequency-domain analysis (FFT)** to capture artifacts invisible in the pixel domain.  

The final product will be a **web-based dashboard** that allows users to upload an image and receive:
- A real vs. AI-generated classification score.  
- A visual explanation (saliency or attention map) showing why the image was flagged.  

---

## 🎯 Project Goal
To create a reliable, interpretable, and accessible AI system that:
1. Accurately detects AI-generated and deepfake images.  
2. Provides visual explanations to promote transparency and trust in AI systems.  
3. Offers a usable web platform for real-time image authenticity verification.  
4. Contributes open research and tools to the broader AI ethics and misinformation-detection community.

---

## 👥 Team

| Name | Role |
|------|------|
| **Lukhsaan Elankumaran** | Project Lead |
| **Md Nafieu Hossain Alif** | ML Engineer |
| **Zuhair Qureshi** | ML Engineer |
| **Krish Bhagirath** | Research & Modeling |
| **Oriana Rueckert** | Data Engineer |
| **Vihaan Singhal** | Data & Deployment Engineer |

---

## 🧩 End Goal
By the end of the term, the project will deliver:
- A trained hybrid **CNN + ViT + FFT-based model** for image authenticity detection.  
- **Explainability tools** such as Grad-CAM and attention heatmaps for visual insights.  
- A **Next.js + FastAPI web app** that performs real-time deepfake detection from user uploads.  
- **Technical documentation** and a short demo video presenting results and key findings.

---

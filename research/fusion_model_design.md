# Fusion Model Design Document

**Project:** DeepFakeDetector
**Purpose:** Combine outputs from multiple submodels (CNN Transfer, ViT Base, DeiT Distilled, Gradient Field CNN) into a single, robust final prediction
**Author:** McMaster AI Society Research Team
**Date:** 2026-02-10

---

## Table of Contents

1. [Fusion Model Choice (What Model & Why)](#1-fusion-model-choice-what-model--why)
2. [Fusion Model Inputs & Outputs](#2-fusion-model-inputs--outputs)
3. [Training Workflow for the Fusion Model](#3-training-workflow-for-the-fusion-model)
4. [How the Fusion Model is Stored on Hugging Face](#4-how-the-fusion-model-is-stored-on-hugging-face)
5. [How the Backend Pulls the Fusion Model from Hugging Face](#5-how-the-backend-pulls-the-fusion-model-from-hugging-face)
6. [End-to-End Inference Flow (CRITICAL)](#6-end-to-end-inference-flow-critical)
7. [Explainability at the Fusion Level](#7-explainability-at-the-fusion-level)

---

## 1. Fusion Model Choice (What Model & Why)

### Overview

We evaluate **two fusion approaches** for combining outputs from four specialized submodels:

1. **Feature-Level Fusion (Primary)**: Concatenate 512-D embedding vectors + meta-model
2. **Probability-Level Stacking (Baseline)**: Logistic regression on output probabilities

Both approaches produce a final classification: **Real** (0) vs **AI-generated/Fake** (1).

---

### Comparative Analysis of Fusion Approaches

| Approach | Complexity | Expressiveness | Explainability | Speed | Training Data Needed | Suitable for Demo/Paper |
|----------|------------|----------------|----------------|-------|---------------------|------------------------|
| **Feature-Level Fusion (Meta-Model)** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ (moderate) | ⭐⭐⭐⭐⭐ |
| **Probability Stacking (Logistic Regression)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (minimal) | ⭐⭐⭐⭐ |
| Rule-based Weighted Average | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | N/A (no training) | ⭐⭐⭐ |
| Attention-Based Fusion | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ (moderate) | ⭐⭐⭐⭐⭐ |

---

### Detailed Comparison

#### 1. **Feature-Level Fusion with Meta-Model (PRIMARY RECOMMENDATION)**

**How it works:**
1. Each submodel outputs a **512-D feature embedding** (extracted before final classification layer)
2. Concatenate all embeddings: `f_combined = [f_cnn; f_vit; f_deit; f_gradfield]` = **2048-D vector** (4 models × 512-D each)
3. Train a **meta-model** (dense neural network) to classify from the combined features

**Meta-Model Architecture:**
```
Input: 2048-D concatenated feature vector
   ↓
Dense Layer 1: 2048 → 512 (ReLU + Dropout 0.5)
   ↓
Dense Layer 2: 512 → 128 (ReLU + Dropout 0.3)
   ↓
Output Layer: 128 → 1 (Sigmoid)
   ↓
Final Output: Probability ∈ [0, 1]
```

**Advantages:**
- **High expressiveness**: Can learn complex non-linear interactions between model features
- **Richer information**: Uses full embedding space, not just final probabilities
- **Better generalization**: More robust to distribution shifts
- **Research novelty**: Suitable for publication (more sophisticated than simple stacking)
- **Attention compatibility**: Can be extended with attention mechanisms

**Disadvantages:**
- **More parameters**: ~1.3M parameters (risk of overfitting)
- **Requires more training data**: Need 5,000+ validation samples
- **Slower inference**: Forward pass through neural network (~2-5ms)
- **Less interpretable**: Harder to explain which model contributed most
- **Training complexity**: Requires careful hyperparameter tuning

**When to use:**
- You have ≥5,000 validation samples for fusion training
- You want state-of-the-art performance for research publication
- You need to model complex interactions (e.g., "trust CNN only when ViT is uncertain")

---

#### 2. **Probability-Level Stacking with Logistic Regression (BASELINE)**

**How it works:**
1. Each submodel outputs a single **probability** (post-softmax) for "fake" class
2. Assemble probability vector: `x = [p_cnn, p_vit, p_deit, p_gradfield]` = **4-D vector**
3. Train a **logistic regression** classifier on these probabilities

**Advantages:**
- **Extreme simplicity**: Only 5 parameters (4 weights + 1 bias)
- **Highly explainable**: Coefficients directly show each model's contribution
- **Fast inference**: Single matrix multiplication (<1ms)
- **Stable training**: Convex optimization, no local minima
- **Low data requirements**: Works with 1,000+ validation samples
- **No overfitting risk**: Too simple to overfit
- **Industry standard**: Used in Kaggle competitions, production systems

**Disadvantages:**
- **Limited expressiveness**: Cannot model non-linear interactions
- **Less information**: Discards rich embedding information
- **Assumes independence**: Treats model predictions as independent features

**When to use:**
- You have <5,000 validation samples
- You need fast, explainable predictions
- You want a strong baseline to compare against
- You need production-ready stability

---

#### 3. **Attention-Based Fusion (RESEARCH EXTENSION)**

**How it works:**
1. Extract feature embeddings from each model: `[f_cnn, f_vit, f_deit, f_gradfield]`
2. Learn **attention weights** via a small network:
   ```
   Attention Module:
   Input: 4 × 512-D feature tensors
   → Linear projection → Softmax → Attention weights [w1, w2, w3, w4]
   ```
3. Compute weighted sum: `f_fused = w1·f_cnn + w2·f_vit + w3·f_deit + w4·f_gradfield`
4. Pass through classification head

**Advantages:**
- **Interpretable weights**: Can visualize which model is trusted per sample
- **Dynamic fusion**: Weights adapt per image (unlike fixed logistic regression)
- **State-of-the-art**: Aligns with modern ML research trends

**Disadvantages:**
- **More complex**: Harder to implement and tune
- **Slower training**: Requires backpropagation through attention mechanism

**When to use:** Phase 2 research extension, after validating simpler approaches

---

### **Recommended Implementation Strategy**

**Phase 1 (MVP - Weeks 1-2):**
- Implement **Probability-Level Stacking** (Logistic Regression)
- Rationale: Fast to implement, stable, explainable
- Success metric: Validation AUROC ≥ 0.88

**Phase 2 (Advanced - Weeks 3-4):**
- Implement **Feature-Level Fusion** (Meta-Model)
- Rationale: Unlock full performance potential
- Success metric: Validation AUROC ≥ 0.92

**Phase 3 (Research Extension - Optional):**
- Implement **Attention-Based Fusion**
- Rationale: Publishable novelty, dynamic explainability

**Decision Rule:**
- If Phase 1 achieves AUROC ≥ 0.90 → Stay with logistic regression (simplicity wins)
- If Phase 1 achieves AUROC < 0.90 → Proceed to Phase 2 (need more power)

---

## 2. Fusion Model Inputs & Outputs

### 2.1 Input Format — Two Variants

#### **Variant A: Feature-Level Fusion (Primary)**

Each submodel must expose a `get_embedding()` method that returns the penultimate layer features:

```python
# CNN Transfer (EfficientNet-B0)
embedding_cnn = cnn_model.get_embedding(image)  # Shape: (512,)

# ViT Base
embedding_vit = vit_model.get_embedding(image)  # Shape: (512,) [CLS token]

# DeiT Distilled
embedding_deit = deit_model.get_embedding(image)  # Shape: (512,) [CLS token]

# Gradient Field CNN
embedding_gradfield = gradfield_model.get_embedding(image)  # Shape: (512,)
```

**Fusion input:**
```python
import numpy as np

# Concatenate all embeddings
x = np.concatenate([
    embedding_cnn,
    embedding_vit,
    embedding_deit,
    embedding_gradfield
])  # Shape: (2048,)
```

---

#### **Variant B: Probability-Level Stacking (Baseline)**

Each submodel outputs a probability via standard `predict()` method:

```python
# Get probabilities for "fake" class (index 1)
prob_cnn = cnn_model.predict(image)["prob"]  # ∈ [0, 1]
prob_vit = vit_model.predict(image)["prob"]
prob_deit = deit_model.predict(image)["prob"]
prob_gradfield = gradfield_model.predict(image)["prob"]
```

**Fusion input:**
```python
x = np.array([prob_cnn, prob_vit, prob_deit, prob_gradfield])  # Shape: (4,)
```

---

### 2.2 Output Format (Unified)

Both fusion variants produce the same output format:

```python
{
    "prob_fake": float,        # ∈ [0, 1], probability image is AI-generated
    "prob_real": float,        # = 1 - prob_fake
    "label": str,              # "real" or "fake"
    "confidence": float,       # ∈ [0, 1], certainty of prediction
    "label_index": int         # 0 = real, 1 = fake
}
```

**Classification threshold:**
```python
threshold = 0.5  # Default, can be tuned on validation set
label_index = 1 if prob_fake >= threshold else 0
label = "fake" if label_index == 1 else "real"
```

**Confidence calculation:**
```python
# Confidence = distance from decision boundary
confidence = abs(prob_fake - 0.5) * 2  # Maps [0.5, 1.0] → [0, 1]

# Examples:
# prob_fake = 0.95 → confidence = 0.90 (very confident it's fake)
# prob_fake = 0.52 → confidence = 0.04 (uncertain)
# prob_fake = 0.05 → confidence = 0.90 (very confident it's real)
```

---

### 2.3 Standardized Submodel Interface

All submodels must implement this interface:

```python
class BaseSubmodel:
    def predict(self, image: np.ndarray) -> dict:
        """
        Standard prediction interface.

        Args:
            image: RGB image array (224, 224, 3) or batch (B, 224, 224, 3)

        Returns:
            {
                "logit": float,         # Raw logit score
                "prob": float,          # Probability of fake ∈ [0, 1]
                "embedding": np.array,  # 512-D feature vector (optional)
                "meta": dict            # Additional metadata
            }
        """
        pass

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature embedding (for feature-level fusion).

        Args:
            image: RGB image array (224, 224, 3)

        Returns:
            embedding: 512-D numpy array
        """
        pass
```

**Implementation note:** Submodels should cache embeddings during `predict()` to avoid redundant forward passes.

---

## 3. Training Workflow for the Fusion Model

### 3.1 Dataset Splitting Strategy

```
Original Dataset (e.g., OpenFake + WildFake)
├── Train Split (70%): Used to train all 4 submodels
├── Validation Split (15%): Used to train fusion model
└── Test Split (15%): Final evaluation only (never seen during training)
```

**Critical rule:** The fusion model must NEVER train on data that the submodels saw during training. This prevents overfitting and ensures proper stacking behavior.

---

### 3.2 Step-by-Step Training Workflow

#### **Step 1: Train All Submodels**

Train each model independently on the **train split** (70%):

| Submodel | Backbone | Training Data | HuggingFace Repo |
|----------|----------|---------------|------------------|
| CNN Transfer | EfficientNet-B0 | Train split (70%) | `DeepFakeDetector/cnn-transfer-efficientnet-b0` |
| ViT Base | ViT-B/16 | Train split (70%) | `DeepFakeDetector/vit-base-patch16-224` |
| DeiT Distilled | DeiT-B/16 distilled | Train split (70%) | `DeepFakeDetector/deit-distilled-patch16-224` |
| Gradient Field CNN | Custom CNN on FFT | Train split (70%) | `DeepFakeDetector/gradient-field-cnn` |

**Each model should save:**
- `weights.pt`: PyTorch model weights
- `config.json`: Configuration (input size, normalization, label mapping)
- `predict.py`: Prediction interface

---

#### **Step 2: Generate Submodel Predictions on Validation Set**

Run inference with all 4 trained submodels on the **validation split** (15%):

**For Feature-Level Fusion:**
```python
import numpy as np
import json
from pathlib import Path

# Load all submodels
cnn_model = load_model("DeepFakeDetector/cnn-transfer-efficientnet-b0")
vit_model = load_model("DeepFakeDetector/vit-base-patch16-224")
deit_model = load_model("DeepFakeDetector/deit-distilled-patch16-224")
gradfield_model = load_model("DeepFakeDetector/gradient-field-cnn")

# Generate embeddings for validation set
validation_embeddings = []

for image_path, true_label in validation_set:
    image = load_and_preprocess(image_path)  # Returns (224, 224, 3) RGB

    # Extract embeddings from each submodel
    emb_cnn = cnn_model.get_embedding(image)         # (512,)
    emb_vit = vit_model.get_embedding(image)         # (512,)
    emb_deit = deit_model.get_embedding(image)       # (512,)
    emb_gradfield = gradfield_model.get_embedding(image)  # (512,)

    # Concatenate into 2048-D feature vector
    features = np.concatenate([emb_cnn, emb_vit, emb_deit, emb_gradfield])

    validation_embeddings.append({
        "image_id": Path(image_path).name,
        "features": features.tolist(),
        "label": 1 if true_label == "fake" else 0
    })

# Save to disk
with open("validation_embeddings.json", "w") as f:
    json.dump(validation_embeddings, f)
```

**For Probability-Level Stacking:**
```python
validation_probabilities = []

for image_path, true_label in validation_set:
    image = load_and_preprocess(image_path)

    # Get probabilities from each submodel
    prob_cnn = cnn_model.predict(image)["prob"]
    prob_vit = vit_model.predict(image)["prob"]
    prob_deit = deit_model.predict(image)["prob"]
    prob_gradfield = gradfield_model.predict(image)["prob"]

    validation_probabilities.append({
        "image_id": Path(image_path).name,
        "features": [prob_cnn, prob_vit, prob_deit, prob_gradfield],
        "label": 1 if true_label == "fake" else 0
    })

with open("validation_probabilities.json", "w") as f:
    json.dump(validation_probabilities, f)
```

---

#### **Step 3: Train the Fusion Model**

##### **Option A: Feature-Level Meta-Model (PyTorch)**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Load cached validation embeddings
with open("validation_embeddings.json", "r") as f:
    data = json.load(f)

X = np.array([d["features"] for d in data])  # Shape: (N, 2048)
y = np.array([d["label"] for d in data])     # Shape: (N,)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y).unsqueeze(1)

# Define meta-model
class MetaModel(nn.Module):
    def __init__(self, input_dim=2048):
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# Initialize model
model = MetaModel(input_dim=2048)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training loop
num_epochs = 50
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size].to(device)
        batch_y = y_train[i:i+batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Validation metrics
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_train.to(device)).cpu().numpy().flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        auroc = roc_auc_score(y, y_pred_proba)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | "
          f"AUROC: {auroc:.3f} | Acc: {acc:.3f} | F1: {f1:.3f}")

# Save trained model
torch.save(model.state_dict(), "fusion_metamodel.pt")
```

**Expected performance:**
- Validation AUROC: **0.92-0.96**
- Validation Accuracy: 0.88-0.93
- Validation F1: 0.88-0.93

---

##### **Option B: Probability Stacking (Logistic Regression)**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np

# Load cached validation probabilities
with open("validation_probabilities.json", "r") as f:
    data = json.load(f)

X = np.array([d["features"] for d in data])  # Shape: (N, 4)
y = np.array([d["label"] for d in data])     # Shape: (N,)

# Train logistic regression
fusion_model = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="liblinear",
    max_iter=1000,
    random_state=42
)

fusion_model.fit(X, y)

# Evaluate
y_pred_proba = fusion_model.predict_proba(X)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

print(f"Validation AUROC: {roc_auc_score(y, y_pred_proba):.3f}")
print(f"Validation Accuracy: {accuracy_score(y, y_pred):.3f}")
print(f"Validation F1: {f1_score(y, y_pred):.3f}")

# Extract learned weights
print(f"\nFusion weights: {fusion_model.coef_[0]}")
print(f"Fusion bias: {fusion_model.intercept_[0]}")

# Save model
import joblib
joblib.dump(fusion_model, "fusion_logreg.pkl", compress=3)
```

**Expected performance:**
- Validation AUROC: **0.88-0.92**
- Validation Accuracy: 0.85-0.90
- Validation F1: 0.85-0.90

---

#### **Step 4: Avoiding Overfitting**

**For Meta-Model:**
- Use dropout (0.3-0.5) in hidden layers
- Apply L2 regularization (weight_decay=1e-4)
- Use early stopping based on validation loss
- Monitor calibration (reliability diagrams)

**For Logistic Regression:**
- Use L2 regularization (built-in via parameter `C`)
- Very low overfitting risk (only 5 parameters)

**Common practices:**
- Never tune threshold on validation set (use default 0.5)
- Reserve test set for final evaluation only
- Monitor calibration curves

---

#### **Step 5: Final Evaluation on Test Set**

```python
# Generate test predictions (same process as validation)
test_predictions = generate_predictions(test_set, all_submodels)

# For meta-model
X_test = torch.FloatTensor([d["features"] for d in test_predictions])
y_test = np.array([d["label"] for d in test_predictions])

model.eval()
with torch.no_grad():
    y_test_proba = model(X_test.to(device)).cpu().numpy().flatten()
y_test_pred = (y_test_proba >= 0.5).astype(int)

# For logistic regression
X_test = np.array([d["features"] for d in test_predictions])
y_test_proba = fusion_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= 0.5).astype(int)

# Report metrics
print(f"Test AUROC: {roc_auc_score(y_test, y_test_proba):.3f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
print(f"Test F1: {f1_score(y_test, y_test_pred):.3f}")
```

**Report these test metrics in the paper.**

---

### 3.3 Evaluation Metrics

| Metric | Purpose | Target (Meta-Model) | Target (Logistic Reg) |
|--------|---------|---------------------|----------------------|
| **AUROC** | Overall discriminative ability | > 0.92 | > 0.88 |
| **Accuracy** | Correct classifications | > 0.89 | > 0.86 |
| **F1 Score** | Balance precision/recall | > 0.89 | > 0.86 |
| **Calibration Error** | Probability reliability | < 0.05 | < 0.05 |
| **Gain over best single model** | Fusion benefit | +5-7% AUROC | +3-5% AUROC |

---

## 4. How the Fusion Model is Stored on Hugging Face

### 4.1 Repository Structure

#### **For Feature-Level Meta-Model:**

```
DeepFakeDetector/fusion-metamodel/
├── fusion_metamodel.pt          # PyTorch model weights
├── model_architecture.py        # MetaModel class definition
├── config.json                  # Metadata and configuration
├── label_map.json               # Class index to name mapping
├── README.md                    # Model card with usage instructions
└── test_metrics.json            # Performance metrics for transparency
```

#### **For Probability Stacking (Logistic Regression):**

```
DeepFakeDetector/fusion-logreg/
├── fusion_logreg.pkl            # Serialized sklearn model
├── config.json                  # Metadata and configuration
├── label_map.json               # Class index to name mapping
├── README.md                    # Model card with usage instructions
└── test_metrics.json            # Performance metrics
```

---

### 4.2 File Specifications

#### **1. fusion_metamodel.pt (Feature-Level)**

PyTorch state dict saved using `torch.save()`:

```python
# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': 2048,
    'architecture': 'MetaModel'
}, "fusion_metamodel.pt")

# Load
checkpoint = torch.load("fusion_metamodel.pt")
model = MetaModel(input_dim=checkpoint['input_dim'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

**Size:** ~5-10 MB

---

#### **2. model_architecture.py**

Contains the MetaModel class definition:

```python
import torch.nn as nn

class MetaModel(nn.Module):
    def __init__(self, input_dim=2048):
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x
```

---

#### **3. config.json**

**For Meta-Model:**
```json
{
  "model_type": "MetaModelFusion",
  "version": "1.0.0",
  "input_features": [
    "embedding_cnn_transfer",
    "embedding_vit_base",
    "embedding_deit_distilled",
    "embedding_gradfield_cnn"
  ],
  "input_dim": 2048,
  "embedding_dim_per_model": 512,
  "num_submodels": 4,
  "threshold": 0.5,
  "architecture": {
    "layer1": "2048 → 512 (ReLU, Dropout 0.5)",
    "layer2": "512 → 128 (ReLU, Dropout 0.3)",
    "output": "128 → 1 (Sigmoid)"
  },
  "submodel_versions": {
    "cnn_transfer": "v1.2",
    "vit_base": "v1.1",
    "deit_distilled": "v1.0",
    "gradfield_cnn": "v1.0"
  },
  "training_date": "2026-02-10",
  "framework": "PyTorch",
  "torch_version": "2.1.0"
}
```

**For Logistic Regression:**
```json
{
  "model_type": "LogisticRegressionStacking",
  "version": "1.0.0",
  "input_features": [
    "prob_cnn_transfer",
    "prob_vit_base",
    "prob_deit_distilled",
    "prob_gradfield_cnn"
  ],
  "num_features": 4,
  "threshold": 0.5,
  "submodel_versions": {
    "cnn_transfer": "v1.2",
    "vit_base": "v1.1",
    "deit_distilled": "v1.0",
    "gradfield_cnn": "v1.0"
  },
  "training_date": "2026-02-10",
  "framework": "scikit-learn",
  "sklearn_version": "1.3.0"
}
```

---

#### **4. label_map.json**

```json
{
  "0": "real",
  "1": "fake"
}
```

---

#### **5. README.md (Model Card)**

**For Meta-Model:**
```markdown
# DeepFake Fusion Model — Feature-Level Meta-Model

## Model Description
Combines 512-D feature embeddings from 4 specialized deepfake detection models:
- CNN Transfer (EfficientNet-B0)
- ViT Base (patch16-224)
- DeiT Distilled (patch16-224)
- Gradient Field CNN

## Architecture
- Input: 2048-D concatenated embeddings (4 models × 512-D)
- Hidden layers: 2048 → 512 → 128
- Output: Sigmoid activation for binary classification

## Usage

```python
import torch
from model_architecture import MetaModel

# Load model
model = MetaModel(input_dim=2048)
checkpoint = torch.load("fusion_metamodel.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Input: concatenated embeddings from 4 submodels
embeddings = torch.FloatTensor([[...]])  # Shape: (1, 2048)

# Predict
with torch.no_grad():
    prob_fake = model(embeddings).item()
    label = "fake" if prob_fake >= 0.5 else "real"
```

## Performance (Test Set)
- AUROC: 0.93
- Accuracy: 0.90
- F1 Score: 0.90
```

---

#### **6. test_metrics.json**

**For Meta-Model:**
```json
{
  "test_auroc": 0.93,
  "test_accuracy": 0.90,
  "test_f1": 0.90,
  "test_samples": 1500,
  "calibration_error": 0.04,
  "training_parameters": {
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "dropout": [0.5, 0.3]
  }
}
```

**For Logistic Regression:**
```json
{
  "test_auroc": 0.90,
  "test_accuracy": 0.88,
  "test_f1": 0.88,
  "test_samples": 1500,
  "fusion_weights": [0.30, 0.42, 0.18, 0.10],
  "fusion_bias": -0.12,
  "calibration_error": 0.03
}
```

---

## 5. How the Backend Pulls the Fusion Model from Hugging Face

### 5.1 Backend Model Loading Strategy

**Approach:** Download once at startup, cache in memory

---

### 5.2 Implementation (FastAPI)

#### **For Feature-Level Meta-Model:**

```python
# backend/models/fusion_loader.py

import torch
import json
from pathlib import Path
from huggingface_hub import snapshot_download
import sys

# Configuration
FUSION_REPO = "DeepFakeDetector/fusion-metamodel"
CACHE_DIR = Path.home() / ".cache" / "deepfake_fusion"

class FusionModelLoader:
    def __init__(self):
        self.model = None
        self.config = None
        self.label_map = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def download_and_load(self, revision="main"):
        """
        Download fusion model from HuggingFace and load into memory.

        Args:
            revision: Git branch/tag/commit hash (default: "main")
        """
        print(f"Downloading fusion model from {FUSION_REPO}...")

        # Download all files
        repo_path = snapshot_download(
            repo_id=FUSION_REPO,
            revision=revision,
            cache_dir=CACHE_DIR,
            local_dir=CACHE_DIR / "fusion_model",
            local_dir_use_symlinks=False
        )

        print(f"Model downloaded to {repo_path}")

        # Add to Python path to import model_architecture
        sys.path.insert(0, str(repo_path))
        from model_architecture import MetaModel

        # Load config
        config_path = Path(repo_path) / "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load model
        model_path = Path(repo_path) / "fusion_metamodel.pt"
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = MetaModel(input_dim=self.config["input_dim"])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load label map
        label_map_path = Path(repo_path) / "label_map.json"
        with open(label_map_path, "r") as f:
            self.label_map = json.load(f)

        print(f"Fusion model ready (version {self.config['version']})")
        return self

    def predict(self, embeddings: list[float]) -> dict:
        """
        Run fusion inference on concatenated embeddings.

        Args:
            embeddings: Concatenated list of 4 × 512-D embeddings (length 2048)

        Returns:
            {
                "prob_fake": float,
                "prob_real": float,
                "label": str,
                "confidence": float,
                "label_index": int
            }
        """
        import numpy as np

        # Validate input
        assert len(embeddings) == self.config["input_dim"], \
            f"Expected {self.config['input_dim']} features, got {len(embeddings)}"

        # Convert to tensor
        X = torch.FloatTensor(embeddings).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            prob_fake = self.model(X).item()

        prob_real = 1 - prob_fake

        # Classify
        threshold = self.config["threshold"]
        label_index = 1 if prob_fake >= threshold else 0
        label = self.label_map[str(label_index)]

        # Confidence
        confidence = abs(prob_fake - 0.5) * 2

        return {
            "prob_fake": float(prob_fake),
            "prob_real": float(prob_real),
            "label": label,
            "confidence": float(confidence),
            "label_index": int(label_index)
        }

# Global singleton
fusion_loader = FusionModelLoader()
```

---

#### **For Probability Stacking (Logistic Regression):**

```python
import joblib
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

FUSION_REPO = "DeepFakeDetector/fusion-logreg"
CACHE_DIR = Path.home() / ".cache" / "deepfake_fusion"

class FusionModelLoader:
    def __init__(self):
        self.model = None
        self.config = None
        self.label_map = None

    def download_and_load(self, revision="main"):
        print(f"Downloading fusion model from {FUSION_REPO}...")

        repo_path = snapshot_download(
            repo_id=FUSION_REPO,
            revision=revision,
            cache_dir=CACHE_DIR,
            local_dir=CACHE_DIR / "fusion_model",
            local_dir_use_symlinks=False
        )

        # Load sklearn model
        model_path = Path(repo_path) / "fusion_logreg.pkl"
        self.model = joblib.load(model_path)

        # Load config
        config_path = Path(repo_path) / "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load label map
        label_map_path = Path(repo_path) / "label_map.json"
        with open(label_map_path, "r") as f:
            self.label_map = json.load(f)

        print(f"Fusion model ready (version {self.config['version']})")
        return self

    def predict(self, submodel_probs: list[float]) -> dict:
        """
        Run fusion inference on submodel probabilities.

        Args:
            submodel_probs: [prob_cnn, prob_vit, prob_deit, prob_gradfield]

        Returns:
            {
                "prob_fake": float,
                "prob_real": float,
                "label": str,
                "confidence": float,
                "label_index": int
            }
        """
        # Validate input
        assert len(submodel_probs) == self.config["num_features"], \
            f"Expected {self.config['num_features']} features, got {len(submodel_probs)}"

        # Reshape for sklearn
        X = np.array(submodel_probs).reshape(1, -1)

        # Predict
        prob_fake = self.model.predict_proba(X)[0, 1]
        prob_real = 1 - prob_fake

        # Classify
        threshold = self.config["threshold"]
        label_index = 1 if prob_fake >= threshold else 0
        label = self.label_map[str(label_index)]

        # Confidence
        confidence = abs(prob_fake - 0.5) * 2

        return {
            "prob_fake": float(prob_fake),
            "prob_real": float(prob_real),
            "label": label,
            "confidence": float(confidence),
            "label_index": int(label_index)
        }

fusion_loader = FusionModelLoader()
```

---

### 5.3 FastAPI Startup Hook

```python
# backend/main.py

from fastapi import FastAPI
from models.fusion_loader import fusion_loader

app = FastAPI(title="DeepFake Detection API")

@app.on_event("startup")
async def startup_event():
    """Load all models at server startup"""
    print("Loading fusion model...")
    fusion_loader.download_and_load(revision="main")  # or pin to commit hash
    print("Fusion model loaded successfully!")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "fusion_model_loaded": fusion_loader.model is not None
    }
```

---

### 5.4 Version Pinning (Production Best Practice)

```python
# Pin to specific commit hash (recommended for production)
fusion_loader.download_and_load(revision="a3f7d9c")

# Or pin to semantic version tag
fusion_loader.download_and_load(revision="v1.0.0")
```

---

## 6. End-to-End Inference Flow (CRITICAL)

This section documents the **exact step-by-step execution trace** from image upload to final prediction.

---

### 6.1 Inference Pipeline Overview

```
1. User uploads image
   ↓
2. FastAPI receives POST /predict request
   ↓
3. Image preprocessing (resize, normalize)
   ↓
4. Parallel inference on all 4 submodels
   ↓  ├─ CNN Transfer → embedding_cnn (512-D) + prob_fake_cnn
   ↓  ├─ ViT Base → embedding_vit (512-D) + prob_fake_vit
   ↓  ├─ DeiT Distilled → embedding_deit (512-D) + prob_fake_deit
   ↓  └─ Gradient Field CNN → embedding_gradfield (512-D) + prob_fake_gradfield
   ↓
5a. [Feature Fusion Path] Concatenate embeddings → [2048-D vector]
   ↓
6a. Meta-model inference → final prob_fake
   ↓
OR
   ↓
5b. [Probability Fusion Path] Assemble probabilities → [4-D vector]
   ↓
6b. Logistic regression inference → final prob_fake
   ↓
7. Backend returns JSON response with:
   - Overall AI-generated probability
   - Per-submodel probabilities
   - Final label
   - Confidence score
   - Explainability (fusion weights/contributions)
```

---

### 6.2 Detailed Step-by-Step Trace

#### **Step 1: Backend Receives Image Upload**

```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file
    image_bytes = await file.read()

    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Move to next step
    return await run_inference(image)
```

**Input:** Raw image file (JPEG/PNG)
**Output:** PIL Image object (RGB mode)

---

#### **Step 2: Image Preprocessing**

```python
from torchvision import transforms

# Standard ImageNet normalization (used by all submodels)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Apply preprocessing
image_tensor = preprocess(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
```

**Input:** PIL Image (variable size)
**Output:** PyTorch tensor (1, 3, 224, 224)

---

#### **Step 3: Run All Submodels in Parallel**

```python
import torch

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)

# Run all models (can be parallelized on multi-GPU setup)
with torch.no_grad():
    # CNN Transfer (EfficientNet-B0)
    result_cnn = cnn_model.predict(image_tensor)
    embedding_cnn = result_cnn["embedding"]  # (512,)
    prob_fake_cnn = result_cnn["prob"]       # scalar ∈ [0, 1]

    # ViT Base
    result_vit = vit_model.predict(image_tensor)
    embedding_vit = result_vit["embedding"]  # (512,)
    prob_fake_vit = result_vit["prob"]

    # DeiT Distilled
    result_deit = deit_model.predict(image_tensor)
    embedding_deit = result_deit["embedding"]  # (512,)
    prob_fake_deit = result_deit["prob"]

    # Gradient Field CNN
    # Note: Gradient Field CNN computes gradient field internally
    result_gradfield = gradfield_model.predict(image_tensor)
    embedding_gradfield = result_gradfield["embedding"]  # (512,)
    prob_fake_gradfield = result_gradfield["prob"]
```

**Input:** Image tensor (1, 3, 224, 224)
**Output:** 4 embeddings (each 512-D) + 4 probabilities (each ∈ [0, 1])

**Timing (on V100 GPU):**
- CNN Transfer: ~15ms
- ViT Base: ~25ms
- DeiT Distilled: ~25ms
- Gradient Field CNN: ~20ms
- **Total: ~85ms** (parallel execution)

---

#### **Step 4a: Prepare Fusion Input (Feature-Level)**

```python
import numpy as np

# Concatenate all embeddings
fusion_input = np.concatenate([
    embedding_cnn.cpu().numpy().flatten(),
    embedding_vit.cpu().numpy().flatten(),
    embedding_deit.cpu().numpy().flatten(),
    embedding_gradfield.cpu().numpy().flatten()
])  # Shape: (2048,)

# Example values
# fusion_input = [0.21, 0.34, ..., 0.87]  # 2048 values
```

**Input:** 4 individual 512-D embeddings
**Output:** Single 2048-D numpy array

---

#### **Step 4b: Prepare Fusion Input (Probability-Level)**

```python
# Assemble probability vector
fusion_input = [
    prob_fake_cnn,
    prob_fake_vit,
    prob_fake_deit,
    prob_fake_gradfield
]

# Example values
# fusion_input = [0.92, 0.87, 0.89, 0.76]
```

**Input:** 4 individual probabilities
**Output:** List[float] of length 4

---

#### **Step 5: Fusion Model Inference**

```python
# Run fusion model
fusion_result = fusion_loader.predict(fusion_input)

# fusion_result = {
#     "prob_fake": 0.91,
#     "prob_real": 0.09,
#     "label": "fake",
#     "confidence": 0.82,
#     "label_index": 1
# }
```

**Input:** 2048-D embedding vector (meta-model) OR 4-D probability vector (logistic reg)
**Output:** Final prediction dict

**Timing:**
- Meta-model: ~2ms (forward pass through 3-layer NN)
- Logistic regression: <1ms (single matrix multiplication)

---

#### **Step 6: Construct Response with Per-Model Breakdown**

```python
response = {
    # Final fusion prediction
    "prediction": {
        "label": fusion_result["label"],
        "probability_ai_generated": fusion_result["prob_fake"],
        "probability_real": fusion_result["prob_real"],
        "confidence": fusion_result["confidence"]
    },

    # Individual submodel predictions (for transparency)
    "submodel_predictions": {
        "cnn_transfer": {
            "probability_fake": prob_fake_cnn,
            "label": "fake" if prob_fake_cnn >= 0.5 else "real"
        },
        "vit_base": {
            "probability_fake": prob_fake_vit,
            "label": "fake" if prob_fake_vit >= 0.5 else "real"
        },
        "deit_distilled": {
            "probability_fake": prob_fake_deit,
            "label": "fake" if prob_fake_deit >= 0.5 else "real"
        },
        "gradient_field_cnn": {
            "probability_fake": prob_fake_gradfield,
            "label": "fake" if prob_fake_gradfield >= 0.5 else "real"
        }
    },

    # Fusion model metadata
    "fusion_info": {
        "model_type": fusion_loader.config["model_type"],
        "model_version": fusion_loader.config["version"],
        "fusion_approach": "feature_level" if "MetaModel" in fusion_loader.config["model_type"] else "probability_level"
    }
}
```

**Example response (Feature-Level Fusion):**

```json
{
  "prediction": {
    "label": "fake",
    "probability_ai_generated": 0.93,
    "probability_real": 0.07,
    "confidence": 0.86
  },
  "submodel_predictions": {
    "cnn_transfer": {
      "probability_fake": 0.92,
      "label": "fake"
    },
    "vit_base": {
      "probability_fake": 0.87,
      "label": "fake"
    },
    "deit_distilled": {
      "probability_fake": 0.89,
      "label": "fake"
    },
    "gradient_field_cnn": {
      "probability_fake": 0.76,
      "label": "fake"
    }
  },
  "fusion_info": {
    "model_type": "MetaModelFusion",
    "model_version": "1.0.0",
    "fusion_approach": "feature_level"
  }
}
```

---

#### **Step 7: Return Response to Frontend**

```python
from fastapi.responses import JSONResponse

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ... (steps 1-6 above)

    return JSONResponse(content=response, status_code=200)
```

**Total latency (GPU):**
- Image preprocessing: ~5ms
- Submodel inference (parallel): ~85ms
- Fusion model (meta-model): ~2ms
- JSON serialization: ~1ms
- **Total: ~93ms** (sub-100ms inference!)

**Total latency (CPU):**
- Image preprocessing: ~10ms
- Submodel inference (sequential): ~800ms
- Fusion model (meta-model): ~3ms
- JSON serialization: ~1ms
- **Total: ~814ms** (under 1 second)

---

## 7. Explainability at the Fusion Level

### Overview

Explainability at the fusion level shows **which submodels contributed most** to the final decision. This complements pixel-level explanations (Grad-CAM, attention maps).

---

### 7.1 Explainability for Logistic Regression (Probability Stacking)

#### **Global Weights (Model-Level)**

The logistic regression coefficients directly show each submodel's global importance:

```python
# After training
fusion_weights = fusion_model.coef_[0]
# Example: [0.30, 0.42, 0.18, 0.10]

# Normalize to percentages
importance = fusion_weights / fusion_weights.sum() * 100
# Example: [30%, 42%, 18%, 10%]
```

**Interpretation Table:**

| Submodel | Weight | Normalized (%) | Interpretation |
|----------|--------|----------------|----------------|
| CNN Transfer | 0.30 | 30% | Moderate importance (local artifacts) |
| **ViT Base** | **0.42** | **42%** | **Highest importance** (global context) |
| DeiT Distilled | 0.18 | 18% | Lower importance (redundant with ViT?) |
| Gradient Field CNN | 0.10 | 10% | Minimal importance (frequency features less reliable) |

**Insight:** ViT Base is the most trusted model. Consider removing DeiT or improving Gradient Field CNN.

---

#### **Per-Prediction Contribution**

Show how much each submodel contributed to a specific prediction:

```python
def explain_logistic_prediction(submodel_probs, fusion_model):
    """
    Compute contribution of each submodel to final prediction.
    """
    import numpy as np

    # Get fusion weights and bias
    weights = fusion_model.coef_[0]  # Shape: (4,)
    bias = fusion_model.intercept_[0]

    # Compute logit contributions
    contributions = weights * np.array(submodel_probs)

    # Normalize to percentages
    total_contribution = np.abs(contributions).sum()
    contribution_pct = np.abs(contributions) / total_contribution * 100

    return {
        "cnn_transfer": contribution_pct[0],
        "vit_base": contribution_pct[1],
        "deit_distilled": contribution_pct[2],
        "gradient_field_cnn": contribution_pct[3]
    }

# Example usage
contributions = explain_logistic_prediction(
    submodel_probs=[0.92, 0.87, 0.89, 0.76],
    fusion_model=fusion_loader.model
)
# Output: {
#     "cnn_transfer": 28.5,
#     "vit_base": 34.2,
#     "deit_distilled": 15.8,
#     "gradient_field_cnn": 21.5
# }
```

---

### 7.2 Explainability for Meta-Model (Feature-Level Fusion)

#### **Gradient-Based Attribution**

Use **Integrated Gradients** to compute feature importance:

```python
import torch
from captum.attr import IntegratedGradients

def explain_metamodel_prediction(embeddings, metamodel):
    """
    Compute per-submodel contributions using Integrated Gradients.

    Args:
        embeddings: Concatenated 2048-D numpy array
        metamodel: Trained MetaModel instance

    Returns:
        dict: Contribution percentage for each submodel
    """
    # Convert to tensor
    input_tensor = torch.FloatTensor(embeddings).unsqueeze(0)
    input_tensor.requires_grad = True

    # Initialize Integrated Gradients
    ig = IntegratedGradients(metamodel)

    # Compute attributions
    attributions = ig.attribute(input_tensor, target=0)  # Target class 0 (fake)
    attributions = attributions.squeeze().detach().numpy()

    # Sum attributions per submodel (each contributes 512 dims)
    contrib_cnn = np.abs(attributions[0:512]).sum()
    contrib_vit = np.abs(attributions[512:1024]).sum()
    contrib_deit = np.abs(attributions[1024:1536]).sum()
    contrib_gradfield = np.abs(attributions[1536:2048]).sum()

    # Normalize to percentages
    total = contrib_cnn + contrib_vit + contrib_deit + contrib_gradfield

    return {
        "cnn_transfer": (contrib_cnn / total) * 100,
        "vit_base": (contrib_vit / total) * 100,
        "deit_distilled": (contrib_deit / total) * 100,
        "gradient_field_cnn": (contrib_gradfield / total) * 100
    }
```

---

#### **Attention Visualization (Alternative)**

If using attention-based fusion:

```python
def visualize_attention_weights(attention_weights):
    """
    Display attention weights for each submodel.

    Args:
        attention_weights: [w1, w2, w3, w4] from attention module
    """
    import matplotlib.pyplot as plt

    models = ["CNN", "ViT", "DeiT", "GradField"]
    weights = attention_weights.cpu().detach().numpy()

    plt.figure(figsize=(8, 4))
    plt.bar(models, weights)
    plt.ylabel("Attention Weight")
    plt.title("Submodel Attention Weights")
    plt.ylim(0, 1)
    plt.show()
```

---

### 7.3 Disagreement Detection

Flag cases where submodels **disagree strongly**:

```python
def check_disagreement(submodel_probs, threshold=0.3):
    """
    Detect if submodels disagree (useful for uncertainty estimation).

    Args:
        submodel_probs: List of 4 probabilities
        threshold: Maximum acceptable disagreement range

    Returns:
        dict: Disagreement metrics
    """
    import numpy as np

    probs = np.array(submodel_probs)
    std_dev = np.std(probs)
    prob_range = probs.max() - probs.min()

    return {
        "is_disagreement": prob_range > threshold,
        "disagreement_score": float(prob_range),
        "standard_deviation": float(std_dev),
        "interpretation": "High disagreement - uncertain prediction" if prob_range > threshold else "Models agree"
    }

# Example 1: Strong agreement
result = check_disagreement([0.92, 0.89, 0.91, 0.88])
# {"is_disagreement": False, "disagreement_score": 0.04}

# Example 2: Strong disagreement (uncertain case)
result = check_disagreement([0.15, 0.82, 0.45, 0.91])
# {"is_disagreement": True, "disagreement_score": 0.76}
```

**Frontend warning:**
```
⚠️ Warning: High model disagreement detected (score: 0.76)
   This prediction may be uncertain. Consider manual review.
```

---

### 7.4 Combined Explainability Response

**Full explanation returned to frontend:**

```json
{
  "prediction": {
    "label": "fake",
    "probability_ai_generated": 0.93,
    "confidence": 0.86
  },
  "submodel_predictions": {
    "cnn_transfer": {"probability_fake": 0.92},
    "vit_base": {"probability_fake": 0.87},
    "deit_distilled": {"probability_fake": 0.89},
    "gradient_field_cnn": {"probability_fake": 0.76}
  },
  "explainability": {
    "fusion_contributions": {
      "cnn_transfer": 29.5,
      "vit_base": 35.2,
      "deit_distilled": 18.3,
      "gradient_field_cnn": 17.0
    },
    "disagreement": {
      "is_disagreement": false,
      "disagreement_score": 0.16,
      "interpretation": "Models show strong agreement"
    },
    "interpretation": "ViT Base (35%) and CNN Transfer (30%) were most influential in this prediction."
  },
  "grad_cam_url": "/explain/gradcam/abc123.png",
  "attention_map_url": "/explain/attention/abc123.png"
}
```

---

### 7.5 Frontend Visualization

**Combined display (fusion + pixel-level):**

```
┌───────────────────────────────────────────────────────────────┐
│ Prediction: FAKE (93% confidence)                             │
├───────────────────────────────────────────────────────────────┤
│ Model Contributions:                                          │
│   • ViT Base:            35% ████████████████████             │
│   • CNN Transfer:        30% ███████████████                  │
│   • Gradient Field:      17% ████████                         │
│   • DeiT Distilled:      18% █████████                        │
├───────────────────────────────────────────────────────────────┤
│ Interpretation:                                               │
│   ViT Base and CNN Transfer agree this image shows strong    │
│   AI-generation artifacts. All models agree (low uncertainty).│
├───────────────────────────────────────────────────────────────┤
│ Visual Explanation (Click to view):                           │
│   [Show Grad-CAM Heatmap]  [Show ViT Attention Map]          │
└───────────────────────────────────────────────────────────────┘
```

---

### 7.6 Complementing Grad-CAM / Attention Maps

**Two levels of explainability:**

1. **Fusion-level (this section):** "Which models drove the decision?"
   - Answers: "Why did the fusion model trust ViT more than CNN?"
   - Useful for: Model debugging, trust calibration, research analysis

2. **Pixel-level (Grad-CAM, attention maps):** "Which pixels look fake?"
   - Answers: "Where in the image are the artifacts?"
   - Useful for: User understanding, visual validation

**Workflow:**
- User uploads image → Sees fusion prediction
- User clicks "Explain" → Sees fusion contributions + Grad-CAM overlay
- User clicks specific model → Sees that model's Grad-CAM / attention map

---

## Summary

### Key Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Primary Fusion Model** | Feature-Level Meta-Model | Maximum expressiveness, research novelty |
| **Baseline Model** | Logistic Regression Stacking | Simple, explainable, fast baseline |
| **Input Features (Primary)** | 2048-D concatenated embeddings | Richer information, better generalization |
| **Input Features (Baseline)** | 4-D probability vector | Minimal complexity, strong baseline |
| **Training Data** | Validation split (15%) | Prevents overfitting via stacking |
| **Storage Format** | PyTorch state dict + sklearn pickle | Framework-native, efficient |
| **Backend Loading** | Startup download + cache | Fast inference, version control |
| **Explainability** | Integrated Gradients + disagreement analysis | Transparent, actionable |

---

### Implementation Roadmap

#### **Phase 1 (Weeks 1-2): Baseline Implementation**
- Train logistic regression stacking model
- Implement backend probability fusion pipeline
- Achieve validation AUROC ≥ 0.88
- Upload to HuggingFace: `DeepFakeDetector/fusion-logreg`

#### **Phase 2 (Weeks 3-4): Advanced Fusion**
- Train meta-model with feature-level fusion
- Implement embedding extraction in all submodels
- Achieve validation AUROC ≥ 0.92
- Upload to HuggingFace: `DeepFakeDetector/fusion-metamodel`

#### **Phase 3 (Week 5): Explainability**
- Implement Integrated Gradients attribution
- Add disagreement detection
- Create frontend visualizations
- A/B test fusion approaches

#### **Phase 4 (Week 6): Optimization & Deployment**
- Optimize inference latency (<100ms GPU)
- Calibration analysis
- Production deployment
- Documentation and demo video

---

### Success Metrics

The fusion model is successful if:

✅ **Performance (Meta-Model):** Test AUROC ≥ 0.92, Accuracy ≥ 0.89
✅ **Performance (Logistic Reg):** Test AUROC ≥ 0.88, Accuracy ≥ 0.86
✅ **Speed:** Inference <100ms on GPU, <1s on CPU
✅ **Explainability:** Fusion contributions are interpretable and stable
✅ **Reliability:** Calibration error <0.05
✅ **Production-ready:** Successfully deployed to backend with no errors

---

## References

1. **Stacking Ensembles:** Wolpert, D. H. (1992). "Stacked generalization." Neural Networks.
2. **Integrated Gradients:** Sundararajan, M., et al. (2017). "Axiomatic Attribution for Deep Networks." ICML.
3. **Model Deployment:** [HuggingFace Hub Documentation](https://huggingface.co/docs/hub)
4. **FastAPI Best Practices:** [FastAPI Documentation](https://fastapi.tiangolo.com/)
5. **Calibration:** Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." ICML.
6. **DeepFake Detection:** See existing research documents in `research/` directory

---

**Document Version:** 1.0
**Last Updated:** 2026-02-10
**Review Status:** Ready for team review and implementation


import numpy as np
import os
import json

def predict(submodel_outputs: dict, **kwargs) -> dict:
    '''
    Standard Fusion Predict Interface
    Args:
        submodel_outputs: dict mapping model_name -> {"prob_fake": float}
    '''
    # Load config to get order
    base_path = os.path.dirname(__file__)
    with open(os.path.join(base_path, "config.json")) as f:
        config = json.load(f)

    order = config["submodel_order"]
    probs = []
    for name in order:
        if name not in submodel_outputs:
            raise ValueError(f"Missing output for {name}")
        probs.append(submodel_outputs[name]["prob_fake"])

    X = np.array([probs]) # (1, n_models)

    # Load Model
    # Detect if logreg or pytorch
    if os.path.exists(os.path.join(base_path, "fusion_logreg.pkl")):
        import joblib
        model = joblib.load(os.path.join(base_path, "fusion_logreg.pkl"))
        prob_fake = model.predict_proba(X)[0, 1]
    elif os.path.exists(os.path.join(base_path, "fusion_model.pt")):
        import torch
        import torch.nn as nn
        # Simple reconstruction of architecture (must match training)
        # For robustness, one might pickle the whole model or save arch config.
        # Here we assume the simple MLP structure used in notebook.
        input_dim = len(order)
        net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        net.load_state_dict(torch.load(os.path.join(base_path, "fusion_model.pt")))
        net.eval()
        with torch.no_grad():
            prob_fake = net(torch.tensor(X, dtype=torch.float32)).item()
    else:
        raise FileNotFoundError("No model file found")

    return {
        "pred": "fake" if prob_fake >= config.get("threshold", 0.5) else "real",
        "pred_int": 1 if prob_fake >= config.get("threshold", 0.5) else 0,
        "prob_fake": float(prob_fake),
        "meta": {"min_prob": float(np.min(probs)), "max_prob": float(np.max(probs))}
    }
    
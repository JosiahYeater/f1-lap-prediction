"""
Evaluation script for the GRU regression model on F1 telemetry data.

This script:
1. Loads the trained GRU model and test dataset.
2. Evaluates the model's performance on unseen test laps.
3. Computes metrics such as MAE, RMSE, R², and MAPE.
4. Generates a per-lap error breakdown.
5. Produces an error-weighted feature heatmap to identify influential inputs.
6. Saves all results (JSON, NumPy, plots) to the model's evaluation directory.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import COLUMN_NAMES
from model import GRURegressor, LapDataset, get_loader, load_feature_names, DEVICE, ROOT_DIR, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

# -----------------------------------------------------------------------------
# Global Constants
# -----------------------------------------------------------------------------

TEST_DIR = ROOT_DIR / "data" / "preprocessed" / "test"
MODELS_DIR = ROOT_DIR / "models"

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():

    # ---- Support for --model and --latest args ---- #
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        help="Timestamp folder under models/, e.g. Aug-05-2025@0154"
    )
    group.add_argument(
        "--latest",
        action="store_true",
        help="Use most recent model in models/"
    )
    args = parser.parse_args()

    if args.latest:
        candidates = [
            p for p in MODELS_DIR.iterdir()
            if p.is_dir() and (p / "model_state_dict.pth").exists()
        ]
        if not candidates:
            raise RuntimeError(f"No models found in {MODELS_DIR}")
        model_dir = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"Using latest model: {model_dir.name}")
    else:
        model_dir = MODELS_DIR / args.model
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # ---- Prepare test dataset & loader ---- #
    test_dataset = LapDataset(TEST_DIR)
    
    if len(test_dataset) == 0:
        raise RuntimeError(f"No test files found in {TEST_DIR}")
    
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda")
    )
    lap_titles = [p.name for p in test_dataset.paths]  # guaranteed to match loader order

    # ---- Load feature names and set input size ---- #
    try:
        feature_names = load_feature_names(
            COLUMN_NAMES[:-5],
            ROOT_DIR / "data" / "metadata" / "vector_encoders.json"
        )
        input_size = len(feature_names)
        print(f"✅ Loaded feature names successfully! INPUT_SIZE={input_size}")
    except Exception as e:
        raise RuntimeError(f"⚠️ Failed to load feature names: {e}")

    # ---- Load trained model ---- #
    model = GRURegressor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    model.load_state_dict(torch.load(model_dir / "model_state_dict.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"Starting evaluation of {model_dir}...")
    try:
        evaluate(model, model_dir, test_loader, lap_titles, feature_names)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

# -----------------------------------------------------------------------------
# Evaluation Function -> produces metrics, per-lap errors, and heatmap
# ----------------------------------------------------------------------------- 

def evaluate(model, model_dir, test_loader, lap_titles, feature_names):
    """
    Evaluate the GRU model on the test set.

    Tracks:
        - Predictions vs. ground truth lap times.
        - Per-lap absolute and percentage errors.
        - Error-weighted influence per feature (for heatmap).

    Saves:
        - metrics.json
        - per_lap_errors.json
        - error_heatmap.npy
        - true_vs_pred.png
        - error_heatmap.png
    """
    y_preds = []
    y_trues = []
    per_lap_errors = []
    weighted_sum_matrix = None
    total_weight = 0.0

    for i, (lap_matrix, y_true) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            lap_matrix = lap_matrix.to(DEVICE)
            y_true = y_true.to(DEVICE)
            y_pred = model(lap_matrix).squeeze()  # scalar prediction

        error = torch.abs(y_pred - y_true).item()

        # Track predictions for metrics
        y_preds.append(y_pred.item())
        y_trues.append(y_true.item())

        # Initialize weighted sum matrix (timesteps × features) if first iteration
        if weighted_sum_matrix is None:
            weighted_sum_matrix = torch.zeros_like(lap_matrix.squeeze(0))

        # Accumulate absolute input values weighted by error magnitude
        weighted_sum_matrix += torch.abs(lap_matrix.squeeze(0)) * error
        total_weight += error

        # Record per-lap error
        per_lap_errors.append({
            "lap": lap_titles[i],
            "true": y_true.item(),
            "pred": y_pred.item(),
            "abs_error": error,
            "percent_error": (error / max(1e-6, y_true.item())) * 100
        })

    # ---- Compute aggregate metrics ---- #
    mae = mean_absolute_error(y_trues, y_preds)
    rmse = mean_squared_error(y_trues, y_preds, squared=False)
    r2 = r2_score(y_trues, y_preds)
    mape = np.mean(np.abs((np.array(y_preds) - np.array(y_trues)) / np.array(y_trues))) * 100

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_absolute_percent_error": mape
    }

    # ---- Normalize error heatmap ---- #
    if total_weight == 0:
        total_weight = 1.0
    error_heatmap = (weighted_sum_matrix / total_weight).cpu().numpy()

    # ---- Save outputs ---- #
    eval_dir = model_dir / "evaluation"
    (eval_dir / "graphs").mkdir(parents=True, exist_ok=True)

    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(eval_dir / "per_lap_errors.json", "w") as f:
        json.dump(per_lap_errors, f, indent=4)

    np.save(eval_dir / "error_heatmap.npy", error_heatmap)

    # ---- Plot predictions vs. true ---- #
    plt.figure()
    plt.scatter(y_trues, y_preds, alpha=0.7, edgecolor='k')
    plt.plot([min(y_trues), max(y_trues)], [min(y_trues), max(y_trues)],
             color='red', linestyle='--', label='Ideal')
    plt.fill_between([min(y_trues), max(y_trues)],
                     [min(y_trues) - mae, max(y_trues) - mae],
                     [min(y_trues) + mae, max(y_trues) + mae],
                     color='orange', alpha=0.2, label='±MAE')
    plt.xlabel("True Lap Time (s)")
    plt.ylabel("Predicted Lap Time (s)")
    plt.title("True vs Predicted Lap Times")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(eval_dir / "graphs" / "true_vs_pred.png")
    plt.close()

    # ---- Plot feature influence heatmap ---- #
    plt.imshow(error_heatmap.T, aspect="auto", cmap="hot")
    plt.title("Error-Weighted Feature Heatmap")
    plt.xlabel("Timestep")
    plt.ylabel("Feature")
    plt.yticks(
        ticks=np.arange(len(feature_names)),
        labels=feature_names,
        fontsize=6 if len(feature_names) < 80 else 4
    )
    plt.colorbar(label="Weighted Influence")
    plt.tight_layout()
    plt.savefig(eval_dir / "graphs" / "error_heatmap.png")
    plt.close()

    print(f"✅ Evaluation complete. Results saved to {eval_dir}")


if __name__ == "__main__":
    main()

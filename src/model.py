"""
Model definition, data loading, training loop, and save utilities for the GRU
lap-time regression project.

This script:
- Defines a GRU-based regressor that predicts lap time from per-timestep features.
- Provides a Dataset and DataLoader for preprocessed .npy lap files.
- Trains the model with MSE loss and an Adam optimizer.
- Uses a ReduceLROnPlateau scheduler and a simple early-stopping heuristic.
- Saves the best-performing model state dict, training metadata, and loss plot.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple
import datetime
import json
import matplotlib.pyplot as plt

from dataset import COLUMN_NAMES
# -----------------------------------------------------------------------------
# Global Constants
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = Path(__file__).parent.parent
TRAIN_DIR = ROOT_DIR / "data" / "preprocessed" / "train"
VAL_DIR = ROOT_DIR / "data" / "preprocessed" / "val"
MODEL_SAVES_DIR = ROOT_DIR / "models"

INITIAL_LR = 0.001
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3

BATCH_SIZE = 32
NUM_EPOCHS = 100
VALIDATION_INTERVAL = 1
VALIDATION_THRESHOLD = 0.0002

INPUT_SIZE = 84 # from data (excluding last column)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for NumPy and PyTorch to ensure reproducible results.

    Args:
        seed (int): The seed value to use for reproducibility. Default is 42.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CUDA (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Load Feature Names (base + one-hot categories for input size and heat map in evaluate.py)
# -----------------------------------------------------------------------------
def load_feature_names(base_feature_names, encoder_path):
    """
    Load and expand feature names by appending one-hot encoded category labels.

    Args:
        base_feature_names (list): List of base telemetry features.
        encoder_path (Path): Path to the JSON file containing category vector encodings.

    Returns:
        list: Full ordered list of feature names (continuous + one-hot encoded categories).
    """
    with open(encoder_path, 'r') as f:
        encoders = json.load(f)

    feature_names = base_feature_names.copy()

    for category, mapping in encoders.items():
        # Reverse-map one-hot indices to category labels
        index_to_label = {}
        for label, vec in mapping.items():
            for i, bit in enumerate(vec):
                if bit == 1:
                    index_to_label[i] = label

        # Append labels in the correct order
        max_index = max(index_to_label.keys())
        for i in range(max_index + 1):
            label = index_to_label.get(i, f"{category}:<unknown_{i}>")
            feature_names.append(f"{category}:{label}")

    return feature_names

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------
class GRURegressor(nn.Module):
    """GRU-based regression model predicting a scalar lap time.

    Args:
        input_size (int): Number of input features per timestep.
        hidden_size (int, optional): GRU hidden size. Defaults to 64.
        num_layers (int, optional): Number of GRU layers. Defaults to 1.
        dropout (float, optional): Dropout between GRU layers. Defaults to 0.2.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout=0.2):
        super(GRURegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer: input → hidden
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Linear layer: final hidden state → scalar lap time
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input of shape (batch, timesteps, input_size).

        Returns:
            torch.Tensor: Predicted lap time of shape (batch,).
        """
        # GRU forward pass; we only use the final hidden state
        out, hidden = self.gru(x)  # hidden: (num_layers, batch, hidden_size)
        last_hidden = hidden[-1]   # (batch, hidden_size)
        output = self.fc(last_hidden)  # (batch, 1)
        return output.squeeze(1)       # (batch,)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
class LapDataset(Dataset):
    """Dataset for preprocessed lap tensors saved as .npy files.

    Each .npy file contains a single lap matrix with shape (T, F_processed),
    where the final column is the constant LapTime target.
    """

    def __init__(self, data_dir: str | Path):
        self.paths = list(Path(data_dir).rglob("*.npy"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        lap = np.load(path)

        # Split into features and target (LapTime is identical across rows)
        inputs = lap[:, :-1]        # (timesteps, features - 1)
        target = lap[0, -1]         # scalar

        x = torch.tensor(inputs, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)
        return x, y


def get_loader(data_dir, batch_size=32, shuffle=True, device='cpu'):
    """Create a DataLoader over all laps within a directory tree.

    Args:
        data_dir (str | Path): Root directory containing .npy laps.
        batch_size (int, optional): Mini-batch size. Defaults to 32.
        shuffle (bool, optional): Shuffle dataset order. Defaults to True.
        device (str, optional): 'cuda' or 'cpu', controls pin_memory. Defaults to 'cpu'.

    Returns:
        DataLoader: Torch DataLoader producing (x, y) batches.
    """
    dataset = LapDataset(data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,            # You can tweak this based on your CPU
        pin_memory=(device == "cuda"),
        drop_last=False
    )
    return loader

# -----------------------------------------------------------------------------
# Loss and Optimizer
# -----------------------------------------------------------------------------
def get_loss_and_optimizer(model: nn.Module, learning_rate: float = INITIAL_LR):
    """Return the MSE loss function and Adam optimizer."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return loss_fn, optimizer

# -----------------------------------------------------------------------------
# Model Save Utils
# -----------------------------------------------------------------------------
def save_model(model_name: str, state_dict: dict = None, epoch: int = 0, best_val_loss: float = float('inf')):
    """Save model state, training metadata, and loss plot to a timestamped folder.

    Args:
        model_name (str): Class name of the model.
        state_dict (dict | None): Model parameters to save.
        epoch (int): Number of epochs completed.
        best_val_loss (float): Best validation loss observed.
    """
    time = datetime.datetime.now().strftime("%b-%d-%Y@%H%M")

    model_save_path = MODEL_SAVES_DIR / f"{time}"
    model_save_path.mkdir(parents=True, exist_ok=True)

    model_info_path = model_save_path / "model_info.json"
    plot_path = model_save_path / "validation_loss_plot.png"
    state_dict_path = model_save_path / "model_state_dict.pth"

    model_info = {
        "model_name": model_name,
        "device": DEVICE,
        "initial_learning_rate": INITIAL_LR,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "validation_interval": VALIDATION_INTERVAL,
        "validation_threshold": VALIDATION_THRESHOLD,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_epochs": epoch,
        "best_val_loss": best_val_loss
    }

    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    try:
        plt.legend()
        plt.yscale('symlog', linthresh=0.01)
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(plot_path, format='png', dpi=300)
    except Exception as e:
        print(f"❌ Failed to save validation loss plot: {e}")

    torch.save(state_dict, state_dict_path)
    print(f"✅ Model saved to {model_save_path}")

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train_and_save(
    model,
    loss_fn,
    optimizer,
    scheduler,
    num_epochs: int,
    device: str = "cpu",
    early_stop_threshold: float = 0.001,
    ):
    """Train the model, track losses, and save the best checkpoint.

    Early stopping triggers when the relative change in validation loss
    drops below `early_stop_threshold` compared to the previous epoch.

    Args:
        model (nn.Module): Model to train.
        loss_fn: Loss function (MSE).
        optimizer: Optimizer (Adam).
        scheduler: LR scheduler (ReduceLROnPlateau).
        num_epochs (int): Maximum number of epochs to train.
        device (str): 'cuda' or 'cpu'.
        early_stop_threshold (float): Relative Δval loss threshold for early stop.
    """
    model.to(device)
    last_val_loss = None

    epochs = []
    avg_val_losses = []
    avg_train_losses = []

    best_val_loss = float('inf')
    best_state_dict = None

    plt.gca().clear()

    for epoch in range(num_epochs):

        train_loader = get_loader(TRAIN_DIR, batch_size=BATCH_SIZE, shuffle=True, device=DEVICE)
        val_loader = get_loader(VAL_DIR, batch_size=BATCH_SIZE, shuffle=False, device=DEVICE)

        # ----- Training ----- #
        print(f"🔄 Epoch {epoch+1} — Training...")
        model.train()
        train_loss = 0.0
        train_length = 0.0

        for x_batch, y_batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            try:
                preds = model(x_batch)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("⚠️ CUDA out of memory!")
                    torch.cuda.empty_cache()
                    continue
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_length += 1


        avg_train_loss = train_loss / train_length

        # ----- Validation ----- #
        print(f"🔄 Epoch {epoch+1} — Validating...")
        model.eval()
        val_loss = 0.0
        val_length = 0.0

        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                try:
                    preds = model(x_batch)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("⚠️ CUDA out of memory!")
                        torch.cuda.empty_cache()
                        continue

                loss = loss_fn(preds, y_batch)
                val_loss += loss.item()
                val_length += 1

        avg_val_loss = val_loss / val_length

        scheduler.step(avg_val_loss)

        epochs.append(epoch + 1)
        avg_val_losses.append(avg_val_loss)
        avg_train_losses.append(avg_train_loss)

        print(f"✅ Epoch {epoch+1} — Done!                        ")
        print(f"📉 Epoch {epoch+1} — Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # ----- Early stopping check ----- #
        if last_val_loss is not None:
            pct_change = abs(avg_val_loss - last_val_loss) / last_val_loss
            if pct_change < early_stop_threshold:
                print(f"🛑 Early stopping triggered — ΔVal Loss < {early_stop_threshold*100:.2f}%")
                break
        last_val_loss = avg_val_loss

        # ----- Save best model ----- #
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()

    # ----- Save model ----- #
    plt.plot(epochs, avg_val_losses, label='Validation Loss')
    plt.plot(epochs, avg_train_losses, label='Training Loss')

    save_model(model.__class__.__name__, best_state_dict, epochs[-1], best_val_loss)

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main():
    """Kick off training with default hyperparameters and config."""

    print("🚀 Starting training...")
    try:
        feature_names = load_feature_names(COLUMN_NAMES[:-5], ROOT_DIR / "data" / "metadata" / "vector_encoders.json")
        INPUT_SIZE = len(feature_names)
        print (f"✅ Loaded feature names successfully! INPUT_SIZE={INPUT_SIZE}")
    except Exception as e:
        print(f"⚠️ Failed to load feature names properly: {e} \n Using hard-coded INPUT_SIZE={INPUT_SIZE}")

    model = GRURegressor(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)

    loss_fn, optimizer = get_loss_and_optimizer(model, learning_rate=INITIAL_LR)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    train_and_save(model, loss_fn, optimizer, scheduler, num_epochs=NUM_EPOCHS, device=DEVICE)

    print("✅ Training complete!")


if __name__ == "__main__":
    set_seed(42)  # Ensure reproducibility
    main()

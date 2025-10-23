"""
Model training module for strategy search.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
from .model import create_model


class StrategyTrainer:
    """
    Trains ML models for trading strategy evaluation.
    """

    def __init__(self, config: Dict, device: str = 'cuda'):
        """
        Args:
            config: Training configuration
            device: 'cuda' or 'cpu'
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def train_model(self, train_data: np.ndarray, epochs: int = 30) -> nn.Module:
        """
        Train a model on provided data.

        Args:
            train_data: Training data (OHLCV normalized)
            epochs: Number of training epochs

        Returns:
            Trained model
        """
        # Create model
        model = create_model(self.config).to(self.device)

        # Prepare data
        X, y = self._prepare_sequences(train_data)
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 0.0001)
        )

        batch_size = self.config.get('batch_size', 256)

        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0

            # Mini-batch training
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                # Forward pass
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches if n_batches > 0 else 0

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        model.eval()
        return model

    def _prepare_sequences(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for training.

        Args:
            data: OHLCV data (normalized)

        Returns:
            (X, y) where X is sequences and y is binary labels (price up/down)
        """
        seq_length = self.config.get('seq_length', 60)

        X, y = [], []

        for i in range(len(data) - seq_length - 1):
            # Input sequence
            seq = data[i:i + seq_length]
            X.append(seq)

            # Target: Did price go up? (binary classification)
            current_price = data[i + seq_length, 3]  # Close price
            future_price = data[i + seq_length + 1, 3]
            label = 1 if future_price > current_price else 0
            y.append(label)

        return np.array(X), np.array(y)


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize OHLCV data using returns.

    Args:
        data: Raw OHLCV data

    Returns:
        Normalized data
    """
    normalized = np.zeros_like(data)

    # Calculate returns for each column
    for col in range(data.shape[1]):
        returns = np.diff(data[:, col]) / (data[:-1, col] + 1e-10)
        normalized[1:, col] = returns
        normalized[0, col] = 0

    return normalized

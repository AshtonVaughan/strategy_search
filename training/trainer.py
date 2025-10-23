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

    def __init__(self, config: Dict, device: str = 'cuda', input_size: int = 5):
        """
        Args:
            config: Training configuration
            device: 'cuda' or 'cpu'
            input_size: Number of input features (default: 5 for OHLCV)
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size

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

    def train_model_with_labels(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 30
    ) -> nn.Module:
        """
        Train a model with pre-generated smart labels.

        Args:
            features: Feature matrix (n_samples x n_features)
            labels: Smart labels (1=buy, -1=sell, 0=hold)
            epochs: Number of training epochs

        Returns:
            Trained model
        """
        # Update config with correct input size
        model_config = self.config.copy()
        model_config['input_size'] = self.input_size

        # Create model
        model = create_model(model_config).to(self.device)

        # Prepare sequences with labels
        X, y = self._prepare_sequences_with_labels(features, labels)

        if len(X) == 0:
            print("  [WARN] No valid training samples, returning untrained model")
            return model

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

            # Shuffle data
            indices = torch.randperm(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for i in range(0, len(X_shuffled), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]

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

    def _prepare_sequences_with_labels(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences using smart labels.

        Args:
            features: Feature matrix (n_samples x n_features)
            labels: Smart labels (1=buy, -1=sell, 0=hold)

        Returns:
            (X, y) where X is sequences and y is binary labels
        """
        seq_length = self.config.get('seq_length', 60)

        X, y = [], []

        for i in range(len(features) - seq_length):
            # Input sequence
            seq = features[i:i + seq_length]

            # Target label at end of sequence
            target_label = labels[i + seq_length]

            # Skip unlabeled samples (hold=0) during training
            # This focuses the model on clear opportunities
            if target_label == 0:
                continue

            X.append(seq)

            # Convert to binary: 1 (buy) -> 1, -1 (sell) -> 0
            binary_label = 1 if target_label == 1 else 0
            y.append(binary_label)

        if len(X) == 0:
            return np.array([]), np.array([])

        return np.array(X), np.array(y)

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

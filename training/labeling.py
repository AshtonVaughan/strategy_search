"""
Smart labeling system for trading ML models.
Only labels significant price moves (>20 pips) to reduce noise.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class SmartLabeler:
    """
    Generate high-quality labels for ML training.
    Only labels clear, significant price moves to reduce noise.
    """

    def __init__(
        self,
        pip_value: float = 0.0001,  # For EURUSD
        threshold_pips: float = 20,  # Minimum move to label
        lookforward_hours: int = 12,  # How far to look ahead
        min_atr_multiple: float = 1.5  # Minimum move relative to ATR
    ):
        """
        Args:
            pip_value: Pip size (0.0001 for EURUSD)
            threshold_pips: Minimum price move in pips to create label
            lookforward_hours: Number of hours to look ahead for target
            min_atr_multiple: Minimum move as multiple of ATR
        """
        self.pip_value = pip_value
        self.threshold_pips = threshold_pips
        self.lookforward_hours = lookforward_hours
        self.min_atr_multiple = min_atr_multiple

    def generate_labels(
        self,
        data: pd.DataFrame,
        labeling_method: str = 'forward_return'
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate smart labels for training.

        Args:
            data: DataFrame with OHLCV and features
            labeling_method: Method for labeling
                - 'forward_return': Label based on forward price movement
                - 'swing_points': Label swing highs/lows
                - 'breakout': Label breakout opportunities

        Returns:
            (data_with_labels, labels) tuple
        """
        print(f"\n[INFO] Generating smart labels (method: {labeling_method})...")

        if labeling_method == 'forward_return':
            data, labels = self._label_forward_return(data)
        elif labeling_method == 'swing_points':
            data, labels = self._label_swing_points(data)
        elif labeling_method == 'breakout':
            data, labels = self._label_breakout(data)
        else:
            raise ValueError(f"Unknown labeling method: {labeling_method}")

        # Calculate statistics
        n_total = len(labels)
        n_labeled = np.sum(labels != 0)
        n_buy = np.sum(labels == 1)
        n_sell = np.sum(labels == -1)
        label_rate = n_labeled / n_total * 100

        print(f"[INFO] Labeling results:")
        print(f"  Total samples: {n_total}")
        print(f"  Labeled samples: {n_labeled} ({label_rate:.1f}%)")
        print(f"  Buy signals: {n_buy} ({n_buy/n_total*100:.1f}%)")
        print(f"  Sell signals: {n_sell} ({n_sell/n_total*100:.1f}%)")
        print(f"  Unlabeled (hold): {n_total - n_labeled} ({(n_total-n_labeled)/n_total*100:.1f}%)")

        return data, labels

    def _label_forward_return(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Label based on significant forward price movements.
        Only label if price moves >threshold_pips in lookforward_hours.
        """
        close = data['Close'].values
        atr = data['atr_14'].values if 'atr_14' in data.columns else np.ones(len(close)) * 0.0001

        labels = np.zeros(len(close), dtype=int)
        forward_returns = np.zeros(len(close))
        significant_move = np.zeros(len(close), dtype=bool)

        # Calculate forward returns
        for i in range(len(close) - self.lookforward_hours):
            future_prices = close[i+1:i+self.lookforward_hours+1]

            # Find max gain and max loss
            max_gain = np.max(future_prices - close[i])
            max_loss = np.min(future_prices - close[i])

            # Check if move is significant
            threshold_price = self.threshold_pips * self.pip_value
            atr_threshold = atr[i] * self.min_atr_multiple

            # Use the larger of fixed threshold or ATR-based threshold
            final_threshold = max(threshold_price, atr_threshold)

            # Label buy if significant upward move
            if max_gain >= final_threshold and abs(max_gain) > abs(max_loss):
                labels[i] = 1
                forward_returns[i] = max_gain
                significant_move[i] = True

            # Label sell if significant downward move
            elif abs(max_loss) >= final_threshold and abs(max_loss) > abs(max_gain):
                labels[i] = -1
                forward_returns[i] = max_loss
                significant_move[i] = True

            # Otherwise: hold (0)

        # Add to dataframe for analysis
        data['label'] = labels
        data['forward_return'] = forward_returns
        data['significant_move'] = significant_move

        return data, labels

    def _label_swing_points(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Label swing highs (sell) and swing lows (buy).
        Only label if swing is significant (>threshold_pips).
        """
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values

        labels = np.zeros(len(close), dtype=int)

        window = 5  # Look N bars before and after

        for i in range(window, len(close) - window):
            # Swing High (potential sell)
            is_swing_high = high[i] == np.max(high[i-window:i+window+1])

            if is_swing_high:
                # Check if move is significant
                recent_low = np.min(low[i-window:i+1])
                move_size = high[i] - recent_low

                if move_size >= self.threshold_pips * self.pip_value:
                    labels[i] = -1  # Sell signal

            # Swing Low (potential buy)
            is_swing_low = low[i] == np.min(low[i-window:i+window+1])

            if is_swing_low:
                # Check if move is significant
                recent_high = np.max(high[i-window:i+1])
                move_size = recent_high - low[i]

                if move_size >= self.threshold_pips * self.pip_value:
                    labels[i] = 1  # Buy signal

        data['label'] = labels

        return data, labels

    def _label_breakout(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Label breakout opportunities from consolidation.
        Only label if breakout is confirmed and significant.
        """
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values if 'Volume' in data.columns else np.ones(len(close))
        atr = data['atr_14'].values if 'atr_14' in data.columns else np.ones(len(close)) * 0.0001

        labels = np.zeros(len(close), dtype=int)

        lookback = 20

        for i in range(lookback, len(close) - self.lookforward_hours):
            # Check if in consolidation
            recent_high = np.max(high[i-lookback:i])
            recent_low = np.min(low[i-lookback:i])
            range_size = recent_high - recent_low

            # Consolidation: range < 2 * ATR
            if range_size < 2 * atr[i]:
                # Upside breakout
                if close[i] > recent_high and volume[i] > np.mean(volume[i-lookback:i]) * 1.2:
                    # Confirm with forward movement
                    future_high = np.max(high[i:i+self.lookforward_hours])
                    if future_high - close[i] >= self.threshold_pips * self.pip_value:
                        labels[i] = 1

                # Downside breakout
                elif close[i] < recent_low and volume[i] > np.mean(volume[i-lookback:i]) * 1.2:
                    # Confirm with forward movement
                    future_low = np.min(low[i:i+self.lookforward_hours])
                    if close[i] - future_low >= self.threshold_pips * self.pip_value:
                        labels[i] = -1

        data['label'] = labels

        return data, labels

    def filter_noisy_labels(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        min_confidence: float = 0.7
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Filter out noisy labels based on market conditions.

        Args:
            data: DataFrame with features
            labels: Current labels
            min_confidence: Minimum confidence to keep label

        Returns:
            (filtered_data, filtered_labels) tuple
        """
        print(f"\n[INFO] Filtering noisy labels (min_confidence: {min_confidence})...")

        filtered_labels = labels.copy()

        # Get volatility and trend indicators
        atr = data['atr_14'].values if 'atr_14' in data.columns else None
        adx = data['adx'].values if 'adx' in data.columns else None
        volume_ratio = data['volume_ratio'].values if 'volume_ratio' in data.columns else None

        n_filtered = 0

        for i in range(len(labels)):
            if labels[i] == 0:
                continue

            # Calculate confidence score
            confidence = 1.0

            # Low ADX = weak trend = less confident
            if adx is not None and adx[i] < 25:
                confidence *= 0.7

            # Low volume = less confident
            if volume_ratio is not None and volume_ratio[i] < 0.8:
                confidence *= 0.8

            # Very high volatility = less confident
            if atr is not None:
                avg_atr = np.mean(atr[max(0, i-50):i+1])
                if atr[i] > avg_atr * 2:
                    confidence *= 0.7

            # Filter if confidence too low
            if confidence < min_confidence:
                filtered_labels[i] = 0
                n_filtered += 1

        print(f"[INFO] Filtered {n_filtered} noisy labels ({n_filtered/len(labels)*100:.1f}%)")

        return data, filtered_labels

    def create_weighted_labels(
        self,
        data: pd.DataFrame,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Create sample weights based on label importance.
        Recent data and stronger signals get higher weight.

        Args:
            data: DataFrame with features
            labels: Labels

        Returns:
            Sample weights array
        """
        weights = np.ones(len(labels))

        # Higher weight for labeled samples
        weights[labels != 0] *= 2.0

        # Higher weight for recent data (exponential decay)
        time_weights = np.exp(np.linspace(-2, 0, len(weights)))
        weights *= time_weights

        # Higher weight for clearer signals (based on ATR)
        if 'forward_return' in data.columns:
            abs_return = np.abs(data['forward_return'].values)
            return_weight = np.clip(abs_return / (self.threshold_pips * self.pip_value), 1, 3)
            weights *= return_weight

        # Normalize
        weights /= np.mean(weights)

        return weights

    def balance_labels(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        method: str = 'undersample'
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Balance buy/sell labels to prevent bias.

        Args:
            data: DataFrame with features
            labels: Labels
            method: 'undersample', 'oversample', or 'class_weight'

        Returns:
            (balanced_data, balanced_labels) tuple
        """
        n_buy = np.sum(labels == 1)
        n_sell = np.sum(labels == -1)

        print(f"\n[INFO] Balancing labels (method: {method})...")
        print(f"  Before: Buy={n_buy}, Sell={n_sell}")

        if method == 'undersample':
            # Undersample majority class
            min_count = min(n_buy, n_sell)

            buy_indices = np.where(labels == 1)[0]
            sell_indices = np.where(labels == -1)[0]
            hold_indices = np.where(labels == 0)[0]

            # Randomly select equal numbers
            np.random.seed(42)
            selected_buy = np.random.choice(buy_indices, min_count, replace=False)
            selected_sell = np.random.choice(sell_indices, min_count, replace=False)

            # Keep all hold samples
            selected_indices = np.concatenate([selected_buy, selected_sell, hold_indices])
            selected_indices = np.sort(selected_indices)

            balanced_data = data.iloc[selected_indices].copy()
            balanced_labels = labels[selected_indices]

        elif method == 'oversample':
            # Oversample minority class (not implemented yet - would need SMOTE)
            print("[WARNING] Oversample not implemented, using original labels")
            balanced_data = data
            balanced_labels = labels

        else:  # class_weight
            # Just return original, weights will be handled in training
            balanced_data = data
            balanced_labels = labels

        n_buy_after = np.sum(balanced_labels == 1)
        n_sell_after = np.sum(balanced_labels == -1)
        print(f"  After: Buy={n_buy_after}, Sell={n_sell_after}")

        return balanced_data, balanced_labels


def test_labeler():
    """Test the smart labeler."""
    print("="*70)
    print("TESTING SMART LABELER")
    print("="*70)

    # Create sample data
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        'Open': 1.0800 + np.random.randn(n) * 0.001,
        'High': 1.0810 + np.random.randn(n) * 0.001,
        'Low': 1.0790 + np.random.randn(n) * 0.001,
        'Close': 1.0800 + np.cumsum(np.random.randn(n) * 0.0002),
        'Volume': 10000 + np.random.randn(n) * 1000,
        'atr_14': 0.0001 + np.random.rand(n) * 0.0001,
        'adx': 20 + np.random.rand(n) * 30,
        'volume_ratio': 0.8 + np.random.rand(n) * 0.4
    })

    labeler = SmartLabeler(threshold_pips=20, lookforward_hours=12)

    # Test forward return labeling
    print("\n1. Testing forward_return method:")
    data_labeled, labels = labeler.generate_labels(data, method='forward_return')

    # Test filtering
    print("\n2. Testing noise filtering:")
    data_filtered, labels_filtered = labeler.filter_noisy_labels(data_labeled, labels)

    # Test weighting
    print("\n3. Testing sample weighting:")
    weights = labeler.create_weighted_labels(data_labeled, labels)
    print(f"  Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    print(f"  Mean weight: {weights.mean():.2f}")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    test_labeler()

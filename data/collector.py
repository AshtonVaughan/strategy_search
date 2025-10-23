"""
Multi-timeframe data collection for forex trading.
Fetches 1H and 4H data, aligns timeframes, and merges for context.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Tuple
from data.features import FeatureEngine


class MultiTimeframeCollector:
    """
    Fetch and align multi-timeframe OHLCV data from Yahoo Finance.
    Combines 1H (primary) and 4H (context) data.
    """

    def __init__(self, symbol: str = 'EURUSD=X'):
        """
        Args:
            symbol: Trading symbol (default: EURUSD=X)
        """
        self.symbol = symbol
        self.feature_engine = FeatureEngine()

    def fetch_multi_timeframe_data(
        self,
        start_date: str,
        end_date: str,
        primary_interval: str = '1h',
        context_interval: str = '4h'
    ) -> pd.DataFrame:
        """
        Fetch and merge multi-timeframe data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            primary_interval: Primary timeframe (default: 1h)
            context_interval: Context timeframe (default: 4h)

        Returns:
            DataFrame with combined 1H + 4H features
        """
        print(f"Fetching multi-timeframe data for {self.symbol}...")
        print(f"  Primary: {primary_interval} | Context: {context_interval}")
        print(f"  Date range: {start_date} to {end_date}")

        # Fetch primary timeframe (1H)
        print(f"\n1. Fetching {primary_interval} data...")
        data_1h = self._fetch_data(start_date, end_date, primary_interval)

        if data_1h is None or len(data_1h) == 0:
            raise ValueError(f"Failed to fetch {primary_interval} data!")

        print(f"   -> Loaded {len(data_1h)} {primary_interval} bars")

        # Fetch context timeframe (4H)
        print(f"\n2. Fetching {context_interval} data...")
        data_4h = self._fetch_data(start_date, end_date, context_interval)

        if data_4h is None or len(data_4h) == 0:
            print(f"   [WARNING] Failed to fetch {context_interval} data, using only {primary_interval}")
            return self._generate_features_single_timeframe(data_1h, primary_interval)

        print(f"   -> Loaded {len(data_4h)} {context_interval} bars")

        # Generate features for both timeframes
        print("\n3. Generating features...")
        print(f"   -> Generating {primary_interval} features...")
        data_1h_features = self.feature_engine.generate_all_features(data_1h)

        print(f"   -> Generating {context_interval} features...")
        data_4h_features = self.feature_engine.generate_all_features(data_4h)

        # Align and merge timeframes
        print("\n4. Aligning and merging timeframes...")
        merged_data = self._merge_timeframes(data_1h_features, data_4h_features, primary_interval, context_interval)

        print(f"   -> Final dataset: {len(merged_data)} rows, {len(merged_data.columns)} columns")
        print(f"\n[OK] Multi-timeframe data ready!")

        return merged_data

    def _fetch_data(self, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            start_date: Start date
            end_date: End date
            interval: Data interval (1h, 4h, 1d)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)

            if data.empty:
                return None

            # Keep only OHLCV
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

            # Remove timezone to avoid comparison issues
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            return data

        except Exception as e:
            print(f"   [ERROR] Failed to fetch {interval} data: {e}")
            return None

    def _generate_features_single_timeframe(self, data: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Generate features for single timeframe (fallback if 4H fails).

        Args:
            data: OHLCV data
            interval: Timeframe

        Returns:
            DataFrame with features
        """
        print(f"\n[WARNING] Using single timeframe ({interval}) only")
        features = self.feature_engine.generate_all_features(data)
        return features

    def _merge_timeframes(
        self,
        data_1h: pd.DataFrame,
        data_4h: pd.DataFrame,
        primary_interval: str,
        context_interval: str
    ) -> pd.DataFrame:
        """
        Merge 1H and 4H timeframes by forward-filling 4H data.

        Args:
            data_1h: 1H data with features
            data_4h: 4H data with features
            primary_interval: Primary timeframe name
            context_interval: Context timeframe name

        Returns:
            Merged DataFrame with both 1H and 4H features
        """
        # Select key 4H features to merge (avoid duplicating all 80+ features)
        # We'll take the most important indicators from 4H for context
        context_features_to_merge = [
            'ema_21', 'ema_50', 'ema_200',  # Trend
            'macd', 'macd_signal', 'macd_hist',  # Momentum
            'rsi_14', 'stoch_k',  # Oscillators
            'atr_14',  # Volatility
            'bb_upper', 'bb_lower',  # Bands
            'adx', 'di_plus', 'di_minus',  # Trend strength
            'obv', 'vwap', 'volume_ratio',  # Volume
            'trend_strength', 'volatility_regime',  # Regime
            'momentum_composite'  # Composite
        ]

        # Ensure features exist in 4H data
        context_features_available = [f for f in context_features_to_merge if f in data_4h.columns]

        # Extract context features from 4H
        data_4h_context = data_4h[context_features_available].copy()

        # Rename 4H features with suffix
        data_4h_context.columns = [f'{col}_4h' for col in data_4h_context.columns]

        # Reindex 4H data to 1H index (forward fill)
        data_4h_aligned = data_4h_context.reindex(data_1h.index, method='ffill')

        # Merge 1H and 4H
        merged = pd.concat([data_1h, data_4h_aligned], axis=1)

        # Fill any remaining NaN
        merged = merged.fillna(method='ffill').fillna(method='bfill')

        return merged

    def prepare_training_data(
        self,
        start_date: str,
        end_date: str,
        normalize: bool = True
    ) -> Tuple[pd.DataFrame, list]:
        """
        Prepare complete training dataset with all features.

        Args:
            start_date: Start date
            end_date: End date
            normalize: Whether to normalize features

        Returns:
            (data, feature_columns) tuple
        """
        # Fetch multi-timeframe data
        data = self.fetch_multi_timeframe_data(start_date, end_date)

        # Get feature columns (exclude OHLCV and index)
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in data.columns if col not in ohlcv_cols]

        print(f"\n[INFO] Total features: {len(feature_cols)}")

        # Normalize features if requested
        if normalize:
            print("[INFO] Normalizing features (rolling z-score)...")
            data = self.feature_engine.normalize_features(data, feature_cols)

        return data, feature_cols

    def get_validation_week_data(
        self,
        week_start: str,
        week_days: int = 7
    ) -> pd.DataFrame:
        """
        Get data for a specific validation week.

        Args:
            week_start: Week start date (YYYY-MM-DD)
            week_days: Number of days

        Returns:
            DataFrame for validation week
        """
        week_start_dt = pd.to_datetime(week_start)
        week_end_dt = week_start_dt + timedelta(days=week_days)

        # Fetch data
        data = self.fetch_multi_timeframe_data(
            start_date=week_start,
            end_date=week_end_dt.strftime('%Y-%m-%d')
        )

        return data

    def get_training_data_for_week(
        self,
        week_start: str,
        lookback_days: int = 365
    ) -> pd.DataFrame:
        """
        Get training data before a validation week (prevent lookahead).

        Args:
            week_start: Validation week start date
            lookback_days: Days of history for training

        Returns:
            Training data
        """
        week_start_dt = pd.to_datetime(week_start)
        train_end_dt = week_start_dt - timedelta(days=1)
        train_start_dt = train_end_dt - timedelta(days=lookback_days)

        # Fetch data
        data = self.fetch_multi_timeframe_data(
            start_date=train_start_dt.strftime('%Y-%m-%d'),
            end_date=train_end_dt.strftime('%Y-%m-%d')
        )

        return data

    def split_train_val(
        self,
        data: pd.DataFrame,
        val_fraction: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and validation sets (time-series aware).

        Args:
            data: Full dataset
            val_fraction: Fraction for validation (default: 0.2)

        Returns:
            (train_data, val_data) tuple
        """
        split_idx = int(len(data) * (1 - val_fraction))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]

        print(f"[INFO] Train: {len(train_data)} rows | Val: {len(val_data)} rows")

        return train_data, val_data


def test_collector():
    """Test the multi-timeframe collector."""
    print("="*70)
    print("TESTING MULTI-TIMEFRAME COLLECTOR")
    print("="*70)

    collector = MultiTimeframeCollector(symbol='EURUSD=X')

    # Test with recent data (within Yahoo's 730-day limit)
    start = '2024-06-01'
    end = '2024-07-01'

    data, feature_cols = collector.prepare_training_data(start, end, normalize=True)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Features: {len(feature_cols)}")
    print(f"\nFirst 5 feature columns:")
    for i, col in enumerate(feature_cols[:5]):
        print(f"  {i+1}. {col}")
    print(f"\n4H context features:")
    context_4h = [col for col in data.columns if col.endswith('_4h')]
    print(f"  Total: {len(context_4h)}")
    for col in context_4h[:5]:
        print(f"    - {col}")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    test_collector()

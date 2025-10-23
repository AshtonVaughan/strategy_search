"""
5-week cross-validation across 5 years to prevent overfitting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta


class CrossValidator:
    """
    Validates trading strategies on 5 different weeks across 5 years.
    Prevents overfitting by testing on diverse market conditions.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Validation configuration
        """
        self.config = config
        self.weeks_config = config['weeks_config']

    def validate_strategy(
        self,
        model,
        data: pd.DataFrame,
        backtester,
        strategy_params: Dict
    ) -> Dict:
        """
        Validate strategy on all 5 weeks and aggregate results.

        Args:
            model: Trained ML model
            data: Full OHLC dataset
            backtester: Backtesting engine
            strategy_params: Trading strategy parameters

        Returns:
            Aggregated metrics across all weeks
        """
        week_results = []

        for week_cfg in self.weeks_config:
            # Extract week data
            week_data = self._get_week_data(data, week_cfg)

            if len(week_data) < 10:  # Skip if not enough data
                continue

            # Generate signals
            signals = backtester.generate_signals(
                model,
                week_data,
                confidence_threshold=strategy_params['confidence_threshold']
            )

            # Backtest
            metrics = backtester.backtest(
                week_data,
                signals,
                stop_loss_pips=strategy_params['stop_loss_pips'],
                take_profit_pips=strategy_params['take_profit_pips'],
                risk_per_trade=strategy_params['risk_per_trade'],
                trailing_stop=strategy_params.get('trailing_stop', False)
            )

            week_results.append({
                'year': week_cfg['year'],
                'metrics': metrics
            })

        # Aggregate results across weeks
        if len(week_results) == 0:
            return self._empty_results()

        aggregated = self._aggregate_results(week_results)

        return aggregated

    def _get_week_data(self, data: pd.DataFrame, week_cfg: Dict) -> pd.DataFrame:
        """
        Extract one week of data for testing.

        Args:
            data: Full OHLC data
            week_cfg: Week configuration

        Returns:
            One week of data
        """
        start = pd.to_datetime(week_cfg['start'])
        end = start + timedelta(days=week_cfg['days'])

        # Data is timezone-naive now
        week_data = data[(data.index >= start) & (data.index < end)]

        return week_data

    def _aggregate_results(self, week_results: List[Dict]) -> Dict:
        """
        Aggregate metrics across all weeks using mean.

        Args:
            week_results: List of results for each week

        Returns:
            Aggregated metrics
        """
        # Extract all metrics
        all_metrics = [wr['metrics'] for wr in week_results]

        # Calculate means
        aggregated = {
            'sharpe_ratio': np.mean([m['sharpe_ratio'] for m in all_metrics]),
            'win_rate': np.mean([m['win_rate'] for m in all_metrics]),
            'max_drawdown': np.mean([m['max_drawdown'] for m in all_metrics]),
            'total_return': np.mean([m['total_return'] for m in all_metrics]),
            'profit_factor': np.mean([m['profit_factor'] for m in all_metrics]),
            'total_trades': np.sum([m['total_trades'] for m in all_metrics]),
            'avg_win': np.mean([m['avg_win'] for m in all_metrics]),
            'avg_loss': np.mean([m['avg_loss'] for m in all_metrics]),

            # Additional: Track variance (consistency)
            'sharpe_std': np.std([m['sharpe_ratio'] for m in all_metrics]),
            'winrate_std': np.std([m['win_rate'] for m in all_metrics]),

            # Number of weeks tested
            'num_weeks': len(week_results),

            # Individual week results (for analysis)
            'week_details': week_results
        }

        return aggregated

    def _empty_results(self) -> Dict:
        """Return empty results if validation fails."""
        return {
            'sharpe_ratio': 0,
            'win_rate': 0,
            'max_drawdown': 1.0,
            'total_return': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'sharpe_std': 0,
            'winrate_std': 0,
            'num_weeks': 0,
            'week_details': []
        }


def get_training_data_for_week(
    data: pd.DataFrame,
    week_cfg: Dict,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Get training data for a specific validation week.
    Uses data BEFORE the validation week to prevent lookahead bias.

    Args:
        data: Full OHLC data
        week_cfg: Week configuration
        lookback_days: How many days of history to use for training

    Returns:
        Training data
    """
    week_start = pd.to_datetime(week_cfg['start'])
    train_end = week_start - timedelta(days=1)  # Day before validation week
    train_start = train_end - timedelta(days=lookback_days)

    # Data is timezone-naive now
    train_data = data[(data.index >= train_start) & (data.index <= train_end)]

    return train_data

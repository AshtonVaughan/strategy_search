"""
Vectorized backtesting engine using vectorbt for speed.
Handles ML model signal generation + trade execution simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# Optional: Try to import vectorbt, fall back to simple backtesting if unavailable
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except Exception as e:
    print(f"Warning: vectorbt not available ({e}). Using simple backtesting fallback.")
    VECTORBT_AVAILABLE = False


class MLBacktester:
    """
    Fast backtesting for ML-based trading strategies.
    Uses vectorbt for vectorized operations (100x faster than loops).
    """

    def __init__(self, initial_capital: float = 10000):
        """
        Args:
            initial_capital: Starting account balance
        """
        self.initial_capital = initial_capital

    def generate_signals(
        self,
        model,
        data: pd.DataFrame,
        confidence_threshold: float = 0.70
    ) -> pd.DataFrame:
        """
        Generate trading signals from ML model predictions.

        Args:
            model: Trained ML model
            data: OHLC data
            confidence_threshold: Minimum confidence to trade

        Returns:
            DataFrame with signals (1=buy, -1=sell, 0=hold)
        """
        import torch

        # Prepare features (same as training)
        features = self._prepare_features(data)

        # Get model predictions
        model.eval()

        # Get device from model
        device = next(model.parameters()).device

        with torch.no_grad():
            predictions = []
            confidences = []

            for i in range(len(features)):
                if i < model.seq_length:
                    predictions.append(0)
                    confidences.append(0.5)
                    continue

                # Get sequence
                seq = features[i-model.seq_length:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)

                # Predict
                pred = model(seq_tensor)
                confidence = torch.sigmoid(pred).item()

                # Generate signal
                if confidence >= confidence_threshold:
                    signal = 1  # Buy
                elif confidence <= (1 - confidence_threshold):
                    signal = -1  # Sell
                else:
                    signal = 0  # Hold

                predictions.append(signal)
                confidences.append(confidence)

        signals_df = pd.DataFrame({
            'signal': predictions,
            'confidence': confidences
        }, index=data.index)

        return signals_df

    def generate_signals_with_features(
        self,
        model,
        data: pd.DataFrame,
        feature_cols: list,
        confidence_threshold: float = 0.70,
        strategy_logic: str = 'simple'
    ) -> pd.DataFrame:
        """
        Generate trading signals using full feature set and advanced logic.

        Args:
            model: Trained ML model
            data: DataFrame with all features
            feature_cols: List of feature column names
            confidence_threshold: Minimum confidence to trade
            strategy_logic: 'simple', 'multi_signal', 'adaptive_threshold', 'trend_confirm'

        Returns:
            DataFrame with signals (1=buy, -1=sell, 0=hold)
        """
        import torch

        # Extract features
        features = data[feature_cols].values

        # Get model predictions
        model.eval()

        # Get device from model
        device = next(model.parameters()).device

        with torch.no_grad():
            predictions = []
            confidences = []

            for i in range(len(features)):
                if i < model.seq_length:
                    predictions.append(0)
                    confidences.append(0.5)
                    continue

                # Get sequence
                seq = features[i-model.seq_length:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)

                # Predict
                pred = model(seq_tensor)
                confidence = torch.sigmoid(pred).item()

                # Apply strategy logic
                signal = self._apply_strategy_logic(
                    confidence=confidence,
                    threshold=confidence_threshold,
                    data_row=data.iloc[i],
                    strategy_logic=strategy_logic
                )

                predictions.append(signal)
                confidences.append(confidence)

        signals_df = pd.DataFrame({
            'signal': predictions,
            'confidence': confidences
        }, index=data.index)

        return signals_df

    def _apply_strategy_logic(
        self,
        confidence: float,
        threshold: float,
        data_row: pd.Series,
        strategy_logic: str
    ) -> int:
        """
        Apply different signal generation logic based on strategy type.

        Args:
            confidence: Model confidence (0-1)
            threshold: Confidence threshold
            data_row: Current data row with all features
            strategy_logic: Strategy type

        Returns:
            Signal: 1 (buy), -1 (sell), 0 (hold)
        """
        if strategy_logic == 'simple':
            # Simple: Model prediction only
            if confidence >= threshold:
                return 1
            elif confidence <= (1 - threshold):
                return -1
            else:
                return 0

        elif strategy_logic == 'multi_signal':
            # Multi-signal: Model + RSI + MACD confirmation
            rsi = data_row.get('rsi_14', 50)
            macd_hist = data_row.get('macd_hist', 0)

            # Buy signal
            if confidence >= threshold:
                # Confirm with RSI (not overbought) and MACD (positive momentum)
                if rsi < 70 and macd_hist > 0:
                    return 1

            # Sell signal
            elif confidence <= (1 - threshold):
                # Confirm with RSI (not oversold) and MACD (negative momentum)
                if rsi > 30 and macd_hist < 0:
                    return -1

            return 0

        elif strategy_logic == 'adaptive_threshold':
            # Adaptive: Adjust threshold based on volatility
            atr = data_row.get('atr_14', 0.0001)
            hist_vol = data_row.get('hist_volatility', 0.01)

            # Lower threshold in high volatility (more opportunities)
            # Higher threshold in low volatility (more selective)
            vol_ratio = hist_vol / 0.01  # Normalize to typical forex volatility
            adjusted_threshold = threshold * (0.8 + 0.4 * min(vol_ratio, 2))  # 0.8x to 1.2x

            if confidence >= adjusted_threshold:
                return 1
            elif confidence <= (1 - adjusted_threshold):
                return -1
            else:
                return 0

        elif strategy_logic == 'trend_confirm':
            # Trend confirmation: Only trade with trend
            ema_21 = data_row.get('ema_21', data_row.get('Close', 1.0))
            ema_50 = data_row.get('ema_50', data_row.get('Close', 1.0))
            close = data_row.get('Close', 1.0)

            # Uptrend: EMAs aligned up + price above 21 EMA
            uptrend = (ema_21 > ema_50) and (close > ema_21)

            # Downtrend: EMAs aligned down + price below 21 EMA
            downtrend = (ema_21 < ema_50) and (close < ema_21)

            # Buy only in uptrend
            if confidence >= threshold and uptrend:
                return 1

            # Sell only in downtrend
            elif confidence <= (1 - threshold) and downtrend:
                return -1

            return 0

        else:
            # Unknown strategy, default to simple
            if confidence >= threshold:
                return 1
            elif confidence <= (1 - threshold):
                return -1
            else:
                return 0

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare OHLCV features for model input.
        Normalize using simple returns.
        """
        features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values

        # Simple normalization (returns)
        normalized = np.zeros_like(features)
        normalized[1:] = (features[1:] - features[:-1]) / (features[:-1] + 1e-10)
        normalized[0] = 0

        return normalized

    def backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        stop_loss_pips: float = 20,
        take_profit_pips: float = 40,
        risk_per_trade: float = 0.02,
        trailing_stop: bool = False
    ) -> Dict:
        """
        Run backtest with position sizing and risk management.

        Args:
            data: OHLC data
            signals: Trading signals from generate_signals()
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips
            risk_per_trade: Risk per trade (fraction of capital)
            trailing_stop: Use trailing stop

        Returns:
            Dictionary of performance metrics
        """
        # Convert pips to price (for EURUSD, 1 pip = 0.0001)
        pip_value = 0.0001
        sl_price = stop_loss_pips * pip_value
        tp_price = take_profit_pips * pip_value

        # Extract signals
        entries = signals['signal'] == 1
        exits = signals['signal'] == -1

        # Calculate position sizes based on risk
        capital = self.initial_capital
        position_sizes = []

        for i in range(len(data)):
            if entries.iloc[i]:
                # Risk-based position sizing
                risk_amount = capital * risk_per_trade
                position_size = risk_amount / sl_price
                position_sizes.append(position_size)
                capital = capital  # Update after trade closes
            else:
                position_sizes.append(0)

        # Use vectorbt Portfolio for fast backtesting if available
        if VECTORBT_AVAILABLE:
            try:
                portfolio = vbt.Portfolio.from_signals(
                    close=data['Close'],
                    entries=entries,
                    exits=exits,
                    init_cash=self.initial_capital,
                    fees=0.0002,  # 2 pips spread
                    slippage=0.0001,  # 1 pip slippage
                )

                # Extract metrics
                metrics = {
                    'total_return': portfolio.total_return(),
                    'sharpe_ratio': portfolio.sharpe_ratio(),
                    'max_drawdown': portfolio.max_drawdown(),
                    'win_rate': portfolio.win_rate(),
                    'total_trades': portfolio.total_trades(),
                    'avg_win': portfolio.winning_trades().avg() if portfolio.winning_trades().count() > 0 else 0,
                    'avg_loss': portfolio.losing_trades().avg() if portfolio.losing_trades().count() > 0 else 0,
                    'profit_factor': portfolio.profit_factor() if portfolio.profit_factor() != np.inf else 0,
                    'final_balance': portfolio.final_value(),
                }

            except Exception as e:
                # Fallback to simple metrics if vectorbt fails
                print(f"Vectorbt error: {e}, using fallback")
                metrics = self._simple_backtest(data, signals, sl_price, tp_price)
        else:
            # Vectorbt not available, use simple backtest
            metrics = self._simple_backtest(data, signals, sl_price, tp_price)

        return metrics

    def _simple_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        sl_price: float,
        tp_price: float
    ) -> Dict:
        """
        Simple backtest fallback if vectorbt fails.
        """
        capital = self.initial_capital
        position = None
        trades = []

        for i in range(len(data)):
            # Entry
            if signals['signal'].iloc[i] == 1 and position is None:
                position = {
                    'entry_price': data['Close'].iloc[i],
                    'entry_time': data.index[i],
                    'direction': 'long'
                }

            # Exit
            elif signals['signal'].iloc[i] == -1 and position is not None:
                exit_price = data['Close'].iloc[i]
                pnl = exit_price - position['entry_price']

                # Apply SL/TP
                if pnl < -sl_price:
                    pnl = -sl_price
                elif pnl > tp_price:
                    pnl = tp_price

                trades.append(pnl)
                position = None

        # Calculate metrics
        if len(trades) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'final_balance': capital,
            }

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]

        total_return = sum(trades) / capital
        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

        # Approximate Sharpe (simplified)
        returns = np.array(trades) / capital
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if len(returns) > 1 else 0

        # Approximate max drawdown
        cumulative = np.cumsum(trades)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) / capital if len(drawdown) > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_balance': capital + sum(trades),
        }


def calculate_composite_fitness(metrics: Dict, config: Dict) -> float:
    """
    Calculate composite fitness score from backtest metrics.

    Args:
        metrics: Backtest performance metrics
        config: Fitness weights from config

    Returns:
        Fitness score (0-1, higher is better)
    """
    # Extract weights
    w_sharpe = config.get('sharpe_weight', 0.4)
    w_winrate = config.get('winrate_weight', 0.3)
    w_dd = config.get('drawdown_weight', 0.3)

    # Normalize metrics to 0-1
    sharpe_norm = min(max(metrics['sharpe_ratio'], 0) / 3.0, 1.0)  # 0-3 → 0-1
    winrate_norm = max((metrics['win_rate'] - 0.5) / 0.3, 0)  # 50-80% → 0-1
    dd_norm = 1.0 - min(metrics['max_drawdown'] / 0.3, 1.0)  # 0-30% → 1-0

    # Composite score
    fitness = (
        w_sharpe * sharpe_norm +
        w_winrate * winrate_norm +
        w_dd * dd_norm
    )

    # Penalize if not enough trades
    min_trades = config.get('min_trades', 10)
    if metrics['total_trades'] < min_trades:
        fitness *= (metrics['total_trades'] / min_trades)

    return fitness

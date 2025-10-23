"""
Comprehensive feature engineering module for forex trading.
Generates 80+ technical indicators across multiple categories.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class FeatureEngine:
    """
    Generate 80+ technical indicators for ML trading models.
    Includes trend, momentum, volatility, volume, price action, and regime features.
    """

    def __init__(self):
        self.feature_names = []

    def generate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all technical indicators from OHLCV data.

        Args:
            data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']

        Returns:
            DataFrame with 80+ feature columns + original OHLCV
        """
        df = data.copy()

        # Track feature names
        self.feature_names = []

        # 1. Trend Indicators (15 features)
        df = self._add_trend_indicators(df)

        # 2. Momentum Indicators (12 features)
        df = self._add_momentum_indicators(df)

        # 3. Volatility Indicators (10 features)
        df = self._add_volatility_indicators(df)

        # 4. Volume Indicators (12 features)
        df = self._add_volume_indicators(df)

        # 5. Price Action Features (15 features)
        df = self._add_price_action_features(df)

        # 6. Market Regime Detection (8 features)
        df = self._add_regime_features(df)

        # 7. Derived Features (10 features)
        df = self._add_derived_features(df)

        # Fill NaN values (forward fill then backward fill)
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    # ========== 1. TREND INDICATORS (15) ==========

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-following indicators.
        EMAs, SMAs, MACD, ADX, Parabolic SAR, Ichimoku.
        """
        close = df['Close']
        high = df['High']
        low = df['Low']

        # EMAs (5)
        df['ema_8'] = close.ewm(span=8, adjust=False).mean()
        df['ema_21'] = close.ewm(span=21, adjust=False).mean()
        df['ema_50'] = close.ewm(span=50, adjust=False).mean()
        df['ema_100'] = close.ewm(span=100, adjust=False).mean()
        df['ema_200'] = close.ewm(span=200, adjust=False).mean()

        # MACD (3)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ADX (3)
        adx_data = self._calculate_adx(high, low, close, period=14)
        df['adx'] = adx_data['adx']
        df['di_plus'] = adx_data['di_plus']
        df['di_minus'] = adx_data['di_minus']

        # Ichimoku Cloud (2)
        df['ichimoku_tenkan'] = self._ichimoku_tenkan(high, low, period=9)
        df['ichimoku_kijun'] = self._ichimoku_kijun(high, low, period=26)

        # Supertrend (2)
        supertrend_data = self._calculate_supertrend(high, low, close, period=10, multiplier=3)
        df['supertrend'] = supertrend_data['supertrend']
        df['supertrend_direction'] = supertrend_data['direction']

        self.feature_names.extend([
            'ema_8', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'di_plus', 'di_minus',
            'ichimoku_tenkan', 'ichimoku_kijun',
            'supertrend', 'supertrend_direction'
        ])

        return df

    def _calculate_adx(self, high, low, close, period=14):
        """Calculate Average Directional Index."""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(span=period, adjust=False).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return {'adx': adx, 'di_plus': plus_di, 'di_minus': minus_di}

    def _ichimoku_tenkan(self, high, low, period=9):
        """Ichimoku Tenkan-sen (Conversion Line)."""
        period_high = high.rolling(window=period).max()
        period_low = low.rolling(window=period).min()
        return (period_high + period_low) / 2

    def _ichimoku_kijun(self, high, low, period=26):
        """Ichimoku Kijun-sen (Base Line)."""
        period_high = high.rolling(window=period).max()
        period_low = low.rolling(window=period).min()
        return (period_high + period_low) / 2

    def _calculate_supertrend(self, high, low, close, period=10, multiplier=3):
        """Calculate Supertrend indicator."""
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        # Basic bands
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)

        # Supertrend calculation
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)

        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1

        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif close.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]

        return {'supertrend': supertrend, 'direction': direction}

    # ========== 2. MOMENTUM INDICATORS (12) ==========

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum oscillators.
        RSI, Stochastic, Williams %R, ROC, CCI, MOM, Ultimate Oscillator.
        """
        close = df['Close']
        high = df['High']
        low = df['Low']

        # RSI (2 periods)
        df['rsi_14'] = self._calculate_rsi(close, period=14)
        df['rsi_28'] = self._calculate_rsi(close, period=28)

        # Stochastic Oscillator (2)
        stoch = self._calculate_stochastic(high, low, close, period=14)
        df['stoch_k'] = stoch['k']
        df['stoch_d'] = stoch['d']

        # Williams %R (1)
        df['williams_r'] = self._calculate_williams_r(high, low, close, period=14)

        # Rate of Change (2)
        df['roc_12'] = ((close - close.shift(12)) / close.shift(12)) * 100
        df['roc_24'] = ((close - close.shift(24)) / close.shift(24)) * 100

        # Commodity Channel Index (1)
        df['cci'] = self._calculate_cci(high, low, close, period=20)

        # Momentum (2)
        df['momentum_10'] = close - close.shift(10)
        df['momentum_20'] = close - close.shift(20)

        # Ultimate Oscillator (1)
        df['ultimate_osc'] = self._calculate_ultimate_oscillator(high, low, close)

        # Money Flow Index (1)
        df['mfi'] = self._calculate_mfi(high, low, close, df['Volume'], period=14)

        self.feature_names.extend([
            'rsi_14', 'rsi_28', 'stoch_k', 'stoch_d', 'williams_r',
            'roc_12', 'roc_24', 'cci', 'momentum_10', 'momentum_20',
            'ultimate_osc', 'mfi'
        ])

        return df

    def _calculate_rsi(self, close, period=14):
        """Calculate Relative Strength Index."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_stochastic(self, high, low, close, period=14):
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=3).mean()
        return {'k': k, 'd': d}

    def _calculate_williams_r(self, high, low, close, period=14):
        """Calculate Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr

    def _calculate_cci(self, high, low, close, period=20):
        """Calculate Commodity Channel Index."""
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return cci

    def _calculate_ultimate_oscillator(self, high, low, close):
        """Calculate Ultimate Oscillator."""
        # Buying Pressure
        bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Averages
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()

        uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
        return uo

    def _calculate_mfi(self, high, low, close, volume, period=14):
        """Calculate Money Flow Index."""
        tp = (high + low + close) / 3
        mf = tp * volume

        mf_pos = mf.where(tp > tp.shift(), 0).rolling(period).sum()
        mf_neg = mf.where(tp < tp.shift(), 0).rolling(period).sum()

        mfi = 100 - (100 / (1 + mf_pos / mf_neg))
        return mfi

    # ========== 3. VOLATILITY INDICATORS (10) ==========

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility measures.
        ATR, Bollinger Bands, Keltner Channels, Donchian Channels.
        """
        close = df['Close']
        high = df['High']
        low = df['Low']

        # ATR (2 periods)
        df['atr_14'] = self._calculate_atr(high, low, close, period=14)
        df['atr_28'] = self._calculate_atr(high, low, close, period=28)

        # Bollinger Bands (3)
        bb = self._calculate_bollinger_bands(close, period=20, std=2)
        df['bb_upper'] = bb['upper']
        df['bb_middle'] = bb['middle']
        df['bb_lower'] = bb['lower']

        # Keltner Channels (2)
        kc = self._calculate_keltner_channels(high, low, close, period=20, atr_mult=2)
        df['kc_upper'] = kc['upper']
        df['kc_lower'] = kc['lower']

        # Donchian Channels (2)
        df['donchian_upper'] = high.rolling(window=20).max()
        df['donchian_lower'] = low.rolling(window=20).min()

        # Historical Volatility (1)
        df['hist_volatility'] = close.pct_change().rolling(window=20).std() * np.sqrt(252)

        self.feature_names.extend([
            'atr_14', 'atr_28', 'bb_upper', 'bb_middle', 'bb_lower',
            'kc_upper', 'kc_lower', 'donchian_upper', 'donchian_lower', 'hist_volatility'
        ])

        return df

    def _calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return atr

    def _calculate_bollinger_bands(self, close, period=20, std=2):
        """Calculate Bollinger Bands."""
        middle = close.rolling(window=period).mean()
        std_dev = close.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return {'upper': upper, 'middle': middle, 'lower': lower}

    def _calculate_keltner_channels(self, high, low, close, period=20, atr_mult=2):
        """Calculate Keltner Channels."""
        middle = close.ewm(span=period, adjust=False).mean()
        atr = self._calculate_atr(high, low, close, period=period)
        upper = middle + (atr_mult * atr)
        lower = middle - (atr_mult * atr)
        return {'upper': upper, 'lower': lower}

    # ========== 4. VOLUME INDICATORS (12) ==========

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.
        OBV, VWAP, AD Line, CMF, Volume EMAs, Force Index.
        """
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # OBV (1)
        df['obv'] = self._calculate_obv(close, volume)

        # VWAP (1)
        df['vwap'] = self._calculate_vwap(high, low, close, volume)

        # Accumulation/Distribution Line (1)
        df['ad_line'] = self._calculate_ad_line(high, low, close, volume)

        # Chaikin Money Flow (1)
        df['cmf'] = self._calculate_cmf(high, low, close, volume, period=20)

        # Volume EMAs (3)
        df['volume_ema_8'] = volume.ewm(span=8, adjust=False).mean()
        df['volume_ema_21'] = volume.ewm(span=21, adjust=False).mean()
        df['volume_ema_50'] = volume.ewm(span=50, adjust=False).mean()

        # Volume Ratio (1)
        df['volume_ratio'] = volume / volume.rolling(window=20).mean()

        # Force Index (2)
        df['force_index_13'] = self._calculate_force_index(close, volume, period=13)
        df['force_index_2'] = self._calculate_force_index(close, volume, period=2)

        # Ease of Movement (1)
        df['eom'] = self._calculate_eom(high, low, volume)

        # Volume Price Trend (1)
        df['vpt'] = self._calculate_vpt(close, volume)

        self.feature_names.extend([
            'obv', 'vwap', 'ad_line', 'cmf', 'volume_ema_8', 'volume_ema_21',
            'volume_ema_50', 'volume_ratio', 'force_index_13', 'force_index_2',
            'eom', 'vpt'
        ])

        return df

    def _calculate_obv(self, close, volume):
        """Calculate On-Balance Volume."""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    def _calculate_vwap(self, high, low, close, volume):
        """Calculate Volume Weighted Average Price."""
        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / volume.cumsum()

    def _calculate_ad_line(self, high, low, close, volume):
        """Calculate Accumulation/Distribution Line."""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad = (clv * volume).cumsum()
        return ad

    def _calculate_cmf(self, high, low, close, volume, period=20):
        """Calculate Chaikin Money Flow."""
        mfv = ((close - low) - (high - close)) / (high - low) * volume
        mfv = mfv.fillna(0)
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf

    def _calculate_force_index(self, close, volume, period=13):
        """Calculate Force Index."""
        fi = close.diff() * volume
        return fi.ewm(span=period, adjust=False).mean()

    def _calculate_eom(self, high, low, volume):
        """Calculate Ease of Movement."""
        distance = ((high + low) / 2 - (high.shift() + low.shift()) / 2)
        box_ratio = (volume / 100000) / (high - low)
        eom = distance / box_ratio
        return eom.ewm(span=14, adjust=False).mean()

    def _calculate_vpt(self, close, volume):
        """Calculate Volume Price Trend."""
        vpt = (volume * close.pct_change()).cumsum()
        return vpt

    # ========== 5. PRICE ACTION FEATURES (15) ==========

    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price action patterns and levels.
        Support/resistance, pivots, candlestick patterns, price position.
        """
        close = df['Close']
        high = df['High']
        low = df['Low']
        open_price = df['Open']

        # Candlestick body and shadows (4)
        df['candle_body'] = abs(close - open_price)
        df['upper_shadow'] = high - pd.concat([open_price, close], axis=1).max(axis=1)
        df['lower_shadow'] = pd.concat([open_price, close], axis=1).min(axis=1) - low
        df['body_ratio'] = df['candle_body'] / (high - low + 1e-10)

        # Price position relative to bands (3)
        df['price_to_bb_upper'] = (close - df['bb_upper']) / df['bb_upper']
        df['price_to_bb_lower'] = (close - df['bb_lower']) / df['bb_lower']
        df['price_to_vwap'] = (close - df['vwap']) / df['vwap']

        # Support/Resistance levels (2)
        df['resistance_20'] = high.rolling(window=20).max()
        df['support_20'] = low.rolling(window=20).min()

        # Pivot Points (3)
        df['pivot_point'] = (high.shift() + low.shift() + close.shift()) / 3
        df['pivot_r1'] = 2 * df['pivot_point'] - low.shift()
        df['pivot_s1'] = 2 * df['pivot_point'] - high.shift()

        # Price gaps (1)
        df['gap'] = open_price - close.shift()

        # Candlestick patterns (2)
        df['doji'] = (abs(close - open_price) < (high - low) * 0.1).astype(int)
        df['hammer'] = ((df['lower_shadow'] > 2 * df['candle_body']) &
                       (df['upper_shadow'] < df['candle_body'])).astype(int)

        self.feature_names.extend([
            'candle_body', 'upper_shadow', 'lower_shadow', 'body_ratio',
            'price_to_bb_upper', 'price_to_bb_lower', 'price_to_vwap',
            'resistance_20', 'support_20', 'pivot_point', 'pivot_r1', 'pivot_s1',
            'gap', 'doji', 'hammer'
        ])

        return df

    # ========== 6. MARKET REGIME DETECTION (8) ==========

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime classification features.
        Trend state, volatility regime, market phase.
        """
        close = df['Close']

        # Trend strength (2)
        df['trend_strength'] = (df['ema_21'] - df['ema_50']) / df['ema_50']
        df['trend_consistency'] = (close > df['ema_21']).rolling(window=20).mean()

        # Volatility regime (2)
        df['volatility_regime'] = (df['atr_14'] / close).rolling(window=50).rank(pct=True)
        df['volatility_expansion'] = df['atr_14'] / df['atr_28']

        # Market phase (2)
        df['price_momentum_20'] = (close - close.shift(20)) / close.shift(20)
        df['volume_momentum_20'] = (df['Volume'] - df['Volume'].shift(20)) / df['Volume'].shift(20)

        # Consolidation detection (1)
        df['consolidation'] = (df['atr_14'] < df['atr_14'].rolling(window=50).quantile(0.3)).astype(int)

        # Breakout potential (1)
        df['breakout_potential'] = (
            (df['Volume'] > df['volume_ema_21'] * 1.5) &
            (df['candle_body'] > df['candle_body'].rolling(window=20).mean() * 1.5)
        ).astype(int)

        self.feature_names.extend([
            'trend_strength', 'trend_consistency', 'volatility_regime', 'volatility_expansion',
            'price_momentum_20', 'volume_momentum_20', 'consolidation', 'breakout_potential'
        ])

        return df

    # ========== 7. DERIVED FEATURES (10) ==========

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features combining multiple indicators.
        Momentum scores, strength indices, composite signals.
        """
        # Momentum composite (2)
        df['momentum_composite'] = (
            (df['rsi_14'] / 100) * 0.3 +
            ((df['stoch_k'] + 50) / 100) * 0.3 +
            ((df['williams_r'] + 50) / 100) * 0.2 +
            ((df['cci'] + 100) / 200) * 0.2
        )

        df['trend_momentum_align'] = (
            (df['Close'] > df['ema_21']).astype(int) *
            (df['macd_hist'] > 0).astype(int)
        )

        # Volume strength (2)
        df['volume_strength'] = (
            (df['obv'] > df['obv'].shift(5)).astype(int) +
            (df['cmf'] > 0).astype(int) +
            (df['ad_line'] > df['ad_line'].shift(5)).astype(int)
        ) / 3

        df['volume_price_confirm'] = (
            ((df['Close'] > df['Close'].shift()) & (df['Volume'] > df['volume_ema_21'])).astype(int)
        )

        # Volatility-adjusted momentum (2)
        df['vol_adj_momentum'] = df['momentum_10'] / (df['atr_14'] + 1e-10)
        df['vol_adj_roc'] = df['roc_12'] / (df['hist_volatility'] + 1e-10)

        # Cross signals (2)
        df['ema_cross_signal'] = ((df['ema_8'] > df['ema_21']).astype(int) -
                                  (df['ema_8'] < df['ema_21']).astype(int))
        df['macd_cross_signal'] = ((df['macd'] > df['macd_signal']).astype(int) -
                                   (df['macd'] < df['macd_signal']).astype(int))

        # Overbought/Oversold composite (1)
        df['overbought_oversold'] = (
            ((df['rsi_14'] > 70).astype(int) - (df['rsi_14'] < 30).astype(int)) * 0.4 +
            ((df['stoch_k'] > 80).astype(int) - (df['stoch_k'] < 20).astype(int)) * 0.3 +
            ((df['williams_r'] > -20).astype(int) - (df['williams_r'] < -80).astype(int)) * 0.3
        )

        # Price-to-moving-average ratios (1)
        df['price_ma_distance'] = (df['Close'] - df['ema_50']) / df['ema_50']

        self.feature_names.extend([
            'momentum_composite', 'trend_momentum_align', 'volume_strength', 'volume_price_confirm',
            'vol_adj_momentum', 'vol_adj_roc', 'ema_cross_signal', 'macd_cross_signal',
            'overbought_oversold', 'price_ma_distance'
        ])

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of all generated feature names."""
        return self.feature_names

    def normalize_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Normalize features using rolling z-score (online normalization).

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names to normalize

        Returns:
            DataFrame with normalized features
        """
        df_norm = df.copy()

        for col in feature_cols:
            if col in df_norm.columns:
                # Rolling z-score normalization (lookback 100 periods)
                rolling_mean = df_norm[col].rolling(window=100, min_periods=10).mean()
                rolling_std = df_norm[col].rolling(window=100, min_periods=10).std()
                df_norm[col] = (df_norm[col] - rolling_mean) / (rolling_std + 1e-10)

        return df_norm

# Libraries
import polars as pl
import numpy as np
from typing import Dict, List


class FeatureEngineer:
    """Calculate technical indicators and statistical features."""

    def __init__(self, config: Dict):
        self.config = config

    def calculate_all_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate all features for the given DataFrame."""
        df = self.calculate_returns(df)
        df = self.calculate_moving_averages(df)
        df = self.calculate_rsi(df)
        df = self.calculate_macd(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_atr(df)
        df = self.calculate_stochastic(df)
        df = self.calculate_williams_r(df)
        df = self.calculate_cci(df)
        df = self.calculate_obv(df)
        df = self.calculate_advanced_volatility(df)
        df = self.calculate_statistical_features(df)
        df = self.calculate_price_patterns(df)
        df = self.calculate_advanced_indicators(df)
        df = self.calculate_market_microstructure(df)
        df = self.calculate_time_based_features(df)
        df = self.calculate_interaction_features(df)
        df = self.calculate_regime_features(df)
        return df

    def calculate_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate various return periods."""
        df = df.with_columns([
            (pl.col("close").pct_change()).alias("return_1d"),
            (pl.col("close").pct_change(5)).alias("return_5d"),
            (pl.col("close").pct_change(20)).alias("return_20d"),
        ])
        return df

    def calculate_moving_averages(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Simple and Exponential Moving Averages."""
        # Simple Moving Averages
        for period in self.config["sma_periods"]:
            df = df.with_columns(
                pl.col("close").rolling_mean(period).alias(f"sma_{period}")
            )

        # Exponential Moving Averages
        for period in self.config["ema_periods"]:
            df = df.with_columns(
                pl.col("close").ewm_mean(span=period).alias(f"ema_{period}")
            )

        return df

    def calculate_rsi(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Calculate Relative Strength Index."""
        # Calculate price changes
        delta = pl.col("close").diff()

        # Separate gains and losses
        gains = pl.when(delta > 0).then(delta).otherwise(0)
        losses = pl.when(delta < 0).then(-delta).otherwise(0)

        # Calculate average gains and losses
        avg_gains = gains.rolling_mean(period)
        avg_losses = losses.rolling_mean(period)

        # Calculate RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        df = df.with_columns(rsi.alias("rsi"))
        return df

    def calculate_macd(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        # Calculate 12-day and 26-day EMAs
        ema_12 = pl.col("close").ewm_mean(span=12)
        ema_26 = pl.col("close").ewm_mean(span=26)

        # MACD line
        macd_line = ema_12 - ema_26

        # Signal line (9-day EMA of MACD)
        signal_line = macd_line.ewm_mean(span=9)

        # MACD histogram
        macd_histogram = macd_line - signal_line

        df = df.with_columns([
            macd_line.alias("macd"),
            signal_line.alias("macd_signal"),
            macd_histogram.alias("macd_histogram")
        ])

        return df

    def calculate_bollinger_bands(self, df: pl.DataFrame, period: int = 20, std_dev: int = 2) -> pl.DataFrame:
        """Calculate Bollinger Bands."""
        # Middle Band (SMA)
        middle_band = pl.col("close").rolling_mean(period)

        # Standard deviation
        std = pl.col("close").rolling_std(period)

        # Upper and Lower Bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        df = df.with_columns([
            middle_band.alias("bb_middle"),
            upper_band.alias("bb_upper"),
            lower_band.alias("bb_lower"),
            ((pl.col("close") - lower_band) / (upper_band - lower_band)).alias("bb_position")
        ])

        return df

    def calculate_atr(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Calculate Average True Range for volatility."""
        # True Range calculation
        high_low = pl.col("high") - pl.col("low")
        high_close = (pl.col("high") - pl.col("close").shift(1)).abs()
        low_close = (pl.col("low") - pl.col("close").shift(1)).abs()

        true_range = pl.max_horizontal([high_low, high_close, low_close])

        # ATR (exponential moving average of TR)
        atr = true_range.ewm_mean(span=period)

        df = df.with_columns(atr.alias("atr"))
        return df

    def calculate_stochastic(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Calculate Stochastic Oscillator."""
        # Calculate rolling min/max
        rolling_low = pl.col("low").rolling_min(period)
        rolling_high = pl.col("high").rolling_max(period)
        
        # %K calculation
        k_percent = ((pl.col("close") - rolling_low) / (rolling_high - rolling_low)) * 100
        
        # %D calculation (3-period SMA of %K)
        d_percent = k_percent.rolling_mean(3)
        
        df = df.with_columns([
            k_percent.alias("stoch_k"),
            d_percent.alias("stoch_d")
        ])
        
        return df
    
    def calculate_williams_r(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Calculate Williams %R indicator."""
        # Calculate rolling min/max
        rolling_low = pl.col("low").rolling_min(period)
        rolling_high = pl.col("high").rolling_max(period)
        
        # Williams %R calculation
        williams_r = ((rolling_high - pl.col("close")) / (rolling_high - rolling_low)) * -100
        
        df = df.with_columns(williams_r.alias("williams_r"))
        
        return df
    
    def calculate_cci(self, df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
        """Calculate Commodity Channel Index."""
        # Typical Price
        typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
        
        # Moving Average of Typical Price
        ma_typical = typical_price.rolling_mean(period)
        
        # Mean Deviation
        mean_dev = (typical_price - ma_typical).abs().rolling_mean(period)
        
        # CCI calculation
        cci = (typical_price - ma_typical) / (0.015 * mean_dev)
        
        df = df.with_columns(cci.alias("cci"))
        
        return df
    
    def calculate_obv(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate On-Balance Volume."""
        # Volume direction based on price change
        volume_direction = pl.when(pl.col("close") > pl.col("close").shift(1)).then(pl.col("volume"))
        volume_direction = volume_direction.when(pl.col("close") < pl.col("close").shift(1)).then(-pl.col("volume"))
        volume_direction = volume_direction.otherwise(0)
        
        # Cumulative sum for OBV
        obv = volume_direction.cum_sum()
        
        # OBV momentum (rate of change)
        obv_momentum = obv.pct_change(10)
        
        df = df.with_columns([
            obv.alias("obv"),
            obv_momentum.alias("obv_momentum")
        ])
        
        return df
    
    def calculate_advanced_volatility(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate advanced volatility measures."""
        # Parkinson volatility (using high-low range)
        parkinson_vol = ((pl.col("high") / pl.col("low")).log() ** 2).rolling_mean(20).sqrt() * np.sqrt(252)
        
        # Garman-Klass volatility
        gk_vol = (
            0.5 * (pl.col("high") / pl.col("low")).log() ** 2 -
            (2 * np.log(2) - 1) * ((pl.col("close") / pl.col("open")).log() ** 2)
        ).rolling_mean(20).sqrt() * np.sqrt(252)
        
        # Volatility ratio (short-term vs long-term)
        if "volatility_20d" in df.columns and "volatility_50d" in df.columns:
            vol_ratio = pl.col("volatility_20d") / pl.col("volatility_50d")
        else:
            vol_ratio = pl.lit(1.0)  # Default ratio of 1
        
        df = df.with_columns([
            parkinson_vol.alias("parkinson_vol"),
            gk_vol.alias("gk_vol"),
            vol_ratio.alias("vol_ratio")
        ])
        
        return df
    
    def calculate_statistical_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate statistical features."""
        # Rolling statistics
        for window in [20, 50]:
            # Check if we have enough data for the window
            if len(df) >= window:
                df = df.with_columns([
                    pl.col("return_1d").rolling_std(window).alias(f"volatility_{window}d"),
                    pl.col("return_1d").rolling_skew(window).alias(f"skew_{window}d"),
                    pl.col("return_1d").rolling_sum(window).alias(f"momentum_{window}d")
                ])
            else:
                # Create columns with reduced window or null values
                actual_window = min(window, max(5, len(df) // 2))  # Use at least 5 days or half the data
                df = df.with_columns([
                    pl.col("return_1d").rolling_std(actual_window).alias(f"volatility_{window}d"),
                    pl.col("return_1d").rolling_skew(actual_window).alias(f"skew_{window}d"),
                    pl.col("return_1d").rolling_sum(actual_window).alias(f"momentum_{window}d")
                ])

        # Price relative to moving averages
        if "sma_20" in df.columns and "sma_50" in df.columns:
            df = df.with_columns([
                (pl.col("close") / pl.col("sma_20") - 1).alias("price_to_sma20"),
                (pl.col("close") / pl.col("sma_50") - 1).alias("price_to_sma50")
            ])
        else:
            # Create default columns if SMAs don't exist
            df = df.with_columns([
                pl.lit(0).alias("price_to_sma20"),
                pl.lit(0).alias("price_to_sma50")
            ])

        return df

    def calculate_price_patterns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate price pattern indicators."""
        # Higher highs and lower lows
        hh = (pl.col("high") > pl.col("high").shift(1)) & (pl.col("high").shift(1) > pl.col("high").shift(2))
        ll = (pl.col("low") < pl.col("low").shift(1)) & (pl.col("low").shift(1) < pl.col("low").shift(2))
        
        # Trend consistency (using rolling correlation coefficient)
        # Create a simple trend measure using price change direction consistency
        price_changes = pl.col("close").diff()
        trend_consistency = price_changes.sign().rolling_mean(20)
        
        # Price acceleration
        price_acceleration = pl.col("return_1d").diff()
        
        df = df.with_columns([
            hh.cast(pl.Int8).alias("higher_high"),
            ll.cast(pl.Int8).alias("lower_low"),
            trend_consistency.alias("trend_consistency"),
            price_acceleration.alias("price_acceleration")
        ])
        
        return df

    def calculate_advanced_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate advanced technical indicators."""
        # Relative Vigor Index (RVI)
        close_open = pl.col("close") - pl.col("open")
        high_low = pl.col("high") - pl.col("low")
        rvi_numerator = close_open.rolling_mean(4)
        rvi_denominator = high_low.rolling_mean(4)
        rvi = rvi_numerator / (rvi_denominator + 1e-8)
        
        # Chande Momentum Oscillator (CMO)
        price_changes = pl.col("close").diff()
        gains = pl.when(price_changes > 0).then(price_changes).otherwise(0)
        losses = pl.when(price_changes < 0).then(-price_changes).otherwise(0)
        gain_sum = gains.rolling_sum(14)
        loss_sum = losses.rolling_sum(14)
        cmo = 100 * (gain_sum - loss_sum) / (gain_sum + loss_sum + 1e-8)
        
        # Market Facilitation Index
        mfi = (pl.col("high") - pl.col("low")) / pl.col("volume")
        
        # Ehlers Filter (Simple implementation)
        alpha = 0.07
        ehlers_filter = pl.col("close").ewm_mean(alpha=alpha)
        
        # Adaptive Moving Average
        efficiency_ratio = (pl.col("close") - pl.col("close").shift(10)).abs() / \
                          pl.col("close").diff().abs().rolling_sum(10)
        
        df = df.with_columns([
            rvi.alias("rvi"),
            cmo.alias("cmo"),
            mfi.alias("mfi"),
            ehlers_filter.alias("ehlers_filter"),
            efficiency_ratio.alias("efficiency_ratio")
        ])
        
        return df
    
    def calculate_market_microstructure(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate market microstructure features."""
        # Typical price spread (proxy for bid-ask spread)
        price_spread = (pl.col("high") - pl.col("low")) / pl.col("close")
        
        # Intraday returns
        open_close_return = (pl.col("close") - pl.col("open")) / pl.col("open")
        high_close_return = (pl.col("high") - pl.col("close")) / pl.col("close")
        low_close_return = (pl.col("low") - pl.col("close")) / pl.col("close")
        
        # Volume-weighted features
        vwap = ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).rolling_sum(20) / \
               pl.col("volume").rolling_sum(20)
        
        # Order flow proxy
        buy_pressure = pl.when(pl.col("close") > pl.col("open")).then(pl.col("volume")).otherwise(0)
        sell_pressure = pl.when(pl.col("close") < pl.col("open")).then(pl.col("volume")).otherwise(0)
        order_flow_ratio = buy_pressure.rolling_sum(10) / (sell_pressure.rolling_sum(10) + 1e-8)
        
        # Liquidity proxy
        liquidity_proxy = pl.col("volume") / (pl.col("high") - pl.col("low") + 1e-8)
        
        df = df.with_columns([
            price_spread.alias("price_spread"),
            open_close_return.alias("open_close_return"),
            high_close_return.alias("high_close_return"),
            low_close_return.alias("low_close_return"),
            vwap.alias("vwap"),
            order_flow_ratio.alias("order_flow_ratio"),
            liquidity_proxy.alias("liquidity_proxy")
        ])
        
        return df
    
    def calculate_time_based_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate time-based features."""
        # Convert datetime to useful time features
        if "datetime" in df.columns:
            df = df.with_columns([
                pl.col("datetime").dt.weekday().alias("day_of_week"),
                pl.col("datetime").dt.month().alias("month"),
                pl.col("datetime").dt.hour().alias("hour")
            ])
            
            # Create cyclical features for better ML performance
            df = df.with_columns([
                (2 * np.pi * pl.col("day_of_week") / 7).sin().alias("day_of_week_sin"),
                (2 * np.pi * pl.col("day_of_week") / 7).cos().alias("day_of_week_cos"),
                (2 * np.pi * pl.col("month") / 12).sin().alias("month_sin"),
                (2 * np.pi * pl.col("month") / 12).cos().alias("month_cos")
            ])
        else:
            # Create default columns if datetime not available
            df = df.with_columns([
                pl.lit(1).alias("day_of_week"),
                pl.lit(1).alias("month"),
                pl.lit(0).alias("hour"),
                pl.lit(0).alias("day_of_week_sin"),
                pl.lit(1).alias("day_of_week_cos"),
                pl.lit(0).alias("month_sin"),
                pl.lit(1).alias("month_cos")
            ])
        
        # Rolling windows for different timeframes
        short_window_return = pl.col("return_1d").rolling_mean(3)
        medium_window_return = pl.col("return_1d").rolling_mean(10)
        long_window_return = pl.col("return_1d").rolling_mean(30)
        
        # Lag features
        lag_features = []
        for lag in [1, 2, 3, 5, 10]:
            lag_features.extend([
                pl.col("return_1d").shift(lag).alias(f"return_1d_lag_{lag}"),
                pl.col("rsi").shift(lag).alias(f"rsi_lag_{lag}"),
                pl.col("volatility_20d").shift(lag).alias(f"volatility_lag_{lag}")
            ])
        
        df = df.with_columns([
            short_window_return.alias("short_window_return"),
            medium_window_return.alias("medium_window_return"),
            long_window_return.alias("long_window_return")
        ] + lag_features)
        
        return df
    
    def calculate_interaction_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create feature interactions for better ML performance."""
        # Momentum-volatility interactions
        rsi_vol_interaction = pl.col("rsi") * pl.col("volatility_20d")
        macd_bb_interaction = pl.col("macd") * pl.col("bb_position")
        
        # Price-volume interactions
        price_volume_interaction = pl.col("return_1d") * pl.col("volume").pct_change()
        
        # Trend-momentum interactions
        sma_rsi_interaction = (pl.col("close") / pl.col("sma_20")) * pl.col("rsi")
        
        # Volatility regime interactions
        vol_momentum_interaction = pl.col("volatility_20d") * pl.col("momentum_20d")
        
        # Cross-timeframe interactions
        short_long_momentum = pl.col("short_window_return") * pl.col("long_window_return")
        
        # Technical indicator interactions
        stoch_williams_interaction = pl.col("stoch_k") * pl.col("williams_r")
        cci_rsi_interaction = pl.col("cci") * pl.col("rsi")
        
        df = df.with_columns([
            rsi_vol_interaction.alias("rsi_vol_interaction"),
            macd_bb_interaction.alias("macd_bb_interaction"),
            price_volume_interaction.alias("price_volume_interaction"),
            sma_rsi_interaction.alias("sma_rsi_interaction"),
            vol_momentum_interaction.alias("vol_momentum_interaction"),
            short_long_momentum.alias("short_long_momentum"),
            stoch_williams_interaction.alias("stoch_williams_interaction"),
            cci_rsi_interaction.alias("cci_rsi_interaction")
        ])
        
        return df
    
    def calculate_regime_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate market regime features."""
        # Volatility regime detection
        vol_rolling_mean = pl.col("volatility_20d").rolling_mean(50)
        vol_rolling_std = pl.col("volatility_20d").rolling_std(50)
        vol_zscore = (pl.col("volatility_20d") - vol_rolling_mean) / (vol_rolling_std + 1e-8)
        
        # Trend regime detection
        trend_score = (pl.col("close") / pl.col("sma_50") - 1) * 100
        trend_consistency_score = pl.col("trend_consistency").rolling_mean(10)
        
        # Momentum regime
        momentum_zscore = (pl.col("momentum_20d") - pl.col("momentum_20d").rolling_mean(50)) / \
                         (pl.col("momentum_20d").rolling_std(50) + 1e-8)
        
        # Market stress indicator
        stress_indicator = (vol_zscore.abs() + momentum_zscore.abs()) / 2
        
        # Regime classification
        vol_regime = pl.when(vol_zscore > 1).then(2)  # High volatility
        vol_regime = vol_regime.when(vol_zscore < -1).then(0)  # Low volatility
        vol_regime = vol_regime.otherwise(1)  # Normal volatility
        
        trend_regime = pl.when(trend_score > 5).then(2)  # Strong uptrend
        trend_regime = trend_regime.when(trend_score < -5).then(0)  # Strong downtrend
        trend_regime = trend_regime.otherwise(1)  # Sideways
        
        df = df.with_columns([
            vol_zscore.alias("vol_zscore"),
            trend_score.alias("trend_score"),
            trend_consistency_score.alias("trend_consistency_score"),
            momentum_zscore.alias("momentum_zscore"),
            stress_indicator.alias("stress_indicator"),
            vol_regime.alias("vol_regime"),
            trend_regime.alias("trend_regime")
        ])
        
        return df
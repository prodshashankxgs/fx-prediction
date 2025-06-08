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
        df = self.calculate_statistical_features(df)
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

    def calculate_statistical_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate statistical features."""
        # Rolling statistics
        for window in [20, 50]:
            df = df.with_columns([
                pl.col("return_1d").rolling_std(window).alias(f"volatility_{window}d"),
                pl.col("return_1d").rolling_skew(window).alias(f"skew_{window}d"),
                pl.col("return_1d").rolling_sum(window).alias(f"momentum_{window}d")
            ])

        # Price relative to moving averages
        df = df.with_columns([
            (pl.col("close") / pl.col("sma_20") - 1).alias("price_to_sma20"),
            (pl.col("close") / pl.col("sma_50") - 1).alias("price_to_sma50")
        ])

        return df
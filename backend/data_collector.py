# Libraries
import polars as pl
from datetime import datetime, timedelta
from typing import Optional
import requests
import os

# Import configuration
from backend.config import CACHE_EXPIRY_HOURS


class PolygonClient:
    """Client for interacting with Polygon.io API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def get_forex_data(
            self,
            pair: str,
            timeframe: str = "day",
            start_date: str = None,
            end_date: str = None,
            limit: int = 5000
    ) -> pl.DataFrame:
        """Fetch historical forex data from Polygon.io."""

        # Parse currency pair
        from_currency, to_currency = pair.split(":")

        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Construct API endpoint
        endpoint = f"{self.base_url}/v2/aggs/ticker/C:{from_currency}{to_currency}/range/1/{timeframe}/{start_date}/{end_date}"

        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": limit
        }

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "OK":
                error_msg = data.get("error", "Unknown API error")
                raise ValueError(f"API error for {pair}: {error_msg}")

            if not data.get("results"):
                raise ValueError(
                    f"No data available for {pair}. This pair may not be supported by the API or may not have recent data.")

            # Convert to Polars DataFrame
            df = pl.DataFrame(data["results"])

            # Rename columns for clarity
            df = df.rename({
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "t": "timestamp",
                "n": "transactions"
            })

            # Convert timestamp to datetime
            df = df.with_columns(
                pl.from_epoch("timestamp", time_unit="ms").alias("datetime")
            )

            return df.sort("datetime")

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def validate_currency_pair(self, pair: str) -> bool:
        """Validate if the currency pair format is correct."""
        # Check if pair has correct format (XXX:YYY)
        if ":" not in pair:
            return False

        parts = pair.split(":")
        if len(parts) != 2:
            return False

        # Check if both parts are 3-letter currency codes
        from_currency, to_currency = parts
        if len(from_currency) != 3 or len(to_currency) != 3:
            return False

        # Check if both parts are alphabetic
        if not (from_currency.isalpha() and to_currency.isalpha()):
            return False

        return True


def save_data_to_parquet(data: pl.DataFrame, filepath: str):
    """Save DataFrame to Parquet format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.write_parquet(filepath)


def load_cached_data(filepath: str) -> Optional[pl.DataFrame]:
    """Load cached data if available and not expired."""
    if not os.path.exists(filepath):
        return None

    # Check if cache is expired
    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    if datetime.now() - file_time > timedelta(hours=CACHE_EXPIRY_HOURS):
        return None

    return pl.read_parquet(filepath)
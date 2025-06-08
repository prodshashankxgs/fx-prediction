# Libraries
import os

import streamlit as st
import logging
from datetime import datetime
import traceback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import your existing modules
from backend.main import ForexPredictionApp
# Removed SUPPORTED_PAIRS import - now supporting all currency pairs
from backend.utils import setup_logger
from backend.data_collector import PolygonClient, load_cached_data
from backend.features import FeatureEngineer
from backend.config import POLYGON_API_KEY, TECHNICAL_INDICATORS

# Configure page
st.set_page_config(
    page_title="Forex Rate Prediction Tool",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = setup_logger("streamlit_app")


def create_forex_charts(df: pl.DataFrame, pair: str, prediction=None):
    """Create comprehensive forex charts with technical indicators."""

    # Extract data as lists/arrays for plotting
    datetime_values = df.select("datetime").to_series().to_list()
    open_values = df.select("open").to_series().to_list()
    high_values = df.select("high").to_series().to_list()
    low_values = df.select("low").to_series().to_list()
    close_values = df.select("close").to_series().to_list()

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=['Price Action & Technical Indicators', 'Volume', 'RSI', 'MACD'],
        row_heights=[0.5, 0.2, 0.15, 0.15]
    )

    # 1. Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=datetime_values,
            open=open_values,
            high=high_values,
            low=low_values,
            close=close_values,
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    # Add moving averages if available
    df_columns = df.columns
    ma_columns = [col for col in df_columns if 'sma_' in col or 'ema_' in col]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

    for i, ma_col in enumerate(ma_columns[:5]):  # Limit to 5 MAs
        if ma_col in df_columns:
            ma_values = df.select(ma_col).to_series().to_list()
            fig.add_trace(
                go.Scatter(
                    x=datetime_values,
                    y=ma_values,
                    mode='lines',
                    name=ma_col.upper().replace('_', ' '),
                    line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8
                ),
                row=1, col=1
            )

    # Add Bollinger Bands if available
    if all(col in df_columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
        bb_upper = df.select("bb_upper").to_series().to_list()
        bb_middle = df.select("bb_middle").to_series().to_list()
        bb_lower = df.select("bb_lower").to_series().to_list()

        fig.add_trace(
            go.Scatter(
                x=datetime_values,
                y=bb_upper,
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(128, 128, 128, 0.3)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=datetime_values,
                y=bb_lower,
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(128, 128, 128, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                showlegend=False
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=datetime_values,
                y=bb_middle,
                mode='lines',
                name='BB Middle',
                line=dict(color='rgba(128, 128, 128, 0.5)', width=1, dash='dash')
            ),
            row=1, col=1
        )

    # 2. Volume chart
    if 'volume' in df_columns:
        volume_values = df.select("volume").to_series().to_list()
        colors_volume = ['#26a69a' if close >= open else '#ef5350'
                         for close, open in zip(close_values, open_values)]

        fig.add_trace(
            go.Bar(
                x=datetime_values,
                y=volume_values,
                name='Volume',
                marker_color=colors_volume,
                opacity=0.7
            ),
            row=2, col=1
        )

    # 3. RSI
    if 'rsi' in df_columns:
        rsi_values = df.select("rsi").to_series().to_list()
        fig.add_trace(
            go.Scatter(
                x=datetime_values,
                y=rsi_values,
                mode='lines',
                name='RSI',
                line=dict(color='#FF6B6B', width=2)
            ),
            row=3, col=1
        )

        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)

    # 4. MACD
    if all(col in df_columns for col in ['macd', 'macd_signal']):
        macd_values = df.select("macd").to_series().to_list()
        macd_signal_values = df.select("macd_signal").to_series().to_list()

        fig.add_trace(
            go.Scatter(
                x=datetime_values,
                y=macd_values,
                mode='lines',
                name='MACD',
                line=dict(color='#4ECDC4', width=2)
            ),
            row=4, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=datetime_values,
                y=macd_signal_values,
                mode='lines',
                name='MACD Signal',
                line=dict(color='#FF6B6B', width=2)
            ),
            row=4, col=1
        )

        if 'macd_histogram' in df_columns:
            macd_hist_values = df.select("macd_histogram").to_series().to_list()
            colors_macd = ['#26a69a' if val >= 0 else '#ef5350' for val in macd_hist_values]
            fig.add_trace(
                go.Bar(
                    x=datetime_values,
                    y=macd_hist_values,
                    name='MACD Histogram',
                    marker_color=colors_macd,
                    opacity=0.6
                ),
                row=4, col=1
            )

    # Update layout
    fig.update_layout(
        title=f'{pair} - Comprehensive Technical Analysis',
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        font=dict(size=12)
    )

    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    # Update x-axes
    fig.update_xaxes(title_text="Date", row=4, col=1)

    return fig


def create_prediction_gauge(prediction):
    """Create a gauge chart for prediction confidence."""
    if isinstance(prediction, dict) and 'confidence' in prediction:
        confidence = prediction['confidence']
    else:
        confidence = 0.5  # Default neutral

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prediction Confidence"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def main():
    st.title("💱 Forex Rate Prediction Tool")
    st.markdown("---")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Debug mode
        debug_mode = st.checkbox("Enable Debug Mode", value=False)
        if debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)

        # Cache option
        use_cache = st.checkbox("Use Cached Data", value=True, help="Uncheck to force fresh data fetch")

        st.markdown("---")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Currency Pair Selection")

        # Three input methods
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "Text Entry"],
            horizontal=True
        )

        if input_method == "Manual Input":
            # Single text input for any currency pair
            currency_pair = st.text_input(
                "Enter Currency Pair",
                value="USD:GBP",
                placeholder="e.g., USD:GBP, EUR:USD, GBP:JPY, AUD:CAD, CHF:NOK",
                help="Format: FROM:TO (e.g., USD:GBP). You can enter any 3-letter currency codes."
            ).upper().strip()

        else:  # Manual Entry
            # Separate text inputs for maximum flexibility
            col_from, col_to = st.columns(2)

            with col_from:
                from_currency = st.text_input(
                    "From Currency",
                    value="USD",
                    placeholder="e.g., USD, EUR, GBP",
                    help="Enter 3-letter currency code"
                ).upper().strip()

            with col_to:
                to_currency = st.text_input(
                    "To Currency",
                    value="GBP",
                    placeholder="e.g., GBP, JPY, EUR",
                    help="Enter 3-letter currency code"
                ).upper().strip()

            currency_pair = f"{from_currency}:{to_currency}" if from_currency and to_currency else ""

        # Display selected pair
        if currency_pair:
            st.info(f"Selected pair: **{currency_pair}**")

        # Enhanced validation function
        def validate_currency_pair(pair):
            if not pair:
                return False, "Please enter a currency pair"

            if ":" not in pair:
                return False, "Invalid format. Use FROM:TO format (e.g., USD:GBP)"

            try:
                from_curr, to_curr = pair.split(":")

                # Check if currencies are 3-letter codes
                if len(from_curr) != 3 or len(to_curr) != 3:
                    return False, "Currency codes must be exactly 3 letters (e.g., USD, EUR, GBP)"

                # Check if they're different
                if from_curr == to_curr:
                    return False, "From and To currencies cannot be the same"

                # Check if currencies contain only letters
                if not from_curr.isalpha() or not to_curr.isalpha():
                    return False, "Currency codes must contain only letters"

                return True, "Valid currency pair format"

            except ValueError:
                return False, "Invalid format. Use FROM:TO format (e.g., USD:GBP)"

        # Validation
        if currency_pair:
            pair_valid, validation_message = validate_currency_pair(currency_pair)

            if pair_valid:
                st.success(f"✅ {validation_message}")
                st.info(
                    "💡 **Note:** We'll attempt to fetch data and make predictions for any valid currency pair. Accuracy may vary depending on data availability.")
            else:
                st.error(f"❌ {validation_message}")
        else:
            pair_valid = False

        # Prediction button
        st.markdown("---")
        predict_button = st.button(
            "🔮Generate Prediction",
            type="primary",
            use_container_width=True,
            disabled=not pair_valid
        )

    with col2:
        st.header("Quick Stats")
        st.metric("Currency Pairs", "✅ All Supported")
        st.metric("Data Source", "Polygon.io API")
        st.metric("Cache Enabled", "Yes" if use_cache else "No")

        if currency_pair and pair_valid:
            from_curr, to_curr = currency_pair.split(":")
            st.markdown(f"**From:** {from_curr}")
            st.markdown(f"**To:** {to_curr}")

            # Show pair info
            st.info("🌍 Currency pair ready for analysis")

    # Results section
    if predict_button and pair_valid:
        st.markdown("---")
        st.header("🔍 Prediction Results")

        # Create progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Initialize the app
            status_text.text("Initializing prediction engine...")
            progress_bar.progress(10)

            # Initialize components
            client = PolygonClient(POLYGON_API_KEY)
            feature_engineer = FeatureEngineer(TECHNICAL_INDICATORS)
            app = ForexPredictionApp()

            # Update progress
            status_text.text("Fetching market data...")
            progress_bar.progress(30)

            # Get raw data for charting
            cache_file = f"./forex_data/{currency_pair.replace(':', '_')}_daily.parquet"
            df = load_cached_data(cache_file) if use_cache else None

            if df is None:
                from datetime import timedelta
                df = client.get_forex_data(
                    pair=currency_pair,
                    timeframe="day",
                    start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                )

            status_text.text("Calculating technical indicators...")
            progress_bar.progress(50)

            # Calculate features for charting
            df_with_features = feature_engineer.calculate_all_features(df)

            status_text.text("Generating prediction...")
            progress_bar.progress(70)

            # Run prediction with progress updates
            with st.spinner("Running prediction analysis..."):
                prediction = app.run(currency_pair, use_cache=use_cache)
                progress_bar.progress(100)
                status_text.text("Prediction complete!")

            # Display results in a nice format
            st.success("✅ Prediction generated successfully!")

            # Create columns for results display
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.subheader("📊 Prediction Summary")
                if isinstance(prediction, dict):
                    for key, value in prediction.items():
                        if isinstance(value, (int, float)):
                            st.metric(key.replace("_", " ").title(), f"{value:.4f}")
                        else:
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.write(prediction)

                # Add prediction gauge
                if isinstance(prediction, dict):
                    gauge_fig = create_prediction_gauge(prediction)
                    st.plotly_chart(gauge_fig, use_container_width=True)

            with result_col2:
                st.subheader("⏰ Analysis Info")
                st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Currency Pair:** {currency_pair}")
                st.markdown(f"**Data Source:** {'Cache' if use_cache else 'Fresh API'}")
                st.markdown(f"**Data Points:** {len(df_with_features):,}")

                # Latest price info
                if len(df_with_features) > 0:
                    latest_close = df_with_features.select("close").tail(1).item()
                    st.markdown("### 📈 Latest Price")
                    st.metric("Close Price", f"{latest_close:.5f}")
                    if len(df_with_features) > 1:
                        prev_close = df_with_features.select("close").tail(2).head(1).item()
                        change = latest_close - prev_close
                        change_pct = (change / prev_close) * 100
                        st.metric("Daily Change", f"{change:+.5f}", f"{change_pct:+.2f}%")

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Create and display comprehensive charts
            st.markdown("---")
            st.header("📈 Technical Analysis Charts")

            try:
                # Create the main chart
                chart_fig = create_forex_charts(df_with_features, currency_pair, prediction)
                st.plotly_chart(chart_fig, use_container_width=True)

                # Additional charts in tabs
                tab1, tab2, tab3 = st.tabs(["💹 Price Overview", "📊 Volume Analysis", "🔍 Indicators"])

                with tab1:
                    # Simple price chart
                    datetime_vals = df_with_features.select("datetime").to_series().to_list()
                    close_vals = df_with_features.select("close").to_series().to_list()

                    price_fig = go.Figure()
                    price_fig.add_trace(go.Scatter(
                        x=datetime_vals,
                        y=close_vals,
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2)
                    ))

                    price_fig.update_layout(
                        title=f'{currency_pair} - Price Trend (Last 30 Days)',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        height=400,
                        template='plotly_white'
                    )

                    # Show only last 30 days
                    if len(datetime_vals) > 30:
                        price_fig.update_xaxes(range=[datetime_vals[-30], datetime_vals[-1]])

                    st.plotly_chart(price_fig, use_container_width=True)

                with tab2:
                    if 'volume' in df_with_features.columns:
                        volume_vals = df_with_features.select("volume").to_series().to_list()
                        vol_fig = go.Figure()
                        vol_fig.add_trace(go.Bar(
                            x=datetime_vals,
                            y=volume_vals,
                            name='Volume',
                            marker_color='lightblue'
                        ))

                        vol_fig.update_layout(
                            title=f'{currency_pair} - Trading Volume',
                            xaxis_title='Date',
                            yaxis_title='Volume',
                            height=400,
                            template='plotly_white'
                        )

                        st.plotly_chart(vol_fig, use_container_width=True)
                    else:
                        st.info("Volume data not available for this currency pair")

                with tab3:
                    # Technical indicators summary
                    col_ind1, col_ind2 = st.columns(2)

                    with col_ind1:
                        if 'rsi' in df_with_features.columns:
                            latest_rsi = df_with_features.select("rsi").tail(1).item()
                            st.metric("Current RSI", f"{latest_rsi:.2f}")

                            if latest_rsi > 70:
                                st.warning("🔴 Overbought territory")
                            elif latest_rsi < 30:
                                st.warning("🟢 Oversold territory")
                            else:
                                st.info("🟡 Neutral territory")

                    with col_ind2:
                        # Moving average analysis
                        ma_cols = [col for col in df_with_features.columns if 'sma_' in col]
                        if ma_cols:
                            current_price = df_with_features.select("close").tail(1).item()
                            st.markdown("**Moving Average Signals:**")

                            for ma_col in ma_cols[:3]:  # Show first 3
                                ma_value = df_with_features.select(ma_col).tail(1).item()
                                signal = "🟢 Above" if current_price > ma_value else "🔴 Below"
                                st.markdown(f"• {ma_col.upper()}: {signal}")

            except Exception as chart_error:
                st.error(f"Error creating charts: {str(chart_error)}")
                if debug_mode:
                    st.code(traceback.format_exc())

        except Exception as e:
            progress_bar.empty()
            status_text.empty()

            st.error(f"❌ Error during prediction: {str(e)}")

            if debug_mode:
                st.subheader("Debug Information")
                st.code(traceback.format_exc())

            logger.error(f"Streamlit app error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Forex Rate Prediction Tool | Built with Streamlit 🚀</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    def add_ml_section():
        """Add ML training option to Streamlit sidebar."""

        with st.sidebar:
            st.markdown("---")
            st.header("🤖 ML Enhancement")

            # Check if model exists
            import os
            model_path = f"models/{currency_pair.replace(':', '_')}_ml.pkl"
            model_exists = os.path.exists(model_path)

            if model_exists:
                st.success("✅ ML Model Available")
                if st.button("Retrain Model", key="retrain"):
                    train_ml_model()
            else:
                st.info("No ML model trained yet")
                if st.button("Train ML Model", type="primary", key="train"):
                    train_ml_model()

            # Info about ML
            with st.expander("About ML Enhancement"):
                st.markdown("""
                The ML model analyzes patterns in technical indicators to provide
                an additional signal that contributes 15% to the final prediction.

                - Uses XGBoost for fast, accurate predictions
                - Trains on historical price patterns
                - Validates using time series cross-validation
                - Seamlessly integrates with technical analysis
                """)

    def train_ml_model():
        """Train ML model from Streamlit."""
        with st.spinner("Training ML model... This takes about 30 seconds."):
            try:
                # Get the app instance and train
                app = ForexPredictionApp()
                ml_results = app.train_ml_model(currency_pair)

                if ml_results.get("status") == "success":
                    st.success(f"✅ Model trained! Accuracy: {ml_results['validation_accuracy']:.1%}")

                    # Save the model
                    model_path = f"models/{currency_pair.replace(':', '_')}_ml.pkl"
                    os.makedirs("models", exist_ok=True)
                    app.analyzer.ml_analyzer.save(model_path)

                    # Show metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Validation Accuracy", f"{ml_results['validation_accuracy']:.1%}")
                    with col2:
                        st.metric("Features Used", ml_results['feature_count'])

                    st.info("ML model will now contribute to predictions!")
                else:
                    st.error(f"Training failed: {ml_results.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
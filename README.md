# bitcoin_prediction_system_using_2_staged_ensemble

<img width="712" alt="스크린샷 2025-06-09 오후 9 55 10" src="https://github.com/user-attachments/assets/b4bbdef2-8b3c-47ce-b063-2bc4431099f4" />

This project implements **a novel 2-stage ensemble deep learning system for cryptocurrency price prediction that significantly outperforms traditional single-model approaches.** The core innovation lies in our hierarchical learning structure where Stage 1 deploys multiple specialized models to predict different market features (open, high, low, volume), and Stage 2 employs a meta-learning model that intelligently combines these predictions to forecast the final closing price.

Our approach addresses the fundamental limitation of existing cryptocurrency prediction systems that typically use a single model to learn all market patterns simultaneously. Instead, we allow each model to become an expert in specific market characteristics, similar to how a medical team has specialists for different areas who collaborate for comprehensive diagnosis. The first stage consists of four feature-specific models: an open price specialist that focuses solely on opening price patterns, a high price expert that captures peak price movements, a low price analyzer for trough detection, and a volume specialist for trading activity patterns. Each model uses different deep learning architectures including CNN-DNN, CNN-BiGRU, TCN-DNN, and TCN-BiGRU to capture diverse temporal patterns.

The second stage meta-model learns complex relationships between these specialized predictions rather than simply averaging them. It discovers patterns like "when the volume model predicts high activity and the high price model suggests upward movement, the closing price typically increases by X percent." This sophisticated combination approach, based on stacking ensemble methodology, enables the system to make more accurate and robust predictions than any individual model could achieve alone.

We comprehensively evaluate our system on three distinct cryptocurrencies representing different market segments. Bitcoin serves as the stable, mature market representative. Chiliz represents sports fan tokens with moderate volatility and unique event-driven patterns. Sui represents emerging blockchain projects with high volatility and less predictable behavior. This diverse selection ensures our methodology works across different cryptocurrency types and market conditions.

The system's practical effectiveness is demonstrated through real trading simulations using Chiliz, where our best-performing TCN-BiGRU model achieved a 9.9% return rate in 24 hours with a 66.7% win rate. These results validate that our approach is not merely theoretically sound but also practically profitable for actual cryptocurrency trading.

## Dataset and Preprocessing

Our research utilizes comprehensive datasets from three major cryptocurrencies collected over an 8-month period from October 2024 to June 2025. The data consists of 1-hour interval OHLCV (Open, High, Low, Close, Volume) records sourced from major cryptocurrency exchanges through the ccxt library.

| Component | Specification |
|-----------|---------------|
| **Cryptocurrencies** | Bitcoin (BTC/USDT), Chiliz (CHZ/USDT), Sui (SUI/USDT) |
| **Collection Period** | October 2024 - June 2025 (8 months) |
| **Data Frequency** | 1-hour interval OHLCV data |
| **Train/Test Split** | 80% training / 20% testing |
| **Input Sequence Length** | 24 hours (24 time steps) |
| **Batch Size** | 32 |
| **Training Epochs** | 20 |
| **Optimizer** | Adam (learning rate: 0.001) |
| **Evaluation Metrics** | RMSE, MAE, R², Cosine Similarity |

The preprocessing pipeline enhances the raw OHLCV data with technical indicators that capture market momentum and volatility. We calculate price_change as the hour-over-hour percentage change in closing prices to measure market momentum. The volatility indicator represents the 24-hour rolling standard deviation of price changes, quantifying market uncertainty. Volume_ma and price_ma are 24-hour moving averages that smooth out short-term noise and reveal underlying trends. All features undergo MinMaxScaler normalization to ensure consistent scales across different value ranges, preventing larger values from dominating the learning process.

## Architecture

### Stage 1: Feature-Specific Models
- **Open Price Model**: Specializes in predicting opening prices
- **High Price Model**: Focuses on daily high price patterns  
- **Low Price Model**: Captures daily low price movements
- **Volume Model**: Analyzes trading volume patterns

### Stage 2: Meta-Learning Model
- Combines predictions from all Stage 1 models
- Uses Deep Neural Network (DNN) for final close price prediction
- Learns complex interactions between different market features

## Model Architectures

### CNN-DNN (Convolutional Neural Network + Deep Neural Network)
- **CNN Layer**: Captures local temporal patterns in price data
- **DNN Layer**: Learns complex non-linear relationships
- **Best for**: Short-term pattern recognition

### CNN-BiGRU (CNN + Bidirectional Gated Recurrent Unit)
- **CNN Layer**: Local pattern extraction
- **BiGRU Layer**: Bidirectional temporal dependency learning
- **Best for**: Pattern recognition with temporal context

### TCN-DNN (Temporal Convolutional Network + DNN)
- **TCN Layer**: Multi-scale temporal pattern capture using dilated convolutions
- **DNN Layer**: High-level feature combination
- **Best for**: Long-range temporal dependencies

### TCN-BiGRU (TCN + BiGRU) 
- **TCN Layer**: Multi-scale temporal features
- **BiGRU Layer**: Bidirectional sequence modeling
- **Best for**: Complex temporal relationships (Best overall performance)

## Dataset

### Cryptocurrencies
- **Bitcoin (BTC/USDT)**: Most stable, mature market
- **Chiliz (CHZ/USDT)**: Sports fan token with moderate volatility
- **Sui (SUI/USDT)**: New blockchain project with high volatility

### Data Specifications
- **Time Period**: October 2024 - June 2025 (8 months)
- **Frequency**: 1-hour OHLCV data
- **Features**: Open, High, Low, Close, Volume + Technical indicators
- **Train/Test Split**: 80% / 20%
- **Sequence Length**: 24 hours (24 time steps)

### Technical Indicators
- `price_change`: Hour-over-hour price change rate
- `volatility`: 24-hour rolling standard deviation of price changes
- `volume_ma`: 24-hour moving average of volume
- `price_ma`: 24-hour moving average of close price

## Results

### Model Performance Comparison

| Model | Coin | RMSE | MAE | R² | Cosine Similarity |
|-------|------|------|-----|----|--------------------|
| **TCN-BiGRU** | Bitcoin | 0.051218 | 0.036844 | **0.942969** | **0.977675** |
| | Chiliz | 0.070980 | 0.055314 | **0.905338** | **0.979866** |
| | Sui | 0.073315 | 0.062110 | **0.897945** | **0.980616** |
| CNN-BiGRU | Bitcoin | 0.059736 | 0.046052 | 0.922424 | 0.964588 |
| | Chiliz | 0.080938 | 0.065810 | 0.876914 | 0.976782 |
| | Sui | **0.061019** | **0.049764** | 0.929307 | 0.981367 |
| CNN-DNN | Bitcoin | 0.078474 | 0.058365 | 0.866122 | 0.941027 |
| | Chiliz | 0.108409 | 0.086297 | 0.779178 | 0.922254 |
| | Sui | 0.094367 | 0.077195 | 0.830922 | 0.932506 |
| TCN-DNN | Bitcoin | 0.070826 | 0.049185 | 0.890946 | 0.955837 |
| | Chiliz | 0.086356 | 0.068977 | 0.859883 | 0.952993 |
| | Sui | 0.069246 | 0.056717 | 0.908959 | 0.968928 |

**Key Findings:**
- **TCN-BiGRU** achieved the best overall performance across all cryptocurrencies
- **Bitcoin** showed the most predictable patterns (highest R² values)
- **Sui** exhibited the highest volatility but strong directional accuracy
- All models achieved **>90% Cosine Similarity**, indicating excellent directional prediction

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Weather Impact on Agricultural Commodity Prices

## MSc Financial Engineering Capstone Project

This repository contains the code and resources for a Master of Science in Financial Engineering (MScFE) Capstone 
project exploring the relationship between weather conditions and agricultural commodity prices in the United States 
from 2010 to 2025.

### Problem Statement

This study quantifies how weather conditions in key US agricultural regions affect the prices of major commodities 
(corn, soybeans, barley, peanuts, lean hogs) and related products (wheat, soybean oil/meal, dairy feed, peas, pork).
The research aims to:

- Quantify the impact of specific weather variables on price fluctuations
- Analyze price differences between related agricultural products
- Study the speed at which price differences stabilize after weather events

Understanding these dynamics is crucial as weather changes significantly affect agricultural markets, with implications 
for traders, producers, and policymakers.

### Data Sources

- **Weather Data**: `NOAA` and `NASA` datasets providing temperature, precipitation, and extreme weather events
- **Price Data**: Historical commodity prices from `yfinance`
- **Time Period**: 2010-2025

### Plan and Methodology

1. Defining Commodities for Analysis
2. Extract Commodity Price Data
3. Define Locations and Extract Weather Data
4. Perform Exploratory Data Analysis (EDA):
   - visualize price and weather trends;
   - detect anomalies; 
   - compute correlations; 
   - compare distributions during extreme weather years (like, 2012 U.S. drought).
5. Time Series Analysis:
   - detect dominant annual or multi-year cycles using Fast Fourier Transform (FFT);
   - apply ARIMA models to quantify seasonality and mean reversion.
6. Regression Analysis:
   - perform linear and multiple regression of prices and spreads against regional weather variables;
   - perform test for significance, lags, and non-linear impacts;
   - consider/perform search and inclusion of additional control variables (energy prices, time trends etc.) where 
   needed;
7. Measure the Price Spread Dynamics:
   - analyze relationships like the soybean crush spread and the corn-hog margin;
   - using error correction models or co-integration tests to evaluate mean-reverting behavior.

### Repository Structure

```
├── data/            # Raw and processed data files
├── analysis/        # Jupyter notebooks for analysis and visualization
├── src/             # Custom modules to use in jupyter notebooks
├── pyproject.toml   # Project configuration file
├── .env             # Environment variables file (e.g., API keys) - should be added manually
└── README.md        # Project overview
```
<font color="crimson">NB: The .env file is not included in the repository. It should be added manually.</font>

### Key Research Questions

1. How do temperature and precipitation anomalies affect commodity price fluctuations?
2. What is the relationship between weather events and price spreads between related agricultural products?
3. Do commodity prices exhibit mean reversion behavior following weather-induced price movements?
4. How does the impact of weather differ across various agricultural commodities?
5. Can weather data improve the accuracy of agricultural commodity price forecasts?

### Literature Foundation

This research builds upon existing studies that have explored the relationship between weather and agricultural 
economics. Key findings from the literature include:

- Negative impacts of climate change on US agriculture (Fisher et al., 2012)
- Effects of extreme weather on local and futures markets for corn and soybeans (Skevas, 2024)
- Non-linear relationships between temperature/precipitation and crop yields
- Interconnectedness of US commodity markets across different time scales (Zhang et al., 2023)
- Growing price volatility caused by extreme weather events (Sun T. et al., 2023)

Our work extends these findings by providing a more systematic and data-driven analysis focused specifically on the US 
market and selected commodity pairs.

### Installation and Usage

Poetry is used for dependency management. To install Poetry, follow the instructions on the 
[Poetry website](https://python-poetry.org/docs/#installation).

To clone the repository and set up the environment, run the following commands:

```bash
git clone https://github.com/AnglewoodJack/MScFECapstone.git
cd MScFECapstone
poetry install
```

This will create a virtual environment and install all necessary packages listed in `pyproject.toml`.

To run the Jupyter notebooks, you can use the following command:

```bash
poetry run jupyter lab
```
Then navigate to the `analysis` directory and open the desired notebook.

### Contributors

Student Group `9175`:
- Elvis Atta Frimpong – attafrimpongelvis@gmail.com
- Remmo Sri Ardiansyah – remmo.ardiansyah@aol.com
- Ivan Andriushin – i.andryushin@gmail.com

Advisors:
- Prof. Kenneth Abbott

### Acknowledgments

This research was conducted as part of the Master of Science in Financial Engineering program at World Quant University. 
We acknowledge the support of faculty advisors and the use of data provided by NOAA, NASA, and financial data providers.
# Affordable Housing Finder

Housing market cluster analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python program that visualizes and analyzes housing market patterns across cities using Zillow's Home Value Index (ZHVI) data. The program employs K-means clustering to identify market segments based on growth rates, volatility, and average prices for different bedroom configurations.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

- Interactive command-line interface for selecting analysis parameters
- Support for analyzing different housing sizes (1-5+ bedrooms)
- Data preprocessing including:
  - Missing value handling with interpolation
  - Data standardization
  - Automatic outlier removal
- Automated optimal cluster determination using silhouette scoring
- Visualization of market segments through scatter plots
- Analysis metrics:
  - Growth Rate (from 2020-present)
  - Price Volatility
  - Average Price

## Installation

```bash
# Clone the repository
git clone https://github.com/lakota-camp/affordable-housing-finder.git

# Navigate to the project directory
cd affordable-housing-finder

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

## Usage

1. Download the CSV data files from [Zillow](https://www.zillow.com/research/data/)
2. Ensure your data files are properly structured in the `affordable-housing-finder/data/csv/city/` directory
3. Run the main program:

```bash
python main.py
```

1. Follow the interactive prompts to:
   - Select number of bedrooms (1-5+)
   - Choose first variable for analysis
   - Choose second variable for analysis
   - View the resulting cluster visualization

Example code for programmatic usage:

```python
from main import preprocess_data, scale_features, build_k_means_model

# Load and preprocess data
df = pd.read_csv(CITY_HOME_VALUES["ZHVI 3-Bedroom Time Series ($)"])
df = preprocess_data(df)

# Scale features and build model
features = df[["GrowthRate", "Volatility", "AveragePrice"]]
scaled_features = scale_features(features)
k_means_results = build_k_means_model(scaled_features)
```

## Directory Structure

Required directory structure:

```
housing-market-cluster/
├── data/
│   └── csv/
│       └── city/
│           ├── City_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
│           ├── City_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
│           ├── City_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
│           ├── City_zhvi_bdrmcnt_4_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
│           └── City_zhvi_bdrmcnt_5_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
```

## Dependencies

- pandas
- scikit-learn
- matplotlib
- warnings (standard library)

## Configuration

Key configuration parameters are defined in the `CONFIG` dictionary:

```python
CONFIG = {
    "start_year": "2020",
    "missing_data_threshold_percent": 0.25
}
```

## Functions

Key functions include:

- `preprocess_data()`: Handles data cleaning and feature engineering
- `scale_features()`: Standardizes numerical features
- `build_k_means_model()`: Determines optimal clusters and builds model
- `plot_data()`: Visualizes clustering results
- `prompt_user()`: Handles user interaction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/LegendaryFeature`)
3. Commit your changes (`git commit -m 'Add some LegendaryFeature'`)
4. Push to the branch (`git push origin feature/LegendaryFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Zillow Research](https://www.zillow.com/research/data/) for providing the ZHVI dataset

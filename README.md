# StockMetrics: Stock Price vs. Fundamental Variables Analysis

## Project Overview

StockMetrics is a comprehensive data analysis project that examines the relationship between stock prices and fundamental financial metrics for five major IT companies. The project implements correlation analysis and linear regression to identify which financial variables have a statistically significant impact on stock performance.

## Key Features

* **Correlation Analysis:** Calculates Pearson correlation coefficients between each company's stock returns and five fundamental variables
* **Linear Regression:** Performs multiple linear regression using scikit-learn to model the relationship between stock prices and fundamental variables
* **Statistical Significance:** Identifies the most statistically significant variables for each company using F-statistic p-values
* **Data Visualization:** Generates comprehensive visualizations including correlation heatmaps, R-squared scores, and significant variable charts

## Data Sources

The analysis uses data from the provided `data.xls` Excel file, which contains:

* **Sheet 1:** Daily stock prices for five IT companies from 2005 to 2025
* **Sheet 2:** Annual fundamental financial data, including Sales, EBITDA, and PAT

The five companies included in the analysis are:
* Infosys Ltd.
* Wipro Ltd.
* HCL Technologies Ltd.
* Tata Consultancy Services Ltd.
* Tech Mahindra Ltd.

## Calculated Fundamental Variables

From the raw financial data, five key fundamental variables are calculated:
* **Sales Growth:** Year-over-year percentage change in sales
* **EBITDA Growth:** Year-over-year percentage change in EBITDA
* **EBITDA Margin Change:** Change in EBITDA/Sales ratio
* **PAT Growth:** Year-over-year percentage change in Profit After Tax
* **PAT Margin Change:** Change in PAT/Sales ratio

## Methodology

### Data Preprocessing

1. **Data Loading and Validation**
   * Raw data loaded from Excel file with validation checks
   * Empty dataframes and missing required columns detected
   * Data structure validated before analysis

2. **Stock Price Processing**
   * Cleaned by removing headers and converting to numeric format
   * Dates converted to datetime format
   * Resampled to yearly frequency and converted to returns

3. **Fundamental Variables Calculation**
   * Five key metrics calculated as listed above
   * Time series alignment performed between stock and fundamental data

### Statistical Analysis

1. **Correlation Analysis**
   * Pearson correlation coefficients calculated between stock returns and fundamental variables
   * Implementation using NumPy's corrcoef function
   * Results visualized through heatmaps and summary statistics

2. **Linear Regression Analysis**
   * Implemented using scikit-learn's LinearRegression
   * Stock returns modeled as a function of the five fundamental variables
   * Evaluation metrics:
     * R-squared (coefficient of determination)
     * Mean Squared Error (MSE)
   * Feature significance determined using F-statistics and p-values
   * Significance threshold: α = 0.05

## Libraries and Tools Used

1. **Data Manipulation**
   * pandas: Data loading, manipulation, and preprocessing
   * numpy: Numerical operations and calculations

2. **Machine Learning and Statistics**
   * scikit-learn:
     * LinearRegression: Regression modeling
     * r2_score, mean_squared_error: Model evaluation
     * f_regression: Feature significance testing

3. **Visualization**
   * matplotlib: Creating plots and charts
   * seaborn: Enhanced visualization, particularly heatmaps

4. **File Handling**
   * os: File path operations
   * openpyxl: Excel file reading (used by pandas)

## Visualization Techniques

The project generates several visualizations to help interpret the results:

1. **Correlation Heatmap:** Shows the correlation between each company's stock returns and the fundamental variables
2. **R-squared Bar Chart:** Displays the R-squared values for each company's regression model
3. **MSE Bar Chart:** Shows the Mean Squared Error for each regression model
4. **Significant Variables Chart:** Highlights the top 3 most statistically significant variables
5. **Summary Statistics:** Provides an overview of key findings from the analysis

All visualizations are saved in the `output` directory.

## Project Structure

```
StockMetrics/
├── README.md                         # Project documentation
├── data.xls                          # Source data file
├── output/                           # Analysis results
│   ├── complete_analysis_results.png # Visualization of all results
│   ├── complete_correlation_matrix.csv
│   ├── complete_regression_results.csv
│   └── top_3_significant_variables.csv
├── run_analysis.py                   # Entry point script
└── src/                              # Source code
    ├── analysis/
    │   └── analysis.py               # Core analysis functions
    ├── constants.py                  # Project constants
    ├── main.py                       # Main execution logic
    └── utils/
        └── data_loader.py            # Data loading utilities
```

## Installation and Usage

To run the analysis, you will need to have Python 3 and the following libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

Once the dependencies are installed, you can run the analysis by executing:

```bash
python run_analysis.py
```

The script will perform the analysis and save the results in the `output` directory.

## Results and Findings

The analysis identifies which fundamental financial metrics have the strongest correlation with stock price movements and which variables are statistically significant in predicting stock returns through regression analysis.

The complete results can be found in the following files:
* `output/complete_correlation_matrix.csv`: Full correlation matrix for all companies and variables
* `output/complete_regression_results.csv`: Detailed regression results including coefficients and p-values
* `output/top_3_significant_variables.csv`: The three most significant variables for each company
* `output/complete_analysis_results.png`: Visual summary of all analysis results

## Conclusion

The StockMetrics project provides a structured approach to analyzing the relationship between stock performance and fundamental financial metrics. By implementing both correlation analysis and linear regression, it offers insights into which financial variables have the strongest influence on stock returns for the analyzed IT companies.

This analysis can help investors and financial analysts make more informed decisions by understanding which fundamental metrics are most closely tied to stock performance in the IT sector.

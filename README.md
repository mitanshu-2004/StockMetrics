# Stock Price vs. Fundamental Variables Analysis

This project analyzes the relationship between the stock prices of five major IT companies and a set of key fundamental financial variables. The goal is to determine which, if any, of these variables have a statistically significant impact on stock performance.

## Key Features

*   **Correlation Analysis:** Calculates the correlation between each company's stock returns and five fundamental variables.
*   **Linear Regression:** Performs a multiple linear regression to model the relationship between stock prices and the fundamental variables.
*   **Statistical Significance:** Identifies the most statistically significant variables for each company using p-values.
*   **Data Visualization:** Generates a comprehensive set of plots to visualize the results, including a correlation matrix heatmap and R-squared scores for each company.

## Data

The analysis uses data from the provided `data.xls` Excel file, which contains two sheets:

*   **Sheet 1:** Daily stock prices for the five companies from 2005 to 2025.
*   **Sheet 2:** Annual fundamental financial data, including Sales, EBITDA, and PAT.

The five companies included in the analysis are:

*   Infosys Ltd.
*   Wipro Ltd.
*   HCL Technologies Ltd.
*   Tata Consultancy Services Ltd.
*   Tech Mahindra Ltd.

From this raw data, the following five fundamental variables are calculated:

*   Sales Growth
*   EBITDA Growth
*   EBITDA Margin Change
*   PAT Growth
*   PAT Margin Change

## Installation and Usage

To run the analysis, you will need to have Python 3 and the following libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels openpyxl
```

Once the dependencies are installed, you can run the analysis by executing the following command in your terminal:

```bash
python run_analysis.py
```

The script will perform the analysis and save the results in the `output` directory.

## Results and Findings

The analysis generated a correlation matrix and a set of linear regression models for each company. The key findings are summarized below.

### Correlation Results

*   **Wipro:** Shows a strong positive correlation between stock returns and both PAT Margin Change (0.627) and EBITDA Margin Change (0.565). It also has a strong negative correlation with Sales Growth (-0.501).
*   **TCS:** Has a moderate negative correlation with Sales Growth (-0.450) and a moderate positive correlation with PAT Margin Change (0.364).
*   **Other Companies:** Show weaker and more mixed correlation patterns.

### Regression Results

The regression analysis, using a more statistically rigorous approach with the `statsmodels` library, found that **no variables were statistically significant at the 5% level**. This suggests that, based on the available data, there is not enough evidence to conclude that these specific fundamental variables have a strong, direct impact on stock returns.

This is a valuable finding in itself, as it indicates that other factors not included in this analysis may be more influential in driving stock prices.

## Project Structure

```
data_analyst/
├── src/
│   ├── main.py                    # Main script for the analysis
│   ├── constants.py                    # Defined stock symbols
│   ├── analysis/
│   │   └── analysis.py            # Functions for correlation and regression
│   └── utils/
│       └── data_loader.py         # Function for loading and cleaning data
├── run_analysis.py                # Entry point to run the analysis
├── data.xls                       # Input data file
├── output/                        # Directory for all output files
└── README.md                      # This file
```

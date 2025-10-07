import os
import pandas as pd
from src.utils.data_loader import load_data, clean_stock_data
from src.analysis.analysis import (
    calculate_fundamental_metrics,
    run_correlation_analysis,
    run_regression_analysis,
    create_visualizations
)

def main():
    print("="*80)
    print("STOCK PRICE vs FUNDAMENTAL VARIABLES ANALYSIS")
    print("="*80)
    
    print("\n1. LOADING AND PROCESSING DATA")
    print("-" * 40)
    print("Loading Excel data...")
    stock_raw, fund_raw = load_data()
    
    print("Processing stock price data...")
    stock_returns = clean_stock_data(stock_raw)
    
    print("Calculating fundamental variables...")
    fund_metrics = calculate_fundamental_metrics(fund_raw)
    print(f"Fundamental variables calculated: {list(fund_metrics.keys())}")
    
    print("\n2. CORRELATION ANALYSIS")
    print("-" * 40)
    print("Calculating correlations...")
    corr_df = run_correlation_analysis(stock_returns, fund_metrics)
    
    print("\n3. LINEAR REGRESSION ANALYSIS")
    print("-" * 40)
    print("Running linear regressions...")
    reg_results, top_3_vars = run_regression_analysis(stock_returns, fund_metrics)
    
    print("\n4. CREATING VISUALIZATIONS")
    print("-" * 40)
    print("Generating charts and graphs...")
    create_visualizations(corr_df, reg_results, top_3_vars)
    
    print("\n5. SAVING RESULTS")
    print("-" * 40)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    
    if not corr_df.empty:
        corr_df.to_csv(os.path.join(output_dir, 'complete_correlation_matrix.csv'))
        print("Correlation matrix saved.")
    
    reg_list = []
    for comp, results in reg_results.items():
        reg_list.append({
            'Company': comp,
            'R2_Score': results['r2'],
            'MSE': results['mse'],
            'Observations': results['n_obs'],
            'Features': ', '.join(results['features'])
        })
    
    reg_df = pd.DataFrame(reg_list)
    reg_df.to_csv(os.path.join(output_dir, 'complete_regression_results.csv'), index=False)
    print("Regression results saved.")
    
    if top_3_vars:
        top_vars_df = pd.DataFrame(top_3_vars)
        top_vars_df.to_csv(os.path.join(output_dir, 'top_3_significant_variables.csv'), index=False)
        print("Top 3 significant variables saved.")
    
    print("\n6. RESULTS SUMMARY")
    print("-" * 40)
    
    print("\nCORRELATION MATRIX:")
    print(corr_df.round(3))
    
    print(f"\nREGRESSION RESULTS:")
    print(reg_df.round(3))
    
    if top_3_vars:
        print(f"\nTOP 3 MOST SIGNIFICANT VARIABLES:")
        for i, var in enumerate(top_3_vars, 1):
            print(f"{i}. {var['variable']} ({var['company']})")
            print(f"   Coefficient: {var['coefficient']:.4f}, p-value: {var['p_value']:.3f}")
    else:
        print("\nNo variables found to be statistically significant at the 5% level.")
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
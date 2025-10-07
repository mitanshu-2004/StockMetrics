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
    print("Starting Stock vs Fundamentals Analysis...")
    
    stock_raw, fund_raw = load_data()
    stock_returns = clean_stock_data(stock_raw)
    fund_metrics = calculate_fundamental_metrics(fund_raw)
    
    print(f"Metrics calculated: {list(fund_metrics.keys())}")
    
    corr_df = run_correlation_analysis(stock_returns, fund_metrics)
    reg_results, top_3_vars, _ = run_regression_analysis(stock_returns, fund_metrics)
    
    create_visualizations(corr_df, reg_results, top_3_vars)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    
    corr_df.to_csv(os.path.join(output_dir, 'complete_correlation_matrix.csv'))
    
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
    
    if top_3_vars:
        top_vars_df = pd.DataFrame(top_3_vars)
        top_vars_df.to_csv(os.path.join(output_dir, 'top_3_significant_variables.csv'), index=False)
    
    print("\n--- Results Summary ---")
    print("\nCorrelation Description:")
    print(corr_df.describe().round(3))
    
    print("\nRegression Results:")
    for company, results in reg_results.items():
        print(f"\n{company}:")
        print(f"  R-squared: {results['r2']:.3f}, MSE: {results['mse']:.3f}, Observations: {results['n_obs']}")
        if results['significant_vars']:
            print("  Significant Variables (p<0.05):")
            for var, p_val in results['significant_vars']:
                print(f"    - {var} (p-value: {p_val:.3f})")
        else:
            print("  No significant variables found.")

    print("\nAnalysis complete. Results saved to 'output' directory.")

if __name__ == "__main__":
    main()
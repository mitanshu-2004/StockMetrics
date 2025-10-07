from utils.data_loader import load_data, clean_stock_data, clean_fund_data
from analysis.correlation_analysis import calc_correlations
from analysis.regression_analysis import run_regressions
import pandas as pd
import os

def create_summary():
    print("Running analysis...")
    
    # Load and process data
    stock_raw, fund_raw = load_data()
    stock_returns = clean_stock_data(stock_raw)
    fund_metrics = clean_fund_data(fund_raw)
    corr_results = calc_correlations(stock_returns, fund_metrics)
    reg_results = run_regressions(stock_returns, fund_metrics)
    
    print("\n" + "="*60)
    print("STOCK ANALYSIS SUMMARY")
    print("="*60)
    
    # Correlation summary
    print("\nCORRELATION MATRIX:")
    print("-" * 30)
    
    companies = list(corr_results.keys())
    if companies:
        metrics = list(corr_results[companies[0]].keys())
        corr_df = pd.DataFrame(index=companies, columns=metrics)
        
        for comp, corrs in corr_results.items():
            for metric, corr in corrs.items():
                corr_df.loc[comp, metric] = corr
        
        print(corr_df.round(3))
        
        # Average correlations
        print(f"\nAverage Correlations:")
        avg_corrs = corr_df.mean(axis=0)
        for metric, avg_corr in avg_corrs.items():
            print(f"  {metric}: {avg_corr:.3f}")
    
    # Regression summary
    print(f"\nREGRESSION RESULTS:")
    print("-" * 30)
    
    reg_list = []
    for comp, results in reg_results.items():
        reg_list.append({
            'Company': comp,
            'R2_Score': results['r2'],
            'MSE': results['mse'],
            'Observations': results['n_obs']
        })
    
    reg_df = pd.DataFrame(reg_list)
    print(reg_df.round(3))
    
    # Top performers
    print(f"\nTOP PERFORMING COMPANIES (by R²):")
    print("-" * 40)
    top_comps = reg_df.sort_values('R2_Score', ascending=False)
    print(top_comps.round(3))
    
    # Save results
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, 'output')
    
    corr_df.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
    reg_df.to_csv(os.path.join(output_dir, 'regression_results.csv'))
    
    print(f"\nResults saved to output/ directory")

if __name__ == "__main__":
    create_summary()